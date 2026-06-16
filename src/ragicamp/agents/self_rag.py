"""Self-RAG Agent - Adaptive retrieval with batched operations.

Batched architecture (three clean phases, minimal model loads):
  Phase 1 (Assess):   Load generator → batch assess ALL queries → unload
  Phase 2 (Retrieve): Load embedder  → batch encode retrieval group → unload
                       → batch search
  Phase 3 (Generate): Load generator → batch generate ALL (RAG + direct)
                       → optionally batch verify RAG answers
                       → batch regenerate fallbacks → unload
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragicamp.cache.retrieval_store import RetrievalStore
    from ragicamp.models.providers.reranker import RerankerProvider
    from ragicamp.rag.query_transform.base import QueryTransformer

from ragicamp.agents.base import (
    Agent,
    AgentResult,
    Query,
    RetrievedDocInfo,
    Step,
    StepTimer,
    apply_reranking,
    batch_transform_embed_and_search,
    is_hybrid_searcher,
)
from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.core.step_types import (
    ASSESS_RETRIEVAL,
    BATCH_ASSESS,
    BATCH_FALLBACK_GENERATE,
    BATCH_GENERATE,
    BATCH_VERIFY,
    FALLBACK_GENERATE,
    GENERATE,
    VERIFY,
)
from ragicamp.core.types import Document, SearchBackend
from ragicamp.models.providers import EmbedderProvider, GeneratorProvider
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder, PromptConfig

logger = get_logger(__name__)


# Prompts for self-RAG decision making
RETRIEVAL_DECISION_PROMPT = """Analyze this question and decide if you need to retrieve external information to answer it accurately.

Question: {query}

Consider:
- Is this asking for factual information that could be outdated or specific?
- Is this a general knowledge question you're confident about?
- Would external sources improve the answer's accuracy?

Respond with a confidence score from 0.0 to 1.0 indicating how confident you are that you can answer WITHOUT retrieval.
- 1.0 = Completely confident, no retrieval needed
- 0.5 = Uncertain, retrieval would help
- 0.0 = Definitely need retrieval

Format: CONFIDENCE: [score]
Then briefly explain your reasoning.

Response:"""

VERIFICATION_PROMPT = """Verify if the following answer is supported by the provided context.

Context:
{context}

Question: {query}

Answer: {answer}

Is this answer:
1. SUPPORTED - The answer is directly supported by information in the context
2. PARTIALLY_SUPPORTED - The answer is somewhat related but adds information not in context
3. NOT_SUPPORTED - The answer contradicts or is unrelated to the context

Respond with one of: SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED
Then briefly explain.

Verification:"""


# ---------------------------------------------------------------------------
# Per-query state tracker
# ---------------------------------------------------------------------------


@dataclass
class _QueryState:
    """Mutable state for a single query through the three phases."""

    query: Query
    confidence: float = 0.3
    needs_retrieval: bool = True
    docs: list[Document] = field(default_factory=list)
    docs_info: list[RetrievedDocInfo] = field(default_factory=list)
    context_text: str = ""
    prompt: str = ""
    answer: str = ""
    verification: str | None = None
    steps: list[Step] = field(default_factory=list)


class SelfRAGAgent(Agent):
    """Self-RAG agent with adaptive retrieval decision.

    Processes ALL queries in three batched phases:
      1. Assess  -- one generator load to decide retrieval need for ALL queries
      2. Retrieve -- one embedder load for the retrieval group
      3. Generate -- one generator load for ALL answers, verification, and fallbacks

    Model loads per experiment (N queries):
        - 1 embedder load  (was up to N)
        - 2 generator loads (was up to 3N)
    """

    def __init__(
        self,
        name: str,
        embedder_provider: EmbedderProvider,
        generator_provider: GeneratorProvider,
        index: SearchBackend,
        top_k: int = 5,
        retrieval_threshold: float = 0.5,
        verify_answer: bool = False,
        fallback_to_direct: bool = True,
        prompt_builder: PromptBuilder | None = None,
        retrieval_store: RetrievalStore | None = None,
        retriever_name: str | None = None,
        reranker_provider: RerankerProvider | None = None,
        fetch_k: int | None = None,
        query_transformer: QueryTransformer | None = None,
    ):
        """Initialize agent with providers.

        Args:
            name: Agent identifier
            embedder_provider: Provides embedder with lazy loading
            generator_provider: Provides generator with lazy loading
            index: Search backend (VectorIndex, HybridSearcher, etc.)
            top_k: Number of documents to retrieve
            retrieval_threshold: Confidence threshold for skipping retrieval
            verify_answer: Whether to verify answer is grounded
            fallback_to_direct: If verification fails, fall back to direct
            prompt_builder: For building prompts
            retrieval_store: Optional RetrievalStore for caching retrieval results
            retriever_name: Retriever identifier for cache keys
            reranker_provider: Optional RerankerProvider for cross-encoder reranking
            fetch_k: Documents to retrieve before reranking (None = same as top_k)
            query_transformer: Optional QueryTransformer for query expansion
        """
        super().__init__(name)

        self.embedder_provider = embedder_provider
        self.generator_provider = generator_provider
        self.index = index
        self.top_k = top_k
        self.retrieval_threshold = retrieval_threshold
        self.verify_answer = verify_answer
        self.fallback_to_direct = fallback_to_direct
        self.prompt_builder = prompt_builder or PromptBuilder(PromptConfig())
        self.retrieval_store = retrieval_store
        self.retriever_name = retriever_name
        self.reranker_provider = reranker_provider
        self.fetch_k = fetch_k if fetch_k is not None else top_k
        self.query_transformer = query_transformer

        # Detect hybrid searcher (needs query text for sparse leg)
        self._is_hybrid = is_hybrid_searcher(index)

    def run(
        self,
        queries: list[Query],
        *,
        on_result: Callable[[AgentResult], None] | None = None,
        checkpoint_path: Path | None = None,
        show_progress: bool = True,
    ) -> list[AgentResult]:
        """Process queries with batched adaptive retrieval."""
        # Load checkpoint if resuming
        results, completed_idx = [], set()
        if checkpoint_path:
            results, completed_idx = self._load_checkpoint(checkpoint_path)

        pending = [q for q in queries if q.idx not in completed_idx]

        if not pending:
            logger.info("All queries already completed")
            return results

        from contextlib import ExitStack
        from time import perf_counter as _pc

        logger.info(
            "SelfRAG: Processing %d queries (threshold=%.2f, batched)",
            len(pending),
            self.retrieval_threshold,
        )
        _run_t0 = _pc()

        # Initialise per-query state
        states: dict[int, _QueryState] = {q.idx: _QueryState(query=q) for q in pending}

        # NOTE: We intentionally do NOT pre-load the generator in an ExitStack.
        # Each phase loads/unloads models independently so they don't compete
        # for GPU memory on single-GPU setups (generator + embedder both at 95%
        # would OOM).

        # Phase 1: Batch assess  (generator load/unload)
        _phase_t0 = _pc()
        self._phase_assess(states)
        logger.info("Phase 1 (assess) completed in %.1fs", _pc() - _phase_t0)

        # Split into retrieval / direct groups
        retrieval_group = {idx: s for idx, s in states.items() if s.needs_retrieval}
        direct_group = {idx: s for idx, s in states.items() if not s.needs_retrieval}

        logger.info(
            "SelfRAG split: %d retrieval, %d direct",
            len(retrieval_group),
            len(direct_group),
        )

        # Phase 2: Batch retrieve  (embedder load/unload, only for retrieval group)
        if retrieval_group:
            with ExitStack() as stack:
                if self.reranker_provider is not None:
                    stack.enter_context(self.reranker_provider.load())
                _phase_t0 = _pc()
                self._phase_retrieve(retrieval_group)
                logger.info("Phase 2 (retrieve) completed in %.1fs", _pc() - _phase_t0)

        # Build prompts for all queries
        for state in retrieval_group.values():
            state.prompt = self.prompt_builder.build_rag(state.query.text, state.context_text)
        for state in direct_group.values():
            state.prompt = self.prompt_builder.build_direct(state.query.text)

        # Phase 3: Batch generate + verify + fallback  (generator load/unload)
        _phase_t0 = _pc()
        self._phase_generate_and_verify(states, retrieval_group, direct_group)
        logger.info("Phase 3 (generate+verify) completed in %.1fs", _pc() - _phase_t0)

        # Build results in original query order
        new_results = self._build_results(states, pending)

        for result in new_results:
            results.append(result)
            if on_result:
                on_result(result)

        if checkpoint_path:
            self._save_checkpoint(results, checkpoint_path)

        logger.info("Completed %d queries in %.1fs", len(new_results), _pc() - _run_t0)
        return results

    # ------------------------------------------------------------------
    # Phase 1: Assess
    # ------------------------------------------------------------------

    def _phase_assess(self, states: dict[int, _QueryState]) -> None:
        """Batch assess retrieval need for all queries."""
        idx_order = list(states.keys())
        prompts = [
            RETRIEVAL_DECISION_PROMPT.format(query=states[idx].query.text) for idx in idx_order
        ]

        with self.generator_provider.load() as generator:
            with StepTimer(BATCH_ASSESS, model=self.generator_provider.model_name) as step:
                responses = generator.batch_generate(prompts, max_tokens=150)
                step.input = {"n_queries": len(prompts)}
                step.output = {"n_responses": len(responses)}

        for idx, response in zip(idx_order, responses, strict=True):
            confidence = self._parse_confidence(response)
            state = states[idx]
            state.confidence = confidence
            state.needs_retrieval = confidence <= self.retrieval_threshold

            state.steps.append(
                Step(
                    type=ASSESS_RETRIEVAL,
                    input={"query": state.query.text},
                    output={"confidence": confidence},
                    model=self.generator_provider.model_name,
                )
            )

            logger.debug(
                "SelfRAG: q=%d confidence=%.2f -> %s",
                idx,
                confidence,
                "retrieve" if state.needs_retrieval else "direct",
            )

    # ------------------------------------------------------------------
    # Phase 2: Retrieve
    # ------------------------------------------------------------------

    def _phase_retrieve(self, retrieval_group: dict[int, _QueryState]) -> None:
        """Batch embed + search for the retrieval group."""
        idx_order = list(retrieval_group.keys())
        texts = [retrieval_group[idx].query.text for idx in idx_order]

        # When reranking, retrieve more docs (fetch_k) then trim after rerank
        retrieve_k = self.fetch_k if self.reranker_provider else self.top_k

        retrievals, embed_search_steps = batch_transform_embed_and_search(
            query_transformer=self.query_transformer,
            embedder_provider=self.embedder_provider,
            index=self.index,
            query_texts=texts,
            top_k=retrieve_k,
            is_hybrid=self._is_hybrid,
            retrieval_store=self.retrieval_store,
            retriever_name=self.retriever_name,
        )

        # Apply cross-encoder reranking if configured
        if self.reranker_provider is not None:
            retrievals, rerank_step = apply_reranking(
                self.reranker_provider,
                texts,
                retrievals,
                top_k=self.top_k,
                fetch_k=self.fetch_k,
            )
            embed_search_steps.append(rerank_step)

        # Broadcast all embed/search steps (and optional transform step) to each query
        for idx in idx_order:
            retrieval_group[idx].steps.extend(embed_search_steps)

        for idx, results in zip(idx_order, retrievals, strict=True):
            state = retrieval_group[idx]
            state.docs = [r.document for r in results]
            state.docs_info = RetrievedDocInfo.from_search_results(results)
            state.context_text = ContextFormatter.format_numbered_from_docs(state.docs)

    # ------------------------------------------------------------------
    # Phase 3: Generate + Verify + Fallback
    # ------------------------------------------------------------------

    def _phase_generate_and_verify(
        self,
        all_states: dict[int, _QueryState],
        retrieval_group: dict[int, _QueryState],
        direct_group: dict[int, _QueryState],
    ) -> None:
        """Batch generate answers, optionally verify and fallback."""
        idx_order = list(all_states.keys())
        prompts = [all_states[idx].prompt for idx in idx_order]

        with self.generator_provider.load() as generator:
            # Batch generate ALL answers
            with StepTimer(BATCH_GENERATE, model=self.generator_provider.model_name) as step:
                answers = generator.batch_generate(prompts)
                step.input = {"n_queries": len(prompts)}
                step.output = {"n_answers": len(answers)}

            for idx, answer in zip(idx_order, answers, strict=True):
                state = all_states[idx]
                state.answer = answer
                state.steps.append(
                    Step(
                        type=GENERATE,
                        input={
                            "query": state.query.text,
                            "with_context": state.needs_retrieval,
                        },
                        output=answer,
                        model=self.generator_provider.model_name,
                    )
                )

            # Batch verify RAG answers (only retrieval group, if enabled)
            if self.verify_answer and retrieval_group:
                self._batch_verify_and_fallback(generator, retrieval_group)

    def _batch_verify_and_fallback(
        self,
        generator,
        retrieval_group: dict[int, _QueryState],
    ) -> None:
        """Batch verify RAG answers and regenerate fallbacks if needed."""
        idx_order = list(retrieval_group.keys())

        # Build verification prompts
        verify_prompts = []
        for idx in idx_order:
            state = retrieval_group[idx]
            verify_prompts.append(
                VERIFICATION_PROMPT.format(
                    context=state.context_text[: Defaults.MAX_VERIFICATION_CONTEXT_CHARS],
                    query=state.query.text,
                    answer=state.answer,
                )
            )

        with StepTimer(BATCH_VERIFY, model=self.generator_provider.model_name) as step:
            verify_responses = generator.batch_generate(verify_prompts, max_tokens=100)
            step.input = {"n_queries": len(verify_prompts)}

        # Parse verification results
        fallback_ids: list[int] = []
        for idx, response in zip(idx_order, verify_responses, strict=True):
            verdict = self._parse_verification(response)
            state = retrieval_group[idx]
            state.verification = verdict
            state.steps.append(
                Step(
                    type=VERIFY,
                    input={"answer": state.answer[:100]},
                    output={"verification": verdict},
                    model=self.generator_provider.model_name,
                )
            )

            if verdict == "NOT_SUPPORTED" and self.fallback_to_direct:
                fallback_ids.append(idx)

        # Batch regenerate fallbacks as direct (same generator load)
        if fallback_ids:
            logger.debug(
                "SelfRAG: %d answers not supported, falling back to direct", len(fallback_ids)
            )
            fallback_prompts = [
                self.prompt_builder.build_direct(retrieval_group[idx].query.text)
                for idx in fallback_ids
            ]

            with StepTimer(
                BATCH_FALLBACK_GENERATE, model=self.generator_provider.model_name
            ) as step:
                fallback_answers = generator.batch_generate(fallback_prompts)
                step.input = {"n_queries": len(fallback_prompts)}

            for idx, answer in zip(fallback_ids, fallback_answers, strict=True):
                state = retrieval_group[idx]
                state.answer = answer
                state.prompt = fallback_prompts[fallback_ids.index(idx)]
                state.docs = []
                state.docs_info = []
                state.context_text = ""
                state.steps.append(
                    Step(
                        type=FALLBACK_GENERATE,
                        input={"query": state.query.text, "fallback": True},
                        output=answer,
                        model=self.generator_provider.model_name,
                    )
                )

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    def _build_results(
        self,
        states: dict[int, _QueryState],
        original_order: list[Query],
    ) -> list[AgentResult]:
        """Build AgentResult list in original query order."""
        result_map: dict[int, AgentResult] = {}

        for idx, state in states.items():
            result_map[idx] = AgentResult(
                query=state.query,
                answer=state.answer,
                steps=state.steps,
                prompt=state.prompt,
                retrieved_docs=state.docs_info if state.docs_info else None,
                metadata={
                    "used_retrieval": state.needs_retrieval,
                    "confidence": state.confidence,
                    "num_docs": len(state.docs),
                    "verification": state.verification,
                },
            )

        return [result_map[q.idx] for q in original_order if q.idx in result_map]

    # ------------------------------------------------------------------
    # Parsing helpers (unchanged logic from original)
    # ------------------------------------------------------------------

    _CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

    @classmethod
    def _parse_confidence(cls, response: str) -> float:
        """Parse confidence score from generator response."""
        match = cls._CONFIDENCE_RE.search(response)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        return 0.3  # Default: uncertain, should retrieve

    @staticmethod
    def _parse_verification(response: str) -> str:
        """Parse verification verdict from generator response."""
        response_upper = response.upper()
        if "NOT_SUPPORTED" in response_upper:
            return "NOT_SUPPORTED"
        elif "PARTIALLY_SUPPORTED" in response_upper:
            return "PARTIALLY_SUPPORTED"
        elif "SUPPORTED" in response_upper:
            return "SUPPORTED"
        else:
            return "UNKNOWN"
