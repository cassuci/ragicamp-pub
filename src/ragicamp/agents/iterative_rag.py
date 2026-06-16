"""Iterative RAG Agent - Multi-turn query refinement with batched operations.

Batched architecture (one model load per phase per iteration):
- Each iteration: load embedder → batch encode ALL active queries → unload
                   → batch search → load generator → batch sufficiency check
                   → batch refine insufficient → unload
- Final: load generator → batch generate ALL answers → unload

Queries that converge ("sufficient") drop out of the active set each iteration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    BATCH_GENERATE,
    BATCH_REFINE,
    BATCH_SUFFICIENCY,
    EVALUATE_SUFFICIENCY,
    GENERATE,
    REFINE_QUERY,
)
from ragicamp.core.types import Document, SearchBackend
from ragicamp.models.providers import EmbedderProvider, GeneratorProvider
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder, PromptConfig

logger = get_logger(__name__)

# Prompts for iterative refinement
SUFFICIENCY_PROMPT = """Based on the following context, determine if it contains enough information to answer the question.

Context:
{context}

Question: {query}

Instructions:
- If the context provides sufficient information to answer the question, respond with: SUFFICIENT
- If the context is missing key information needed to answer, respond with: INSUFFICIENT
- Then briefly explain your reasoning in one sentence.

Response:"""

REFINEMENT_PROMPT = """The retrieved context was insufficient to answer the question. Generate a refined search query that might find the missing information.

Original Question: {query}
Previous Query: {previous_query}

Retrieved Context (insufficient):
{context}

What key information is missing? Generate a better search query to find it.
Be specific and focus on the missing information, not what was already found.

Refined Query:"""


# ---------------------------------------------------------------------------
# Per-query state tracker used during the batched iteration loop
# ---------------------------------------------------------------------------


@dataclass
class _QueryState:
    """Mutable state for a single query across iterations."""

    query: Query
    current_text: str  # evolves as queries are refined
    docs: list[Document] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    iterations_info: list[dict[str, Any]] = field(default_factory=list)
    stopped_reason: str | None = None


class IterativeRAGAgent(Agent):
    """Iterative RAG agent with multi-turn query refinement.

    Processes ALL queries together at each iteration using batched model
    operations, then splits into "sufficient" (done) and "needs refinement"
    (continue) groups.

    Model loads per experiment (max_iterations=2, N queries):
        - ~3 embedder loads  (was N * 3)
        - ~4 generator loads (was N * 5)
    """

    def __init__(
        self,
        name: str,
        embedder_provider: EmbedderProvider,
        generator_provider: GeneratorProvider,
        index: SearchBackend,
        top_k: int = 5,
        max_iterations: int = 2,
        stop_on_sufficient: bool = True,
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
            top_k: Documents to retrieve per iteration
            max_iterations: Maximum refinement iterations
            stop_on_sufficient: Stop early if context is sufficient
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
        self.max_iterations = max_iterations
        self.stop_on_sufficient = stop_on_sufficient
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
        """Process queries with batched iterative refinement.

        Each iteration is fully batched: one embedder load, one generator
        load.  Queries that converge drop out of the active set.
        """
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
            "IterativeRAG: Processing %d queries (max %d iterations, batched)",
            len(pending),
            self.max_iterations,
        )
        _run_t0 = _pc()

        # --- Initialise per-query state ---
        active: dict[int, _QueryState] = {
            q.idx: _QueryState(query=q, current_text=q.text) for q in pending
        }
        converged: dict[int, _QueryState] = {}

        # NOTE: We intentionally do NOT pre-load embedder + generator together.
        # Each phase loads/unloads models independently so they don't compete
        # for GPU memory on single-GPU setups (generator + embedder both at 95%
        # would OOM).  This means each iteration does a load/unload cycle, but
        # it's the only safe approach for single-GPU.

        # --- Iteration loop (batched) ---
        for iteration in range(self.max_iterations):
            if not active:
                break

            logger.info(
                "=== Iteration %d: %d active queries ===",
                iteration,
                len(active),
            )
            _iter_t0 = _pc()

            # Phase 1: Batch embed + search  (embedder load/unload)
            with ExitStack() as stack:
                if self.reranker_provider is not None:
                    stack.enter_context(self.reranker_provider.load())
                _phase_t0 = _pc()
                self._phase_embed_and_search(active, iteration)
                logger.info(
                    "Iteration %d embed+search completed in %.1fs", iteration, _pc() - _phase_t0
                )

            # Last iteration: skip refinement, go straight to final generation
            if iteration == self.max_iterations - 1:
                for state in active.values():
                    state.iterations_info[-1]["stopped"] = "max_iterations"
                break

            # Phase 2: Batch sufficiency check + refine  (generator load/unload)
            _phase_t0 = _pc()
            if self.stop_on_sufficient:
                newly_sufficient = self._phase_evaluate_and_refine(
                    active,
                    iteration,
                )
                # Move converged queries out of the active set
                for idx in newly_sufficient:
                    converged[idx] = active.pop(idx)
            else:
                # No sufficiency check -- just refine all
                self._phase_refine_only(active, iteration)
            logger.info(
                "Iteration %d evaluate+refine completed in %.1fs",
                iteration,
                _pc() - _phase_t0,
            )

            logger.info("Iteration %d total: %.1fs", iteration, _pc() - _iter_t0)

        # Merge remaining active into converged
        converged.update(active)

        # Phase 3: Batch generate ALL final answers  (generator load/unload)
        logger.info("=== Final generation: %d queries ===", len(converged))
        _phase_t0 = _pc()
        new_results = self._phase_generate_answers(converged, pending)
        logger.info("Final generation completed in %.1fs", _pc() - _phase_t0)

        # Stream results and checkpoint
        for result in new_results:
            results.append(result)
            if on_result:
                on_result(result)

        if checkpoint_path:
            self._save_checkpoint(results, checkpoint_path)

        logger.info("Completed %d queries in %.1fs", len(new_results), _pc() - _run_t0)
        return results

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _phase_embed_and_search(
        self,
        active: dict[int, _QueryState],
        iteration: int,
    ) -> None:
        """Batch encode + search for all active queries.  Updates state in-place."""
        idx_order = list(active.keys())
        texts = [active[idx].current_text for idx in idx_order]

        # When reranking, retrieve more docs (fetch_k) then trim after rerank.
        use_reranker = self.reranker_provider is not None
        retrieve_k = self.fetch_k if use_reranker else self.top_k

        # Only use retrieval cache on iteration 0 (original queries).
        # Subsequent iterations have LLM-refined queries that won't hit cache.
        use_cache = iteration == 0 and self.retrieval_store is not None

        # IMPORTANT: Query transforms (HyDE, multiquery) are only applied on
        # iteration 0.  On later iterations the queries have been LLM-refined
        # and are already document-like — applying HyDE again is both redundant
        # (the refined query IS a hypothetical answer) and expensive (extra
        # generator load per iteration).  See docs/AGENT_PERFORMANCE_ANALYSIS.md.
        use_transformer = self.query_transformer if iteration == 0 else None
        if iteration > 0 and self.query_transformer is not None:
            logger.debug(
                "Skipping query transform on iteration %d (only applied on iteration 0)",
                iteration,
            )

        retrievals, embed_search_steps = batch_transform_embed_and_search(
            query_transformer=use_transformer,
            embedder_provider=self.embedder_provider,
            index=self.index,
            query_texts=texts,
            top_k=retrieve_k,
            is_hybrid=self._is_hybrid,
            retrieval_store=self.retrieval_store if use_cache else None,
            retriever_name=self.retriever_name if use_cache else None,
        )
        # Apply cross-encoder reranking if configured
        if use_reranker:
            retrievals, rerank_step = apply_reranking(
                self.reranker_provider,
                texts,
                retrievals,
                top_k=self.top_k,
                fetch_k=self.fetch_k,
            )
            embed_search_steps.append(rerank_step)

        # Broadcast embed/search steps (and optional transform step) to each query
        for idx in idx_order:
            active[idx].steps.extend(embed_search_steps)

        # Merge results into per-query state
        for idx, results in zip(idx_order, retrievals, strict=True):
            state = active[idx]
            new_docs = [r.document for r in results]
            state.docs = self._merge_documents(state.docs, new_docs)

            state.iterations_info.append(
                {
                    "iteration": iteration,
                    "query": state.current_text,
                    "docs_retrieved": len(new_docs),
                    "total_docs": len(state.docs),
                }
            )

    def _batch_refine(
        self,
        generator,
        idx_list: list[int],
        active: dict[int, _QueryState],
        iteration: int,
    ) -> None:
        """Batch refine a set of active queries using the generator.

        Shared helper for both _phase_evaluate_and_refine and _phase_refine_only.
        """
        refine_prompts = []
        for idx in idx_list:
            state = active[idx]
            context = ContextFormatter.format_numbered_from_docs(state.docs[: self.top_k * 2])
            refine_prompts.append(
                REFINEMENT_PROMPT.format(
                    query=state.query.text,
                    previous_query=state.current_text,
                    context=context[: Defaults.MAX_CONTEXT_CHARS],
                )
            )

        with StepTimer(BATCH_REFINE, model=self.generator_provider.model_name) as step:
            refined = generator.batch_generate(refine_prompts, max_tokens=100)
            step.input = {"n_queries": len(refine_prompts), "iteration": iteration}

        for idx, new_text in zip(idx_list, refined, strict=True):
            new_text = new_text.strip()
            if new_text.startswith('"') and new_text.endswith('"'):
                new_text = new_text[1:-1]
            state = active[idx]
            state.current_text = new_text
            state.steps.append(
                Step(
                    type=REFINE_QUERY,
                    input={"original_query": state.query.text, "iteration": iteration},
                    output={"refined_query": new_text},
                    model=self.generator_provider.model_name,
                )
            )

    def _phase_evaluate_and_refine(
        self,
        active: dict[int, _QueryState],
        iteration: int,
    ) -> list[int]:
        """Batch sufficiency check, then batch refine insufficient queries.

        Returns list of idx that converged (sufficient).
        """
        idx_order = list(active.keys())
        if not idx_order:
            return []

        sufficient_ids: list[int] = []

        with self.generator_provider.load() as generator:
            # --- Batch sufficiency check ---
            suff_prompts = []
            for idx in idx_order:
                state = active[idx]
                # C2 fix: use best-scored docs, not insertion order
                scored_docs = sorted(
                    state.docs, key=lambda d: getattr(d, "score", 0.0), reverse=True
                )
                context = ContextFormatter.format_numbered_from_docs(scored_docs[: self.top_k * 2])
                suff_prompts.append(
                    SUFFICIENCY_PROMPT.format(context=context, query=state.query.text)
                )

            with StepTimer(BATCH_SUFFICIENCY, model=self.generator_provider.model_name) as step:
                suff_responses = generator.batch_generate(suff_prompts, max_tokens=100)
                step.input = {"n_queries": len(suff_prompts), "iteration": iteration}

            # Parse responses
            for idx, response in zip(idx_order, suff_responses, strict=True):
                resp_upper = response.upper()
                is_sufficient = "SUFFICIENT" in resp_upper and "INSUFFICIENT" not in resp_upper
                state = active[idx]
                state.iterations_info[-1]["sufficient"] = is_sufficient
                state.steps.append(
                    Step(
                        type=EVALUATE_SUFFICIENCY,
                        input={"query": state.query.text, "iteration": iteration},
                        output={"sufficient": is_sufficient},
                        model=self.generator_provider.model_name,
                    )
                )

                if is_sufficient:
                    state.iterations_info[-1]["stopped"] = "sufficient"
                    sufficient_ids.append(idx)

            # --- Batch refine remaining (those NOT sufficient) ---
            needs_refine = [idx for idx in idx_order if idx not in sufficient_ids]

            if needs_refine and iteration < self.max_iterations:
                self._batch_refine(generator, needs_refine, active, iteration)

        return sufficient_ids

    def _phase_refine_only(
        self,
        active: dict[int, _QueryState],
        iteration: int,
    ) -> None:
        """Batch refine all active queries (no sufficiency check)."""
        idx_order = list(active.keys())
        if not idx_order or iteration >= self.max_iterations:
            return

        with self.generator_provider.load() as generator:
            self._batch_refine(generator, idx_order, active, iteration)

    def _phase_generate_answers(
        self,
        all_states: dict[int, _QueryState],
        original_order: list[Query],
    ) -> list[AgentResult]:
        """Batch generate final answers for all queries."""
        idx_order = list(all_states.keys())

        # Build prompts
        prompts: list[str] = []
        final_docs_per_query: list[list[Document]] = []
        for idx in idx_order:
            state = all_states[idx]
            final_docs = state.docs[: self.top_k * 2]
            final_docs_per_query.append(final_docs)
            context = ContextFormatter.format_numbered_from_docs(final_docs)
            prompts.append(self.prompt_builder.build_rag(state.query.text, context))

        # Batch generate
        with self.generator_provider.load() as generator:
            with StepTimer(BATCH_GENERATE, model=self.generator_provider.model_name) as step:
                answers = generator.batch_generate(prompts)
                step.input = {"n_queries": len(prompts)}
                step.output = {"n_answers": len(answers)}

        # Build AgentResult list in original query order
        result_map: dict[int, AgentResult] = {}
        for idx, answer, prompt, final_docs in zip(
            idx_order, answers, prompts, final_docs_per_query, strict=True
        ):
            state = all_states[idx]
            state.steps.append(
                Step(
                    type=GENERATE,
                    input={"query": state.query.text},
                    output=answer,
                    model=self.generator_provider.model_name,
                )
            )

            retrieved_docs_info = [
                RetrievedDocInfo(
                    rank=i + 1,
                    doc_id=doc.id if hasattr(doc, "id") else None,
                    content=doc.text if hasattr(doc, "text") else str(doc),
                    score=None,
                )
                for i, doc in enumerate(final_docs)
            ]

            result_map[idx] = AgentResult(
                query=state.query,
                answer=answer,
                steps=state.steps,
                prompt=prompt,
                retrieved_docs=retrieved_docs_info,
                metadata={
                    "iterations": state.iterations_info,
                    "total_docs": len(state.docs),
                    "final_docs_used": len(final_docs),
                },
            )

        # Return in original query order
        return [result_map[q.idx] for q in original_order if q.idx in result_map]

    @staticmethod
    def _merge_documents(
        existing: list[Document],
        new_docs: list[Document],
    ) -> list[Document]:
        """Merge documents, removing duplicates based on content."""
        seen_content: set[str] = set()
        merged: list[Document] = []

        for doc in existing + new_docs:
            content_key = hash(doc.text) if doc.text else ""
            if content_key not in seen_content:
                seen_content.add(content_key)
                merged.append(doc)

        return merged
