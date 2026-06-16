"""Base classes for RAG agents.

Clean Architecture Design:
- Agents receive providers (not loaded models)
- Agents manage their own GPU lifecycle via context managers
- Each agent optimizes its own loading/batching strategy
- All intermediate steps are captured for analysis
- Simple interface: agent.run(queries) → results
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from ragicamp.core.logging import get_logger

if TYPE_CHECKING:
    from ragicamp.cache.retrieval_store import RetrievalStore
    from ragicamp.rag.query_transform.base import QueryTransformer
from ragicamp.core.step_types import BATCH_ENCODE, BATCH_SEARCH, QUERY_TRANSFORM, RERANK

logger = get_logger(__name__)


# =============================================================================
# Core Data Types
# =============================================================================


@dataclass
class Query:
    """Input query for agent processing.

    Attributes:
        idx: Unique index for ordering and checkpointing
        text: The query text
        expected: Expected answers for evaluation (optional)
    """

    idx: int
    text: str
    expected: list[str] | None = None


@dataclass
class Step:
    """Intermediate step in agent processing.

    Every operation (retrieval, generation, reranking, etc.) is logged
    for later analysis.
    """

    type: str  # "retrieve", "generate", "rerank", "hyde", "encode", etc.
    input: Any = None
    output: Any = None
    timing_ms: float = 0.0
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StepTimer:
    """Context manager for timing steps.

    Usage:
        with StepTimer("retrieve", model="bge-large") as step:
            docs = retriever.retrieve(query)
            step.output = docs

    Automatically logs the step type and elapsed time on exit.
    """

    def __init__(self, step_type: str, model: str | None = None, **metadata):
        self.step = Step(type=step_type, model=model, metadata=metadata)
        self._start: float = 0.0

    def __enter__(self) -> Step:
        self._start = perf_counter()
        return self.step

    def __exit__(self, *args):
        self.step.timing_ms = (perf_counter() - self._start) * 1000
        model_str = f" ({self.step.model})" if self.step.model else ""
        logger.info(
            "Step [%s]%s completed in %.1fs",
            self.step.type,
            model_str,
            self.step.timing_ms / 1000,
        )


@dataclass
class RetrievedDocInfo:
    """Retrieved document info for logging.

    Captures all stages of retrieval pipeline for analysis.
    """

    rank: int  # Final rank (1-indexed)
    doc_id: str | None = None
    content: str | None = None  # Can be truncated for disk space
    score: float | None = None  # Final score
    retrieval_score: float | None = None  # Before reranking
    retrieval_rank: int | None = None  # Before reranking
    rerank_score: float | None = None  # Cross-encoder score

    def to_dict(self) -> dict[str, Any]:
        d = {"rank": self.rank}
        if self.doc_id is not None:
            d["doc_id"] = self.doc_id
        if self.content is not None:
            d["content"] = self.content
        if self.score is not None:
            d["score"] = self.score
        if self.retrieval_score is not None:
            d["retrieval_score"] = self.retrieval_score
        if self.retrieval_rank is not None:
            d["retrieval_rank"] = self.retrieval_rank
        if self.rerank_score is not None:
            d["rerank_score"] = self.rerank_score
        return d

    @classmethod
    def from_search_results(cls, search_results: list) -> list[RetrievedDocInfo]:
        """Build a list of RetrievedDocInfo from SearchResult objects.

        This is the shared factory for all RAG agents, eliminating the
        previously duplicated construction logic.

        Args:
            search_results: List of SearchResult objects.

        Returns:
            List of RetrievedDocInfo in rank order.
        """
        return [
            cls(
                rank=i + 1,
                doc_id=sr.document.id if sr.document else None,
                content=sr.document.text if sr.document else None,
                score=sr.score,
                retrieval_score=sr.score,
                retrieval_rank=i + 1,
            )
            for i, sr in enumerate(search_results)
        ]


@dataclass
class AgentResult:
    """Result from processing a single query.

    Contains the answer and all intermediate steps for analysis.

    Attributes:
        query: The input query
        answer: Generated answer
        steps: Pipeline steps with timing
        prompt: Full prompt sent to LLM
        retrieved_docs: Structured retrieved doc info (RAG only)
        metadata: Additional metadata (flexible)
    """

    query: Query
    answer: str
    steps: list[Step] = field(default_factory=list)
    prompt: str | None = None
    retrieved_docs: list[RetrievedDocInfo] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_content: bool = True, max_content_len: int = 500) -> dict[str, Any]:
        """Serialize for checkpointing/storage.

        Args:
            include_content: Whether to include document content (disk space tradeoff)
            max_content_len: Max chars per doc content (0 = full)
        """
        d = {
            "idx": self.query.idx,
            "question": self.query.text,
            "answer": self.answer,
            "expected": self.query.expected,
            "prompt": self.prompt,
            "steps": [
                {
                    "type": s.type,
                    "timing_ms": s.timing_ms,
                    "model": s.model,
                    **({"metadata": s.metadata} if s.metadata else {}),
                }
                for s in self.steps
            ],
            "metadata": self.metadata,
        }

        # Add retrieved docs if present
        if self.retrieved_docs:
            docs_list = []
            for doc in self.retrieved_docs:
                doc_dict = doc.to_dict()
                # Optionally truncate content
                if "content" in doc_dict and include_content:
                    if max_content_len > 0 and len(doc_dict["content"]) > max_content_len:
                        doc_dict["content"] = doc_dict["content"][:max_content_len] + "..."
                elif not include_content:
                    doc_dict.pop("content", None)
                docs_list.append(doc_dict)
            d["retrieved_docs"] = docs_list

        return d


# =============================================================================
# Shared Utilities
# =============================================================================


def is_hybrid_searcher(index: Any) -> bool:
    """Check if a search backend is a HybridSearcher.

    Used by all RAG agents to decide whether to pass query texts
    alongside embeddings during search.
    """
    from ragicamp.retrievers.hybrid import HybridSearcher
    from ragicamp.retrievers.lazy import LazySearchBackend

    if isinstance(index, HybridSearcher):
        return True
    if isinstance(index, LazySearchBackend):
        return getattr(index, "_is_hybrid", False)
    return False


def batch_transform_embed_and_search(
    query_transformer: QueryTransformer | None,
    embedder_provider: Any,
    index: Any,
    query_texts: list[str],
    top_k: int,
    is_hybrid: bool,
    retrieval_store: RetrievalStore | None = None,
    retriever_name: str | None = None,
) -> tuple[list[list], list[Step]]:
    """Query-transform-aware wrapper around :func:`batch_embed_and_search`.

    If *query_transformer* is ``None`` this simply delegates to
    :func:`batch_embed_and_search` and returns the steps list.

    When a transformer is provided:
      1. Records a ``query_transform`` :class:`Step` with timing.
      2. Calls ``transformer.batch_transform(query_texts)`` to expand each
         original query into one-or-more search queries.
      3. Flattens the expanded queries, calls :func:`batch_embed_and_search`.
      4. Merges results back: for each original query, collects results
         from all its expansions, deduplicates by document content, sorts
         by score descending, and takes the top *top_k*.

    Args:
        query_transformer: Optional :class:`QueryTransformer` instance.
        embedder_provider: EmbedderProvider instance.
        index: Search backend (VectorIndex, HybridSearcher, etc.).
        query_texts: List of original user query strings.
        top_k: Number of results per original query.
        is_hybrid: Whether the index is a HybridSearcher.
        retrieval_store: Optional retrieval cache (disabled when transforming).
        retriever_name: Retriever identifier for cache keys.

    Returns:
        Tuple of (search_results_per_query, list_of_steps).
    """
    if query_transformer is None:
        retrievals, encode_step, search_step = batch_embed_and_search(
            embedder_provider=embedder_provider,
            index=index,
            query_texts=query_texts,
            top_k=top_k,
            is_hybrid=is_hybrid,
            retrieval_store=retrieval_store,
            retriever_name=retriever_name,
        )
        return retrievals, [encode_step, search_step]

    # --- Transform queries -----------------------------------------------
    with StepTimer(QUERY_TRANSFORM) as transform_step:
        transform_step.input = {"n_queries": len(query_texts)}
        expanded: list[list[str]] = query_transformer.batch_transform(query_texts)
        total_expanded = sum(len(eq) for eq in expanded)
        transform_step.output = {
            "total_expanded": total_expanded,
            "transformer": repr(query_transformer),
        }

    # Build flat list + index mapping (original_idx → slice range)
    flat_queries: list[str] = []
    boundaries: list[tuple[int, int]] = []  # (start, end) in flat list
    for eq_list in expanded:
        start = len(flat_queries)
        flat_queries.extend(eq_list)
        boundaries.append((start, len(flat_queries)))

    logger.info(
        "Query transform: %d queries -> %d expanded queries",
        len(query_texts),
        len(flat_queries),
    )

    # --- Embed + search expanded queries (no cache — transformed) --------
    flat_retrievals, encode_step, search_step = batch_embed_and_search(
        embedder_provider=embedder_provider,
        index=index,
        query_texts=flat_queries,
        top_k=top_k,
        is_hybrid=is_hybrid,
    )

    # --- Merge results back to original queries --------------------------
    merged_retrievals: list[list] = []
    for start, end in boundaries:
        # Collect all results from this query's expansions
        seen_contents: set[str] = set()
        candidates: list[tuple[float, Any]] = []  # (score, SearchResult)
        for i in range(start, end):
            for sr in flat_retrievals[i]:
                content_key = sr.document.text if sr.document and sr.document.text else ""
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    candidates.append((sr.score if sr.score is not None else 0.0, sr))

        # Sort by score descending, take top_k
        candidates.sort(key=lambda x: x[0], reverse=True)
        merged_retrievals.append([sr for _, sr in candidates[:top_k]])

    steps = [transform_step, encode_step, search_step]
    return merged_retrievals, steps


def batch_embed_and_search(
    embedder_provider: Any,
    index: Any,
    query_texts: list[str],
    top_k: int,
    is_hybrid: bool,
    retrieval_store: Any = None,
    retriever_name: str | None = None,
) -> tuple[list[list], Step, Step]:
    """Shared embed + search logic for all RAG agents.

    Loads the embedder, encodes queries, then searches the index.
    Supports dense, hybrid, and hierarchical search backends.

    When ``retrieval_store`` and ``retriever_name`` are provided, results
    are checked in the SQLite cache first.  Only cache misses trigger
    embedding + search.  This avoids redundant retrieval when multiple
    experiments share the same (retriever, queries, top_k) but differ
    only in LLM model or prompt template.

    Args:
        embedder_provider: EmbedderProvider instance.
        index: Search backend (VectorIndex, HybridSearcher, etc.).
        query_texts: List of query strings to encode and search.
        top_k: Number of results per query.
        is_hybrid: Whether the index is a HybridSearcher (needs query_texts).
        retrieval_store: Optional :class:`RetrievalStore` for caching results.
        retriever_name: Retriever identifier for cache keys (required if store is set).

    Returns:
        Tuple of (search_results, encode_step, search_step).
    """
    from time import perf_counter as _pc

    import numpy as np

    from ragicamp.core.types import Document, SearchResult

    # --- Check retrieval cache -------------------------------------------
    cached_results: list | None = None
    miss_indices: list[int] | None = None

    if retrieval_store is not None and retriever_name:
        _cache_t0 = _pc()
        cached_batch, hit_mask = retrieval_store.get_batch(
            retriever_name,
            query_texts,
            top_k,
        )
        _cache_lookup_s = _pc() - _cache_t0
        n_hits = sum(hit_mask)
        logger.info(
            "Retrieval cache lookup: %.2fs (%d queries)",
            _cache_lookup_s,
            len(query_texts),
        )

        if n_hits == len(query_texts):
            # 100% cache hit — skip embedding and search entirely
            logger.info(
                "Retrieval cache: 100%% hit (%d queries, retriever=%s, k=%d)",
                len(query_texts),
                retriever_name,
                top_k,
            )
            retrievals = [
                [
                    SearchResult(
                        document=Document.from_dict(sr["document"]),
                        score=sr["score"],
                        rank=sr.get("rank", 0),
                    )
                    for sr in result_dicts
                ]
                for result_dicts in cached_batch
            ]
            # Return synthetic steps (no actual work done)
            encode_step = Step(
                type=BATCH_ENCODE,
                model=embedder_provider.model_name,
                input={"n_queries": len(query_texts)},
                output={"cache_hit": True},
                timing_ms=0.0,
            )
            search_step = Step(
                type=BATCH_SEARCH,
                input={"n_queries": len(query_texts), "top_k": top_k},
                output={"n_results": sum(len(r) for r in retrievals), "cache_hit": True},
                timing_ms=0.0,
            )
            return retrievals, encode_step, search_step

        if n_hits > 0:
            # Partial hit — only embed+search the misses
            cached_results = cached_batch
            miss_indices = [i for i, hit in enumerate(hit_mask) if not hit]
            logger.info(
                "Retrieval cache: %d/%d hits, searching %d misses (retriever=%s, k=%d)",
                n_hits,
                len(query_texts),
                len(miss_indices),
                retriever_name,
                top_k,
            )
        else:
            miss_indices = None  # Treat as full miss (simpler code path)
            logger.info(
                "Retrieval cache: 0/%d hits, searching all (retriever=%s, k=%d)",
                len(query_texts),
                retriever_name,
                top_k,
            )

    # --- Determine which queries need embedding+search -------------------
    if miss_indices is not None:
        texts_to_search = [query_texts[i] for i in miss_indices]
    else:
        texts_to_search = query_texts

    # --- Embed -----------------------------------------------------------
    logger.info(
        "Embedding %d queries (model=%s)...",
        len(texts_to_search),
        embedder_provider.model_name,
    )
    with embedder_provider.load() as embedder:
        with StepTimer(BATCH_ENCODE, model=embedder_provider.model_name) as encode_step:
            encode_step.input = {"n_queries": len(texts_to_search)}
            embeddings = embedder.batch_encode(texts_to_search)
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.asarray(embeddings, dtype=np.float32)
            encode_step.output = {"embedding_shape": list(embeddings.shape)}
    logger.info("Embedding done in %.1fs", encode_step.timing_ms / 1000)

    # --- Search ----------------------------------------------------------
    logger.info(
        "Searching index (%d queries, top_k=%d)...",
        len(texts_to_search),
        top_k,
    )
    with StepTimer(BATCH_SEARCH) as search_step:
        search_step.input = {"n_queries": len(texts_to_search), "top_k": top_k}
        if is_hybrid:
            search_results = index.batch_search(
                embeddings, top_k=top_k, query_texts=texts_to_search
            )
        else:
            search_results = index.batch_search(embeddings, top_k=top_k)
        search_step.output = {"n_results": sum(len(r) for r in search_results)}
    logger.info(
        "Search done in %.1fs (%d results)",
        search_step.timing_ms / 1000,
        search_step.output["n_results"],
    )

    # --- Merge cached + fresh results ------------------------------------
    if miss_indices is not None and cached_results is not None:
        # Reconstruct full results list in original order
        retrievals: list[list] = []
        miss_iter = iter(search_results)
        for i in range(len(query_texts)):
            if cached_results[i] is not None:
                # Deserialize cached result
                retrievals.append(
                    [
                        SearchResult(
                            document=Document.from_dict(sr["document"]),
                            score=sr["score"],
                            rank=sr.get("rank", 0),
                        )
                        for sr in cached_results[i]
                    ]
                )
            else:
                retrievals.append(next(miss_iter))
    else:
        retrievals = search_results

    # --- Store fresh results in cache ------------------------------------
    if retrieval_store is not None and retriever_name:
        _store_t0 = _pc()
        if miss_indices is not None:
            # Only store the misses
            miss_queries = [query_texts[i] for i in miss_indices]
            miss_results_dicts = [[sr.to_dict() for sr in results] for results in search_results]
        else:
            # Store everything
            miss_queries = query_texts
            miss_results_dicts = [[sr.to_dict() for sr in results] for results in retrievals]
        retrieval_store.put_batch(
            retriever_name,
            miss_queries,
            miss_results_dicts,
            top_k,
        )
        logger.info(
            "Retrieval cache store: %.2fs (%d queries)",
            _pc() - _store_t0,
            len(miss_queries),
        )

    return retrievals, encode_step, search_step


def apply_reranking(
    reranker_provider: Any,
    query_texts: list[str],
    retrievals: list[list],
    top_k: int,
    fetch_k: int | None = None,
) -> tuple[list[list], Step]:
    """Shared reranking logic for all RAG agents.

    Loads the reranker (via ref-counted provider), reranks documents per query,
    and rebuilds SearchResult lists with reranker scores.

    Args:
        reranker_provider: RerankerProvider instance.
        query_texts: List of query strings (one per retrieval list).
        retrievals: Per-query lists of SearchResult objects.
        top_k: Number of documents to keep after reranking.
        fetch_k: Original fetch count (for logging). Defaults to top_k.

    Returns:
        Tuple of (reranked_retrievals, rerank_step).
    """
    from ragicamp.core.types import Document, SearchResult

    if fetch_k is None:
        fetch_k = top_k

    with StepTimer(RERANK, model=reranker_provider.config.model_name) as step:
        with reranker_provider.load() as reranker:
            docs_lists: list[list[Document]] = [[sr.document for sr in srs] for srs in retrievals]
            reranked_docs = reranker.batch_rerank(
                query_texts,
                docs_lists,
                top_k=top_k,
            )

    reranked_retrievals: list[list[SearchResult]] = []
    for docs in reranked_docs:
        reranked_retrievals.append(
            [
                SearchResult(
                    document=doc,
                    score=getattr(doc, "score", 0.0),
                    rank=rank + 1,
                )
                for rank, doc in enumerate(docs)
            ]
        )

    step.input = {"n_queries": len(query_texts), "fetch_k": fetch_k}
    step.output = {"top_k": top_k}

    logger.info(
        "Reranked %d queries (%d -> %d docs each) in %.1fs",
        len(query_texts),
        fetch_k,
        top_k,
        step.timing_ms / 1000,
    )

    return reranked_retrievals, step


# =============================================================================
# Agent Base Class
# =============================================================================


class Agent(ABC):
    """Base class for all agents.

    Agents receive all queries and manage their own resources.
    Each agent type implements its optimal strategy:

    - FixedRAGAgent: batch_retrieve → unload_embedder → batch_generate
    - IterativeRAGAgent: per-query with multiple rounds
    - SelfRAGAgent: per-query with conditional retrieval
    - DirectLLMAgent: batch_generate only (no retrieval)
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(
        self,
        queries: list[Query],
        *,
        on_result: Callable[[AgentResult], None] | None = None,
        checkpoint_path: Path | None = None,
        show_progress: bool = True,
    ) -> list[AgentResult]:
        """Process all queries with agent-specific optimization.

        This is THE interface for running agents. Each agent type
        implements its own optimal strategy for:
        - Model loading/unloading (GPU optimization)
        - Batching strategy (throughput)
        - Checkpointing (resume capability)

        Args:
            queries: All queries to process
            on_result: Called after each result (for streaming/incremental save)
            checkpoint_path: Enables resume from crashes
            show_progress: Show progress bar

        Returns:
            Results with answers and all intermediate steps
        """
        ...

    def _load_checkpoint(self, path: Path) -> tuple[list[AgentResult], set[int]]:
        """Load checkpoint and return completed results."""
        import json

        if not path.exists():
            return [], set()

        with open(path) as f:
            data = json.load(f)

        results = []
        for r in data.get("results", []):
            # to_dict() writes "question"; accept both for robustness
            text = r.get("question") or r.get("query", "")
            query = Query(idx=r["idx"], text=text, expected=r.get("expected"))
            # Restore steps from checkpoint
            steps = [
                Step(
                    type=s["type"],
                    timing_ms=s.get("timing_ms", 0.0),
                    model=s.get("model"),
                    metadata=s.get("metadata", {}),
                )
                for s in r.get("steps", [])
            ]
            result = AgentResult(
                query=query,
                answer=r["answer"],
                steps=steps,
                prompt=r.get("prompt"),
                metadata=r.get("metadata", {}),
            )
            results.append(result)

        completed_idx = {r.query.idx for r in results}
        logger.info("Loaded checkpoint: %d completed", len(completed_idx))
        return results, completed_idx

    def _save_checkpoint(self, results: list[AgentResult], path: Path) -> None:
        """Save checkpoint atomically."""
        import json

        data = {"results": [r.to_dict() for r in results]}
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
