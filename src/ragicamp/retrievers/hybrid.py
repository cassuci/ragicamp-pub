"""Hybrid search combining dense and sparse retrieval.

Uses Reciprocal Rank Fusion (RRF) to merge results.
Works with the clean VectorIndex architecture.
"""

from ragicamp.core.logging import get_logger
from ragicamp.core.types import Document, SearchResult
from ragicamp.indexes.sparse import SparseIndex
from ragicamp.indexes.vector_index import VectorIndex

logger = get_logger(__name__)


class HybridSearcher:
    """Combines dense (VectorIndex) and sparse (BM25/TF-IDF) search.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both methods.
    RRF is robust and doesn't require score normalization.

    Usage:
        searcher = HybridSearcher(
            vector_index=index,
            sparse_index=sparse_index,
            alpha=0.7,  # Weight toward dense
        )

        # Search requires external embeddings (from EmbedderProvider)
        results = searcher.search(query_embedding, query_text, top_k=5)
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        sparse_index: SparseIndex,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        """Initialize hybrid searcher.

        Args:
            vector_index: Dense vector index
            sparse_index: Sparse (TF-IDF/BM25) index
            alpha: Weight for dense vs sparse (0=sparse only, 1=dense only)
            rrf_k: RRF constant (higher = more weight to lower ranks)
        """
        self.vector_index = vector_index
        self.sparse_index = sparse_index
        self.alpha = alpha
        self.rrf_k = rrf_k

    def search(
        self,
        query_embedding,
        query_text: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Hybrid search with RRF fusion.

        Args:
            query_embedding: Pre-computed query embedding
            query_text: Query text for sparse search
            top_k: Number of results

        Returns:
            List of SearchResult sorted by RRF score
        """
        candidates = top_k * 3

        dense_results = self.vector_index.search(query_embedding, top_k=candidates)
        sparse_hits = self.sparse_index.search(query_text, top_k=candidates)

        return self._rrf_merge(dense_results, sparse_hits, top_k)

    def batch_search(
        self,
        query_embeddings,
        top_k: int = 10,
        query_texts: list[str] | None = None,
    ) -> list[list[SearchResult]]:
        """Batch hybrid search.

        Args:
            query_embeddings: Pre-computed query embeddings, shape (n, dim)
            top_k: Number of results per query
            query_texts: Query texts for sparse search (required for hybrid)

        Returns:
            List of SearchResult lists

        Raises:
            ValueError: If query_texts is None (required for hybrid search)
        """
        if query_texts is None:
            raise ValueError("HybridSearcher.batch_search requires query_texts")
        candidates = top_k * 3

        all_dense = self.vector_index.batch_search(query_embeddings, top_k=candidates)
        all_sparse_hits = self.sparse_index.batch_search(query_texts, top_k=candidates)

        return [
            self._rrf_merge(all_dense[i], all_sparse_hits[i], top_k)
            for i in range(len(query_texts))
        ]

    def _rrf_merge(
        self,
        dense_results: list[SearchResult],
        sparse_hits: list[tuple[int, float]],
        top_k: int,
    ) -> list[SearchResult]:
        """Merge dense SearchResults and sparse (index, score) hits via RRF.

        Computes RRF scores directly from dense results and sparse (idx, score)
        tuples without creating intermediate SearchResult objects for sparse hits.

        Args:
            dense_results: Dense search results.
            sparse_hits: Sparse hits as (doc_index, score) tuples.
            top_k: Number of results to return.

        Returns:
            Merged SearchResult list sorted by RRF score.
        """
        documents = self.vector_index.documents
        num_docs = len(documents)
        rrf_scores: dict[str, float] = {}

        # Build doc_map from dense results (reuse existing Document objects)
        doc_map: dict[str, Document] = {}
        for rank, result in enumerate(dense_results, start=1):
            doc_id = result.document.id
            doc_map[doc_id] = result.document
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.alpha / (self.rrf_k + rank)

        # Add sparse scores directly from (idx, score) tuples — no SearchResult creation
        sparse_rank = 0
        for idx, _score in sparse_hits:
            if 0 <= idx < num_docs:
                sparse_rank += 1
                doc = documents[idx]
                doc_id = doc.id
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - self.alpha) / (
                    self.rrf_k + sparse_rank
                )

        # Sort by RRF score descending and build final results
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results: list[SearchResult] = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k]):
            doc = doc_map[doc_id]
            results.append(
                SearchResult(
                    document=Document(
                        id=doc.id,
                        text=doc.text,
                        metadata=doc.metadata.copy(),
                        score=score,
                    ),
                    score=score,
                    rank=rank,
                )
            )

        return results
