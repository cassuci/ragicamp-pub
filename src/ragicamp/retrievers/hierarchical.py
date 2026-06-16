"""Hierarchical search with parent-child chunk structure.

Hierarchical retrieval addresses a fundamental tradeoff in RAG:
- Small chunks: Better for precise semantic matching
- Large chunks: Better context for the LLM to generate answers

Solution: Search with small "child" chunks, return larger "parent" chunks.
"""

from ragicamp.core.logging import get_logger
from ragicamp.core.types import Document, SearchResult
from ragicamp.indexes.hierarchical import HierarchicalIndex

logger = get_logger(__name__)


class HierarchicalSearcher:
    """Searches child chunks but returns parent chunks.

    Works with HierarchicalIndex which stores parent/child relationships.

    Usage:
        searcher = HierarchicalSearcher(index)

        # Search requires external embeddings (from EmbedderProvider)
        results = searcher.search(query_embedding, top_k=5)
    """

    def __init__(self, index: HierarchicalIndex):
        """Initialize searcher.

        Args:
            index: Pre-built HierarchicalIndex
        """
        self.index = index

    def search(self, query_embedding, top_k: int = 5) -> list[SearchResult]:
        """Search child chunks, return parent chunks.

        Args:
            query_embedding: Pre-computed query embedding
            top_k: Number of parent documents to return

        Returns:
            List of SearchResult with parent documents
        """
        hits = self.index.search(query_embedding, top_k=top_k)

        results = []
        for rank, (parent_idx, score, child_idx) in enumerate(hits):
            parent_doc = self.index.get_parent(parent_idx)
            child_doc = self.index.get_child(child_idx)

            if parent_doc:
                result = SearchResult(
                    document=Document(
                        id=parent_doc.id,
                        text=parent_doc.text,
                        metadata={
                            **parent_doc.metadata,
                            "matched_child": child_doc.id if child_doc else None,
                            "matched_child_text": child_doc.text[:200] if child_doc else None,
                        },
                        score=score,
                    ),
                    score=score,
                    rank=rank,
                )
                results.append(result)

        return results

    def batch_search(
        self,
        query_embeddings,
        top_k: int = 5,
    ) -> list[list[SearchResult]]:
        """Batch search for multiple queries.

        Args:
            query_embeddings: Pre-computed query embeddings, shape (n, dim)
            top_k: Number of parent documents per query

        Returns:
            List of SearchResult lists
        """
        all_hits = self.index.batch_search(query_embeddings, top_k=top_k)

        all_results = []
        for hits in all_hits:
            results = []
            for rank, (parent_idx, score, child_idx) in enumerate(hits):
                parent_doc = self.index.get_parent(parent_idx)
                child_doc = self.index.get_child(child_idx)

                if parent_doc:
                    result = SearchResult(
                        document=Document(
                            id=parent_doc.id,
                            text=parent_doc.text,
                            metadata={
                                **parent_doc.metadata,
                                "matched_child": child_doc.id if child_doc else None,
                                "matched_child_text": child_doc.text[:200] if child_doc else None,
                            },
                            score=score,
                        ),
                        score=score,
                        rank=rank,
                    )
                    results.append(result)

            all_results.append(results)

        return all_results
