"""Core data types used throughout ragicamp.

Single source of truth for Document, Chunk, and related types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Union, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from ragicamp.indexes.vector_index import VectorIndex
    from ragicamp.retrievers.hierarchical import HierarchicalSearcher
    from ragicamp.retrievers.hybrid import HybridSearcher


@dataclass
class Document:
    """A document or chunk in the retrieval system.

    Used for:
    - Raw documents from corpus
    - Chunked text for indexing
    - Search results (with score)

    Attributes:
        id: Unique identifier
        text: Document text content
        metadata: Additional info (source, title, parent_id, etc.)
        score: Retrieval score (only set in search results)
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            score=data.get("score"),
        )


@dataclass
class SearchResult:
    """Result from index search.

    Contains the document and its relevance score.
    """

    document: Document
    score: float
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "rank": self.rank,
        }


@runtime_checkable
class Searcher(Protocol):
    """Protocol for all search backends.

    Implemented by:
    - VectorIndex: Dense semantic search
    - HybridSearcher: Dense + sparse with RRF fusion
    - HierarchicalSearcher: Child search, parent return

    Searchers take embeddings externally (from EmbedderProvider)
    so they don't own any GPU resources.
    """

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
        query_texts: list[str] | None = None,
    ) -> list[list[SearchResult]]:
        """Search for multiple queries at once.

        Args:
            query_embeddings: Query vectors, shape (n_queries, dim)
            top_k: Number of results per query
            query_texts: Optional raw query strings (required by HybridSearcher
                for sparse retrieval; ignored by dense-only backends)

        Returns:
            List of SearchResult lists, one per query
        """
        ...


# Type alias for all supported search backends.
# Defined here as the single source of truth -- agents, factories, etc.
# should import from core.types rather than defining their own.
SearchBackend = Union["VectorIndex", "HybridSearcher", "HierarchicalSearcher", Searcher]
