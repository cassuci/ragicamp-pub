"""Base class for rerankers."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragicamp.core.types import Document


class Reranker(ABC):
    """Base class for document reranking strategies.

    Rerankers take an initial set of retrieved documents and reorder them
    based on more sophisticated relevance scoring. Common approaches:
    - Cross-encoders: Score (query, document) pairs jointly
    - LLM-based: Use an LLM to judge relevance
    - Learned rankers: Use features from query and document
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list["Document"],
        top_k: int,
    ) -> list["Document"]:
        """Rerank documents based on relevance to query.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents (top_k or fewer)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
