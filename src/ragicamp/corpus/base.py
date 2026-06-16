"""Base classes for document corpora.

A DocumentCorpus provides documents for retrieval WITHOUT answer information.
This is distinct from QADatasets which contain question-answer pairs for evaluation.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ragicamp.core.types import Document


@dataclass
class CorpusConfig:
    """Configuration for a document corpus.

    Attributes:
        name: Human-readable corpus identifier
        source: Source identifier (e.g., "wikimedia/wikipedia")
        version: Version/snapshot identifier (e.g., "20231101.simple")
        max_docs: Optional limit on documents to load (for testing)
        metadata: Additional corpus-specific configuration
    """

    name: str
    source: str
    version: str = "latest"
    max_docs: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name} ({self.source}:{self.version})"


class DocumentCorpus(ABC):
    """Base class for document corpora.

    A corpus provides documents for retrieval. Critically, corpora should NOT
    contain answer information - they are pure document sources.

    This prevents data leakage where the retrieval system has access to
    training answers.

    Example:
        >>> config = CorpusConfig(
        ...     name="wikipedia_simple",
        ...     source="wikimedia/wikipedia",
        ...     version="20231101.simple"
        ... )
        >>> corpus = WikipediaCorpus(config)
        >>> for doc in corpus.load(max_docs=10):
        ...     print(doc.text[:100])
    """

    def __init__(self, config: CorpusConfig):
        """Initialize corpus with configuration.

        Args:
            config: Corpus configuration
        """
        self.config = config

    @abstractmethod
    def load(self, max_docs: int | None = None) -> Iterator[Document]:
        """Load documents from corpus.

        Args:
            max_docs: Optional limit on number of documents to yield.
                     If None, uses config.max_docs. If both None, yields all.

        Yields:
            Document objects WITHOUT answer information
        """
        ...

    def get_info(self) -> dict[str, Any]:
        """Get corpus metadata.

        Returns:
            Dictionary with corpus information including:
            - name: Corpus name
            - source: Source identifier
            - version: Version identifier
            - Any additional metadata from config
        """
        return {
            "name": self.config.name,
            "source": self.config.source,
            "version": self.config.version,
            **self.config.metadata,
        }

    def __str__(self) -> str:
        return str(self.config)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config})"
