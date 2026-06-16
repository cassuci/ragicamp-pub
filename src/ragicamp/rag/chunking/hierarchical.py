"""Hierarchical chunking for parent-child document structure.

Hierarchical chunking creates two levels of chunks:
- Parent chunks: Larger chunks (e.g., 1024 chars) for context
- Child chunks: Smaller chunks (e.g., 256 chars) for precise matching

During retrieval:
1. Search is performed against child chunks (better precision)
2. Parent chunks are returned (better context for LLM)

This gives the best of both worlds: precise matching with rich context.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from ragicamp.core.types import Document
from ragicamp.corpus.chunking import ChunkConfig, RecursiveChunker


@dataclass
class HierarchicalChunk:
    """A chunk with parent-child relationship information.

    Attributes:
        text: The chunk text
        parent_text: Text of the parent chunk (for child chunks)
        is_parent: Whether this is a parent chunk
        parent_id: ID of the parent chunk (for child chunks)
        child_ids: IDs of child chunks (for parent chunks)
    """

    text: str
    parent_text: str | None = None
    is_parent: bool = False
    parent_id: str | None = None
    child_ids: list[str] = None

    def __post_init__(self):
        if self.child_ids is None:
            self.child_ids = []


class HierarchicalChunker:
    """Create parent-child chunk hierarchies for retrieval.

    This chunker creates two levels:
    - Parents: Larger chunks for context (returned to LLM)
    - Children: Smaller chunks for matching (used for search)

    The retrieval flow is:
    1. Query is matched against child chunk embeddings
    2. Matching children are mapped to their parents
    3. Parent chunks (with full context) are returned
    """

    def __init__(
        self,
        parent_chunk_size: int = 1024,
        parent_chunk_overlap: int = 100,
        child_chunk_size: int = 256,
        child_chunk_overlap: int = 50,
        min_chunk_size: int = 50,
    ):
        """Initialize hierarchical chunker.

        Args:
            parent_chunk_size: Size of parent chunks in characters
            parent_chunk_overlap: Overlap between parent chunks
            child_chunk_size: Size of child chunks in characters
            child_chunk_overlap: Overlap between child chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Create chunkers for each level
        self.parent_chunker = RecursiveChunker(
            ChunkConfig(
                strategy="recursive",
                chunk_size=parent_chunk_size,
                chunk_overlap=parent_chunk_overlap,
                min_chunk_size=min_chunk_size,
            )
        )

        self.child_chunker = RecursiveChunker(
            ChunkConfig(
                strategy="recursive",
                chunk_size=child_chunk_size,
                chunk_overlap=child_chunk_overlap,
                min_chunk_size=min_chunk_size // 2,  # Allow smaller children
            )
        )

    def chunk_document(
        self,
        document: Document,
    ) -> tuple[list[Document], list[Document], dict[str, str]]:
        """Chunk a document into parent and child chunks.

        Args:
            document: The document to chunk

        Returns:
            Tuple of:
            - List of parent Document objects
            - List of child Document objects
            - Dict mapping child_id -> parent_id
        """
        parent_docs = []
        child_docs = []
        child_to_parent: dict[str, str] = {}

        # First, create parent chunks
        parent_texts = self.parent_chunker.chunk(document.text)

        for p_idx, parent_text in enumerate(parent_texts):
            if len(parent_text.strip()) < self.min_chunk_size:
                continue

            parent_id = f"{document.id}_parent_{p_idx}"

            parent_doc = Document(
                id=parent_id,
                text=parent_text.strip(),
                metadata={
                    **document.metadata,
                    "chunk_type": "parent",
                    "parent_index": p_idx,
                    "source_doc_id": document.id,
                },
            )
            parent_docs.append(parent_doc)

            # Create child chunks from this parent
            child_texts = self.child_chunker.chunk(parent_text)

            for c_idx, child_text in enumerate(child_texts):
                if len(child_text.strip()) < self.min_chunk_size // 2:
                    continue

                child_id = f"{parent_id}_child_{c_idx}"

                child_doc = Document(
                    id=child_id,
                    text=child_text.strip(),
                    metadata={
                        **document.metadata,
                        "chunk_type": "child",
                        "child_index": c_idx,
                        "parent_id": parent_id,
                        "source_doc_id": document.id,
                    },
                )
                child_docs.append(child_doc)
                child_to_parent[child_id] = parent_id

        return parent_docs, child_docs, child_to_parent

    def chunk_documents(
        self,
        documents: Iterator[Document],
    ) -> tuple[list[Document], list[Document], dict[str, str]]:
        """Chunk multiple documents into parent and child chunks.

        Args:
            documents: Iterator of documents to chunk

        Returns:
            Tuple of:
            - List of all parent Document objects
            - List of all child Document objects
            - Dict mapping child_id -> parent_id
        """
        all_parents = []
        all_children = []
        all_mappings: dict[str, str] = {}

        for doc in documents:
            parents, children, mappings = self.chunk_document(doc)
            all_parents.extend(parents)
            all_children.extend(children)
            all_mappings.update(mappings)

        return all_parents, all_children, all_mappings

    def get_info(self) -> dict:
        """Get chunking configuration info."""
        return {
            "strategy": "hierarchical",
            "parent_chunk_size": self.parent_chunk_size,
            "parent_chunk_overlap": self.parent_chunk_overlap,
            "child_chunk_size": self.child_chunk_size,
            "child_chunk_overlap": self.child_chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
        }

    def __repr__(self) -> str:
        return (
            f"HierarchicalChunker(parent={self.parent_chunk_size}, child={self.child_chunk_size})"
        )
