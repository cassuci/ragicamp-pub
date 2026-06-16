"""Document chunking strategies for RAG retrieval.

Chunking long documents improves retrieval quality by:
1. Better semantic matching (smaller chunks = more precise vectors)
2. Fitting context windows (avoid exceeding LLM limits)
3. Reducing noise (irrelevant sections don't dilute retrieval)
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.core.types import Document

logger = get_logger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for document chunking.

    Attributes:
        strategy: Chunking strategy ('fixed', 'sentence', 'paragraph', 'recursive')
        chunk_size: Target size in characters (all strategies, including sentence)
        chunk_overlap: Overlap between consecutive chunks (prevents context loss)
        min_chunk_size: Minimum chunk size (discard smaller chunks)
        separators: Custom separators for recursive strategy
    """

    strategy: str = "recursive"
    chunk_size: int = Defaults.CHUNK_SIZE
    chunk_overlap: int = Defaults.CHUNK_OVERLAP
    min_chunk_size: int = 50
    separators: list[str] | None = None

    def __post_init__(self):
        if self.separators is None:
            # Default separators for recursive splitting (most to least specific)
            self.separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


class ChunkingStrategy(ABC):
    """Base class for document chunking strategies."""

    def __init__(self, config: ChunkConfig):
        self.config = config

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        pass

    def chunk_document(self, document: Document) -> Iterator[Document]:
        """Chunk a document, preserving metadata.

        Args:
            document: Input document

        Yields:
            Document objects for each chunk, with updated metadata
        """
        chunks = self.chunk(document.text)

        for i, chunk_text in enumerate(chunks):
            # Skip chunks that are too small
            if len(chunk_text.strip()) < self.config.min_chunk_size:
                continue

            yield Document(
                id=f"{document.id}_chunk_{i}",
                text=chunk_text.strip(),
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "parent_doc_id": document.id,
                    "chunk_strategy": self.config.strategy,
                },
            )


class FixedSizeChunker(ChunkingStrategy):
    """Split text into fixed-size character chunks with overlap.

    Simple and predictable. Best for uniform document processing.
    """

    def chunk(self, text: str) -> list[str]:
        """Split text into fixed-size chunks."""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.config.chunk_size, text_len)

            # Try to end at a word boundary
            if end < text_len:
                space_pos = text.rfind(" ", start, end)
                if space_pos > start + self.config.chunk_size // 2:
                    end = space_pos

            chunks.append(text[start:end])

            # CRITICAL: Break if we've reached the end (prevents infinite loop)
            if end >= text_len:
                break
            # Move start, accounting for overlap
            start = end - self.config.chunk_overlap if self.config.chunk_overlap > 0 else end

        return chunks


class SentenceChunker(ChunkingStrategy):
    """Split text by sentences, grouping into chunks.

    Better semantic boundaries than fixed-size. ``chunk_size`` (in characters)
    is converted to an approximate sentence count (~80 chars/sentence).
    """

    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def chunk(self, text: str) -> list[str]:
        """Split text into sentence-based chunks."""
        sentences = self.SENTENCE_PATTERN.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text] if text.strip() else []

        chunks = []
        # chunk_size is target characters; convert to sentence count.
        # Average English sentence is ~80-100 chars.
        avg_sentence_len = 80
        sentences_per_chunk = max(1, self.config.chunk_size // avg_sentence_len)
        overlap_sentences = max(0, self.config.chunk_overlap // avg_sentence_len)

        i = 0
        while i < len(sentences):
            end = min(i + sentences_per_chunk, len(sentences))
            chunk_sentences = sentences[i:end]
            chunks.append(" ".join(chunk_sentences))

            # Move forward with overlap
            i = end - overlap_sentences
            if i <= end - sentences_per_chunk:
                i = end

        return chunks


class ParagraphChunker(ChunkingStrategy):
    """Split text by paragraphs (double newlines).

    Respects natural document structure. May produce variable chunk sizes.
    """

    def chunk(self, text: str) -> list[str]:
        """Split text into paragraph-based chunks."""
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [text] if text.strip() else []

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            # If this paragraph alone exceeds chunk_size, split it
            if para_size > self.config.chunk_size and not current_chunk:
                sub_chunks = self._split_large_paragraph(para)
                chunks.extend(sub_chunks)
                continue

            # If adding this paragraph exceeds chunk_size, start new chunk
            if current_size + para_size > self.config.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                overlap_text = current_chunk[-1] if current_chunk else ""
                if len(overlap_text) <= self.config.chunk_overlap:
                    current_chunk = [overlap_text] if overlap_text else []
                    current_size = len(overlap_text)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _split_large_paragraph(self, text: str) -> list[str]:
        """Split a large paragraph using sentence boundaries."""
        sentence_chunker = SentenceChunker(
            ChunkConfig(
                strategy="sentence",
                chunk_size=5,
                chunk_overlap=1,
                min_chunk_size=self.config.min_chunk_size,
            )
        )
        return sentence_chunker.chunk(text)


class RecursiveChunker(ChunkingStrategy):
    """Recursively split text using a hierarchy of separators.

    The most sophisticated strategy. Tries to preserve semantic structure
    by splitting on the most specific separator first, then falling back
    to less specific ones.

    Default separator hierarchy:
    1. Paragraph breaks (\\n\\n)
    2. Line breaks (\\n)
    3. Sentence endings (. ! ?)
    4. Clause separators (; ,)
    5. Spaces

    This is similar to LangChain's RecursiveCharacterTextSplitter.
    """

    # Safety limits
    MAX_RECURSION_DEPTH = 10
    MAX_CHUNKS_PER_DOC = 2000

    def chunk(self, text: str) -> list[str]:
        """Recursively split text into chunks."""
        return self._recursive_split(text, self.config.separators, depth=0)

    def _recursive_split(self, text: str, separators: list[str], depth: int = 0) -> list[str]:
        """Recursively split text using separator hierarchy."""
        if not text.strip():
            return []

        # Base case: text fits in chunk
        if len(text) <= self.config.chunk_size:
            return [text]

        # Safety: bail out if recursion is too deep
        if depth >= self.MAX_RECURSION_DEPTH:
            return self._hard_split(text)

        # Try each separator in order
        for i, sep in enumerate(separators):
            if sep in text:
                # Skip extremely frequent separators to avoid huge lists
                sep_count = text.count(sep)
                if sep_count > 50_000:
                    continue

                splits = text.split(sep)
                chunks = []
                current_chunk = ""

                for split in splits:
                    # Safety: bail out if we have too many chunks
                    if len(chunks) >= self.MAX_CHUNKS_PER_DOC:
                        if current_chunk:
                            chunks.append(current_chunk)
                        return chunks

                    test_chunk = (current_chunk + sep + split).strip() if current_chunk else split

                    if len(test_chunk) <= self.config.chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)

                        if len(split) > self.config.chunk_size:
                            remaining_seps = (
                                separators[i + 1 :] if i + 1 < len(separators) else [" "]
                            )
                            sub_chunks = self._recursive_split(
                                split, remaining_seps, depth=depth + 1
                            )
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = split

                if current_chunk:
                    chunks.append(current_chunk)

                return self._apply_overlap(chunks)

        # Fallback: hard split at chunk_size
        return self._hard_split(text)

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply overlap between consecutive chunks."""
        if self.config.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            overlap = (
                prev_chunk[-self.config.chunk_overlap :]
                if len(prev_chunk) > self.config.chunk_overlap
                else prev_chunk
            )

            if not curr_chunk.startswith(overlap):
                space_idx = overlap.find(" ")
                if space_idx > 0:
                    overlap = overlap[space_idx + 1 :]
                result.append(overlap + " " + curr_chunk)
            else:
                result.append(curr_chunk)

        return result

    def _hard_split(self, text: str) -> list[str]:
        """Last resort: split at exact chunk_size boundaries."""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.config.chunk_size, text_len)
            chunks.append(text[start:end])

            # CRITICAL: Break when we reach the end (prevents infinite loop)
            if end >= text_len:
                break
            start = end - self.config.chunk_overlap if self.config.chunk_overlap > 0 else end

        return chunks


def get_chunker(config: ChunkConfig) -> ChunkingStrategy:
    """Get a chunking strategy based on config."""
    strategies = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceChunker,
        "paragraph": ParagraphChunker,
        "recursive": RecursiveChunker,
    }

    if config.strategy not in strategies:
        raise ValueError(
            f"Unknown chunking strategy: {config.strategy}. Available: {list(strategies.keys())}"
        )

    return strategies[config.strategy](config)


class DocumentChunker:
    """High-level interface for chunking documents.

    Example:
        >>> config = ChunkConfig(strategy="recursive", chunk_size=512, chunk_overlap=50)
        >>> chunker = DocumentChunker(config)
        >>> for chunk_doc in chunker.chunk_documents(documents):
        ...     print(f"Chunk {chunk_doc.id}: {len(chunk_doc.text)} chars")
    """

    def __init__(self, config: ChunkConfig | None = None):
        """Initialize document chunker."""
        self.config = config or ChunkConfig()
        self.strategy = get_chunker(self.config)

    def chunk_documents(
        self,
        documents: Iterator[Document],
        show_progress: bool = True,
    ) -> Iterator[Document]:
        """Chunk multiple documents sequentially.

        Args:
            documents: Input documents to chunk
            show_progress: Whether to show progress

        Yields:
            Chunked Document objects
        """
        total_chunks = 0
        doc_count = 0

        for doc in documents:
            doc_count += 1
            doc_chunks = list(self.strategy.chunk_document(doc))
            total_chunks += len(doc_chunks)

            yield from doc_chunks

            if show_progress and doc_count % 100 == 0:
                logger.info("  Processed %s docs -> %s chunks", doc_count, total_chunks)

        if show_progress:
            logger.info("Chunking complete: %s docs -> %s chunks", doc_count, total_chunks)
            logger.info("  Avg chunks per doc: %.1f", total_chunks / max(doc_count, 1))

    def get_info(self) -> dict[str, Any]:
        """Get chunking configuration info."""
        return {
            "strategy": self.config.strategy,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "min_chunk_size": self.config.min_chunk_size,
        }
