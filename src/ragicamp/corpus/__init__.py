"""Document corpus module for RAG systems.

This module provides abstractions for document sources used in retrieval.
Important: Corpora provide documents WITHOUT answer information.
"""

from ragicamp.corpus.base import CorpusConfig, DocumentCorpus
from ragicamp.corpus.chunking import (
    ChunkConfig,
    ChunkingStrategy,
    DocumentChunker,
    FixedSizeChunker,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
    get_chunker,
)
from ragicamp.corpus.wikipedia import WikipediaCorpus

__all__ = [
    "DocumentCorpus",
    "CorpusConfig",
    "WikipediaCorpus",
    # Chunking
    "ChunkConfig",
    "ChunkingStrategy",
    "DocumentChunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "ParagraphChunker",
    "RecursiveChunker",
    "get_chunker",
]
