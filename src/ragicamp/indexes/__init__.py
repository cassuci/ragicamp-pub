"""Index classes for storing and searching document embeddings.

Clean Architecture:
- VectorIndex: Pure data (FAISS + documents), no model ownership
- IndexBuilder: Builds indexes with provider pattern
- IndexConfig: Configuration for indexes

The index does NOT own the embedder. Embeddings are provided externally.
This allows clean GPU lifecycle management by agents.
"""

from ragicamp.indexes.index_builder import IndexBuilder, build_index
from ragicamp.indexes.vector_index import IndexConfig, VectorIndex

__all__ = [
    # Core types
    "VectorIndex",
    "IndexConfig",
    # Builder
    "IndexBuilder",
    "build_index",
]
