"""Retrieval components.

Clean Architecture:
- VectorIndex: Pure data for dense search
- SparseIndex: TF-IDF/BM25 for sparse search
- HybridSearcher: Combines dense + sparse with RRF
- HierarchicalSearcher: Child chunk search, parent chunk return

Searchers don't own embedders. Embeddings are provided externally
by EmbedderProvider, allowing clean GPU lifecycle management.
"""

from ragicamp.indexes.sparse import SparseIndex, SparseMethod
from ragicamp.indexes.vector_index import VectorIndex
from ragicamp.retrievers.hierarchical import HierarchicalSearcher
from ragicamp.retrievers.hybrid import HybridSearcher
from ragicamp.retrievers.lazy import LazySearchBackend

__all__ = [
    "VectorIndex",
    "SparseIndex",
    "SparseMethod",
    "HybridSearcher",
    "HierarchicalSearcher",
    "LazySearchBackend",
]
