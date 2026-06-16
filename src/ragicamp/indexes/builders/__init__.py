"""Index builders for creating various types of indexes.

This package provides specialized builders for:
- Embedding indexes: Dense vector indexes for semantic search
- Hierarchical indexes: Parent-child chunk relationships

Common utilities:
- EmbeddingStorage: Incremental disk storage for embeddings
- CheckpointManager: Resume support for long-running builds
"""

from ragicamp.indexes.builders.checkpoint import (
    CheckpointManager,
    EmbeddingCheckpoint,
    HierarchicalCheckpoint,
)
from ragicamp.indexes.builders.embedding_builder import build_embedding_index
from ragicamp.indexes.builders.hierarchical_builder import build_hierarchical_index
from ragicamp.indexes.builders.storage import EmbeddingStorage

__all__ = [
    "build_embedding_index",
    "build_hierarchical_index",
    "EmbeddingStorage",
    "CheckpointManager",
    "EmbeddingCheckpoint",
    "HierarchicalCheckpoint",
]
