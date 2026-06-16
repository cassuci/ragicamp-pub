"""Transparent caching layer for ragicamp pipeline components.

The cache module provides disk-backed KV stores that wrap existing providers,
making caching invisible to agents and pipelines.

Architecture:
    Each cache layer follows the same pattern:
    1. A KV store backed by a SQLite table (subprocess-safe via WAL mode)
    2. A CachedXProvider wrapper implementing the same interface as XProvider
    3. A one-liner in the factory to wire it in

Currently implemented layers:
    - EmbeddingStore + CachedEmbedderProvider: caches query embeddings by (model, text)
    - RetrievalStore: caches retrieval results by (retriever, query, top_k)

Usage:
    # Wrap any EmbedderProvider (automatic via ProviderFactory when RAGICAMP_CACHE=1)
    from ragicamp.cache import CachedEmbedderProvider, EmbeddingStore

    store = EmbeddingStore.default()
    cached_provider = CachedEmbedderProvider(original_provider, store)

    # Use exactly like the original -- agents don't know about caching
    with cached_provider.load() as embedder:
        embeddings = embedder.batch_encode(texts)  # checks cache first
"""

from ragicamp.cache.cached_embedder import CachedEmbedder, CachedEmbedderProvider
from ragicamp.cache.embedding_store import EmbeddingStore
from ragicamp.cache.retrieval_store import RetrievalStore

__all__ = [
    "CachedEmbedder",
    "CachedEmbedderProvider",
    "EmbeddingStore",
    "RetrievalStore",
]
