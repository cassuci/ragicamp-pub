"""Transparent cache wrapper for EmbedderProvider.

``CachedEmbedderProvider`` is a **drop-in replacement** for any
``EmbedderProvider`` (same ``load()`` / ``model_name`` interface).  It yields
a ``CachedEmbedder`` that checks the SQLite KV store on every
``batch_encode`` call and **lazy-loads the real embedding model only when
there is a cache miss**.

If 100 % of queries are cached the GPU is never touched.

Integration (single line in ``ProviderFactory.create_embedder``):

.. code-block:: python

    provider = CachedEmbedderProvider(provider, store)

All agents, pipelines and index builders that receive the wrapped provider
benefit automatically -- zero code changes required.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from ragicamp.core.logging import get_logger
from ragicamp.models.providers import ManagedEmbedder, ModelProvider

if TYPE_CHECKING:
    from ragicamp.cache.embedding_store import EmbeddingStore
    from ragicamp.models.providers import EmbedderProvider

logger = get_logger(__name__)


class CachedEmbedderProvider(ModelProvider):
    """Drop-in wrapper around :class:`EmbedderProvider` that adds a KV cache.

    The underlying model is only loaded when there is at least one cache miss
    in a ``batch_encode`` call.  If every text is already cached the model is
    **never** loaded and no GPU memory is consumed.

    Parameters
    ----------
    inner : EmbedderProvider
        The real provider to delegate to on cache misses.
    store : EmbeddingStore
        Shared SQLite store for persisting embeddings across processes.
    """

    def __init__(self, inner: EmbedderProvider, store: EmbeddingStore) -> None:
        self.inner = inner
        self.store = store

    # -- ModelProvider interface ------------------------------------------

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    @contextmanager
    def load(self, gpu_fraction: float | None = None) -> Iterator[CachedEmbedder]:
        """Yield a :class:`CachedEmbedder`.

        The real model loads lazily inside ``batch_encode`` only if needed.
        On context-manager exit the real model is unloaded (if it was loaded).
        """
        cached = CachedEmbedder(
            provider=self.inner,
            store=self.store,
            gpu_fraction=gpu_fraction,
        )
        try:
            yield cached
        finally:
            cached.cleanup()

    # -- Forward config attribute access for compatibility ----------------

    @property
    def config(self) -> Any:
        return self.inner.config


class CachedEmbedder(ManagedEmbedder):
    """Embedder implementation that checks the KV store before computing.

    On ``batch_encode(texts)``:

    1. Look up every text in the SQLite store.
    2. If 100 % hit -> return immediately (no model loaded).
    3. If partial/no hit -> **lazy-load** the real model, encode only the
       misses, store them, and merge with the cached vectors.
    """

    def __init__(
        self,
        provider: EmbedderProvider,
        store: EmbeddingStore,
        gpu_fraction: float | None = None,
    ) -> None:
        self._provider = provider
        self._store = store
        self._gpu_fraction = gpu_fraction
        self._real_embedder: ManagedEmbedder | None = None
        self._real_ctx = None  # context manager for the real embedder

    # -- Embedder interface -----------------------------------------------

    def batch_encode(self, texts: list[str]) -> Any:
        """Encode texts to embeddings, using cache where possible.

        Returns
        -------
        np.ndarray
            Embedding matrix of shape ``(len(texts), dim)``.
        """
        model_name = self._provider.model_name

        # 1. Check cache -------------------------------------------------
        cached_embs, hit_mask = self._store.get_batch(model_name, texts)

        if cached_embs is not None and hit_mask.all():
            logger.info(
                "Embedding cache: 100%% hit (%d texts, model=%s)",
                len(texts),
                model_name,
            )
            return cached_embs

        # 2. Identify misses ---------------------------------------------
        miss_indices = [i for i, hit in enumerate(hit_mask) if not hit]
        miss_texts = [texts[i] for i in miss_indices]

        # 3. Lazy-load real model ----------------------------------------
        if self._real_embedder is None:
            hit_count = int(hit_mask.sum()) if cached_embs is not None else 0
            logger.info(
                "Embedding cache: %d/%d hits, loading model for %d misses (model=%s)",
                hit_count,
                len(texts),
                len(miss_texts),
                model_name,
            )
            self._real_embedder = self._load_real_model()

        # 4. Encode misses -----------------------------------------------
        miss_embs = self._real_embedder.batch_encode(miss_texts)
        if not isinstance(miss_embs, np.ndarray):
            miss_embs = np.asarray(miss_embs, dtype=np.float32)

        # 5. Store new embeddings ----------------------------------------
        self._store.put_batch(model_name, miss_texts, miss_embs)

        # 6. Merge cached + new ------------------------------------------
        if cached_embs is not None:
            result = cached_embs  # already correct shape
        else:
            # No prior hits -- allocate fresh result array
            dim = miss_embs.shape[1]
            result = np.zeros((len(texts), dim), dtype=np.float32)

        for local_idx, global_idx in enumerate(miss_indices):
            result[global_idx] = miss_embs[local_idx]

        hit_pct = (len(texts) - len(miss_indices)) / len(texts) * 100 if texts else 0
        logger.info(
            "Embedding cache: %.0f%% hit (%d/%d), encoded %d new (model=%s)",
            hit_pct,
            len(texts) - len(miss_indices),
            len(texts),
            len(miss_indices),
            model_name,
        )
        return result

    def get_dimension(self) -> int:
        """Return embedding dimension.

        Tries the store first (no model load needed).  Falls back to loading
        the real model.
        """
        # Try cached dimension first
        dim = self._store.get_dimension(self._provider.model_name)
        if dim is not None:
            return dim

        # Must load the real model to discover dimension
        if self._real_embedder is None:
            self._real_embedder = self._load_real_model()
        return self._real_embedder.get_dimension()

    def unload(self) -> None:
        """Alias for :meth:`cleanup` -- matches the ``ManagedEmbedder`` protocol."""
        self.cleanup()

    # -- Internal ---------------------------------------------------------

    def _load_real_model(self) -> ManagedEmbedder:
        """Load the real embedder via the inner provider's context manager."""
        ctx = self._provider.load(gpu_fraction=self._gpu_fraction)
        embedder = ctx.__enter__()
        # Only store the context after successful entry so cleanup()
        # never calls __exit__() on a context that failed to enter.
        self._real_ctx = ctx
        return embedder

    def cleanup(self) -> None:
        """Unload the real model if it was loaded, freeing GPU memory."""
        if self._real_ctx is not None:
            try:
                self._real_ctx.__exit__(None, None, None)
            except Exception:
                logger.warning("Error unloading cached embedder", exc_info=True)
            self._real_ctx = None
            self._real_embedder = None
