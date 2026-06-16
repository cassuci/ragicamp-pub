"""Lazy proxy for search backends — defers disk loading until first use.

When the retrieval cache gives 100% hits, ``batch_embed_and_search`` returns
before ever calling ``index.batch_search``.  With a ``LazySearchBackend``
wrapping the real index, the 3-minute pickle/FAISS deserialization is skipped
entirely in those cases.

Usage in ``Experiment.from_spec``::

    # Instead of:
    #   index = HierarchicalSearcher(HierarchicalIndex.load(...))
    # Use:
    index = LazySearchBackend(
        loader=lambda: HierarchicalSearcher(HierarchicalIndex.load(name, path)),
        is_hybrid=False,
    )

The proxy forwards all attribute access to the loaded backend.  The first
call to *any* method (typically ``batch_search``) triggers the load.

The ``is_hybrid`` flag controls the result of
``is_hybrid_searcher(index)`` — which checks ``hasattr(index, "sparse_index")``
— **without** triggering a load.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)


class LazySearchBackend:
    """Transparent lazy-loading proxy for any search backend.

    Parameters
    ----------
    loader : callable
        Zero-argument callable that returns the real search backend
        (``VectorIndex``, ``HierarchicalSearcher``, ``HybridSearcher``, …).
    is_hybrid : bool
        If ``True``, the proxy exposes a ``sparse_index`` attribute so that
        ``is_hybrid_searcher()`` returns ``True`` without loading the backend.
    """

    # Attributes that live on the proxy itself (not forwarded).
    _PROXY_ATTRS = frozenset(
        {
            "_loader",
            "_backend",
            "_is_hybrid",
            "_PROXY_ATTRS",
        }
    )

    def __init__(self, loader: Callable[[], Any], *, is_hybrid: bool = False) -> None:
        # Use object.__setattr__ to avoid triggering __getattr__
        object.__setattr__(self, "_loader", loader)
        object.__setattr__(self, "_backend", None)
        object.__setattr__(self, "_is_hybrid", is_hybrid)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> Any:
        backend = object.__getattribute__(self, "_backend")
        if backend is None:
            loader = object.__getattribute__(self, "_loader")
            logger.info("LazySearchBackend: loading index on first access...")
            backend = loader()
            object.__setattr__(self, "_backend", backend)
            logger.info("LazySearchBackend: index ready")
        return backend

    @property
    def is_loaded(self) -> bool:
        """Check whether the real backend has been loaded (without triggering a load)."""
        return object.__getattribute__(self, "_backend") is not None

    # ------------------------------------------------------------------
    # Attribute forwarding
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Handle the is_hybrid_searcher() check without loading
        if name == "sparse_index":
            if not object.__getattribute__(self, "_is_hybrid"):
                raise AttributeError(name)
            # Must load to provide the real sparse_index
            return getattr(self._ensure_loaded(), name)

        # Everything else: load and forward
        return getattr(self._ensure_loaded(), name)

    def __len__(self) -> int:
        return len(self._ensure_loaded())

    def __repr__(self) -> str:
        backend = object.__getattribute__(self, "_backend")
        if backend is not None:
            return f"LazySearchBackend(loaded={backend!r})"
        return "LazySearchBackend(not loaded)"
