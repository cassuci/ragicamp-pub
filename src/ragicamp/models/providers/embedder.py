"""Embedder provider with lazy loading and lifecycle management."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.utils.resource_manager import ResourceManager

from .base import ModelProvider

logger = get_logger(__name__)


@dataclass
class EmbedderConfig:
    """Configuration for embedder."""

    model_name: str
    backend: str = "vllm"  # "vllm" or "sentence_transformers"
    trust_remote_code: bool = True


class EmbedderProvider(ModelProvider):
    """Provides embedder with lazy loading and proper cleanup.

    Usage:
        provider = EmbedderProvider(EmbedderConfig("BAAI/bge-large-en-v1.5"))

        with provider.load() as embedder:
            embeddings = embedder.batch_encode(texts)
        # Embedder unloaded, GPU memory freed
    """

    def __init__(self, config: EmbedderConfig):
        self.config = config
        self._embedder = None
        self._refcount: int = 0

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @contextmanager
    def load(self, gpu_fraction: float | None = None) -> Iterator["ManagedEmbedder"]:
        """Load embedder, yield it, then unload.

        Supports ref-counting: nested ``with provider.load()`` calls reuse
        the already-loaded model and only unload when the outermost context
        exits.
        """
        self._refcount += 1
        try:
            if self._refcount == 1:
                # First caller — actually load the model
                from time import perf_counter as _pc

                if gpu_fraction is None:
                    gpu_fraction = Defaults.VLLM_GPU_MEMORY_FRACTION_FULL

                logger.info(
                    "Loading embedder: %s (gpu=%.0f%%)", self.model_name, gpu_fraction * 100
                )
                _t0 = _pc()

                if self.config.backend == "vllm":
                    embedder = self._load_vllm(gpu_fraction)
                elif self.config.backend == "sentence_transformers":
                    embedder = self._load_sentence_transformers()
                else:
                    raise ValueError(
                        f"Unknown embedder backend: '{self.config.backend}'. "
                        f"Supported: 'vllm', 'sentence_transformers'"
                    )

                _load_s = _pc() - _t0
                logger.info("Embedder loaded in %.1fs: %s", _load_s, self.model_name)
                self._embedder = embedder
            else:
                logger.debug(
                    "Embedder already loaded (refcount=%d): %s", self._refcount, self.model_name
                )

            yield self._embedder

        finally:
            self._refcount -= 1
            if self._refcount == 0:
                self._unload()

    def _load_vllm(self, gpu_fraction: float) -> "ManagedEmbedder":
        """Load vLLM embedder."""
        from ragicamp.models.vllm_embedder import VLLMEmbedder

        vllm_embedder = VLLMEmbedder(
            model_name=self.config.model_name,
            gpu_memory_fraction=gpu_fraction,
            trust_remote_code=self.config.trust_remote_code,
        )
        return VLLMEmbedderWrapper(vllm_embedder)

    def _load_sentence_transformers(self) -> "ManagedEmbedder":
        """Load sentence-transformers embedder."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.config.model_name)
        return SentenceTransformerWrapper(model)

    def _unload(self):
        """Unload embedder and free GPU memory."""
        if self._embedder is not None:
            if hasattr(self._embedder, "unload"):
                self._embedder.unload()
            self._embedder = None

        ResourceManager.clear_gpu_memory()
        logger.info("Embedder unloaded: %s", self.model_name)


class ManagedEmbedder(ABC):
    """ABC for provider-managed embedders with batch operations.

    Distinct from ``ragicamp.models.embedder.Embedder`` (Protocol for raw backends).
    Wrappers produced by ``EmbedderProvider.load()`` implement this interface.
    """

    @abstractmethod
    def batch_encode(self, texts: list[str]) -> Any:
        """Encode multiple texts to embeddings."""
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        ...

    def unload(self) -> None:  # noqa: B027
        """Unload model (optional override)."""


class VLLMEmbedderWrapper(ManagedEmbedder):
    """Wrapper around VLLMEmbedder implementing Embedder protocol."""

    def __init__(self, embedder):
        self._embedder = embedder

    def batch_encode(self, texts: list[str]) -> Any:
        return self._embedder.encode(texts)

    def get_dimension(self) -> int:
        return self._embedder.get_sentence_embedding_dimension()

    def unload(self):
        self._embedder.unload()


class SentenceTransformerWrapper(ManagedEmbedder):
    """Wrapper around SentenceTransformer implementing Embedder protocol."""

    def __init__(self, model):
        self._model = model

    def batch_encode(self, texts: list[str]) -> Any:
        return self._model.encode(texts, show_progress_bar=True)

    def get_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def unload(self) -> None:
        """Move model off GPU and free references."""
        import gc

        import torch

        if hasattr(self._model, "to"):
            self._model.to("cpu")
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
