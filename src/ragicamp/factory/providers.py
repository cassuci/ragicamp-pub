"""Provider factory for creating model providers from configuration.

Creates EmbedderProvider, GeneratorProvider, and RerankerProvider for the
clean architecture where models are loaded/unloaded via context managers.

Embedding cache:
    When ``RAGICAMP_CACHE`` is set to ``"1"`` (the default), the factory
    automatically wraps every ``EmbedderProvider`` with a
    ``CachedEmbedderProvider`` backed by a shared SQLite KV store.  This
    caches query embeddings across Optuna trials / subprocesses and avoids
    loading the embedding model when all queries are already cached.

    Disable with ``RAGICAMP_CACHE=0``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ragicamp.core.logging import get_logger

if TYPE_CHECKING:
    from ragicamp.cache.cached_embedder import CachedEmbedderProvider
from ragicamp.models.providers import (
    EmbedderConfig,
    EmbedderProvider,
    GeneratorConfig,
    GeneratorProvider,
    ModelProvider,
)
from ragicamp.models.providers.reranker import RerankerConfig, RerankerProvider

logger = get_logger(__name__)


class ProviderFactory:
    """Factory for creating model providers from configuration.

    Usage:
        # From model spec string
        gen_provider = ProviderFactory.create_generator("vllm:meta-llama/Llama-3.2-3B")
        emb_provider = ProviderFactory.create_embedder("Alibaba-NLP/gte-Qwen2-1.5B-instruct")

        # Use with context managers
        with gen_provider.load() as generator:
            answers = generator.batch_generate(prompts)
    """

    @staticmethod
    def parse_generator_spec(
        spec: str,
        **kwargs: Any,
    ) -> GeneratorConfig:
        """Parse a model spec string into GeneratorConfig.

        Args:
            spec: Model spec like 'vllm:meta-llama/Llama-3.2-3B' or 'hf:google/gemma-2b-it'
            **kwargs: Additional config params

        Returns:
            GeneratorConfig
        """
        if ":" in spec:
            provider, model_name = spec.split(":", 1)
        else:
            provider, model_name = "vllm", spec

        backend = "vllm" if provider == "vllm" else "hf"

        return GeneratorConfig(
            model_name=model_name,
            backend=backend,
            dtype=kwargs.get("dtype", "auto"),
            trust_remote_code=kwargs.get("trust_remote_code", True),
            max_model_len=kwargs.get("max_model_len"),
        )

    @staticmethod
    def create_generator(
        spec: str | dict[str, Any],
        **kwargs: Any,
    ) -> GeneratorProvider:
        """Create a GeneratorProvider from spec.

        Args:
            spec: Model spec string or config dict
            **kwargs: Additional params

        Returns:
            GeneratorProvider ready for .load()
        """
        if isinstance(spec, dict):
            config = GeneratorConfig(
                model_name=spec.get("model_name", spec.get("name", "")),
                backend=spec.get("backend", spec.get("type", "vllm")),
                dtype=spec.get("dtype", "auto"),
                trust_remote_code=spec.get("trust_remote_code", True),
                max_model_len=spec.get("max_model_len"),
            )
        else:
            config = ProviderFactory.parse_generator_spec(spec, **kwargs)

        logger.info("Creating generator provider: %s (%s)", config.model_name, config.backend)
        return GeneratorProvider(config)

    @staticmethod
    def parse_embedder_spec(
        spec: str,
        backend: str = "vllm",
        **kwargs: Any,
    ) -> EmbedderConfig:
        """Parse an embedder spec into EmbedderConfig.

        Args:
            spec: Model name like 'BAAI/bge-large-en-v1.5'
            backend: 'vllm' or 'sentence_transformers'
            **kwargs: Additional config params

        Returns:
            EmbedderConfig
        """
        return EmbedderConfig(
            model_name=spec,
            backend=backend,
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )

    @staticmethod
    def create_embedder(
        spec: str | dict[str, Any],
        backend: str = "vllm",
        **kwargs: Any,
    ) -> EmbedderProvider | CachedEmbedderProvider:
        """Create an EmbedderProvider from spec.

        When the embedding cache is enabled (``RAGICAMP_CACHE=1``, the
        default), the returned provider is automatically wrapped with a
        :class:`~ragicamp.cache.CachedEmbedderProvider` so that embeddings
        are looked up in a shared SQLite KV store before loading the model.

        Args:
            spec: Model name or config dict
            backend: 'vllm' or 'sentence_transformers'
            **kwargs: Additional params

        Returns:
            EmbedderProvider (or CachedEmbedderProvider) ready for .load()
        """
        if isinstance(spec, dict):
            config = EmbedderConfig(
                model_name=spec.get("model_name", spec.get("embedding_model", "")),
                backend=spec.get("backend", spec.get("embedding_backend", backend)),
                trust_remote_code=spec.get("trust_remote_code", True),
            )
        else:
            config = ProviderFactory.parse_embedder_spec(spec, backend, **kwargs)

        logger.info("Creating embedder provider: %s (%s)", config.model_name, config.backend)
        provider: ModelProvider = EmbedderProvider(config)

        # Wrap with embedding cache if enabled
        if os.environ.get("RAGICAMP_CACHE", "1") == "1":
            try:
                from ragicamp.cache import CachedEmbedderProvider, EmbeddingStore

                store = EmbeddingStore.default()
                provider = CachedEmbedderProvider(provider, store)
                logger.info("Embedding cache enabled (db=%s)", store.db_path)
            except Exception:
                logger.warning(
                    "Failed to enable embedding cache, falling back to uncached",
                    exc_info=True,
                )

        return provider

    @staticmethod
    def create_reranker(
        spec: str,
        **kwargs: Any,
    ) -> RerankerProvider:
        """Create a RerankerProvider from spec.

        Args:
            spec: Reranker name (e.g. ``'bge'``, ``'bge-v2'``, ``'ms-marco'``)
                  or a full HuggingFace model path.
            **kwargs: Additional config params (e.g. ``batch_size``).

        Returns:
            RerankerProvider ready for ``.load()``
        """
        config = RerankerConfig(
            model_name=spec,
            batch_size=kwargs.get("batch_size", 32),
        )
        logger.info("Creating reranker provider: %s (%s)", spec, config.model_name)
        return RerankerProvider(config)
