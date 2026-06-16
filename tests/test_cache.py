"""Tests for the embedding cache module.

Tests cover:
    - EmbeddingStore: round-trip put/get, partial hits, stats, clear
    - CachedEmbedder: lazy model loading, 100% hit (no model), partial hit
    - CachedEmbedderProvider: interface compatibility, cleanup lifecycle
    - Factory integration: RAGICAMP_CACHE env var toggling
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ragicamp.cache.embedding_store import EmbeddingStore, _text_hash

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def store(tmp_path: Path) -> EmbeddingStore:
    """Create a fresh EmbeddingStore in a temp directory."""
    return EmbeddingStore(tmp_path / "test_cache.db")


@pytest.fixture
def sample_texts() -> list[str]:
    return [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "What year did WW2 end?",
    ]


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Deterministic 384-dim embeddings for 3 texts."""
    rng = np.random.RandomState(42)
    return rng.randn(3, 384).astype(np.float32)


# ============================================================================
# EmbeddingStore tests
# ============================================================================


class TestEmbeddingStore:
    """Tests for the SQLite KV store."""

    def test_roundtrip_put_get(
        self,
        store: EmbeddingStore,
        sample_texts: list[str],
        sample_embeddings: np.ndarray,
    ):
        """Store embeddings and retrieve them -- values should match exactly."""
        model = "test-model"
        written = store.put_batch(model, sample_texts, sample_embeddings)
        assert written == 3

        embs, mask = store.get_batch(model, sample_texts)
        assert embs is not None
        assert mask.all()
        np.testing.assert_array_almost_equal(embs, sample_embeddings)

    def test_get_empty_store(self, store: EmbeddingStore, sample_texts: list[str]):
        """Getting from an empty store returns None + all-False mask."""
        embs, mask = store.get_batch("no-model", sample_texts)
        assert embs is None
        assert not mask.any()

    def test_get_empty_texts(self, store: EmbeddingStore):
        """Getting with empty text list returns empty results."""
        embs, mask = store.get_batch("any-model", [])
        assert embs is None
        assert len(mask) == 0

    def test_partial_hit(
        self,
        store: EmbeddingStore,
        sample_texts: list[str],
        sample_embeddings: np.ndarray,
    ):
        """Store 2 of 3 texts, then get all 3 -- should get partial hits."""
        model = "test-model"
        store.put_batch(model, sample_texts[:2], sample_embeddings[:2])

        embs, mask = store.get_batch(model, sample_texts)
        assert embs is not None
        assert mask[0] and mask[1] and not mask[2]
        np.testing.assert_array_almost_equal(embs[0], sample_embeddings[0])
        np.testing.assert_array_almost_equal(embs[1], sample_embeddings[1])
        # The miss slot should be zeros
        np.testing.assert_array_equal(embs[2], np.zeros(384, dtype=np.float32))

    def test_different_models_isolated(
        self,
        store: EmbeddingStore,
        sample_texts: list[str],
        sample_embeddings: np.ndarray,
    ):
        """Embeddings stored under one model should not leak to another."""
        store.put_batch("model-a", sample_texts, sample_embeddings)

        embs, mask = store.get_batch("model-b", sample_texts)
        assert embs is None
        assert not mask.any()

    def test_idempotent_put(
        self,
        store: EmbeddingStore,
        sample_texts: list[str],
        sample_embeddings: np.ndarray,
    ):
        """Putting the same texts twice should not fail or overwrite."""
        model = "test-model"
        written1 = store.put_batch(model, sample_texts, sample_embeddings)
        assert written1 == 3

        # Second put with same data -- INSERT OR IGNORE
        written2 = store.put_batch(model, sample_texts, sample_embeddings)
        assert written2 == 0  # nothing new

        # Values should still be the originals
        embs, mask = store.get_batch(model, sample_texts)
        np.testing.assert_array_almost_equal(embs, sample_embeddings)

    def test_stats(
        self,
        store: EmbeddingStore,
        sample_texts: list[str],
        sample_embeddings: np.ndarray,
    ):
        """Stats should reflect stored data accurately."""
        store.put_batch("model-a", sample_texts[:2], sample_embeddings[:2])
        store.put_batch("model-b", sample_texts[2:], sample_embeddings[2:])

        info = store.stats()
        assert info["total_entries"] == 3
        assert info["total_size_mb"] >= 0  # tiny test data rounds to 0.00 MB
        assert "model-a" in info["models"]
        assert "model-b" in info["models"]
        assert info["models"]["model-a"]["entries"] == 2
        assert info["models"]["model-b"]["entries"] == 1

    def test_clear_all(
        self,
        store: EmbeddingStore,
        sample_texts: list[str],
        sample_embeddings: np.ndarray,
    ):
        """Clear should remove all entries."""
        store.put_batch("model-a", sample_texts, sample_embeddings)
        deleted = store.clear()
        assert deleted == 3

        info = store.stats()
        assert info["total_entries"] == 0

    def test_clear_specific_model(
        self,
        store: EmbeddingStore,
        sample_texts: list[str],
        sample_embeddings: np.ndarray,
    ):
        """Clear with model filter should only remove that model."""
        store.put_batch("model-a", sample_texts[:2], sample_embeddings[:2])
        store.put_batch("model-b", sample_texts[2:], sample_embeddings[2:])

        deleted = store.clear(model="model-a")
        assert deleted == 2

        info = store.stats()
        assert info["total_entries"] == 1
        assert "model-b" in info["models"]
        assert "model-a" not in info["models"]

    def test_get_dimension(
        self,
        store: EmbeddingStore,
        sample_texts: list[str],
        sample_embeddings: np.ndarray,
    ):
        """get_dimension should return the stored dim for a model."""
        store.put_batch("test-model", sample_texts, sample_embeddings)
        assert store.get_dimension("test-model") == 384
        assert store.get_dimension("unknown-model") is None

    def test_text_hash_deterministic(self):
        """Hash of the same text should always be the same."""
        h1 = _text_hash("hello world")
        h2 = _text_hash("hello world")
        assert h1 == h2
        assert len(h1) == 32

    def test_text_hash_different(self):
        """Different texts should produce different hashes."""
        h1 = _text_hash("hello")
        h2 = _text_hash("world")
        assert h1 != h2

    def test_default_store_path(self, tmp_path, monkeypatch):
        """Default store should use RAGICAMP_CACHE_DIR if set."""
        monkeypatch.setenv("RAGICAMP_CACHE_DIR", str(tmp_path / "custom_cache"))
        store = EmbeddingStore.default()
        assert "custom_cache" in str(store.db_path)
        store.close()


# ============================================================================
# CachedEmbedder / CachedEmbedderProvider tests
# ============================================================================


class MockEmbedder:
    """Minimal embedder for testing -- returns sequential embeddings."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.call_count = 0
        self.last_texts: list[str] = []

    def batch_encode(self, texts: list[str]) -> np.ndarray:
        self.call_count += 1
        self.last_texts = texts
        # Return deterministic embeddings based on text content
        rng = np.random.RandomState(abs(hash(tuple(texts))) % (2**31))
        return rng.randn(len(texts), self.dim).astype(np.float32)

    def get_dimension(self) -> int:
        return self.dim

    def unload(self) -> None:
        pass


class MockEmbedderProvider:
    """Minimal provider for testing -- yields MockEmbedder."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._mock_embedder: MockEmbedder | None = None

    @property
    def model_name(self) -> str:
        return "mock-embedder"

    @contextmanager
    def load(self, gpu_fraction: float | None = None) -> Iterator[MockEmbedder]:
        self._mock_embedder = MockEmbedder(self.dim)
        try:
            yield self._mock_embedder
        finally:
            self._mock_embedder = None

    @property
    def config(self):
        return MagicMock(model_name="mock-embedder", backend="mock")


class TestCachedEmbedder:
    """Tests for CachedEmbedder with lazy loading."""

    def test_full_cache_hit_no_model_load(
        self,
        store: EmbeddingStore,
        sample_texts: list[str],
        sample_embeddings: np.ndarray,
    ):
        """When all texts are cached, the real model should never be loaded."""
        from ragicamp.cache.cached_embedder import CachedEmbedder

        # Pre-fill the cache
        store.put_batch("mock-embedder", sample_texts, sample_embeddings)

        provider = MockEmbedderProvider()
        cached = CachedEmbedder(provider=provider, store=store)

        result = cached.batch_encode(sample_texts)

        # Result should match cached values
        np.testing.assert_array_almost_equal(result, sample_embeddings)
        # Real model should never have been loaded
        assert cached._real_embedder is None

        cached.cleanup()

    def test_full_miss_loads_model(
        self,
        store: EmbeddingStore,
        sample_texts: list[str],
    ):
        """When nothing is cached, the model should be loaded and results stored."""
        from ragicamp.cache.cached_embedder import CachedEmbedder

        provider = MockEmbedderProvider()
        cached = CachedEmbedder(provider=provider, store=store)

        result = cached.batch_encode(sample_texts)

        # Model should have been loaded
        assert cached._real_embedder is not None
        assert result.shape == (3, 384)

        # Should now be in the cache
        embs, mask = store.get_batch("mock-embedder", sample_texts)
        assert mask.all()
        np.testing.assert_array_almost_equal(embs, result)

        cached.cleanup()

    def test_partial_hit_only_encodes_misses(
        self,
        store: EmbeddingStore,
        sample_texts: list[str],
        sample_embeddings: np.ndarray,
    ):
        """Partial hit: model encodes only the miss texts."""
        from ragicamp.cache.cached_embedder import CachedEmbedder

        # Cache 2 of 3
        store.put_batch("mock-embedder", sample_texts[:2], sample_embeddings[:2])

        provider = MockEmbedderProvider()
        cached = CachedEmbedder(provider=provider, store=store)

        result = cached.batch_encode(sample_texts)

        # First 2 should match cached values
        np.testing.assert_array_almost_equal(result[:2], sample_embeddings[:2])
        # Model should have been loaded for the 1 miss
        assert cached._real_embedder is not None
        assert cached._real_embedder.last_texts == [sample_texts[2]]

        # The miss should now be in cache
        embs, mask = store.get_batch("mock-embedder", sample_texts)
        assert mask.all()

        cached.cleanup()

    def test_cleanup_unloads_model(self, store: EmbeddingStore, sample_texts: list[str]):
        """cleanup() should set _real_embedder back to None."""
        from ragicamp.cache.cached_embedder import CachedEmbedder

        provider = MockEmbedderProvider()
        cached = CachedEmbedder(provider=provider, store=store)

        # Force model load
        cached.batch_encode(sample_texts)
        assert cached._real_embedder is not None

        cached.cleanup()
        assert cached._real_embedder is None

    def test_get_dimension_from_cache(
        self,
        store: EmbeddingStore,
        sample_texts: list[str],
        sample_embeddings: np.ndarray,
    ):
        """get_dimension() should use cached dim without loading model."""
        from ragicamp.cache.cached_embedder import CachedEmbedder

        store.put_batch("mock-embedder", sample_texts, sample_embeddings)

        provider = MockEmbedderProvider()
        cached = CachedEmbedder(provider=provider, store=store)

        dim = cached.get_dimension()
        assert dim == 384
        # Model should NOT have been loaded
        assert cached._real_embedder is None

        cached.cleanup()


class TestCachedEmbedderProvider:
    """Tests for the provider wrapper."""

    def test_provider_interface_compatible(self, store: EmbeddingStore):
        """CachedEmbedderProvider should have same interface as EmbedderProvider."""
        from ragicamp.cache.cached_embedder import CachedEmbedderProvider

        inner = MockEmbedderProvider()
        cached_provider = CachedEmbedderProvider(inner, store)

        assert cached_provider.model_name == "mock-embedder"
        assert cached_provider.config is not None

    def test_context_manager_lifecycle(
        self,
        store: EmbeddingStore,
        sample_texts: list[str],
    ):
        """load() context manager should yield CachedEmbedder and cleanup on exit."""
        from ragicamp.cache.cached_embedder import CachedEmbedder, CachedEmbedderProvider

        inner = MockEmbedderProvider()
        cached_provider = CachedEmbedderProvider(inner, store)

        with cached_provider.load() as embedder:
            assert isinstance(embedder, CachedEmbedder)
            # Use it
            result = embedder.batch_encode(sample_texts)
            assert result.shape == (3, 384)

        # After exit, model should be cleaned up
        # (the CachedEmbedder's cleanup was called)

    def test_100pct_hit_no_gpu_usage(
        self,
        store: EmbeddingStore,
        sample_texts: list[str],
        sample_embeddings: np.ndarray,
    ):
        """Full cache hit scenario: no model loading happens at all."""
        from ragicamp.cache.cached_embedder import CachedEmbedderProvider

        store.put_batch("mock-embedder", sample_texts, sample_embeddings)

        inner = MockEmbedderProvider()
        cached_provider = CachedEmbedderProvider(inner, store)

        with cached_provider.load() as embedder:
            result = embedder.batch_encode(sample_texts)
            np.testing.assert_array_almost_equal(result, sample_embeddings)
            # Real model was never loaded
            assert embedder._real_embedder is None


# ============================================================================
# Factory integration test
# ============================================================================


class TestFactoryIntegration:
    """Test that ProviderFactory correctly wraps with cache."""

    def test_cache_enabled_by_default(self, monkeypatch, tmp_path):
        """With RAGICAMP_CACHE=1 (default), factory should wrap with cache."""
        from ragicamp.cache.cached_embedder import CachedEmbedderProvider

        monkeypatch.setenv("RAGICAMP_CACHE", "1")
        monkeypatch.setenv("RAGICAMP_CACHE_DIR", str(tmp_path))

        from ragicamp.factory.providers import ProviderFactory

        provider = ProviderFactory.create_embedder("test/model", backend="vllm")
        assert isinstance(provider, CachedEmbedderProvider)

    def test_cache_disabled(self, monkeypatch):
        """With RAGICAMP_CACHE=0, factory should return raw provider."""
        from ragicamp.models.providers import EmbedderProvider

        monkeypatch.setenv("RAGICAMP_CACHE", "0")

        from ragicamp.factory.providers import ProviderFactory

        provider = ProviderFactory.create_embedder("test/model", backend="vllm")
        assert isinstance(provider, EmbedderProvider)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
