"""Tests for provider ref-counting behavior.

All three providers (Generator, Embedder, Reranker) support ref-counting:
nested ``with provider.load()`` calls reuse the already-loaded model and only
unload when the outermost context exits.

These tests verify:
- Backward compatibility: single load/unload works identically to before
- Nested loads reuse the model (same object, loaded once)
- Unload only happens when the outermost context exits
- Exception during first load leaves clean state (refcount 0)
- Exception in user code still decrements refcount correctly
- Refcount never goes negative
"""

from unittest.mock import MagicMock, patch

import pytest

from ragicamp.models.providers.embedder import EmbedderConfig, EmbedderProvider
from ragicamp.models.providers.generator import GeneratorConfig, GeneratorProvider
from ragicamp.models.providers.reranker import RerankerConfig, RerankerProvider

# ---------------------------------------------------------------------------
# Helpers: patch internal load methods to avoid real model instantiation
# ---------------------------------------------------------------------------


class FakeGenerator:
    """Lightweight stand-in for a real Generator."""

    def __init__(self):
        self.model_name = "fake"

    def batch_generate(self, prompts, **kwargs):
        return ["answer"] * len(prompts)

    def unload(self):
        pass


class FakeEmbedder:
    """Lightweight stand-in for a real Embedder."""

    def batch_encode(self, texts):
        import numpy as np

        return np.zeros((len(texts), 4), dtype="float32")

    def get_dimension(self):
        return 4

    def unload(self):
        pass


class FakeRerankerModel:
    """Lightweight stand-in for CrossEncoder."""

    def predict(self, pairs, **kwargs):
        return [0.5] * len(pairs)


@pytest.fixture()
def gen_provider():
    """GeneratorProvider with patched internals (no GPU needed)."""
    provider = GeneratorProvider(GeneratorConfig(model_name="fake/gen", backend="hf"))
    fake = FakeGenerator()
    with patch.object(provider, "_load_hf", return_value=fake):
        with patch.object(provider, "_unload", wraps=provider._unload) as mock_unload:
            provider._mock_unload = mock_unload
            provider._fake = fake
            yield provider


@pytest.fixture()
def emb_provider():
    """EmbedderProvider with patched internals (no GPU needed)."""
    provider = EmbedderProvider(
        EmbedderConfig(model_name="fake/emb", backend="sentence_transformers")
    )
    fake = FakeEmbedder()
    with patch.object(provider, "_load_sentence_transformers", return_value=fake):
        with patch.object(provider, "_unload", wraps=provider._unload) as mock_unload:
            provider._mock_unload = mock_unload
            provider._fake = fake
            yield provider


@pytest.fixture()
def rerank_provider():
    """RerankerProvider with patched internals (no GPU needed).

    The reranker imports torch and CrossEncoder locally inside load(),
    so we patch them via the builtins __import__ path isn't viable.
    Instead we mock the whole first-load block by patching at the
    sentence_transformers and torch import targets.
    """
    provider = RerankerProvider(RerankerConfig(model_name="bge"))

    fake_model = FakeRerankerModel()
    fake_cross_encoder = MagicMock(return_value=fake_model)
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    with patch.dict(
        "sys.modules",
        {
            "torch": fake_torch,
            "sentence_transformers": MagicMock(CrossEncoder=fake_cross_encoder),
        },
    ):
        with patch.object(provider, "_unload", wraps=provider._unload) as mock_unload:
            provider._mock_unload = mock_unload
            yield provider


# ===================================================================
# GeneratorProvider — thorough tests
# ===================================================================


class TestGeneratorRefcount:
    """Test ref-counting on GeneratorProvider."""

    def test_single_load_unload(self, gen_provider):
        """Single load/unload — backward compatible behavior."""
        assert gen_provider._refcount == 0
        assert gen_provider._generator is None

        with gen_provider.load() as gen:
            assert gen is gen_provider._fake
            assert gen_provider._refcount == 1

        assert gen_provider._refcount == 0
        assert gen_provider._mock_unload.call_count == 1

    def test_nested_load_reuses_model(self, gen_provider):
        """Nested load() yields the same model object, loads only once."""
        with gen_provider.load() as gen_outer:
            assert gen_provider._refcount == 1
            with gen_provider.load() as gen_inner:
                assert gen_provider._refcount == 2
                assert gen_inner is gen_outer  # same object
            # Inner exited — still loaded
            assert gen_provider._refcount == 1
            assert gen_provider._mock_unload.call_count == 0

        # Outer exited — now unloaded
        assert gen_provider._refcount == 0
        assert gen_provider._mock_unload.call_count == 1

    def test_triple_nesting(self, gen_provider):
        """Three levels of nesting: refcount 1→2→3→2→1→0."""
        with gen_provider.load() as g1:
            assert gen_provider._refcount == 1
            with gen_provider.load():
                assert gen_provider._refcount == 2
                with gen_provider.load() as g3:
                    assert gen_provider._refcount == 3
                    assert g3 is g1
                assert gen_provider._refcount == 2
            assert gen_provider._refcount == 1
            assert gen_provider._mock_unload.call_count == 0

        assert gen_provider._refcount == 0
        assert gen_provider._mock_unload.call_count == 1

    def test_exception_in_user_code_decrements_refcount(self, gen_provider):
        """Exception in user code still decrements refcount correctly."""
        with pytest.raises(ValueError, match="boom"):
            with gen_provider.load():
                with gen_provider.load():
                    assert gen_provider._refcount == 2
                    raise ValueError("boom")

        assert gen_provider._refcount == 0
        assert gen_provider._mock_unload.call_count == 1

    def test_exception_during_first_load_resets_state(self):
        """If model loading fails, refcount goes back to 0."""
        provider = GeneratorProvider(GeneratorConfig(model_name="fail/model", backend="hf"))

        with patch.object(provider, "_load_hf", side_effect=RuntimeError("OOM")):
            with pytest.raises(RuntimeError, match="OOM"):
                with provider.load():
                    pass  # pragma: no cover

        assert provider._refcount == 0
        assert provider._generator is None

    def test_load_after_failed_load_works(self):
        """After a failed load, a subsequent load attempt works normally."""
        provider = GeneratorProvider(GeneratorConfig(model_name="retry/model", backend="hf"))
        fake = FakeGenerator()

        # First attempt: fails
        with patch.object(provider, "_load_hf", side_effect=RuntimeError("OOM")):
            with pytest.raises(RuntimeError):
                with provider.load():
                    pass  # pragma: no cover

        assert provider._refcount == 0

        # Second attempt: succeeds
        with patch.object(provider, "_load_hf", return_value=fake):
            with provider.load() as gen:
                assert gen is fake
                assert provider._refcount == 1

        assert provider._refcount == 0

    def test_refcount_never_negative(self, gen_provider):
        """Refcount should never go below zero."""
        with gen_provider.load():
            pass

        assert gen_provider._refcount == 0
        # Even after multiple exits, refcount stays at 0 (no double-unload)

    def test_sequential_load_unload_cycles(self, gen_provider):
        """Multiple sequential load/unload cycles work correctly."""
        for _ in range(3):
            with gen_provider.load() as gen:
                assert gen is gen_provider._fake
                assert gen_provider._refcount == 1
            assert gen_provider._refcount == 0

        assert gen_provider._mock_unload.call_count == 3

    def test_yielded_model_is_usable(self, gen_provider):
        """Model returned by nested load() is functional."""
        with gen_provider.load() as gen:
            result1 = gen.batch_generate(["hello"])
            with gen_provider.load() as gen_inner:
                result2 = gen_inner.batch_generate(["world"])
                assert result1 == ["answer"]
                assert result2 == ["answer"]


# ===================================================================
# EmbedderProvider — verify same ref-counting behavior
# ===================================================================


class TestEmbedderRefcount:
    """Test ref-counting on EmbedderProvider."""

    def test_single_load_unload(self, emb_provider):
        """Single load/unload works."""
        with emb_provider.load() as emb:
            assert emb is emb_provider._fake
            assert emb_provider._refcount == 1

        assert emb_provider._refcount == 0
        assert emb_provider._mock_unload.call_count == 1

    def test_nested_load_reuses_model(self, emb_provider):
        """Nested load reuses model, unloads only at outermost exit."""
        with emb_provider.load() as emb_outer:
            with emb_provider.load() as emb_inner:
                assert emb_inner is emb_outer
                assert emb_provider._refcount == 2
            assert emb_provider._refcount == 1
            assert emb_provider._mock_unload.call_count == 0

        assert emb_provider._refcount == 0
        assert emb_provider._mock_unload.call_count == 1

    def test_exception_cleanup(self, emb_provider):
        """Exception cleans up refcount correctly."""
        with pytest.raises(RuntimeError):
            with emb_provider.load():
                with emb_provider.load():
                    raise RuntimeError("fail")

        assert emb_provider._refcount == 0

    def test_exception_during_first_load(self):
        """Failed model load leaves clean state."""
        provider = EmbedderProvider(
            EmbedderConfig(model_name="fail", backend="sentence_transformers")
        )

        with patch.object(provider, "_load_sentence_transformers", side_effect=RuntimeError("OOM")):
            with pytest.raises(RuntimeError):
                with provider.load():
                    pass  # pragma: no cover

        assert provider._refcount == 0
        assert provider._embedder is None


# ===================================================================
# RerankerProvider — verify same ref-counting behavior
# ===================================================================


class TestRerankerRefcount:
    """Test ref-counting on RerankerProvider."""

    def test_single_load_unload(self, rerank_provider):
        """Single load/unload works."""
        with rerank_provider.load() as reranker:
            assert reranker is not None
            assert rerank_provider._refcount == 1

        assert rerank_provider._refcount == 0
        assert rerank_provider._mock_unload.call_count == 1

    def test_nested_load_reuses_model(self, rerank_provider):
        """Nested load reuses model, unloads only at outermost exit."""
        with rerank_provider.load() as rr_outer:
            with rerank_provider.load() as rr_inner:
                assert rr_inner is rr_outer
                assert rerank_provider._refcount == 2
            assert rerank_provider._refcount == 1
            assert rerank_provider._mock_unload.call_count == 0

        assert rerank_provider._refcount == 0
        assert rerank_provider._mock_unload.call_count == 1

    def test_exception_cleanup(self, rerank_provider):
        """Exception cleans up refcount correctly."""
        with pytest.raises(ValueError):
            with rerank_provider.load():
                with rerank_provider.load():
                    raise ValueError("test")

        assert rerank_provider._refcount == 0
