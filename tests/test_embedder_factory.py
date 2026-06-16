"""Tests for embedder factory and protocol."""

import pytest

from ragicamp.models.embedder import Embedder


class TestEmbedderProtocol:
    """Tests for Embedder protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that Embedder is runtime checkable."""

        # Create a minimal mock that satisfies the protocol
        class MockEmbedder:
            def encode(self, sentences, **kwargs):
                import numpy as np

                return np.zeros((len(sentences), 128))

            def get_sentence_embedding_dimension(self):
                return 128

            def unload(self):
                pass

        mock = MockEmbedder()
        assert isinstance(mock, Embedder)

    def test_non_conforming_class_fails_check(self):
        """Test that non-conforming class fails isinstance check."""

        class NotAnEmbedder:
            pass

        assert not isinstance(NotAnEmbedder(), Embedder)


class TestCreateEmbedder:
    """Tests for create_embedder factory.

    Note: These tests use mocking to avoid loading actual models.
    """

    def test_factory_returns_vllm_for_vllm_backend(self, monkeypatch):
        """Test factory returns VLLMEmbedder for vllm backend."""

        # Mock VLLMEmbedder to avoid loading actual model
        class MockVLLMEmbedder:
            def __init__(self, model_name, gpu_memory_fraction, enforce_eager):
                self.model_name = model_name
                self.gpu_memory_fraction = gpu_memory_fraction
                self.enforce_eager = enforce_eager

        import ragicamp.models.embedder as embedder_module

        monkeypatch.setattr(
            "ragicamp.models.vllm_embedder.VLLMEmbedder",
            MockVLLMEmbedder,
        )

        # Need to also patch in the embedder module's import
        original_create = embedder_module.create_embedder

        def patched_create(model_name, backend="vllm", **kwargs):
            if backend == "vllm":
                return MockVLLMEmbedder(
                    model_name=model_name,
                    gpu_memory_fraction=kwargs.get("gpu_memory_fraction", 0.9),
                    enforce_eager=kwargs.get("enforce_eager", False),
                )
            return original_create(model_name, backend, **kwargs)

        monkeypatch.setattr(embedder_module, "create_embedder", patched_create)

        embedder = embedder_module.create_embedder(
            model_name="test/model",
            backend="vllm",
            gpu_memory_fraction=0.8,
        )

        assert isinstance(embedder, MockVLLMEmbedder)
        assert embedder.model_name == "test/model"
        assert embedder.gpu_memory_fraction == 0.8

    def test_factory_returns_st_for_sentence_transformers_backend(self, monkeypatch):
        """Test factory returns SentenceTransformerEmbedder for sentence_transformers backend."""

        class MockSTEmbedder:
            def __init__(self, model_name, use_flash_attn, use_compile):
                self.model_name = model_name
                self.use_flash_attn = use_flash_attn
                self.use_compile = use_compile

        import ragicamp.models.embedder as embedder_module

        original_create = embedder_module.create_embedder

        def patched_create(model_name, backend="vllm", **kwargs):
            if backend == "sentence_transformers":
                return MockSTEmbedder(
                    model_name=model_name,
                    use_flash_attn=kwargs.get("use_flash_attn", True),
                    use_compile=kwargs.get("use_compile", True),
                )
            return original_create(model_name, backend, **kwargs)

        monkeypatch.setattr(embedder_module, "create_embedder", patched_create)

        embedder = embedder_module.create_embedder(
            model_name="test/model",
            backend="sentence_transformers",
            use_flash_attn=False,
        )

        assert isinstance(embedder, MockSTEmbedder)
        assert embedder.model_name == "test/model"
        assert embedder.use_flash_attn is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
