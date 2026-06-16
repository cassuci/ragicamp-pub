"""Tests for configuration validation module."""

import pytest

from ragicamp.config.validation import (
    VALID_DATASETS,
    VALID_PROVIDERS,
    ConfigError,
    validate_config,
    validate_dataset,
    validate_model_spec,
)


class TestValidateModelSpec:
    """Tests for model spec validation."""

    def test_valid_hf_spec(self):
        """Test valid HuggingFace model spec."""
        validate_model_spec("hf:meta-llama/Llama-3.2-3B-Instruct")
        # Should not raise

    def test_valid_openai_spec(self):
        """Test valid OpenAI model spec."""
        validate_model_spec("openai:gpt-4o-mini")
        # Should not raise

    def test_valid_vllm_spec(self):
        """Test valid vLLM model spec."""
        validate_model_spec("vllm:meta-llama/Llama-2-7b")
        # Should not raise

    def test_invalid_spec_no_colon(self):
        """Test that spec without colon raises error."""
        with pytest.raises(ConfigError) as exc_info:
            validate_model_spec("invalid-spec")

        assert "Invalid model spec" in str(exc_info.value)
        assert "provider:model_name" in str(exc_info.value)

    def test_invalid_provider(self):
        """Test that unknown provider raises error."""
        with pytest.raises(ConfigError) as exc_info:
            validate_model_spec("anthropic:claude-3")

        assert "Unknown model provider" in str(exc_info.value)
        assert "anthropic" in str(exc_info.value)


class TestValidateDataset:
    """Tests for dataset validation."""

    def test_valid_datasets(self):
        """Test all valid datasets."""
        for ds in VALID_DATASETS:
            validate_dataset(ds)
            # Should not raise

    def test_invalid_dataset(self):
        """Test that unknown dataset raises error."""
        with pytest.raises(ConfigError) as exc_info:
            validate_dataset("unknown_dataset")

        assert "Unknown dataset" in str(exc_info.value)


class TestValidateConfig:
    """Tests for config validation."""

    def test_minimal_valid_config(self):
        """Test minimal valid config."""
        config = {
            "name": "test_study",
            "datasets": ["nq"],
        }
        warnings = validate_config(config)
        assert len(warnings) == 0

    def test_missing_name_raises(self):
        """Test that missing name raises error."""
        config = {"datasets": ["nq"]}

        with pytest.raises(ConfigError) as exc_info:
            validate_config(config)

        assert "name" in str(exc_info.value)

    def test_no_datasets_warning(self):
        """Test that no datasets produces warning."""
        config = {"name": "test"}
        warnings = validate_config(config)

        assert any("No datasets" in w for w in warnings)

    def test_invalid_dataset_raises(self):
        """Test that invalid dataset raises error."""
        config = {
            "name": "test",
            "datasets": ["invalid_ds"],
        }

        with pytest.raises(ConfigError):
            validate_config(config)

    def test_direct_enabled_no_models_warning(self):
        """Test warning when direct enabled without models."""
        config = {
            "name": "test",
            "datasets": ["nq"],
            "direct": {"enabled": True},
        }
        warnings = validate_config(config)

        assert any("no models" in w.lower() for w in warnings)

    def test_rag_enabled_no_retrievers_warning(self):
        """Test warning when RAG enabled without retrievers."""
        config = {
            "name": "test",
            "datasets": ["nq"],
            "rag": {"enabled": True, "models": ["hf:test/model"]},
        }
        warnings = validate_config(config)

        assert any("no retrievers" in w.lower() for w in warnings)


class TestConstants:
    """Test validation constants are complete."""

    def test_valid_datasets_not_empty(self):
        """Test VALID_DATASETS is not empty."""
        assert len(VALID_DATASETS) > 0
        assert "nq" in VALID_DATASETS
        assert "hotpotqa" in VALID_DATASETS

    def test_valid_providers_not_empty(self):
        """Test VALID_PROVIDERS is not empty."""
        assert len(VALID_PROVIDERS) > 0
        assert "hf" in VALID_PROVIDERS
        assert "openai" in VALID_PROVIDERS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
