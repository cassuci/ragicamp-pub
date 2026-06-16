"""Tests for configuration schemas and validation."""

import pytest
from pydantic import ValidationError

from ragicamp.config.schemas import (
    AgentConfig,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    MetricConfig,
    ModelConfig,
)


class TestEvaluationConfig:
    """Test evaluation configuration and modes."""

    def test_default_mode_is_both(self):
        """Test that default mode is 'both'."""
        config = EvaluationConfig()
        assert config.mode == "both"

    def test_generate_mode(self):
        """Test generate mode configuration."""
        config = EvaluationConfig(mode="generate", batch_size=32)
        assert config.mode == "generate"
        assert config.batch_size == 32

    def test_evaluate_mode_requires_predictions_file(self):
        """Test that evaluate mode requires predictions_file."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationConfig(mode="evaluate")

        assert "predictions_file must be set" in str(exc_info.value)

    def test_evaluate_mode_with_predictions_file(self):
        """Test evaluate mode with predictions_file."""
        config = EvaluationConfig(mode="evaluate", predictions_file="outputs/predictions_raw.json")
        assert config.mode == "evaluate"
        assert config.predictions_file == "outputs/predictions_raw.json"

    def test_both_mode(self):
        """Test both mode configuration."""
        config = EvaluationConfig(mode="both", batch_size=8, num_examples=100)
        assert config.mode == "both"
        assert config.batch_size == 8
        assert config.num_examples == 100

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationConfig(mode="invalid_mode")

        assert "mode must be one of" in str(exc_info.value)

    def test_valid_modes(self):
        """Test all valid modes."""
        valid_modes = ["generate", "evaluate", "both"]

        for mode in valid_modes:
            if mode == "evaluate":
                config = EvaluationConfig(mode=mode, predictions_file="test.json")
            else:
                config = EvaluationConfig(mode=mode)
            assert config.mode == mode


class TestModelConfig:
    """Test model configuration."""

    def test_huggingface_model_config(self):
        """Test HuggingFace model configuration."""
        config = ModelConfig(
            type="huggingface", model_name="google/gemma-2-2b-it", device="cuda", load_in_8bit=True
        )

        assert config.type == "huggingface"
        assert config.model_name == "google/gemma-2-2b-it"
        assert config.device == "cuda"
        assert config.load_in_8bit is True

    def test_openai_model_config(self):
        """Test OpenAI model configuration."""
        config = ModelConfig(type="openai", model_name="gpt-4o", temperature=0.0)

        assert config.type == "openai"
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.0

    def test_model_config_defaults(self):
        """Test model configuration defaults."""
        config = ModelConfig(model_name="test-model")

        assert config.type == "huggingface"  # default
        assert config.device == "cuda"  # default
        assert config.load_in_8bit is False  # default
        assert config.temperature == 0.7  # default


class TestAgentConfig:
    """Test agent configuration."""

    def test_direct_llm_agent_config(self):
        """Test DirectLLM agent configuration."""
        config = AgentConfig(
            type="direct_llm", name="baseline_agent", system_prompt="You are helpful"
        )

        assert config.type == "direct_llm"
        assert config.name == "baseline_agent"
        assert config.system_prompt == "You are helpful"

    def test_rag_agent_config(self):
        """Test RAG agent configuration."""
        config = AgentConfig(type="fixed_rag", name="rag_agent", top_k=10)

        assert config.type == "fixed_rag"
        assert config.top_k == 10


class TestDatasetConfig:
    """Test dataset configuration."""

    def test_dataset_config(self):
        """Test dataset configuration."""
        config = DatasetConfig(
            name="natural_questions", split="validation", num_examples=100, filter_no_answer=True
        )

        assert config.name == "natural_questions"
        assert config.split == "validation"
        assert config.num_examples == 100
        assert config.filter_no_answer is True

    def test_dataset_config_defaults(self):
        """Test dataset configuration defaults."""
        config = DatasetConfig(name="natural_questions")

        assert config.split == "validation"  # default
        assert config.filter_no_answer is True  # default


class TestMetricConfig:
    """Test metric configuration."""

    def test_metric_config_from_string(self):
        """Test creating metric config from string."""
        # This is handled by ExperimentConfig normalization
        pass

    def test_metric_config_with_params(self):
        """Test metric config with parameters."""
        config = MetricConfig(
            name="llm_judge_qa", params={"judgment_type": "binary", "batch_size": 8}
        )

        assert config.name == "llm_judge_qa"
        assert config.params["judgment_type"] == "binary"
        assert config.params["batch_size"] == 8


class TestExperimentConfig:
    """Test complete experiment configuration."""

    def test_minimal_experiment_config(self):
        """Test minimal valid experiment configuration."""
        config = ExperimentConfig(
            agent={"type": "direct_llm", "name": "test"},
            model={"model_name": "test-model"},
            dataset={"name": "natural_questions"},
            metrics=["exact_match", "f1"],
        )

        assert config.agent.type == "direct_llm"
        assert config.model.model_name == "test-model"
        assert config.dataset.name == "natural_questions"
        assert len(config.metrics) == 2

    def test_experiment_config_with_generate_mode(self):
        """Test experiment config with generate mode."""
        config = ExperimentConfig(
            agent={"type": "direct_llm", "name": "test"},
            model={"model_name": "test-model"},
            dataset={"name": "natural_questions"},
            metrics=["exact_match"],
            evaluation={"mode": "generate", "batch_size": 32},
        )

        assert config.evaluation.mode == "generate"
        assert config.evaluation.batch_size == 32

    def test_experiment_config_with_evaluate_mode(self):
        """Test experiment config with evaluate mode."""
        config = ExperimentConfig(
            agent={"type": "direct_llm", "name": "test"},
            model={"model_name": "test-model"},
            dataset={"name": "natural_questions"},
            metrics=["exact_match"],
            evaluation={"mode": "evaluate", "predictions_file": "outputs/predictions.json"},
        )

        assert config.evaluation.mode == "evaluate"
        assert config.evaluation.predictions_file == "outputs/predictions.json"

    def test_metrics_normalization(self):
        """Test that metrics are normalized to dicts."""
        config = ExperimentConfig(
            agent={"type": "direct_llm", "name": "test"},
            model={"model_name": "test-model"},
            dataset={"name": "natural_questions"},
            metrics=["exact_match", {"name": "bertscore", "params": {"model": "bert-base"}}],
        )

        # All metrics should be dicts with 'name' and 'params'
        assert all(isinstance(m, dict) for m in config.metrics)
        assert config.metrics[0]["name"] == "exact_match"
        assert config.metrics[1]["name"] == "bertscore"

    def test_rag_agent_requires_retriever(self):
        """Test that RAG agents work without retriever in config (retriever optional at config level)."""
        # This is now valid - retriever validation happens at runtime, not config time
        config = ExperimentConfig(
            agent={"type": "fixed_rag", "name": "test"},
            model={"model_name": "test-model"},
            dataset={"name": "natural_questions"},
            metrics=["exact_match"],
        )

        # Config is valid, retriever is optional at config level
        assert config.agent.type == "fixed_rag"
        assert config.retriever is None

    def test_rag_agent_with_retriever(self):
        """Test RAG agent with retriever configuration."""
        config = ExperimentConfig(
            agent={"type": "fixed_rag", "name": "test"},
            model={"model_name": "test-model"},
            dataset={"name": "natural_questions"},
            metrics=["exact_match"],
            retriever={"type": "dense", "embedding_model": "all-MiniLM-L6-v2"},
        )

        assert config.retriever is not None
        assert config.retriever.type == "dense"


class TestConfigValidation:
    """Test configuration validation."""

    def test_missing_required_fields(self):
        """Test that missing required fields raise errors."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                agent={"type": "direct_llm", "name": "test"},
                # Missing model, dataset, metrics
            )

    def test_extra_fields_not_allowed_at_top_level(self):
        """Test that extra fields are not allowed at top level."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                agent={"type": "direct_llm", "name": "test"},
                model={"model_name": "test-model"},
                dataset={"name": "natural_questions"},
                metrics=["exact_match"],
                unknown_field="should_fail",  # Extra field
            )

    def test_extra_fields_allowed_in_sub_configs(self):
        """Test that extra fields are allowed in sub-configs (for extensibility)."""
        config = ExperimentConfig(
            agent={"type": "direct_llm", "name": "test"},
            model={
                "model_name": "test-model",
                "custom_param": "value",  # Extra param, should be allowed
            },
            dataset={"name": "natural_questions"},
            metrics=["exact_match"],
        )

        # Should not raise error
        assert config.model.model_name == "test-model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
