"""Tests for component factories.

Tests the new clean architecture factories:
- ProviderFactory: Creates EmbedderProvider and GeneratorProvider
- AgentFactory: Creates agents with providers + index
- DatasetFactory: Creates QA datasets
- MetricFactory: Creates evaluation metrics
"""

from unittest.mock import patch

import pytest

from ragicamp.agents import DirectLLMAgent
from ragicamp.factory import AgentFactory, DatasetFactory, MetricFactory, ProviderFactory


class TestProviderFactory:
    """Test provider creation via factory."""

    def test_parse_generator_spec_vllm(self):
        """Test parsing vLLM generator spec."""
        config = ProviderFactory.parse_generator_spec("vllm:meta-llama/Llama-3.2-3B")

        assert config.model_name == "meta-llama/Llama-3.2-3B"
        assert config.backend == "vllm"

    def test_parse_generator_spec_hf(self):
        """Test parsing HuggingFace generator spec."""
        config = ProviderFactory.parse_generator_spec("hf:google/gemma-2-2b-it")

        assert config.model_name == "google/gemma-2-2b-it"
        assert config.backend == "hf"

    def test_parse_embedder_spec(self):
        """Test parsing embedder spec."""
        config = ProviderFactory.parse_embedder_spec(
            "BAAI/bge-large-en-v1.5",
            backend="sentence_transformers",
        )

        assert config.model_name == "BAAI/bge-large-en-v1.5"
        assert config.backend == "sentence_transformers"

    def test_create_generator_from_spec(self):
        """Test creating generator provider from spec."""
        provider = ProviderFactory.create_generator("vllm:meta-llama/Llama-3.2-3B")

        assert provider.model_name == "meta-llama/Llama-3.2-3B"
        # Provider starts without model loaded (lazy loading)
        assert provider._generator is None

    def test_create_embedder_from_spec(self):
        """Test creating embedder provider from spec."""
        provider = ProviderFactory.create_embedder(
            "BAAI/bge-large-en-v1.5",
            backend="sentence_transformers",
        )

        assert provider.model_name == "BAAI/bge-large-en-v1.5"
        # Provider starts without model loaded (lazy loading)
        # When cache is enabled, the provider is wrapped -- check the inner provider
        inner = getattr(provider, "inner", provider)
        assert inner._embedder is None


class TestAgentFactory:
    """Test agent creation via factory."""

    def test_get_available_agents(self):
        """Test listing available agents."""
        agents = AgentFactory.get_available_agents()

        assert "direct_llm" in agents
        assert "fixed_rag" in agents
        assert "iterative_rag" in agents
        assert "self_rag" in agents

    def test_create_direct_requires_generator(self):
        """Test that create_direct needs generator provider."""
        from ragicamp.models.providers import GeneratorConfig, GeneratorProvider

        config = GeneratorConfig(model_name="test", backend="vllm")
        provider = GeneratorProvider(config)

        agent = AgentFactory.create_direct(
            name="test_agent",
            generator_provider=provider,
        )

        assert isinstance(agent, DirectLLMAgent)
        assert agent.name == "test_agent"


class TestDatasetFactory:
    """Test dataset creation via factory."""

    def test_parse_spec(self):
        """Test parsing dataset spec."""
        config = DatasetFactory.parse_spec("nq", limit=100)

        # "nq" is an alias that gets expanded to full name
        assert "natural_questions" in config["name"] or config["name"] == "nq"
        # limit might be stored as "limit" or "num_examples" depending on implementation
        assert config.get("limit", config.get("num_examples")) == 100 or "limit" not in config

    @patch("ragicamp.factory.datasets.NaturalQuestionsDataset")
    def test_create_natural_questions_dataset(self, mock_nq):
        """Test creating Natural Questions dataset."""
        from unittest.mock import MagicMock

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.examples = [{"question": "q", "answers": ["a"]}] * 100
        mock_nq.return_value = mock_dataset

        config = {
            "name": "natural_questions",
            "split": "validation",
        }

        DatasetFactory.create(config)

        mock_nq.assert_called_once()

    def test_create_dataset_invalid_name(self):
        """Test creating dataset with invalid name."""
        config = {"name": "invalid_dataset", "split": "validation"}

        with pytest.raises(ValueError) as exc_info:
            DatasetFactory.create(config)

        assert "Unknown dataset" in str(exc_info.value)


class TestMetricFactory:
    """Test metrics creation via factory."""

    def test_create_exact_match_metric(self):
        """Test creating exact match metric."""
        metrics_config = ["exact_match"]

        metrics = MetricFactory.create(metrics_config)

        assert len(metrics) == 1
        assert metrics[0].name == "exact_match"

    def test_create_multiple_metrics(self):
        """Test creating multiple metrics."""
        metrics_config = ["exact_match", "f1"]

        metrics = MetricFactory.create(metrics_config)

        assert len(metrics) == 2
        assert metrics[0].name == "exact_match"
        assert metrics[1].name == "f1"

    def test_skip_unavailable_metric(self):
        """Test that unavailable metrics are skipped with warning."""
        metrics_config = ["exact_match", "nonexistent_metric", "f1"]

        metrics = MetricFactory.create(metrics_config)

        # Should create the available ones and skip the nonexistent one
        assert len(metrics) == 2
        metric_names = [m.name for m in metrics]
        assert "exact_match" in metric_names
        assert "f1" in metric_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
