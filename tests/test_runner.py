"""Tests for experiment execution runner module."""

import pytest

from ragicamp.spec import ExperimentSpec, build_specs, name_direct, name_rag

# Backward compatibility alias for tests
ExpSpec = ExperimentSpec


class TestExpSpec:
    """Tests for ExpSpec dataclass."""

    def test_exp_spec_defaults(self):
        """Test ExpSpec default values."""
        spec = ExperimentSpec(
            name="test_exp",
            exp_type="direct",
            model="hf:test/model",
            dataset="nq",
            prompt="default",
        )

        assert spec.name == "test_exp"
        assert spec.exp_type == "direct"
        assert spec.retriever is None
        assert spec.top_k == 5
        assert spec.query_transform is None
        assert spec.reranker is None
        assert spec.batch_size == 8

    def test_exp_spec_with_rag_params(self):
        """Test ExpSpec with RAG parameters."""
        spec = ExperimentSpec(
            name="rag_exp",
            exp_type="rag",
            model="hf:test/model",
            dataset="hotpotqa",
            prompt="concise",
            retriever="dense_bge",
            top_k=10,
            query_transform="hyde",
            reranker="bge",
            reranker_model="BAAI/bge-reranker-large",
        )

        assert spec.retriever == "dense_bge"
        assert spec.top_k == 10
        assert spec.query_transform == "hyde"
        assert spec.reranker == "bge"
        assert spec.reranker_model == "BAAI/bge-reranker-large"


class TestBuildSpecs:
    """Tests for build_specs function."""

    def test_build_direct_specs(self):
        """Test building direct experiment specs."""
        config = {
            "datasets": ["nq"],
            "direct": {
                "enabled": True,
                "models": ["hf:test/model"],
                "prompts": ["default"],
                "quantization": ["4bit"],
            },
        }

        specs = build_specs(config)

        assert len(specs) == 1
        assert specs[0].exp_type == "direct"
        assert specs[0].model == "hf:test/model"
        assert specs[0].dataset == "nq"

    def test_build_rag_specs(self):
        """Test building RAG experiment specs."""
        config = {
            "datasets": ["hotpotqa"],
            "rag": {
                "enabled": True,
                "models": ["hf:test/model"],
                "retrievers": [{"name": "dense_bge", "type": "dense"}],
                "top_k_values": [5],
                "prompts": ["concise"],
                "quantization": ["4bit"],
            },
        }

        specs = build_specs(config)

        assert len(specs) == 1
        assert specs[0].exp_type == "rag"
        assert specs[0].retriever == "dense_bge"

    def test_build_specs_multiple_datasets(self):
        """Test building specs with multiple datasets."""
        config = {
            "datasets": ["nq", "hotpotqa"],
            "direct": {
                "enabled": True,
                "models": ["hf:test/model"],
                "prompts": ["default"],
                "quantization": ["4bit"],
            },
        }

        specs = build_specs(config)

        assert len(specs) == 2
        datasets = [s.dataset for s in specs]
        assert "nq" in datasets
        assert "hotpotqa" in datasets

    def test_build_specs_multiple_top_k(self):
        """Test building specs with multiple top_k values."""
        config = {
            "datasets": ["nq"],
            "rag": {
                "enabled": True,
                "models": ["hf:test/model"],
                "retrievers": [{"name": "dense_bge"}],
                "top_k_values": [3, 5, 10],
                "prompts": ["default"],
                "quantization": ["4bit"],
            },
        }

        specs = build_specs(config)

        assert len(specs) == 3
        top_ks = [s.top_k for s in specs]
        assert 3 in top_ks
        assert 5 in top_ks
        assert 10 in top_ks

    def test_build_specs_with_query_transform(self):
        """Test building specs with query transforms."""
        config = {
            "datasets": ["nq"],
            "rag": {
                "enabled": True,
                "models": ["hf:test/model"],
                "retrievers": [{"name": "dense_bge"}],
                "top_k_values": [5],
                "prompts": ["default"],
                "quantization": ["4bit"],
                "query_transform": ["none", "hyde"],
            },
        }

        specs = build_specs(config)

        assert len(specs) == 2
        transforms = [s.query_transform for s in specs]
        assert None in transforms  # "none" becomes None
        assert "hyde" in transforms

    def test_build_specs_with_reranker(self):
        """Test building specs with rerankers."""
        config = {
            "datasets": ["nq"],
            "rag": {
                "enabled": True,
                "models": ["hf:test/model"],
                "retrievers": [{"name": "dense_bge"}],
                "top_k_values": [5],
                "prompts": ["default"],
                "quantization": ["4bit"],
                "reranker": {
                    "configs": [
                        {"enabled": False, "name": "none"},
                        {"enabled": True, "name": "bge", "model": "BAAI/bge-reranker-large"},
                    ]
                },
            },
        }

        specs = build_specs(config)

        assert len(specs) == 2
        rerankers = [s.reranker for s in specs]
        assert None in rerankers  # disabled reranker becomes None
        assert "bge" in rerankers

    def test_build_specs_openai_skips_non_4bit(self):
        """Test that OpenAI models skip non-4bit quantization."""
        config = {
            "datasets": ["nq"],
            "direct": {
                "enabled": True,
                "models": ["openai:gpt-4o-mini"],
                "prompts": ["default"],
                "quantization": ["4bit", "8bit"],
            },
        }

        specs = build_specs(config)

        # Should create 1 spec
        assert len(specs) == 1


class TestNamingFunctions:
    """Tests for experiment naming functions."""

    def test_name_direct(self):
        """Test direct experiment naming uses hash-based format."""
        name = name_direct("hf:meta-llama/Llama-3.2", "default", "nq")

        assert name.startswith("direct_")
        assert "_nq_" in name
        # Hash suffix: 8 hex chars at end
        assert len(name.split("_")[-1]) == 8

    def test_name_direct_deterministic(self):
        """Same inputs produce same name."""
        n1 = name_direct("hf:meta-llama/Llama-3.2", "default", "nq")
        n2 = name_direct("hf:meta-llama/Llama-3.2", "default", "nq")
        assert n1 == n2

    def test_name_direct_different_params_different_hash(self):
        """Different inputs produce different names."""
        n1 = name_direct("hf:test", "default", "nq")
        n2 = name_direct("hf:test", "concise", "nq")
        assert n1 != n2

    def test_name_rag(self):
        """Test RAG experiment naming uses hash-based format."""
        name = name_rag("hf:test", "default", "nq", "dense_bge", 5)

        assert name.startswith("rag_")
        assert "_nq_" in name
        assert len(name.split("_")[-1]) == 8

    def test_name_rag_deterministic(self):
        """Same RAG params produce same name."""
        n1 = name_rag("hf:test", "default", "nq", "dense", 5, query_transform="hyde")
        n2 = name_rag("hf:test", "default", "nq", "dense", 5, query_transform="hyde")
        assert n1 == n2

    def test_name_rag_with_query_transform(self):
        """Different query transform produces different hash."""
        n1 = name_rag("hf:test", "default", "nq", "dense", 5, query_transform="none")
        n2 = name_rag("hf:test", "default", "nq", "dense", 5, query_transform="hyde")
        assert n1 != n2

    def test_name_rag_with_reranker(self):
        """Different reranker produces different hash."""
        n1 = name_rag("hf:test", "default", "nq", "dense", 5, reranker="none")
        n2 = name_rag("hf:test", "default", "nq", "dense", 5, reranker="bge")
        assert n1 != n2

    def test_name_rag_agent_type_prefix(self):
        """Non-fixed_rag agent types get different prefix."""
        n1 = name_rag("hf:test", "default", "nq", "dense", 5, agent_type="iterative_rag")
        assert n1.startswith("iterative_")

        n2 = name_rag("hf:test", "default", "nq", "dense", 5, agent_type="self_rag")
        assert n2.startswith("self_")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
