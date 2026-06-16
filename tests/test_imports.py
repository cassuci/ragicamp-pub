"""Tests for module imports.

These tests ensure all key modules can be imported without errors.
This catches issues like missing re-exports or incorrect import paths.

Updated for clean architecture migration.
"""


class TestCLIImports:
    """Test that CLI modules import correctly."""

    def test_import_cli_main(self):
        """Test importing cli.main module."""
        from ragicamp.cli.main import main

        assert callable(main)

    def test_import_cli_study(self):
        """Test importing cli.study module."""
        from ragicamp.cli import study

        assert hasattr(study, "run_study")

    def test_cli_study_has_build_specs(self):
        """Test that cli.study exports build_specs."""
        from ragicamp.cli.study import build_specs

        assert callable(build_specs)

    def test_cli_study_has_run_spec(self):
        """Test that cli.study exports run_spec."""
        from ragicamp.cli.study import run_spec

        assert callable(run_spec)


class TestModelImports:
    """Test that model modules import correctly."""

    def test_import_huggingface_model(self):
        """Test importing HuggingFaceModel."""
        from ragicamp.models import HuggingFaceModel

        assert HuggingFaceModel is not None

    def test_import_vllm_model(self):
        """Test importing VLLMModel."""
        from ragicamp.models import VLLMModel

        assert VLLMModel is not None

    def test_import_openai_model(self):
        """Test importing OpenAIModel."""
        from ragicamp.models import OpenAIModel

        assert OpenAIModel is not None

    def test_import_providers(self):
        """Test importing provider classes."""
        from ragicamp.models import EmbedderProvider, GeneratorProvider

        assert EmbedderProvider is not None
        assert GeneratorProvider is not None


class TestIndexImports:
    """Test that index modules import correctly."""

    def test_import_vector_index(self):
        """Test importing VectorIndex."""
        from ragicamp.indexes import VectorIndex

        assert VectorIndex is not None

    def test_import_index_builder(self):
        """Test importing IndexBuilder."""
        from ragicamp.indexes import IndexBuilder

        assert IndexBuilder is not None

    def test_import_index_config(self):
        """Test importing IndexConfig."""
        from ragicamp.indexes import IndexConfig

        assert IndexConfig is not None


class TestRetrieverImports:
    """Test that retriever modules import correctly."""

    def test_import_hybrid_searcher(self):
        """Test importing HybridSearcher."""
        from ragicamp.retrievers import HybridSearcher

        assert HybridSearcher is not None

    def test_import_sparse_index(self):
        """Test importing SparseIndex."""
        from ragicamp.retrievers import SparseIndex

        assert SparseIndex is not None


class TestFactoryImports:
    """Test that factory modules import correctly."""

    def test_import_provider_factory(self):
        """Test importing ProviderFactory."""
        from ragicamp.factory import ProviderFactory

        assert ProviderFactory is not None

    def test_import_agent_factory(self):
        """Test importing AgentFactory."""
        from ragicamp.factory import AgentFactory

        assert AgentFactory is not None

    def test_import_metric_factory(self):
        """Test importing MetricFactory."""
        from ragicamp.factory import MetricFactory

        assert MetricFactory is not None

    def test_import_dataset_factory(self):
        """Test importing DatasetFactory."""
        from ragicamp.factory import DatasetFactory

        assert DatasetFactory is not None


class TestAgentImports:
    """Test that agent modules import correctly."""

    def test_import_agent_protocol(self):
        """Test importing Agent protocol."""
        from ragicamp.agents import Agent

        assert Agent is not None

    def test_import_direct_llm_agent(self):
        """Test importing DirectLLMAgent."""
        from ragicamp.agents import DirectLLMAgent

        assert DirectLLMAgent is not None

    def test_import_fixed_rag_agent(self):
        """Test importing FixedRAGAgent."""
        from ragicamp.agents import FixedRAGAgent

        assert FixedRAGAgent is not None

    def test_import_query_and_result(self):
        """Test importing Query and AgentResult."""
        from ragicamp.agents import AgentResult, Query

        assert Query is not None
        assert AgentResult is not None


class TestCoreImports:
    """Test that core type imports work."""

    def test_import_document(self):
        """Test importing Document."""
        from ragicamp.core.types import Document

        assert Document is not None

    def test_import_search_result(self):
        """Test importing SearchResult."""
        from ragicamp.core.types import SearchResult

        assert SearchResult is not None


class TestSpecImports:
    """Test that spec modules import correctly."""

    def test_import_experiment_spec(self):
        """Test importing ExperimentSpec."""
        from ragicamp.spec import ExperimentSpec

        assert ExperimentSpec is not None

    def test_import_build_specs(self):
        """Test importing build_specs."""
        from ragicamp.spec import build_specs

        assert callable(build_specs)

    def test_import_naming_functions(self):
        """Test importing naming functions."""
        from ragicamp.spec import name_direct, name_rag

        assert callable(name_direct)
        assert callable(name_rag)


class TestExecutionImports:
    """Test that execution modules import correctly."""

    def test_import_run_spec(self):
        """Test importing run_spec."""
        from ragicamp.execution.runner import run_spec

        assert callable(run_spec)


class TestRerankerImports:
    """Test that reranker modules import correctly."""

    def test_import_cross_encoder_reranker(self):
        """Test importing CrossEncoderReranker."""
        from ragicamp.rag.rerankers import CrossEncoderReranker

        assert CrossEncoderReranker is not None
