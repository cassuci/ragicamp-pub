"""Tests for InitHandler metadata completeness.

Verifies that the INIT phase writes all spec fields to metadata.json,
so that even failed/stalled experiments have complete metadata.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ragicamp.execution.phases.init_phase import InitHandler
from ragicamp.spec.experiment import ExperimentSpec
from ragicamp.state import ExperimentPhase, ExperimentState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(output_path: Path, dataset_name: str = "mock_dataset") -> MagicMock:
    """Build a minimal ExecutionContext mock."""
    ctx = MagicMock()
    ctx.output_path = output_path

    # Dataset mock
    dataset = MagicMock()
    dataset.name = dataset_name
    dataset.__iter__ = MagicMock(return_value=iter([]))
    ctx.dataset = dataset

    # Agent mock
    agent = MagicMock()
    agent.name = "mock_agent"
    ctx.agent = agent

    # Metrics mock — list of objects with .name
    metric = MagicMock()
    metric.name = "f1"
    ctx.metrics = [metric]

    return ctx


def _make_state() -> ExperimentState:
    """Build a minimal ExperimentState."""
    state = ExperimentState(
        phase=ExperimentPhase.INIT,
        total_questions=0,
        started_at="2025-01-01T00:00:00",
        updated_at="2025-01-01T00:00:00",
    )
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitHandlerMetadata:
    """Verify InitHandler writes complete metadata.json from spec."""

    def test_metadata_includes_model(self, temp_dir):
        """Model field must be persisted at INIT so failed experiments are identifiable."""
        spec = ExperimentSpec(
            name="direct_test",
            exp_type="direct",
            model="vllm:google/gemma-2-2b-it",
            dataset="nq",
            prompt="concise",
        )
        ctx = _make_context(temp_dir)
        handler = InitHandler()
        handler.execute(spec, _make_state(), ctx)

        metadata = json.loads((temp_dir / "metadata.json").read_text())
        assert metadata["model"] == "vllm:google/gemma-2-2b-it"

    def test_metadata_includes_type(self, temp_dir):
        """Experiment type must be persisted."""
        spec = ExperimentSpec(
            name="rag_test",
            exp_type="rag",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            retriever="dense_bge",
        )
        ctx = _make_context(temp_dir)
        handler = InitHandler()
        handler.execute(spec, _make_state(), ctx)

        metadata = json.loads((temp_dir / "metadata.json").read_text())
        assert metadata["type"] == "rag"

    def test_metadata_includes_prompt(self, temp_dir):
        """Prompt style must be persisted."""
        spec = ExperimentSpec(
            name="direct_test",
            exp_type="direct",
            model="vllm:test/model",
            dataset="nq",
            prompt="fewshot_3",
        )
        ctx = _make_context(temp_dir)
        handler = InitHandler()
        handler.execute(spec, _make_state(), ctx)

        metadata = json.loads((temp_dir / "metadata.json").read_text())
        assert metadata["prompt"] == "fewshot_3"

    def test_metadata_includes_rag_fields(self, temp_dir):
        """RAG-specific fields (retriever, top_k, etc.) must be persisted."""
        spec = ExperimentSpec(
            name="rag_test",
            exp_type="rag",
            model="vllm:test/model",
            dataset="hotpotqa",
            prompt="concise",
            retriever="dense_bge_large_512",
            top_k=10,
            fetch_k=40,
            query_transform="hyde",
            reranker="bge",
            reranker_model="BAAI/bge-reranker-v2-m3",
        )
        ctx = _make_context(temp_dir)
        handler = InitHandler()
        handler.execute(spec, _make_state(), ctx)

        metadata = json.loads((temp_dir / "metadata.json").read_text())
        assert metadata["retriever"] == "dense_bge_large_512"
        assert metadata["top_k"] == 10
        assert metadata["fetch_k"] == 40
        assert metadata["query_transform"] == "hyde"
        assert metadata["reranker"] == "bge"
        assert metadata["reranker_model"] == "BAAI/bge-reranker-v2-m3"

    def test_metadata_uses_spec_dataset_not_context(self, temp_dir):
        """Dataset in metadata must come from spec (canonical short name),
        not from context.dataset.name (class name like 'natural_questions')."""
        spec = ExperimentSpec(
            name="direct_test",
            exp_type="direct",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
        )
        # Simulate the NQ dataset class which has name="natural_questions"
        ctx = _make_context(temp_dir, dataset_name="natural_questions")
        handler = InitHandler()
        handler.execute(spec, _make_state(), ctx)

        metadata = json.loads((temp_dir / "metadata.json").read_text())
        assert metadata["dataset"] == "nq", (
            "metadata.json should use spec.dataset ('nq'), "
            "not context.dataset.name ('natural_questions')"
        )

    def test_metadata_includes_agent_type(self, temp_dir):
        """agent_type from spec must be persisted for singleton experiments."""
        spec = ExperimentSpec(
            name="iterative_test",
            exp_type="rag",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            retriever="dense_bge",
            agent_type="iterative_rag",
        )
        ctx = _make_context(temp_dir)
        handler = InitHandler()
        handler.execute(spec, _make_state(), ctx)

        metadata = json.loads((temp_dir / "metadata.json").read_text())
        assert metadata["agent_type"] == "iterative_rag"

    def test_metadata_preserves_agent_name(self, temp_dir):
        """The runtime agent name should still be recorded."""
        spec = ExperimentSpec(
            name="direct_test",
            exp_type="direct",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
        )
        ctx = _make_context(temp_dir)
        ctx.agent.name = "DirectLLMAgent"
        handler = InitHandler()
        handler.execute(spec, _make_state(), ctx)

        metadata = json.loads((temp_dir / "metadata.json").read_text())
        assert metadata["agent"] == "DirectLLMAgent"

    def test_metadata_none_fields_for_direct(self, temp_dir):
        """Direct experiments should have None for RAG-only fields."""
        spec = ExperimentSpec(
            name="direct_test",
            exp_type="direct",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
        )
        ctx = _make_context(temp_dir)
        handler = InitHandler()
        handler.execute(spec, _make_state(), ctx)

        metadata = json.loads((temp_dir / "metadata.json").read_text())
        assert metadata["retriever"] is None
        assert metadata["reranker"] is None
        assert metadata["query_transform"] is None


class TestInitHandlerMinimalSpec:
    """Verify InitHandler works safely with _MinimalSpec (programmatic API)."""

    def test_minimal_spec_does_not_crash(self, temp_dir):
        """When spec has only 'name', init should still write metadata without errors."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class MinimalSpec:
            name: str

        spec = MinimalSpec(name="programmatic_test")
        ctx = _make_context(temp_dir, dataset_name="natural_questions")
        handler = InitHandler()
        handler.execute(spec, _make_state(), ctx)

        metadata = json.loads((temp_dir / "metadata.json").read_text())
        assert metadata["name"] == "programmatic_test"
        # Falls back to context.dataset.name when spec.dataset is None
        assert metadata["dataset"] == "natural_questions"
        # Other fields should be None, not crash
        assert metadata["model"] is None
        assert metadata["type"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
