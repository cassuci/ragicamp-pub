"""Tests for RAG agents.

Tests the new clean architecture agents:
- DirectLLMAgent: No retrieval, just LLM
- FixedRAGAgent: Fixed retrieve-then-generate
- IterativeRAGAgent: Multi-iteration refinement
- SelfRAGAgent: Adaptive retrieval

All agents use provider pattern for GPU lifecycle management.
"""

import pytest

from ragicamp.agents import (
    AgentResult,
    DirectLLMAgent,
    FixedRAGAgent,
    Query,
    Step,
)
from tests.shared_mocks import FakeEmbedder, FakeGenerator, FakeIndex, FakeProvider


class TestQuery:
    """Tests for Query dataclass."""

    def test_query_creation(self):
        """Test creating a Query."""
        query = Query(idx=0, text="What is the capital of France?", expected="Paris")

        assert query.idx == 0
        assert query.text == "What is the capital of France?"
        assert query.expected == "Paris"

    def test_query_without_expected(self):
        """Test creating a Query without expected answer."""
        query = Query(idx=1, text="Random question?")

        assert query.idx == 1
        assert query.expected is None


class TestStep:
    """Tests for Step dataclass."""

    def test_step_creation(self):
        """Test creating a Step."""
        step = Step(type="retrieve", timing_ms=100.5, model="embedder")

        assert step.type == "retrieve"
        assert step.timing_ms == 100.5
        assert step.model == "embedder"


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_result_creation(self):
        """Test creating an AgentResult."""
        query = Query(idx=0, text="Q?", expected="A")
        steps = [Step(type="generate", timing_ms=50.0, model="llm")]

        result = AgentResult(
            query=query,
            answer="Answer",
            prompt="Full prompt",
            steps=steps,
            metadata={"key": "value"},
        )

        assert result.query == query
        assert result.answer == "Answer"
        assert result.prompt == "Full prompt"
        assert len(result.steps) == 1
        assert result.metadata["key"] == "value"


class TestDirectLLMAgent:
    """Tests for DirectLLMAgent."""

    def test_initialization(self):
        """Test DirectLLMAgent can be instantiated."""
        provider = FakeProvider(FakeGenerator(), "fake-generator")

        agent = DirectLLMAgent(
            name="test_direct",
            generator_provider=provider,
        )

        assert agent.name == "test_direct"

    def test_run_single_query(self):
        """Test running a single query."""
        provider = FakeProvider(FakeGenerator(), "fake-generator")

        agent = DirectLLMAgent(
            name="test_direct",
            generator_provider=provider,
        )

        queries = [Query(idx=0, text="What is 2+2?", expected="4")]
        results = agent.run(queries)

        assert len(results) == 1
        assert isinstance(results[0], AgentResult)
        assert results[0].answer is not None

    def test_run_batch(self):
        """Test running multiple queries."""
        provider = FakeProvider(FakeGenerator(), "fake-generator")

        agent = DirectLLMAgent(
            name="test_direct",
            generator_provider=provider,
        )

        queries = [
            Query(idx=0, text="Q1?"),
            Query(idx=1, text="Q2?"),
            Query(idx=2, text="Q3?"),
        ]
        results = agent.run(queries)

        assert len(results) == 3
        assert all(isinstance(r, AgentResult) for r in results)


class TestFixedRAGAgent:
    """Tests for FixedRAGAgent."""

    def test_initialization(self):
        """Test FixedRAGAgent can be instantiated."""
        agent = FixedRAGAgent(
            name="test_rag",
            embedder_provider=FakeProvider(FakeEmbedder(), "fake-embedder"),
            generator_provider=FakeProvider(FakeGenerator(), "fake-generator"),
            index=FakeIndex(),
            top_k=3,
        )

        assert agent.name == "test_rag"
        assert agent.top_k == 3

    def test_run_with_retrieval(self):
        """Test running with retrieval."""
        agent = FixedRAGAgent(
            name="test_rag",
            embedder_provider=FakeProvider(FakeEmbedder(), "fake-embedder"),
            generator_provider=FakeProvider(FakeGenerator(), "fake-generator"),
            index=FakeIndex(),
            top_k=2,
        )

        queries = [Query(idx=0, text="What is the capital of France?")]
        results = agent.run(queries)

        assert len(results) == 1
        assert isinstance(results[0], AgentResult)
        # Should have encode, search and generate steps
        step_types = [s.type for s in results[0].steps]
        # New architecture uses batch_encode and batch_search
        assert "batch_encode" in step_types or "retrieve" in step_types
        assert "generate" in step_types


class TestAgentFactoryIntegration:
    """Tests for AgentFactory with new architecture."""

    def test_available_agents(self):
        """Test listing available agent types."""
        from ragicamp.factory import AgentFactory

        agents = AgentFactory.get_available_agents()

        assert "direct_llm" in agents
        assert "fixed_rag" in agents
        assert "iterative_rag" in agents
        assert "self_rag" in agents

    def test_create_direct_agent(self):
        """Test creating DirectLLMAgent via factory."""
        from ragicamp.factory import AgentFactory

        agent = AgentFactory.create_direct(
            name="test_agent",
            generator_provider=FakeProvider(FakeGenerator(), "fake-generator"),
        )

        assert isinstance(agent, DirectLLMAgent)
        assert agent.name == "test_agent"

    def test_create_rag_agent(self):
        """Test creating FixedRAGAgent via factory."""
        from ragicamp.factory import AgentFactory

        agent = AgentFactory.create_rag(
            agent_type="fixed_rag",
            name="test_rag",
            embedder_provider=FakeProvider(FakeEmbedder(), "fake-embedder"),
            generator_provider=FakeProvider(FakeGenerator(), "fake-generator"),
            index=FakeIndex(),
            top_k=5,
        )

        assert isinstance(agent, FixedRAGAgent)
        assert agent.name == "test_rag"
        assert agent.top_k == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
