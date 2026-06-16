"""Tests for IterativeRAGAgent.

Tests multi-iteration query refinement with batched operations:
- Iteration loop mechanics (max_iterations respected)
- Convergence via sufficiency checks
- Query refinement on insufficient contexts
- Document deduplication across iterations
- Query transformer applied only on iteration 0
- Proper result structure and metadata
"""

import pytest

from ragicamp.agents.base import Query
from ragicamp.agents.iterative_rag import IterativeRAGAgent
from ragicamp.core.types import Document
from tests.shared_mocks import (
    FakeEmbedder,
    FakeGenerator,
    FakeIndex,
    FakeProvider,
    FakeQueryTransformer,
)


class TestIterativeRAGInitialization:
    def test_initialization(self):
        embedder_provider = FakeProvider(FakeEmbedder(), "fake-embedder")
        generator_provider = FakeProvider(FakeGenerator(), "fake-generator")
        index = FakeIndex()

        agent = IterativeRAGAgent(
            name="test_iterative",
            embedder_provider=embedder_provider,
            generator_provider=generator_provider,
            index=index,
            top_k=3,
            max_iterations=2,
            stop_on_sufficient=True,
        )

        assert agent.name == "test_iterative"
        assert agent.top_k == 3
        assert agent.max_iterations == 2
        assert agent.stop_on_sufficient is True


class TestIterativeRAGExecution:
    def test_single_iteration_produces_results(self):
        embedder_provider = FakeProvider(FakeEmbedder(), "fake-embedder")
        generator_provider = FakeProvider(FakeGenerator(), "fake-generator")
        index = FakeIndex()

        agent = IterativeRAGAgent(
            name="test",
            embedder_provider=embedder_provider,
            generator_provider=generator_provider,
            index=index,
            top_k=3,
            max_iterations=1,
        )

        queries = [Query(idx=0, text="What is the capital of France?", expected=["Paris"])]
        results = agent.run(queries)

        assert len(results) == 1
        assert results[0].answer is not None
        assert results[0].answer != ""
        assert len(results[0].steps) > 0

    def test_max_iterations_respected(self):
        embedder_provider = FakeProvider(FakeEmbedder(), "fake-embedder")
        generator = FakeGenerator(
            {
                "determine if it contains enough": "INSUFFICIENT. Missing key information.",
                "Refined Query": "What is the refined query?",
            }
        )
        generator_provider = FakeProvider(generator, "fake-generator")
        index = FakeIndex()

        agent = IterativeRAGAgent(
            name="test",
            embedder_provider=embedder_provider,
            generator_provider=generator_provider,
            index=index,
            top_k=3,
            max_iterations=2,
            stop_on_sufficient=True,
        )

        queries = [Query(idx=0, text="Test question", expected=[])]
        results = agent.run(queries)

        assert len(results) == 1
        # Check iterations metadata
        iterations = results[0].metadata.get("iterations", [])
        assert len(iterations) == 2, f"Expected 2 iterations, got {len(iterations)}"

    def test_convergence_on_sufficient(self):
        embedder_provider = FakeProvider(FakeEmbedder(), "fake-embedder")
        generator = FakeGenerator(
            {
                "determine if it contains enough": "SUFFICIENT. The context answers the question.",
            }
        )
        generator_provider = FakeProvider(generator, "fake-generator")
        index = FakeIndex()

        agent = IterativeRAGAgent(
            name="test",
            embedder_provider=embedder_provider,
            generator_provider=generator_provider,
            index=index,
            top_k=3,
            max_iterations=3,
            stop_on_sufficient=True,
        )

        queries = [Query(idx=0, text="Test question", expected=[])]
        results = agent.run(queries)

        assert len(results) == 1
        iterations = results[0].metadata.get("iterations", [])
        # Should stop early (iteration 0 + sufficient check = 1 iteration recorded)
        assert len(iterations) == 1
        assert iterations[0].get("stopped") == "sufficient"

    def test_insufficient_triggers_refinement(self):
        embedder_provider = FakeProvider(FakeEmbedder(), "fake-embedder")
        generator = FakeGenerator(
            {
                "determine if it contains enough": "INSUFFICIENT. Missing details.",
                "Refined Query": "refined version of the query",
            }
        )
        generator_provider = FakeProvider(generator, "fake-generator")
        index = FakeIndex()

        agent = IterativeRAGAgent(
            name="test",
            embedder_provider=embedder_provider,
            generator_provider=generator_provider,
            index=index,
            top_k=3,
            max_iterations=2,
            stop_on_sufficient=True,
        )

        queries = [Query(idx=0, text="Original question", expected=[])]
        results = agent.run(queries)

        assert len(results) == 1
        # Check that refinement step exists
        step_types = [s.type for s in results[0].steps]
        assert "refine_query" in step_types

    def test_all_queries_completed(self):
        embedder_provider = FakeProvider(FakeEmbedder(), "fake-embedder")
        generator_provider = FakeProvider(FakeGenerator(), "fake-generator")
        index = FakeIndex()

        agent = IterativeRAGAgent(
            name="test",
            embedder_provider=embedder_provider,
            generator_provider=generator_provider,
            index=index,
            top_k=3,
            max_iterations=1,
        )

        queries = [
            Query(idx=0, text="Q1", expected=[]),
            Query(idx=1, text="Q2", expected=[]),
            Query(idx=2, text="Q3", expected=[]),
        ]
        results = agent.run(queries)

        assert len(results) == 3
        assert {r.query.idx for r in results} == {0, 1, 2}

    def test_result_has_iterations_metadata(self):
        embedder_provider = FakeProvider(FakeEmbedder(), "fake-embedder")
        generator_provider = FakeProvider(FakeGenerator(), "fake-generator")
        index = FakeIndex()

        agent = IterativeRAGAgent(
            name="test",
            embedder_provider=embedder_provider,
            generator_provider=generator_provider,
            index=index,
            top_k=3,
            max_iterations=2,
        )

        queries = [Query(idx=0, text="Test", expected=[])]
        results = agent.run(queries)

        assert len(results) == 1
        metadata = results[0].metadata
        assert "iterations" in metadata
        assert isinstance(metadata["iterations"], list)
        assert "total_docs" in metadata
        assert "final_docs_used" in metadata

    def test_empty_queries_returns_empty(self):
        embedder_provider = FakeProvider(FakeEmbedder(), "fake-embedder")
        generator_provider = FakeProvider(FakeGenerator(), "fake-generator")
        index = FakeIndex()

        agent = IterativeRAGAgent(
            name="test",
            embedder_provider=embedder_provider,
            generator_provider=generator_provider,
            index=index,
            top_k=3,
            max_iterations=2,
        )

        results = agent.run([])

        assert len(results) == 0

    def test_stop_on_sufficient_false(self):
        embedder_provider = FakeProvider(FakeEmbedder(), "fake-embedder")
        generator = FakeGenerator(
            {
                "Refined Query": "refined query text",
            }
        )
        generator_provider = FakeProvider(generator, "fake-generator")
        index = FakeIndex()

        agent = IterativeRAGAgent(
            name="test",
            embedder_provider=embedder_provider,
            generator_provider=generator_provider,
            index=index,
            top_k=3,
            max_iterations=2,
            stop_on_sufficient=False,
        )

        queries = [Query(idx=0, text="Test question", expected=[])]
        results = agent.run(queries)

        assert len(results) == 1
        iterations = results[0].metadata.get("iterations", [])
        # With stop_on_sufficient=False, all iterations run
        assert len(iterations) == 2
        # No sufficiency checks should have been performed
        step_types = [s.type for s in results[0].steps]
        assert "evaluate_sufficiency" not in step_types


class TestIterativeRAGQueryTransform:
    def test_query_transform_only_iteration_0(self):
        embedder_provider = FakeProvider(FakeEmbedder(), "fake-embedder")
        generator = FakeGenerator(
            {
                "determine if it contains enough": "INSUFFICIENT. Missing details.",
                "Refined Query": "refined version",
            }
        )
        generator_provider = FakeProvider(generator, "fake-generator")
        index = FakeIndex()

        transformer = FakeQueryTransformer()

        agent = IterativeRAGAgent(
            name="test",
            embedder_provider=embedder_provider,
            generator_provider=generator_provider,
            index=index,
            top_k=3,
            max_iterations=2,
            stop_on_sufficient=True,
            query_transformer=transformer,
        )

        queries = [Query(idx=0, text="Test question", expected=[])]
        agent.run(queries)

        # Query transformer should be called exactly once (iteration 0 only)
        assert transformer.call_count == 1
        assert transformer.last_queries == ["Test question"]


class TestDocumentMerging:
    def test_merge_documents_deduplicates(self):
        doc1 = Document(id="1", text="Same content")
        doc2 = Document(id="2", text="Same content")
        doc3 = Document(id="3", text="Different content")

        existing = [doc1]
        new_docs = [doc2, doc3]

        merged = IterativeRAGAgent._merge_documents(existing, new_docs)

        # Should have 2 docs: doc1 and doc3 (doc2 is duplicate of doc1)
        assert len(merged) == 2
        assert merged[0].id == "1"
        assert merged[1].id == "3"

    def test_merge_documents_preserves_order(self):
        doc1 = Document(id="1", text="Content A")
        doc2 = Document(id="2", text="Content B")
        doc3 = Document(id="3", text="Content C")
        doc4 = Document(id="4", text="Content D")

        existing = [doc1, doc2]
        new_docs = [doc3, doc4]

        merged = IterativeRAGAgent._merge_documents(existing, new_docs)

        # Existing docs should come first
        assert len(merged) == 4
        assert [d.id for d in merged] == ["1", "2", "3", "4"]

    def test_merge_documents_handles_empty_existing(self):
        doc1 = Document(id="1", text="Content A")
        doc2 = Document(id="2", text="Content B")

        merged = IterativeRAGAgent._merge_documents([], [doc1, doc2])

        assert len(merged) == 2
        assert merged[0].id == "1"
        assert merged[1].id == "2"

    def test_merge_documents_handles_empty_new(self):
        doc1 = Document(id="1", text="Content A")
        doc2 = Document(id="2", text="Content B")

        merged = IterativeRAGAgent._merge_documents([doc1, doc2], [])

        assert len(merged) == 2
        assert merged[0].id == "1"
        assert merged[1].id == "2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
