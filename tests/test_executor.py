"""Tests for ResilientExecutor batch processing and error handling."""

from unittest.mock import MagicMock, patch

import pytest

from ragicamp.execution.executor import ResilientExecutor


class MockResponse:
    """Simple mock agent response."""

    def __init__(self, answer, prompt=None, metadata=None):
        self.answer = answer
        self.prompt = prompt
        self.metadata = metadata or {}


class BatchAgent:
    """Mock agent with batch_answer support."""

    def __init__(self, answer_fn=None):
        self.answer_fn = answer_fn or (lambda q: f"Answer: {q}")
        self.call_count = 0

    def batch_answer(self, queries, **kwargs):
        self.call_count += 1
        return [MockResponse(self.answer_fn(q)) for q in queries]


class SequentialAgent:
    """Mock agent without batch_answer (sequential mode)."""

    def __init__(self, answer_fn=None):
        self.answer_fn = answer_fn or (lambda q: f"Answer: {q}")
        self.call_count = 0

    def answer(self, query, **kwargs):
        self.call_count += 1
        return MockResponse(self.answer_fn(query))


class FailingAgent:
    """Mock agent that fails on specific calls."""

    def __init__(self, fail_on_calls=None):
        self.fail_on_calls = fail_on_calls or set()
        self.call_count = 0

    def batch_answer(self, queries, **kwargs):
        self.call_count += 1
        if self.call_count in self.fail_on_calls:
            raise RuntimeError(f"Batch failure on call {self.call_count}")
        return [MockResponse(f"Answer: {q}") for q in queries]

    def answer(self, query, **kwargs):
        self.call_count += 1
        if self.call_count in self.fail_on_calls:
            raise RuntimeError(f"Sequential failure on call {self.call_count}")
        return MockResponse(f"Answer: {query}")


@pytest.fixture
def mock_resource_manager():
    with patch("ragicamp.execution.executor.ResourceManager") as mock:
        yield mock


def test_empty_queries():
    agent = BatchAgent()
    executor = ResilientExecutor(agent, batch_size=32)
    results = executor.execute([], progress=False)
    assert results == []


def test_batch_execution_basic(mock_resource_manager):
    agent = BatchAgent(lambda q: f"A: {q}")
    executor = ResilientExecutor(agent, batch_size=2)

    queries = [
        (0, "What is Python?", ["programming language"]),
        (1, "What is RAG?", ["retrieval augmented generation"]),
        (2, "What is FAISS?", ["vector search"]),
    ]

    results = executor.execute(queries, progress=False)

    assert len(results) == 3
    assert results[0]["idx"] == 0
    assert results[0]["query"] == "What is Python?"
    assert results[0]["prediction"] == "A: What is Python?"
    assert results[0]["expected"] == ["programming language"]
    assert results[0]["error"] is None

    assert results[2]["prediction"] == "A: What is FAISS?"


def test_sequential_fallback_no_batch_method(mock_resource_manager):
    agent = SequentialAgent(lambda q: f"Sequential: {q}")
    executor = ResilientExecutor(agent, batch_size=32)

    queries = [
        (0, "Q1", ["A1"]),
        (1, "Q2", ["A2"]),
    ]

    results = executor.execute(queries, progress=False)

    assert len(results) == 2
    assert results[0]["prediction"] == "Sequential: Q1"
    assert results[1]["prediction"] == "Sequential: Q2"
    assert agent.call_count == 2


def test_sequential_fallback_batch_size_one(mock_resource_manager):
    agent = SequentialAgent(lambda q: f"Sequential: {q}")
    executor = ResilientExecutor(agent, batch_size=1)

    queries = [(0, "Q1", ["A1"]), (1, "Q2", ["A2"])]

    results = executor.execute(queries, progress=False)

    assert len(results) == 2
    assert results[0]["prediction"] == "Sequential: Q1"
    assert agent.call_count == 2


def test_single_batch_failure_creates_error_markers(mock_resource_manager):
    agent = FailingAgent(fail_on_calls={1})
    executor = ResilientExecutor(agent, batch_size=2)

    queries = [
        (0, "Q1", ["A1"]),
        (1, "Q2", ["A2"]),
        (2, "Q3", ["A3"]),
    ]

    results = executor.execute(queries, progress=False)

    assert len(results) == 3
    assert results[0]["prediction"].startswith("[ERROR:")
    assert "Batch failure on call 1" in results[0]["error"]
    assert results[1]["prediction"].startswith("[ERROR:")
    assert results[2]["prediction"] == "Answer: Q3"
    assert results[2]["error"] is None


def test_consecutive_failures_abort(mock_resource_manager):
    agent = FailingAgent(fail_on_calls={1, 2, 3, 4, 5, 6})
    executor = ResilientExecutor(agent, batch_size=2)

    queries = [(i, f"Q{i}", [f"A{i}"]) for i in range(12)]

    results = executor.execute(queries, progress=False)

    assert len(results) == 12

    error_count = sum(1 for r in results if r["prediction"].startswith("[ERROR:"))
    aborted_count = sum(1 for r in results if r["prediction"].startswith("[ABORTED:"))

    assert error_count == 8
    assert aborted_count == 4
    assert results[8]["prediction"].startswith("[ABORTED:")
    assert "Aborted after 5 failures" in results[8]["error"]


def test_checkpoint_callback_fires_at_threshold(mock_resource_manager):
    agent = BatchAgent()
    executor = ResilientExecutor(agent, batch_size=2)

    queries = [(i, f"Q{i}", [f"A{i}"]) for i in range(10)]

    checkpoint_calls = []

    def checkpoint_fn(results):
        checkpoint_calls.append(len(results))

    executor.execute(queries, progress=False, checkpoint_every=4, checkpoint_callback=checkpoint_fn)

    assert checkpoint_calls == [4, 8]


def test_checkpoint_callback_not_fired_if_disabled(mock_resource_manager):
    agent = BatchAgent()
    executor = ResilientExecutor(agent, batch_size=2)

    queries = [(i, f"Q{i}", [f"A{i}"]) for i in range(6)]

    checkpoint_fn = MagicMock()

    executor.execute(queries, progress=False, checkpoint_every=0, checkpoint_callback=checkpoint_fn)

    checkpoint_fn.assert_not_called()


def test_gc_collect_every_10_batches(mock_resource_manager):
    agent = BatchAgent()
    executor = ResilientExecutor(agent, batch_size=2)

    queries = [(i, f"Q{i}", [f"A{i}"]) for i in range(25)]

    executor.execute(queries, progress=False)

    assert mock_resource_manager.clear_gpu_memory.call_count == 1


def test_gc_collect_sequential_every_10_items(mock_resource_manager):
    agent = SequentialAgent()
    executor = ResilientExecutor(agent, batch_size=32)

    queries = [(i, f"Q{i}", [f"A{i}"]) for i in range(25)]

    executor.execute(queries, progress=False)

    assert mock_resource_manager.clear_gpu_memory.call_count == 2


def test_result_includes_prompt_and_metadata():
    class RichAgent:
        def batch_answer(self, queries, **kwargs):
            return [
                MockResponse(
                    answer=f"A: {q}",
                    prompt=f"Prompt: {q}",
                    metadata={"model": "test-model", "tokens": 10},
                )
                for q in queries
            ]

    agent = RichAgent()
    executor = ResilientExecutor(agent, batch_size=2)

    queries = [(0, "What is Python?", ["language"])]

    results = executor.execute(queries, progress=False)

    assert results[0]["prompt"] == "Prompt: What is Python?"
    assert results[0]["metadata"] == {"model": "test-model", "tokens": 10}


def test_error_prediction_truncated_to_50_chars():
    class LongErrorAgent:
        def batch_answer(self, queries, **kwargs):
            raise RuntimeError("X" * 100)

    agent = LongErrorAgent()
    executor = ResilientExecutor(agent, batch_size=2)

    queries = [(0, "Q1", ["A1"])]

    results = executor.execute(queries, progress=False)

    assert len(results[0]["prediction"]) == len("[ERROR: ") + 50 + len("]")
    assert results[0]["prediction"].startswith("[ERROR:")
    assert len(results[0]["error"]) == 100
