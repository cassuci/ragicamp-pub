"""Pytest fixtures and configuration for RAGiCamp tests.

This module provides reusable fixtures for testing:
- Mock models, retrievers, and agents
- Sample datasets and examples
- Test configuration helpers

Usage:
    def test_something(mock_model, sample_dataset):
        agent = DirectLLMAgent(model=mock_model)
        results = evaluate(agent, sample_dataset)
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Make notebooks/ importable so tests can use analysis_utils without sys.path hacks
_NOTEBOOKS_DIR = str(Path(__file__).resolve().parent.parent / "notebooks")
if _NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, _NOTEBOOKS_DIR)

# Import RAGiCamp components (E402: after sys.path mutation above)
from ragicamp.core.types import Document  # noqa: E402
from ragicamp.datasets.base import QADataset, QAExample  # noqa: E402
from ragicamp.metrics.base import Metric  # noqa: E402

# Import shared agent mocks (defined in tests/shared_mocks.py)
from tests.shared_mocks import (  # noqa: E402, F401
    FakeEmbedder,
    FakeGenerator,
    FakeIndex,
    FakeProvider,
    FakeQueryTransformer,
)

# === Mock Model ===


class MockLanguageModel:
    """Mock language model for testing.

    Returns predictable responses based on configuration.
    """

    def __init__(
        self,
        model_name: str = "mock_model",
        default_response: str = "Mock answer",
        responses: dict[str, str] | None = None,
    ):
        self.model_name = model_name
        self.default_response = default_response
        self.responses = responses or {}
        self.call_history: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate a mock response."""
        self.call_history.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }
        )

        # Check for specific responses
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return response

        return self.default_response

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings."""
        return [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in texts]

    def count_tokens(self, text: str) -> int:
        """Approximate token count."""
        return len(text.split())


@pytest.fixture
def mock_model() -> MockLanguageModel:
    """Create a basic mock language model."""
    return MockLanguageModel()


@pytest.fixture
def mock_model_with_responses() -> MockLanguageModel:
    """Create a mock model with specific responses."""
    return MockLanguageModel(
        responses={
            "capital of France": "Paris",
            "capital of Germany": "Berlin",
            "capital of Japan": "Tokyo",
        }
    )


# === Mock Retriever ===


class MockRetriever:
    """Mock retriever for testing.

    Returns predefined documents or generates mock ones.
    """

    def __init__(
        self,
        name: str = "mock_retriever",
        documents: list[Document] | None = None,
    ):
        self.name = name
        self.documents = documents or self._default_documents()
        self.call_history: list[dict[str, Any]] = []

    def _default_documents(self) -> list[Document]:
        """Create default mock documents."""
        return [
            Document(
                id="doc1",
                text="Paris is the capital of France. It is known for the Eiffel Tower.",
                metadata={"source": "wikipedia"},
                score=0.95,
            ),
            Document(
                id="doc2",
                text="Berlin is the capital of Germany. It has a rich history.",
                metadata={"source": "wikipedia"},
                score=0.88,
            ),
            Document(
                id="doc3",
                text="Tokyo is the capital of Japan. It is a major metropolitan area.",
                metadata={"source": "wikipedia"},
                score=0.82,
            ),
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> list[Document]:
        """Retrieve mock documents."""
        self.call_history.append(
            {
                "query": query,
                "top_k": top_k,
                **kwargs,
            }
        )
        return self.documents[:top_k]

    def index_documents(self, documents: list[Document]) -> None:
        """Store documents for retrieval."""
        self.documents = documents


@pytest.fixture
def mock_retriever() -> MockRetriever:
    """Create a basic mock retriever."""
    return MockRetriever()


# === Sample Data ===


@pytest.fixture
def sample_qa_examples() -> list[QAExample]:
    """Create sample QA examples for testing."""
    return [
        QAExample(
            id="1",
            question="What is the capital of France?",
            answers=["Paris", "paris"],
            metadata={"topic": "geography"},
        ),
        QAExample(
            id="2",
            question="What is the capital of Germany?",
            answers=["Berlin", "berlin"],
            metadata={"topic": "geography"},
        ),
        QAExample(
            id="3",
            question="What is the capital of Japan?",
            answers=["Tokyo", "tokyo"],
            metadata={"topic": "geography"},
        ),
        QAExample(
            id="4",
            question="Who wrote Hamlet?",
            answers=["Shakespeare", "William Shakespeare"],
            metadata={"topic": "literature"},
        ),
        QAExample(
            id="5",
            question="What year did World War 2 end?",
            answers=["1945"],
            metadata={"topic": "history"},
        ),
    ]


class MockQADataset(QADataset):
    """Mock QA dataset for testing."""

    def __init__(
        self,
        examples: list[QAExample],
        name: str = "mock_dataset",
        split: str = "test",
    ):
        super().__init__(name=name, split=split)
        self.examples = examples

    def load(self) -> None:
        """Already loaded."""
        pass


@pytest.fixture
def sample_dataset(sample_qa_examples) -> MockQADataset:
    """Create a sample QA dataset."""
    return MockQADataset(examples=sample_qa_examples)


@pytest.fixture
def small_dataset(sample_qa_examples) -> MockQADataset:
    """Create a small dataset for quick tests."""
    return MockQADataset(examples=sample_qa_examples[:2])


# === Mock Metrics ===


class MockMetric(Metric):
    """Mock metric for testing.

    Returns predefined scores or calculates simple metrics.
    """

    def __init__(
        self,
        name: str = "mock_metric",
        fixed_score: float | None = None,
    ):
        super().__init__(name=name)
        self.fixed_score = fixed_score
        self.call_count = 0

    def compute(
        self,
        predictions: list[str],
        references: list[list[str]],
        **kwargs,
    ) -> dict[str, float]:
        """Compute mock metric."""
        self.call_count += 1

        if self.fixed_score is not None:
            return {self.name: self.fixed_score}

        # Simple exact match calculation
        matches = sum(
            1
            for pred, refs in zip(predictions, references, strict=False)
            if pred.lower().strip() in [r.lower().strip() for r in refs]
        )
        score = matches / len(predictions) if predictions else 0.0
        return {self.name: score}


@pytest.fixture
def mock_metric() -> MockMetric:
    """Create a basic mock metric."""
    return MockMetric()


@pytest.fixture
def mock_metrics() -> list[MockMetric]:
    """Create a list of mock metrics."""
    return [
        MockMetric(name="mock_em"),
        MockMetric(name="mock_f1"),
    ]


# === Shared Agent Mock Fixtures ===
#
# The mock classes (FakeGenerator, FakeEmbedder, FakeProvider, FakeIndex,
# FakeQueryTransformer) are defined in tests/shared_mocks.py and imported above.
# Test files can either use the fixtures below or import the classes directly
# from tests.shared_mocks.


@pytest.fixture
def fake_generator():
    """Create a FakeGenerator with default (no pattern) responses."""
    return FakeGenerator()


@pytest.fixture
def fake_embedder():
    """Create a FakeEmbedder with default 4-dim output."""
    return FakeEmbedder()


@pytest.fixture
def fake_index():
    """Create a FakeIndex with 5 default documents."""
    return FakeIndex()


@pytest.fixture
def fake_generator_provider(fake_generator):
    """Create a FakeProvider wrapping a FakeGenerator."""
    return FakeProvider(fake_generator, "fake-generator")


@pytest.fixture
def fake_embedder_provider(fake_embedder):
    """Create a FakeProvider wrapping a FakeEmbedder."""
    return FakeProvider(fake_embedder, "fake-embedder")


# === Temporary Files ===


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_file(temp_dir) -> Path:
    """Create a temporary config file."""
    config = {
        "agent": {
            "type": "direct_llm",
            "name": "test_agent",
        },
        "model": {
            "type": "huggingface",
            "model_name": "test/model",
        },
        "dataset": {
            "name": "natural_questions",
            "split": "validation",
            "num_examples": 10,
        },
        "metrics": ["exact_match", "f1"],
    }

    config_path = temp_dir / "test_config.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def temp_predictions_file(temp_dir) -> Path:
    """Create a temporary predictions file."""
    predictions = {
        "predictions": [
            {"question": "Q1?", "prediction": "A1", "references": ["A1", "A2"]},
            {"question": "Q2?", "prediction": "B1", "references": ["B1"]},
        ],
        "metadata": {"model": "test", "timestamp": "2024-01-01"},
    }

    pred_path = temp_dir / "predictions.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f)

    return pred_path


# === Test Utilities ===


@pytest.fixture
def assert_no_warnings():
    """Fixture to assert no warnings are raised."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w

        if w:
            warning_messages = [str(warning.message) for warning in w]
            pytest.fail(f"Unexpected warnings: {warning_messages}")


# === Helper Functions ===


def _has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_openai_key() -> bool:
    """Check if OpenAI API key is set."""
    import os

    return bool(os.environ.get("OPENAI_API_KEY"))


# === Skip Markers ===

requires_gpu = pytest.mark.skipif(not _has_cuda(), reason="Test requires GPU")

requires_openai = pytest.mark.skipif(not _has_openai_key(), reason="Test requires OPENAI_API_KEY")

slow = pytest.mark.slow


# === Configure pytest ===


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
