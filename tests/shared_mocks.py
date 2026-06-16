"""Shared mock classes for agent tests.

These mocks implement the provider-pattern interfaces (batch_generate,
batch_encode, batch_search, context-manager load()) used by all RAG agents.

Used by: test_agents.py, test_iterative_rag.py, test_self_rag.py, conftest.py
"""

from contextlib import contextmanager

import numpy as np

from ragicamp.core.types import Document, SearchResult


class FakeGenerator:
    """Mock generator with configurable pattern-matched responses.

    For each prompt, checks self._responses dict for a substring match.
    Falls back to a default answer if no pattern matches.
    """

    def __init__(self, responses=None):
        self._responses = responses or {}
        self._call_count = 0
        self._prompts_seen: list[str] = []
        self.model_name = "fake-generator"

    def batch_generate(self, prompts, **kwargs):
        self._call_count += 1
        results = []
        for p in prompts:
            self._prompts_seen.append(p)
            matched = False
            for key, val in self._responses.items():
                if key in p:
                    results.append(val)
                    matched = True
                    break
            if not matched:
                results.append("default answer")
        return results


class FakeEmbedder:
    """Mock embedder that returns random fixed-dimension vectors."""

    def __init__(self, dim=4):
        self.dim = dim

    def batch_encode(self, texts):
        return np.random.randn(len(texts), self.dim).astype("float32")

    def get_dimension(self):
        return self.dim


class FakeProvider:
    """Mock provider that yields a model via a context-manager load() method.

    Works as both a generator provider and an embedder provider depending
    on the model object passed in.
    """

    def __init__(self, model, model_name="fake"):
        self._model = model
        self.model_name = model_name
        self.config = type("C", (), {"model_name": model_name})()
        self.is_loaded = False

    @contextmanager
    def load(self, **kwargs):
        self.is_loaded = True
        try:
            yield self._model
        finally:
            self.is_loaded = False


class FakeIndex:
    """Mock search backend that returns deterministic SearchResult lists."""

    def __init__(self, docs=None, num_default=5):
        if docs is not None:
            self._docs = docs
        else:
            self._docs = [
                Document(
                    id=f"doc{i}",
                    text=f"Document {i} content about topic",
                    score=0.9 - i * 0.1,
                )
                for i in range(num_default)
            ]

    @property
    def documents(self):
        return self._docs

    def batch_search(self, embeddings, top_k=5, **kwargs):
        return [
            [
                SearchResult(document=d, score=d.score, rank=i + 1)
                for i, d in enumerate(self._docs[:top_k])
            ]
            for _ in range(len(embeddings))
        ]


class FakeQueryTransformer:
    """Mock query transformer that returns each query as-is and tracks calls."""

    def __init__(self):
        self.call_count = 0
        self.last_queries: list[str] = []

    def batch_transform(self, queries):
        self.call_count += 1
        self.last_queries = queries
        return [[q] for q in queries]
