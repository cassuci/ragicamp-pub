"""Tests for HybridSearcher RRF fusion logic."""

import numpy as np
import pytest

from ragicamp.core.types import Document, SearchResult
from ragicamp.retrievers.hybrid import HybridSearcher


class FakeVectorIndex:
    """Mock VectorIndex for testing.

    Must maintain a documents list that matches the indices used by sparse results.
    """

    def __init__(self, results_per_query: list[list[SearchResult]], all_documents: list[Document]):
        self._results = results_per_query
        self.documents = all_documents

    def search(self, query_embedding, top_k: int, **kwargs) -> list[SearchResult]:
        return self._results[0][:top_k] if self._results else []

    def batch_search(self, query_embeddings, top_k: int, **kwargs) -> list[list[SearchResult]]:
        num_queries = len(query_embeddings)
        return [res[:top_k] for res in self._results[:num_queries]]


class FakeSparseIndex:
    """Mock SparseIndex for testing."""

    def __init__(self, results_per_query: list[list[tuple[int, float]]]):
        # Store as list of (index, score) tuples
        self._results = results_per_query

    def search(self, query_text: str, top_k: int, **kwargs) -> list[tuple[int, float]]:
        return self._results[0][:top_k] if self._results else []

    def batch_search(
        self, query_texts: list[str], top_k: int, **kwargs
    ) -> list[list[tuple[int, float]]]:
        num_queries = len(query_texts)
        return [res[:top_k] for res in self._results[:num_queries]]


@pytest.fixture
def sample_documents():
    return [
        Document(id="doc1", text="Python programming", metadata={}),
        Document(id="doc2", text="Java development", metadata={}),
        Document(id="doc3", text="Machine learning", metadata={}),
        Document(id="doc4", text="Data science", metadata={}),
        Document(id="doc5", text="Web development", metadata={}),
    ]


class TestHybridSearcherRRF:
    """Tests for RRF fusion logic."""

    def test_rrf_merge_basic(self, sample_documents):
        # Dense returns doc1, doc2
        dense_results = [
            [
                SearchResult(document=sample_documents[0], score=0.9, rank=0),
                SearchResult(document=sample_documents[1], score=0.8, rank=1),
            ]
        ]

        # Sparse returns doc3, doc4 (indexes 2, 3 in documents list)
        sparse_results = [[(2, 0.7), (3, 0.6)]]

        vector_index = FakeVectorIndex(dense_results, sample_documents)
        sparse_index = FakeSparseIndex(sparse_results)

        searcher = HybridSearcher(
            vector_index=vector_index,
            sparse_index=sparse_index,
            alpha=0.5,
            rrf_k=60,
        )

        query_embedding = np.array([0.1, 0.2, 0.3])
        results = searcher.search(query_embedding, query_text="test query", top_k=4)

        # Should have all 4 documents
        assert len(results) == 4
        result_ids = {r.document.id for r in results}
        assert result_ids == {"doc1", "doc2", "doc3", "doc4"}

        # All results should have scores
        assert all(r.score > 0 for r in results)

    def test_rrf_merge_with_overlap(self, sample_documents):
        # Both dense and sparse return doc1 (index 0)
        dense_results = [
            [
                SearchResult(document=sample_documents[0], score=0.9, rank=0),
                SearchResult(document=sample_documents[1], score=0.8, rank=1),
            ]
        ]

        # Sparse also includes doc1 (index 0) and doc3 (index 2)
        sparse_results = [[(0, 0.95), (2, 0.7)]]

        vector_index = FakeVectorIndex(dense_results, sample_documents)
        sparse_index = FakeSparseIndex(sparse_results)

        searcher = HybridSearcher(
            vector_index=vector_index,
            sparse_index=sparse_index,
            alpha=0.5,
            rrf_k=60,
        )

        query_embedding = np.array([0.1, 0.2, 0.3])
        results = searcher.search(query_embedding, query_text="test query", top_k=5)

        # Should have 3 unique documents
        assert len(results) == 3
        result_ids = [r.document.id for r in results]

        # doc1 should appear only once
        assert result_ids.count("doc1") == 1

        # doc1 should be first (appears in both, gets both RRF contributions)
        assert results[0].document.id == "doc1"

        # doc1 score should be higher than others (sum of both rank contributions)
        assert results[0].score > results[1].score

    def test_alpha_1_dense_only(self, sample_documents):
        # Dense returns doc1, doc2
        dense_results = [
            [
                SearchResult(document=sample_documents[0], score=0.9, rank=0),
                SearchResult(document=sample_documents[1], score=0.8, rank=1),
            ]
        ]

        # Sparse returns doc3, doc4
        sparse_results = [[(2, 0.7), (3, 0.6)]]

        vector_index = FakeVectorIndex(dense_results, sample_documents)
        sparse_index = FakeSparseIndex(sparse_results)

        searcher = HybridSearcher(
            vector_index=vector_index,
            sparse_index=sparse_index,
            alpha=1.0,  # Dense only
            rrf_k=60,
        )

        query_embedding = np.array([0.1, 0.2, 0.3])
        results = searcher.search(query_embedding, query_text="test query", top_k=5)

        result_ids = [r.document.id for r in results]

        # Dense docs should be first (higher scores from alpha=1.0)
        assert result_ids[0] == "doc1"
        assert result_ids[1] == "doc2"

        # Sparse docs should have score of 0 (alpha=1.0 means (1-alpha)=0 weight)
        sparse_doc_scores = [r.score for r in results if r.document.id in ["doc3", "doc4"]]
        assert all(score == 0.0 for score in sparse_doc_scores)

    def test_alpha_0_sparse_only(self, sample_documents):
        # Dense returns doc1, doc2
        dense_results = [
            [
                SearchResult(document=sample_documents[0], score=0.9, rank=0),
                SearchResult(document=sample_documents[1], score=0.8, rank=1),
            ]
        ]

        # Sparse returns doc3, doc4
        sparse_results = [[(2, 0.7), (3, 0.6)]]

        vector_index = FakeVectorIndex(dense_results, sample_documents)
        sparse_index = FakeSparseIndex(sparse_results)

        searcher = HybridSearcher(
            vector_index=vector_index,
            sparse_index=sparse_index,
            alpha=0.0,  # Sparse only
            rrf_k=60,
        )

        query_embedding = np.array([0.1, 0.2, 0.3])
        results = searcher.search(query_embedding, query_text="test query", top_k=5)

        result_ids = [r.document.id for r in results]

        # Sparse docs should be first (higher scores from (1-alpha)=1.0)
        assert result_ids[0] == "doc3"
        assert result_ids[1] == "doc4"

        # Dense docs should have score of 0 (alpha=0.0 means 0 weight)
        dense_doc_scores = [r.score for r in results if r.document.id in ["doc1", "doc2"]]
        assert all(score == 0.0 for score in dense_doc_scores)

    def test_empty_dense_results(self, sample_documents):
        # Dense returns nothing
        dense_results = [[]]

        # Sparse returns doc3, doc4
        sparse_results = [[(2, 0.7), (3, 0.6)]]

        vector_index = FakeVectorIndex(dense_results, sample_documents)
        sparse_index = FakeSparseIndex(sparse_results)

        searcher = HybridSearcher(
            vector_index=vector_index,
            sparse_index=sparse_index,
            alpha=0.5,
            rrf_k=60,
        )

        query_embedding = np.array([0.1, 0.2, 0.3])
        results = searcher.search(query_embedding, query_text="test query", top_k=5)

        # Should have sparse results
        assert len(results) == 2
        result_ids = {r.document.id for r in results}
        assert result_ids == {"doc3", "doc4"}

    def test_empty_sparse_results(self, sample_documents):
        # Dense returns doc1, doc2
        dense_results = [
            [
                SearchResult(document=sample_documents[0], score=0.9, rank=0),
                SearchResult(document=sample_documents[1], score=0.8, rank=1),
            ]
        ]

        # Sparse returns nothing
        sparse_results = [[]]

        vector_index = FakeVectorIndex(dense_results, sample_documents)
        sparse_index = FakeSparseIndex(sparse_results)

        searcher = HybridSearcher(
            vector_index=vector_index,
            sparse_index=sparse_index,
            alpha=0.5,
            rrf_k=60,
        )

        query_embedding = np.array([0.1, 0.2, 0.3])
        results = searcher.search(query_embedding, query_text="test query", top_k=5)

        # Should have dense results
        assert len(results) == 2
        result_ids = {r.document.id for r in results}
        assert result_ids == {"doc1", "doc2"}

    def test_batch_search_multiple_queries(self, sample_documents):
        # Query 1: dense=[doc1, doc2], sparse=[doc3]
        # Query 2: dense=[doc4], sparse=[doc5, doc1]
        # Query 3: dense=[doc2], sparse=[]
        dense_results = [
            [
                SearchResult(document=sample_documents[0], score=0.9, rank=0),
                SearchResult(document=sample_documents[1], score=0.8, rank=1),
            ],
            [SearchResult(document=sample_documents[3], score=0.85, rank=0)],
            [SearchResult(document=sample_documents[1], score=0.75, rank=0)],
        ]

        sparse_results = [
            [(2, 0.7)],  # doc3
            [(4, 0.8), (0, 0.6)],  # doc5, doc1
            [],  # empty
        ]

        vector_index = FakeVectorIndex(dense_results, sample_documents)
        sparse_index = FakeSparseIndex(sparse_results)

        searcher = HybridSearcher(
            vector_index=vector_index,
            sparse_index=sparse_index,
            alpha=0.5,
            rrf_k=60,
        )

        query_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        query_texts = ["query1", "query2", "query3"]

        results = searcher.batch_search(query_embeddings, query_texts=query_texts, top_k=5)

        # Should have 3 result lists
        assert len(results) == 3

        # Query 1 should have 3 unique docs
        assert len(results[0]) == 3
        ids_q1 = {r.document.id for r in results[0]}
        assert ids_q1 == {"doc1", "doc2", "doc3"}

        # Query 2 should have 3 unique docs (doc1 appears in both)
        assert len(results[1]) == 3
        ids_q2 = {r.document.id for r in results[1]}
        assert ids_q2 == {"doc1", "doc4", "doc5"}

        # Query 3 should have 1 doc (only dense)
        assert len(results[2]) == 1
        assert results[2][0].document.id == "doc2"

    def test_top_k_respected(self, sample_documents):
        # Dense returns 3 docs
        dense_results = [
            [
                SearchResult(document=sample_documents[0], score=0.9, rank=0),
                SearchResult(document=sample_documents[1], score=0.8, rank=1),
                SearchResult(document=sample_documents[2], score=0.7, rank=2),
            ]
        ]

        # Sparse returns 3 different docs (need to use indices 3, 4)
        # Note: doc at index 2 is also in dense, so we have 5 unique docs total
        sparse_results = [[(3, 0.75), (4, 0.65), (2, 0.55)]]

        vector_index = FakeVectorIndex(dense_results, sample_documents)
        sparse_index = FakeSparseIndex(sparse_results)

        searcher = HybridSearcher(
            vector_index=vector_index,
            sparse_index=sparse_index,
            alpha=0.5,
            rrf_k=60,
        )

        query_embedding = np.array([0.1, 0.2, 0.3])

        # Request only top 3
        results = searcher.search(query_embedding, query_text="test query", top_k=3)

        # Should return exactly 3 results, not all 5
        assert len(results) == 3

    def test_results_sorted_by_score(self, sample_documents):
        # Create results with known ordering
        dense_results = [
            [
                SearchResult(document=sample_documents[0], score=0.9, rank=0),
                SearchResult(document=sample_documents[1], score=0.7, rank=1),
            ]
        ]

        sparse_results = [[(2, 0.8), (3, 0.6)]]

        vector_index = FakeVectorIndex(dense_results, sample_documents)
        sparse_index = FakeSparseIndex(sparse_results)

        searcher = HybridSearcher(
            vector_index=vector_index,
            sparse_index=sparse_index,
            alpha=0.5,
            rrf_k=60,
        )

        query_embedding = np.array([0.1, 0.2, 0.3])
        results = searcher.search(query_embedding, query_text="test query", top_k=5)

        # Results should be sorted descending by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # Ranks should be 0, 1, 2, 3...
        ranks = [r.rank for r in results]
        assert ranks == list(range(len(results)))
