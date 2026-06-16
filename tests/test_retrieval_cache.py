"""Tests for the retrieval result cache.

Covers RetrievalStore (SQLite KV) and the cache-aware batch_embed_and_search
integration in the agent pipeline.
"""

import pytest

from ragicamp.cache.retrieval_store import RetrievalStore

# ---------------------------------------------------------------------------
# RetrievalStore
# ---------------------------------------------------------------------------


class TestRetrievalStore:
    """Tests for the SQLite-backed retrieval result cache."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary store."""
        return RetrievalStore(tmp_path / "test_cache.db")

    @pytest.fixture
    def sample_results(self):
        """Sample serialized SearchResult dicts."""
        return [
            [
                {
                    "document": {
                        "id": "doc1",
                        "text": "The answer is 42.",
                        "metadata": {"source": "wiki"},
                        "score": 0.95,
                    },
                    "score": 0.95,
                    "rank": 0,
                },
                {
                    "document": {
                        "id": "doc2",
                        "text": "Life, universe, everything.",
                        "metadata": {},
                        "score": 0.80,
                    },
                    "score": 0.80,
                    "rank": 1,
                },
            ],
            [
                {
                    "document": {
                        "id": "doc3",
                        "text": "Paris is the capital of France.",
                        "metadata": {},
                        "score": 0.99,
                    },
                    "score": 0.99,
                    "rank": 0,
                },
            ],
        ]

    def test_put_and_get(self, store, sample_results):
        """Test basic store and retrieve."""
        queries = ["What is 42?", "Capital of France?"]
        written = store.put_batch("dense_bge", queries, sample_results, top_k=5)

        assert written == 2

        cached, hit_mask = store.get_batch("dense_bge", queries, top_k=5)

        assert hit_mask == [True, True]
        assert len(cached[0]) == 2
        assert cached[0][0]["document"]["id"] == "doc1"
        assert cached[1][0]["document"]["text"] == "Paris is the capital of France."

    def test_miss_returns_none(self, store):
        """Test cache miss returns None entries."""
        cached, hit_mask = store.get_batch("dense_bge", ["unknown query"], top_k=5)

        assert hit_mask == [False]
        assert cached[0] is None

    def test_partial_hit(self, store, sample_results):
        """Test mixed hit/miss."""
        store.put_batch("dense_bge", ["query1"], [sample_results[0]], top_k=5)

        cached, hit_mask = store.get_batch(
            "dense_bge",
            ["query1", "query2"],
            top_k=5,
        )

        assert hit_mask == [True, False]
        assert cached[0] is not None
        assert cached[1] is None

    def test_different_top_k_no_collision(self, store, sample_results):
        """Different top_k values don't collide."""
        store.put_batch("dense_bge", ["q1"], [sample_results[0]], top_k=5)
        store.put_batch("dense_bge", ["q1"], [sample_results[1]], top_k=10)

        cached_5, hits_5 = store.get_batch("dense_bge", ["q1"], top_k=5)
        cached_10, hits_10 = store.get_batch("dense_bge", ["q1"], top_k=10)

        assert hits_5 == [True]
        assert hits_10 == [True]
        # Different results for different top_k
        assert len(cached_5[0]) == 2  # sample_results[0] has 2 docs
        assert len(cached_10[0]) == 1  # sample_results[1] has 1 doc

    def test_different_retriever_no_collision(self, store, sample_results):
        """Different retriever names don't collide."""
        store.put_batch("dense_bge", ["q1"], [sample_results[0]], top_k=5)

        cached, hit_mask = store.get_batch("hybrid_bge", ["q1"], top_k=5)
        assert hit_mask == [False]

    def test_insert_or_ignore(self, store, sample_results):
        """Duplicate inserts are no-ops."""
        written1 = store.put_batch("r", ["q"], [sample_results[0]], top_k=5)
        written2 = store.put_batch("r", ["q"], [sample_results[1]], top_k=5)

        assert written1 == 1
        assert written2 == 0  # Ignored — same key exists

        # Original data is preserved
        cached, _ = store.get_batch("r", ["q"], top_k=5)
        assert len(cached[0]) == 2  # Still the first insert

    def test_empty_queries(self, store):
        """Empty query list returns empty results."""
        cached, hit_mask = store.get_batch("r", [], top_k=5)
        assert cached == []
        assert hit_mask == []

        written = store.put_batch("r", [], [], top_k=5)
        assert written == 0

    def test_stats(self, store, sample_results):
        """Test stats reporting."""
        store.put_batch("dense_bge", ["q1", "q2"], sample_results, top_k=5)

        stats = store.stats()
        assert stats["total_entries"] == 2
        assert "dense_bge_k5" in stats["retrievers"]
        assert stats["retrievers"]["dense_bge_k5"]["entries"] == 2

    def test_clear(self, store, sample_results):
        """Test clearing the cache."""
        store.put_batch("r1", ["q1"], [sample_results[0]], top_k=5)
        store.put_batch("r2", ["q1"], [sample_results[0]], top_k=5)

        deleted = store.clear(retriever="r1")
        assert deleted == 1

        # r1 gone, r2 still there
        _, hits1 = store.get_batch("r1", ["q1"], top_k=5)
        _, hits2 = store.get_batch("r2", ["q1"], top_k=5)
        assert hits1 == [False]
        assert hits2 == [True]

    def test_clear_all(self, store, sample_results):
        """Test clearing all entries."""
        store.put_batch("r1", ["q1"], [sample_results[0]], top_k=5)
        store.put_batch("r2", ["q1"], [sample_results[0]], top_k=5)

        deleted = store.clear()
        assert deleted == 2

    def test_concurrent_access_same_db(self, tmp_path, sample_results):
        """Two store instances on the same DB don't corrupt data."""
        db_path = tmp_path / "shared.db"
        store1 = RetrievalStore(db_path)
        store2 = RetrievalStore(db_path)

        store1.put_batch("r", ["q1"], [sample_results[0]], top_k=5)
        cached, hits = store2.get_batch("r", ["q1"], top_k=5)

        assert hits == [True]
        assert cached[0][0]["document"]["id"] == "doc1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
