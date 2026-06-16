"""Tests for LazySearchBackend proxy."""

from ragicamp.retrievers.lazy import LazySearchBackend


class FakeIndex:
    """Minimal stub that mimics a search backend."""

    def __init__(self):
        self.batch_search_calls = 0

    def batch_search(self, embeddings, top_k=5):
        self.batch_search_calls += 1
        return [[] for _ in range(len(embeddings))]

    def __len__(self):
        return 42


class FakeHybridIndex(FakeIndex):
    """Stub that looks like a HybridSearcher (has sparse_index)."""

    sparse_index = "dummy"

    def batch_search(self, embeddings, top_k=5, query_texts=None):
        self.batch_search_calls += 1
        return [[] for _ in range(len(embeddings))]


# ── Core behaviour ──────────────────────────────────────────────


class TestLazySearchBackend:
    def test_does_not_load_on_construction(self):
        """Creating a lazy proxy should NOT call the loader."""
        called = []

        def loader():
            called.append(1)
            return FakeIndex()

        proxy = LazySearchBackend(loader)
        assert not called
        assert not proxy.is_loaded

    def test_loads_on_batch_search(self):
        """First batch_search call should trigger the load."""
        import numpy as np

        real = FakeIndex()
        proxy = LazySearchBackend(lambda: real)

        embeddings = np.zeros((3, 128), dtype=np.float32)
        results = proxy.batch_search(embeddings, top_k=5)

        assert proxy.is_loaded
        assert real.batch_search_calls == 1
        assert len(results) == 3

    def test_loads_only_once(self):
        """Multiple accesses should only call loader once."""
        import numpy as np

        call_count = [0]

        def loader():
            call_count[0] += 1
            return FakeIndex()

        proxy = LazySearchBackend(loader)

        embeddings = np.zeros((2, 64), dtype=np.float32)
        proxy.batch_search(embeddings, top_k=5)
        proxy.batch_search(embeddings, top_k=10)
        _ = len(proxy)

        assert call_count[0] == 1

    def test_len_triggers_load(self):
        real = FakeIndex()
        proxy = LazySearchBackend(lambda: real)

        assert not proxy.is_loaded
        assert len(proxy) == 42
        assert proxy.is_loaded

    def test_arbitrary_attr_forwarded(self):
        real = FakeIndex()
        real.custom_value = "hello"
        proxy = LazySearchBackend(lambda: real)

        assert proxy.custom_value == "hello"
        assert proxy.is_loaded


# ── Hybrid detection ────────────────────────────────────────────


class TestHybridDetection:
    def test_non_hybrid_hides_sparse_index(self):
        """Non-hybrid proxy should NOT expose sparse_index."""
        proxy = LazySearchBackend(lambda: FakeIndex(), is_hybrid=False)
        assert not hasattr(proxy, "sparse_index")
        assert not proxy.is_loaded  # check didn't trigger load

    def test_hybrid_exposes_sparse_index(self):
        """Hybrid proxy should expose sparse_index (triggers load)."""
        real = FakeHybridIndex()
        proxy = LazySearchBackend(lambda: real, is_hybrid=True)

        # hasattr should be True and triggers load
        assert hasattr(proxy, "sparse_index")
        assert proxy.is_loaded
        assert proxy.sparse_index == "dummy"

    def test_is_hybrid_searcher_compat(self):
        """is_hybrid_searcher() pattern should work without loading for non-hybrid."""
        # is_hybrid_searcher checks: hasattr(index, "sparse_index")
        proxy_dense = LazySearchBackend(lambda: FakeIndex(), is_hybrid=False)
        proxy_hybrid = LazySearchBackend(lambda: FakeHybridIndex(), is_hybrid=True)

        assert not hasattr(proxy_dense, "sparse_index")
        assert not proxy_dense.is_loaded  # dense check didn't load

        assert hasattr(proxy_hybrid, "sparse_index")
        # hybrid check DOES load (needs real sparse_index)


# ── Repr ────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_before_load(self):
        proxy = LazySearchBackend(lambda: FakeIndex())
        assert "not loaded" in repr(proxy)

    def test_repr_after_load(self):
        proxy = LazySearchBackend(lambda: FakeIndex())
        _ = len(proxy)  # trigger load
        assert "loaded" in repr(proxy)
        assert "FakeIndex" in repr(proxy)
