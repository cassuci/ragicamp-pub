"""Tests for BM25 sparse index using bm25s library."""

import pytest

from ragicamp.core.types import Document
from ragicamp.indexes.sparse import SparseIndex, SparseMethod


def _make_docs(texts: list[str]) -> list[Document]:
    return [Document(id=f"doc{i}", text=t) for i, t in enumerate(texts)]


CORPUS = [
    "python programming language is widely used",
    "java development for enterprise applications",
    "machine learning with neural networks",
    "data science and statistical analysis",
    "web development using javascript frameworks",
]


@pytest.fixture
def bm25_index():
    docs = _make_docs(CORPUS)
    idx = SparseIndex(name="test_bm25", method=SparseMethod.BM25)
    idx.build(docs, show_progress=False)
    return idx


class TestBM25BatchSearch:
    def test_batch_search_returns_correct_format(self, bm25_index):
        results = bm25_index.batch_search(["python", "java"], top_k=3)

        assert isinstance(results, list)
        assert len(results) == 2
        for query_results in results:
            assert isinstance(query_results, list)
            for item in query_results:
                assert isinstance(item, tuple)
                assert len(item) == 2
                doc_idx, score = item
                assert isinstance(doc_idx, int)
                assert isinstance(score, float)
                assert doc_idx >= 0
                assert score > 0.0

    def test_single_search_delegates(self, bm25_index):
        single = bm25_index.search("python", top_k=3)
        batch = bm25_index.batch_search(["python"], top_k=3)

        assert single == batch[0]

    def test_keyword_matching(self, bm25_index):
        results = bm25_index.search("python programming", top_k=3)

        assert len(results) >= 1
        top_idx = results[0][0]
        assert top_idx == 0  # "python programming language..."

    def test_empty_results_for_unknown_term(self, bm25_index):
        results = bm25_index.search("xyznonexistent", top_k=3)
        assert results == []

    def test_top_k_respected(self, bm25_index):
        results = bm25_index.search("development", top_k=2)
        assert len(results) <= 2


class TestBM25SaveLoad:
    def test_save_and_load_roundtrip(self, bm25_index, tmp_path):
        save_path = tmp_path / "bm25_test"
        bm25_index.save(path=save_path)

        loaded = SparseIndex.load(
            name="test_bm25",
            path=save_path,
            documents=bm25_index.documents,
        )

        original = bm25_index.batch_search(["python", "machine learning"], top_k=3)
        reloaded = loaded.batch_search(["python", "machine learning"], top_k=3)

        assert len(original) == len(reloaded)
        for orig_q, load_q in zip(original, reloaded, strict=True):
            assert len(orig_q) == len(load_q)
            for (oi, os_), (li, ls) in zip(orig_q, load_q, strict=True):
                assert oi == li
                assert abs(os_ - ls) < 1e-5

    def test_old_format_detection(self, tmp_path):
        save_path = tmp_path / "old_bm25"
        save_path.mkdir(parents=True)

        # Create fake old-format files
        (save_path / "config.json").write_text(
            '{"name": "old", "method": "bm25", "max_features": 50000, "num_documents": 1}'
        )
        import pickle

        with open(save_path / "doc_ids.pkl", "wb") as f:
            pickle.dump(["doc0"], f)
        # Old rank_bm25 pickle (just needs to exist)
        (save_path / "bm25.pkl").write_bytes(b"fake")

        with pytest.raises(ValueError, match="old rank_bm25 pickle format"):
            SparseIndex.load(
                name="old",
                path=save_path,
                documents=[Document(id="doc0", text="test")],
            )


class TestBM25Uninitialized:
    def test_search_returns_empty_when_not_built(self):
        idx = SparseIndex(name="empty", method=SparseMethod.BM25)
        assert idx.search("test", top_k=3) == []

    def test_batch_search_returns_empty_when_not_built(self):
        idx = SparseIndex(name="empty", method=SparseMethod.BM25)
        results = idx.batch_search(["test1", "test2"], top_k=3)
        assert results == [[], []]
