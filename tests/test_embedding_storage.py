"""Tests for EmbeddingStorage class."""

from pathlib import Path

import numpy as np
import pytest

from ragicamp.indexes.builders.storage import EmbeddingStorage


class TestEmbeddingStorage:
    """Tests for EmbeddingStorage."""

    def test_append_and_load_embeddings(self, tmp_path: Path):
        """Test appending and loading embeddings."""
        storage = EmbeddingStorage(tmp_path)

        # Append two batches
        emb1 = np.random.rand(10, 128).astype(np.float32)
        emb2 = np.random.rand(5, 128).astype(np.float32)

        storage.append_embeddings(emb1)
        storage.append_embeddings(emb2)

        # Load all
        loaded = storage.load_all_embeddings()

        assert loaded.shape == (15, 128)
        np.testing.assert_array_almost_equal(loaded[:10], emb1)
        np.testing.assert_array_almost_equal(loaded[10:], emb2)

    def test_append_and_load_chunks(self, tmp_path: Path):
        """Test appending and loading chunks."""
        storage = EmbeddingStorage(tmp_path)

        # Append two batches of chunks
        chunks1 = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
        chunks2 = [{"id": 3, "text": "foo"}]

        storage.append_chunks(chunks1)
        storage.append_chunks(chunks2)

        # Load all
        loaded = storage.load_all_chunks()

        assert len(loaded) == 3
        assert loaded[0]["id"] == 1
        assert loaded[2]["text"] == "foo"

    def test_multiple_streams(self, tmp_path: Path):
        """Test multiple named streams (parent/child)."""
        storage = EmbeddingStorage(tmp_path)

        # Append to different streams
        parent_emb = np.random.rand(5, 128).astype(np.float32)
        child_emb = np.random.rand(20, 128).astype(np.float32)

        storage.append_embeddings(parent_emb, key="parent")
        storage.append_embeddings(child_emb, key="child")

        parent_chunks = [{"id": f"p{i}"} for i in range(5)]
        child_chunks = [{"id": f"c{i}"} for i in range(20)]

        storage.append_chunks(parent_chunks, key="parent")
        storage.append_chunks(child_chunks, key="child")

        # Load separately
        loaded_parent_emb = storage.load_all_embeddings(key="parent")
        loaded_child_emb = storage.load_all_embeddings(key="child")
        loaded_parent_chunks = storage.load_all_chunks(key="parent")
        loaded_child_chunks = storage.load_all_chunks(key="child")

        assert loaded_parent_emb.shape == (5, 128)
        assert loaded_child_emb.shape == (20, 128)
        assert len(loaded_parent_chunks) == 5
        assert len(loaded_child_chunks) == 20

    def test_get_embedding_sizes(self, tmp_path: Path):
        """Test tracking of embedding batch sizes."""
        storage = EmbeddingStorage(tmp_path)

        storage.append_embeddings(np.random.rand(10, 64).astype(np.float32))
        storage.append_embeddings(np.random.rand(7, 64).astype(np.float32))
        storage.append_embeddings(np.random.rand(3, 64).astype(np.float32))

        sizes = storage.get_embedding_sizes()
        assert sizes == [10, 7, 3]

    def test_cleanup(self, tmp_path: Path):
        """Test cleanup removes temp files."""
        storage = EmbeddingStorage(tmp_path)

        storage.append_embeddings(np.random.rand(5, 64).astype(np.float32))
        storage.append_chunks([{"id": 1}])

        # Files should exist
        assert (tmp_path / "embeddings_default.npy").exists()
        assert (tmp_path / "chunks_default.pkl").exists()

        storage.cleanup()

        # Files should be removed
        assert not (tmp_path / "embeddings_default.npy").exists()
        assert not (tmp_path / "chunks_default.pkl").exists()

    def test_context_manager(self, tmp_path: Path):
        """Test context manager closes handles."""
        with EmbeddingStorage(tmp_path) as storage:
            storage.append_embeddings(np.random.rand(5, 64).astype(np.float32))
            storage.append_chunks([{"id": 1}])
            saved_sizes = storage.get_embedding_sizes()

        # After context, should be able to load (handles closed properly)
        # Note: New storage instance needs explicit sizes (simulating resume)
        storage2 = EmbeddingStorage(tmp_path)
        loaded = storage2.load_all_embeddings(sizes=saved_sizes)
        assert loaded.shape == (5, 64)

    def test_empty_storage(self, tmp_path: Path):
        """Test loading from empty storage."""
        storage = EmbeddingStorage(tmp_path)

        embeddings = storage.load_all_embeddings()
        chunks = storage.load_all_chunks()

        assert len(embeddings) == 0
        assert len(chunks) == 0

    def test_load_with_explicit_sizes(self, tmp_path: Path):
        """Test loading embeddings with explicit sizes (for resume)."""
        storage = EmbeddingStorage(tmp_path)

        emb1 = np.random.rand(10, 128).astype(np.float32)
        emb2 = np.random.rand(5, 128).astype(np.float32)

        storage.append_embeddings(emb1)
        storage.append_embeddings(emb2)
        storage.close()

        # New storage instance with explicit sizes (simulating resume)
        storage2 = EmbeddingStorage(tmp_path)
        loaded = storage2.load_all_embeddings(sizes=[10, 5])

        assert loaded.shape == (15, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
