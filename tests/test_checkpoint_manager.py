"""Tests for CheckpointManager class."""

from pathlib import Path

import pytest

from ragicamp.indexes.builders.checkpoint import (
    CheckpointManager,
    EmbeddingCheckpoint,
    HierarchicalCheckpoint,
)


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_save_and_load_embedding_checkpoint(self, tmp_path: Path):
        """Test saving and loading embedding checkpoint."""
        manager = CheckpointManager(tmp_path)

        checkpoint = EmbeddingCheckpoint(
            batch_num=5,
            total_docs=5000,
            total_chunks=50000,
            embedding_sizes=[10000, 10000, 10000, 10000, 10000],
        )

        manager.save(checkpoint)
        loaded = manager.load(EmbeddingCheckpoint)

        assert loaded is not None
        assert loaded.batch_num == 5
        assert loaded.total_docs == 5000
        assert loaded.total_chunks == 50000
        assert loaded.embedding_sizes == [10000, 10000, 10000, 10000, 10000]

    def test_save_and_load_hierarchical_checkpoint(self, tmp_path: Path):
        """Test saving and loading hierarchical checkpoint."""
        manager = CheckpointManager(tmp_path)

        checkpoint = HierarchicalCheckpoint(
            batch_num=3,
            total_docs=3000,
            total_parent_chunks=3000,
            total_child_chunks=15000,
            parent_embedding_sizes=[],
            child_embedding_sizes=[5000, 5000, 5000],
        )

        manager.save(checkpoint)
        loaded = manager.load(HierarchicalCheckpoint)

        assert loaded is not None
        assert loaded.batch_num == 3
        assert loaded.total_docs == 3000
        assert loaded.total_parent_chunks == 3000
        assert loaded.total_child_chunks == 15000

    def test_load_nonexistent_returns_none(self, tmp_path: Path):
        """Test loading when no checkpoint exists."""
        manager = CheckpointManager(tmp_path)
        loaded = manager.load(EmbeddingCheckpoint)

        assert loaded is None

    def test_exists(self, tmp_path: Path):
        """Test exists method."""
        manager = CheckpointManager(tmp_path)

        assert not manager.exists()

        checkpoint = EmbeddingCheckpoint(
            batch_num=1,
            total_docs=100,
            total_chunks=1000,
            embedding_sizes=[1000],
        )
        manager.save(checkpoint)

        assert manager.exists()

    def test_clear(self, tmp_path: Path):
        """Test clearing checkpoint."""
        manager = CheckpointManager(tmp_path)

        checkpoint = EmbeddingCheckpoint(
            batch_num=1,
            total_docs=100,
            total_chunks=1000,
            embedding_sizes=[1000],
        )
        manager.save(checkpoint)
        assert manager.exists()

        manager.clear()
        assert not manager.exists()

    def test_type_mismatch_returns_none(self, tmp_path: Path):
        """Test loading wrong checkpoint type returns None."""
        manager = CheckpointManager(tmp_path)

        # Save embedding checkpoint
        checkpoint = EmbeddingCheckpoint(
            batch_num=1,
            total_docs=100,
            total_chunks=1000,
            embedding_sizes=[1000],
        )
        manager.save(checkpoint)

        # Try to load as hierarchical
        loaded = manager.load(HierarchicalCheckpoint)
        assert loaded is None

    def test_overwrite_checkpoint(self, tmp_path: Path):
        """Test that saving overwrites existing checkpoint."""
        manager = CheckpointManager(tmp_path)

        # Save first checkpoint
        checkpoint1 = EmbeddingCheckpoint(
            batch_num=1,
            total_docs=100,
            total_chunks=1000,
            embedding_sizes=[1000],
        )
        manager.save(checkpoint1)

        # Save second checkpoint
        checkpoint2 = EmbeddingCheckpoint(
            batch_num=5,
            total_docs=500,
            total_chunks=5000,
            embedding_sizes=[1000, 1000, 1000, 1000, 1000],
        )
        manager.save(checkpoint2)

        # Load should return latest
        loaded = manager.load(EmbeddingCheckpoint)
        assert loaded.batch_num == 5
        assert loaded.total_docs == 500

    def test_corrupted_checkpoint_returns_none(self, tmp_path: Path):
        """Test that corrupted checkpoint file returns None."""
        manager = CheckpointManager(tmp_path)

        # Write invalid JSON
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint_path.write_text("not valid json {{{")

        loaded = manager.load(EmbeddingCheckpoint)
        assert loaded is None

    def test_checkpoint_path(self, tmp_path: Path):
        """Test getting checkpoint path."""
        manager = CheckpointManager(tmp_path)
        path = manager.get_checkpoint_path()

        assert path == tmp_path / "checkpoint.json"


class TestEmbeddingCheckpoint:
    """Tests for EmbeddingCheckpoint dataclass."""

    def test_default_values(self):
        """Test default values."""
        checkpoint = EmbeddingCheckpoint(
            batch_num=1,
            total_docs=100,
            total_chunks=1000,
        )

        assert checkpoint.embedding_sizes == []
        assert checkpoint.emb_file == "embeddings_default.npy"
        assert checkpoint.chunks_file == "chunks_default.pkl"


class TestHierarchicalCheckpoint:
    """Tests for HierarchicalCheckpoint dataclass."""

    def test_default_values(self):
        """Test default values."""
        checkpoint = HierarchicalCheckpoint(
            batch_num=1,
            total_docs=100,
            total_parent_chunks=100,
            total_child_chunks=500,
        )

        assert checkpoint.parent_embedding_sizes == []
        assert checkpoint.child_embedding_sizes == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
