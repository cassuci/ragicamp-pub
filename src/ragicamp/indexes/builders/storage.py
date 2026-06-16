"""Embedding storage for incremental disk-based processing.

Handles writing embeddings and chunks to disk during Phase 1 (embedding generation)
and reading them back during Phase 2 (normalization + indexing).

Supports multiple named streams for hierarchical builders (parent/child chunks).
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingStorage:
    """Incremental storage of embeddings and chunks to disk.

    Writes embeddings in append-only mode during Phase 1, then reads all back
    during Phase 2. Supports multiple named streams for different chunk types
    (e.g., 'parent', 'child' for hierarchical indexes).

    Example:
        storage = EmbeddingStorage(work_dir)

        # Phase 1: Append batches
        for batch in batches:
            embeddings = encoder.encode(batch)
            storage.append_embeddings(embeddings)
            storage.append_chunks(chunks)

        # Phase 2: Load all
        all_embeddings = storage.load_all_embeddings()
        all_chunks = storage.load_all_chunks()

        # Cleanup
        storage.cleanup()
    """

    def __init__(self, work_dir: Path | str):
        """Initialize storage.

        Args:
            work_dir: Directory to store temp files
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Track embedding sizes per stream (for reading back .npy files)
        self._embedding_sizes: dict[str, list[int]] = {}

        # File handles for append mode
        self._emb_files: dict[str, Any] = {}
        self._chunk_files: dict[str, Any] = {}

    def _get_emb_path(self, key: str = "default") -> Path:
        """Get path for embeddings file."""
        return self.work_dir / f"embeddings_{key}.npy"

    def _get_chunks_path(self, key: str = "default") -> Path:
        """Get path for chunks file."""
        return self.work_dir / f"chunks_{key}.pkl"

    def _get_emb_file(self, key: str = "default"):
        """Get or open file handle for embeddings."""
        if key not in self._emb_files:
            self._emb_files[key] = open(self._get_emb_path(key), "ab")
            self._embedding_sizes[key] = []
        return self._emb_files[key]

    def _get_chunk_file(self, key: str = "default"):
        """Get or open file handle for chunks."""
        if key not in self._chunk_files:
            self._chunk_files[key] = open(self._get_chunks_path(key), "ab")
        return self._chunk_files[key]

    def append_embeddings(self, embeddings: np.ndarray, key: str = "default") -> None:
        """Append embeddings to disk.

        Args:
            embeddings: Embeddings array to append
            key: Stream name (default 'default', use 'parent'/'child' for hierarchical)
        """
        f = self._get_emb_file(key)
        np.save(f, embeddings)
        f.flush()
        self._embedding_sizes[key].append(len(embeddings))

    def append_chunks(self, chunks: list, key: str = "default") -> None:
        """Append chunks to disk.

        Args:
            chunks: List of chunks to append
            key: Stream name
        """
        f = self._get_chunk_file(key)
        pickle.dump(chunks, f)
        f.flush()

    def get_embedding_sizes(self, key: str = "default") -> list[int]:
        """Get the sizes of each batch written (for reading back).

        Args:
            key: Stream name

        Returns:
            List of batch sizes
        """
        return self._embedding_sizes.get(key, [])

    def load_all_embeddings(
        self, key: str = "default", sizes: list[int] | None = None
    ) -> np.ndarray:
        """Load all embeddings from disk.

        Args:
            key: Stream name
            sizes: Batch sizes (required if loading from checkpoint, otherwise uses tracked sizes)

        Returns:
            Stacked numpy array of all embeddings
        """
        # Close file handle if open
        if key in self._emb_files:
            self._emb_files[key].close()
            del self._emb_files[key]

        sizes = sizes or self._embedding_sizes.get(key, [])
        if not sizes:
            return np.array([], dtype=np.float32)

        path = self._get_emb_path(key)
        if not path.exists():
            return np.array([], dtype=np.float32)

        all_embeddings = []
        with open(path, "rb") as f:
            for _ in sizes:
                emb = np.load(f)
                all_embeddings.append(emb)

        if not all_embeddings:
            return np.array([], dtype=np.float32)

        return np.vstack(all_embeddings)

    def load_all_chunks(self, key: str = "default") -> list:
        """Load all chunks from disk.

        Args:
            key: Stream name

        Returns:
            List of all chunks
        """
        # Close file handle if open
        if key in self._chunk_files:
            self._chunk_files[key].close()
            del self._chunk_files[key]

        path = self._get_chunks_path(key)
        if not path.exists():
            return []

        all_chunks = []
        with open(path, "rb") as f:
            while True:
                try:
                    batch = pickle.load(f)
                    all_chunks.extend(batch)
                except EOFError:
                    break

        return all_chunks

    def close(self) -> None:
        """Close all open file handles."""
        for key, f in list(self._emb_files.items()):
            f.close()
            del self._emb_files[key]

        for key, f in list(self._chunk_files.items()):
            f.close()
            del self._chunk_files[key]

    def cleanup(self) -> None:
        """Close handles and remove all temp files."""
        self.close()

        # Remove all files in work dir
        for path in self.work_dir.glob("embeddings_*.npy"):
            path.unlink()
            logger.debug("Removed temp file: %s", path)

        for path in self.work_dir.glob("chunks_*.pkl"):
            path.unlink()
            logger.debug("Removed temp file: %s", path)

        self._embedding_sizes.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close handles but don't cleanup on error."""
        self.close()
        return False
