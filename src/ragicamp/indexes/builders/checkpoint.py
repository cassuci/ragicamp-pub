"""Checkpoint management for resumable index building.

Provides state persistence for long-running embedding jobs, allowing
resume after crashes or interruptions.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeVar

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingCheckpoint:
    """Checkpoint state for embedding index building.

    Tracks progress through Phase 1 (chunking + embedding) so we can resume
    if the process is interrupted.
    """

    batch_num: int  # Number of batches completed
    total_docs: int  # Total documents processed
    total_chunks: int  # Total chunks generated
    embedding_sizes: list[int] = field(
        default_factory=list
    )  # Size of each batch (for .npy reading)

    # Paths to temp files (relative to work_dir)
    emb_file: str = "embeddings_default.npy"
    chunks_file: str = "chunks_default.pkl"


@dataclass
class HierarchicalCheckpoint:
    """Checkpoint state for hierarchical index building.

    Similar to EmbeddingCheckpoint but tracks both parent and child chunks.
    """

    batch_num: int
    total_docs: int
    total_parent_chunks: int
    total_child_chunks: int
    parent_embedding_sizes: list[int] = field(default_factory=list)
    child_embedding_sizes: list[int] = field(default_factory=list)


T = TypeVar("T", EmbeddingCheckpoint, HierarchicalCheckpoint)


class CheckpointManager:
    """Manages checkpoint state persistence.

    Saves and loads checkpoint state to/from disk, enabling resume
    after crashes or interruptions.

    Example:
        checkpoint_mgr = CheckpointManager(work_dir)

        # Check for existing checkpoint
        checkpoint = checkpoint_mgr.load(EmbeddingCheckpoint)
        start_batch = checkpoint.batch_num if checkpoint else 0

        # Process batches
        for batch_num, batch in enumerate(batches):
            if batch_num < start_batch:
                continue  # Skip already processed

            process(batch)
            checkpoint_mgr.save(EmbeddingCheckpoint(
                batch_num=batch_num + 1,
                total_docs=...,
                total_chunks=...,
                embedding_sizes=...,
            ))

        # Clear on success
        checkpoint_mgr.clear()
    """

    CHECKPOINT_FILE = "checkpoint.json"

    def __init__(self, work_dir: Path | str):
        """Initialize checkpoint manager.

        Args:
            work_dir: Directory for checkpoint file
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = self.work_dir / self.CHECKPOINT_FILE

    def save(self, checkpoint: T) -> None:
        """Save checkpoint state to disk.

        Args:
            checkpoint: Checkpoint dataclass to save
        """
        data = {
            "type": type(checkpoint).__name__,
            **asdict(checkpoint),
        }
        with open(self._checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug("Checkpoint saved: batch=%d", checkpoint.batch_num)

    def load(self, checkpoint_type: type[T]) -> T | None:
        """Load checkpoint state from disk.

        Args:
            checkpoint_type: Type of checkpoint to load (EmbeddingCheckpoint or HierarchicalCheckpoint)

        Returns:
            Checkpoint instance if exists and valid, None otherwise
        """
        if not self._checkpoint_path.exists():
            return None

        try:
            with open(self._checkpoint_path) as f:
                data = json.load(f)

            # Verify type matches
            saved_type = data.pop("type", None)
            if saved_type != checkpoint_type.__name__:
                logger.warning(
                    "Checkpoint type mismatch: expected %s, got %s",
                    checkpoint_type.__name__,
                    saved_type,
                )
                return None

            checkpoint = checkpoint_type(**data)
            logger.info(
                "Resumed from checkpoint: batch=%d, docs=%d",
                checkpoint.batch_num,
                checkpoint.total_docs,
            )
            return checkpoint

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning("Failed to load checkpoint: %s", e)
            return None

    def exists(self) -> bool:
        """Check if a checkpoint exists."""
        return self._checkpoint_path.exists()

    def clear(self) -> None:
        """Remove checkpoint file."""
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()
            logger.debug("Checkpoint cleared")

    def get_checkpoint_path(self) -> Path:
        """Get the path to the checkpoint file."""
        return self._checkpoint_path
