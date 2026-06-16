"""Centralized I/O utilities for experiment artifacts.

This module provides a single place for all experiment file operations,
ensuring consistent patterns like atomic writes across the codebase.

All experiment I/O should go through this module:
- Predictions: save_predictions(), load_predictions()
- Results: save_result(), load_result()
- Questions: save_questions(), load_questions()
- State: handled by ExperimentState.save()/load()
"""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragicamp.core.logging import get_logger

if TYPE_CHECKING:
    from ragicamp.experiment import ExperimentResult

logger = get_logger(__name__)


class ExperimentIO:
    """Centralized I/O for experiment artifacts.

    All methods use atomic writes (write to temp, then rename) to prevent
    corruption on crash.

    Example:
        >>> io = ExperimentIO(output_path)
        >>> io.save_predictions(predictions_data)
        >>> data = io.load_predictions()
    """

    def __init__(self, output_path: Path):
        """Initialize with experiment output directory.

        Args:
            output_path: Path to experiment output directory
        """
        self.output_path = Path(output_path)

    @property
    def predictions_path(self) -> Path:
        return self.output_path / "predictions.json"

    @property
    def results_path(self) -> Path:
        return self.output_path / "results.json"

    @property
    def questions_path(self) -> Path:
        return self.output_path / "questions.json"

    @property
    def metadata_path(self) -> Path:
        return self.output_path / "metadata.json"

    @property
    def state_path(self) -> Path:
        return self.output_path / "state.json"

    # =========================================================================
    # Atomic Write Helper
    # =========================================================================

    def _atomic_write(self, data: dict[str, Any], path: Path) -> None:
        """Write JSON atomically (write to temp, then rename)."""
        atomic_write_json(data, path)

    # =========================================================================
    # Predictions
    # =========================================================================

    def save_predictions(self, data: dict[str, Any]) -> None:
        """Save predictions atomically.

        Args:
            data: Predictions data dict with "predictions" list
        """
        self._atomic_write(data, self.predictions_path)
        logger.debug("Saved %d predictions", len(data.get("predictions", [])))

    def load_predictions(self) -> dict[str, Any]:
        """Load predictions from file.

        Returns:
            Predictions data dict

        Raises:
            FileNotFoundError: If predictions file doesn't exist
        """
        with open(self.predictions_path) as f:
            return json.load(f)

    def predictions_exist(self) -> bool:
        """Check if predictions file exists."""
        return self.predictions_path.exists()

    # =========================================================================
    # Results
    # =========================================================================

    def save_result(self, result: "ExperimentResult") -> None:
        """Save experiment result atomically.

        Args:
            result: ExperimentResult to save
        """
        self._atomic_write(result.to_dict(), self.results_path)
        logger.debug("Saved result for %s", result.name)

    def save_result_dict(self, data: dict[str, Any]) -> None:
        """Save result dict atomically.

        Args:
            data: Result data dict
        """
        self._atomic_write(data, self.results_path)

    def load_result(self) -> dict[str, Any]:
        """Load result from file.

        Returns:
            Result data dict
        """
        with open(self.results_path) as f:
            return json.load(f)

    def result_exists(self) -> bool:
        """Check if results file exists."""
        return self.results_path.exists()

    # =========================================================================
    # Questions
    # =========================================================================

    def save_questions(self, questions: list[dict[str, Any]], experiment_name: str) -> None:
        """Save questions list atomically.

        Args:
            questions: List of question dicts with idx, question, expected
            experiment_name: Name of the experiment
        """
        data = {
            "experiment": experiment_name,
            "questions": questions,
            "count": len(questions),
        }
        self._atomic_write(data, self.questions_path)
        logger.debug("Saved %d questions", len(questions))

    def load_questions(self) -> list[dict[str, Any]]:
        """Load questions from file.

        Returns:
            List of question dicts
        """
        with open(self.questions_path) as f:
            data = json.load(f)
        return data["questions"]

    def questions_exist(self) -> bool:
        """Check if questions file exists."""
        return self.questions_path.exists()

    # =========================================================================
    # Metadata
    # =========================================================================

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        """Save experiment metadata atomically.

        Args:
            metadata: Metadata dict
        """
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        self._atomic_write(metadata, self.metadata_path)

    def load_metadata(self) -> dict[str, Any]:
        """Load metadata from file.

        Returns:
            Metadata dict
        """
        with open(self.metadata_path) as f:
            return json.load(f)

    def metadata_exists(self) -> bool:
        """Check if metadata file exists."""
        return self.metadata_path.exists()

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def ensure_dir(self) -> None:
        """Ensure output directory exists."""
        self.output_path.mkdir(parents=True, exist_ok=True)

    def get_completed_indices(self) -> set:
        """Get set of completed prediction indices.

        Returns:
            Set of indices that have predictions
        """
        if not self.predictions_exist():
            return set()

        data = self.load_predictions()
        return {p.get("idx", i) for i, p in enumerate(data.get("predictions", []))}


# =============================================================================
# Module-level convenience functions
# =============================================================================


def atomic_write_json(data: dict[str, Any], path: Path, **kwargs: Any) -> None:
    """Write JSON atomically (write to temp file, then rename).

    This is the canonical atomic write function. Use this everywhere
    instead of bare ``open(path, 'w') + json.dump()``.

    Args:
        data: Dictionary to serialize as JSON
        path: Final destination path
        **kwargs: Extra keyword arguments forwarded to ``json.dump``
            (e.g. ``indent``, ``sort_keys``).  Defaults to ``indent=2``
            when no kwargs are given.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("indent", 2)
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(data, f, **kwargs)
    temp_path.replace(path)


def save_predictions_atomic(data: dict[str, Any], path: Path) -> None:
    """Save predictions atomically (standalone function).

    Args:
        data: Predictions data dict
        path: Path to save to
    """
    atomic_write_json(data, path)


def load_predictions(path: Path) -> dict[str, Any]:
    """Load predictions from file (standalone function).

    Args:
        path: Path to predictions file

    Returns:
        Predictions data dict
    """
    with open(path) as f:
        return json.load(f)
