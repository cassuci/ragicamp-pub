"""Experiment state management for phased execution.

This module provides state tracking for experiments, enabling:
- Resume from any phase after crash/interruption
- Partial re-execution (e.g., recompute only metrics)

Phases:
    INIT -> GENERATING -> GENERATED -> COMPUTING_METRICS -> COMPLETE
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class ExperimentPhase(Enum):
    """Experiment lifecycle phases."""

    INIT = "init"  # Config saved, questions exported
    GENERATING = "generating"  # Predictions in progress
    GENERATED = "generated"  # All predictions complete
    COMPUTING_METRICS = "computing_metrics"  # Metrics being computed
    COMPLETE = "complete"  # All done
    FAILED = "failed"  # Error occurred


# Phase ordering for comparison
PHASE_ORDER = {
    ExperimentPhase.INIT: 0,
    ExperimentPhase.GENERATING: 1,
    ExperimentPhase.GENERATED: 2,
    ExperimentPhase.COMPUTING_METRICS: 3,
    ExperimentPhase.COMPLETE: 4,
    ExperimentPhase.FAILED: -1,
}


@dataclass
class ExperimentState:
    """Persistent state for an experiment.

    Saved to state.json in the experiment output directory.
    """

    phase: ExperimentPhase
    started_at: str
    updated_at: str
    total_questions: int = 0
    predictions_complete: int = 0
    metrics_computed: list[str] = field(default_factory=list)
    metrics_requested: list[str] = field(default_factory=list)
    error: str | None = None

    def save(self, path: Path) -> None:
        """Save state to JSON file atomically."""
        self.updated_at = datetime.now().isoformat()
        data = {
            "phase": self.phase.value,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "total_questions": self.total_questions,
            "predictions_complete": self.predictions_complete,
            "metrics_computed": self.metrics_computed,
            "metrics_requested": self.metrics_requested,
            "error": self.error,
        }
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ExperimentState":
        """Load state from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            phase=ExperimentPhase(data["phase"]),
            started_at=data["started_at"],
            updated_at=data["updated_at"],
            total_questions=data.get("total_questions", 0),
            predictions_complete=data.get("predictions_complete", 0),
            metrics_computed=data.get("metrics_computed", []),
            metrics_requested=data.get("metrics_requested", []),
            error=data.get("error"),
        )

    @classmethod
    def new(cls, total_questions: int = 0, metrics: list[str] | None = None) -> "ExperimentState":
        """Create a new experiment state."""
        now = datetime.now().isoformat()
        return cls(
            phase=ExperimentPhase.INIT,
            started_at=now,
            updated_at=now,
            total_questions=total_questions,
            metrics_requested=metrics or [],
        )

    def advance_to(self, phase: ExperimentPhase) -> None:
        """Advance to a new phase."""
        self.phase = phase
        self.updated_at = datetime.now().isoformat()

    def set_error(self, error: str) -> None:
        """Mark experiment as failed with error."""
        self.error = error
        self.phase = ExperimentPhase.FAILED
        self.updated_at = datetime.now().isoformat()

    def is_at_least(self, phase: ExperimentPhase) -> bool:
        """Check if we're at or past the given phase (phase is at least started)."""
        return PHASE_ORDER.get(self.phase, -1) >= PHASE_ORDER.get(phase, 0)

    def is_past(self, phase: ExperimentPhase) -> bool:
        """Check if we've completed the given phase and moved past it."""
        return PHASE_ORDER.get(self.phase, -1) > PHASE_ORDER.get(phase, 0)
