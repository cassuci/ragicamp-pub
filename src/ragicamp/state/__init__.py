"""State management for experiments.

This package provides state tracking and health checking for experiments:

- ExperimentState: Persistent state for experiments (phase, progress, etc.)
- ExperimentPhase: Lifecycle phases (INIT, GENERATING, GENERATED, COMPUTING_METRICS, COMPLETE)
- ExperimentHealth: Health check result with resume capabilities
- detect_state: Infer state from files on disk
- check_health: Comprehensive health check
"""

from ragicamp.state.experiment_state import (
    PHASE_ORDER,
    ExperimentPhase,
    ExperimentState,
)
from ragicamp.state.health import (
    ExperimentHealth,
    check_health,
    detect_state,
)

__all__ = [
    # State
    "ExperimentPhase",
    "ExperimentState",
    "PHASE_ORDER",
    # Health
    "ExperimentHealth",
    "check_health",
    "detect_state",
]
