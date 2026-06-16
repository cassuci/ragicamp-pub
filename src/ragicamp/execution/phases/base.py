"""Base classes for phase handlers.

Phase handlers implement the Strategy pattern, allowing each phase of experiment
execution to be encapsulated in a separate class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ragicamp.agents.base import Agent
    from ragicamp.datasets.base import QADataset
    from ragicamp.metrics.base import Metric
    from ragicamp.spec import ExperimentSpec
    from ragicamp.state import ExperimentPhase, ExperimentState


@dataclass
class ExecutionContext:
    """Runtime context shared across phases.

    Holds references to components needed during execution,
    as well as runtime configuration.
    """

    output_path: Path
    agent: Optional["Agent"] = None
    dataset: Optional["QADataset"] = None
    metrics: list["Metric"] | None = None
    callbacks: Any | None = None  # ExperimentCallbacks

    # Runtime configuration
    batch_size: int = 8
    checkpoint_every: int = 50
    kwargs: dict[str, Any] = field(default_factory=dict)


class PhaseHandler(ABC):
    """Abstract handler for a single experiment phase.

    Each phase handler is responsible for:
    1. Determining if it can handle a given phase
    2. Executing that phase and returning updated state

    This follows the Strategy pattern, allowing phases to be
    added, removed, or modified independently.
    """

    @abstractmethod
    def can_handle(self, phase: "ExperimentPhase") -> bool:
        """Check if this handler can process the given phase.

        Args:
            phase: The experiment phase to check

        Returns:
            True if this handler can process the phase
        """
        ...

    @abstractmethod
    def execute(
        self,
        spec: "ExperimentSpec",
        state: "ExperimentState",
        context: ExecutionContext,
    ) -> "ExperimentState":
        """Execute the phase, returning updated state.

        Args:
            spec: Experiment specification (immutable config)
            state: Current experiment state (mutable)
            context: Execution context with components and config

        Returns:
            Updated ExperimentState after executing this phase
        """
        ...
