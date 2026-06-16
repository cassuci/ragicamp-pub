"""Phase handlers for experiment execution.

Each phase handler implements a single phase of the experiment lifecycle:
- InitHandler: Export questions and metadata
- GenerationHandler: Generate predictions with checkpointing
- MetricsHandler: Compute evaluation metrics

Usage:
    from ragicamp.execution.phases import (
        PhaseHandler,
        ExecutionContext,
        InitHandler,
        GenerationHandler,
        MetricsHandler,
    )

    handlers = [InitHandler(), GenerationHandler(), MetricsHandler()]
    for phase in phases_to_run:
        handler = next(h for h in handlers if h.can_handle(phase))
        state = handler.execute(spec, state, context)
"""

from ragicamp.execution.phases.base import ExecutionContext, PhaseHandler
from ragicamp.execution.phases.generation import GenerationHandler
from ragicamp.execution.phases.init_phase import InitHandler
from ragicamp.execution.phases.metrics_phase import MetricsHandler

__all__ = [
    "PhaseHandler",
    "ExecutionContext",
    "InitHandler",
    "GenerationHandler",
    "MetricsHandler",
]
