"""Execution module for resilient agent execution and experiment running.

This module provides:
- Resilient execution with automatic batch size reduction
- GPU memory management
- Experiment specification and execution
- Phase handlers for modular experiment phases

Example:
    from ragicamp.execution import ResilientExecutor, ExperimentSpec, build_specs

    executor = ResilientExecutor(agent, batch_size=32)
    results = executor.execute(queries)
"""

from ragicamp.execution.executor import (
    ResilientExecutor,
)
from ragicamp.execution.phases import (
    ExecutionContext,
    GenerationHandler,
    InitHandler,
    MetricsHandler,
    PhaseHandler,
)
from ragicamp.execution.runner import (
    run_generation,
    run_metrics_only,
    run_spec,
    run_spec_subprocess,
)

# Re-export spec types for convenience (canonical location is ragicamp.spec)
from ragicamp.spec import ExperimentSpec, build_specs, name_direct, name_rag

__all__ = [
    # Executor
    "ResilientExecutor",
    # Phase handlers
    "PhaseHandler",
    "ExecutionContext",
    "InitHandler",
    "GenerationHandler",
    "MetricsHandler",
    # Spec (canonical location: ragicamp.spec)
    "ExperimentSpec",
    "build_specs",
    "name_direct",
    "name_rag",
    # Runner
    "run_spec",
    "run_spec_subprocess",
    "run_generation",
    "run_metrics_only",
]
