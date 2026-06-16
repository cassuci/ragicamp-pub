"""Unified Experiment abstraction for RAGiCamp.

Clean Architecture:
    from ragicamp import Experiment

    exp = Experiment.from_spec(spec, output_dir="outputs/")
    results = exp.run()

Or programmatically:

    exp = Experiment(
        name="my_experiment",
        agent=agent,
        dataset=dataset,
        metrics=metrics,
    )
    results = exp.run()

Phases:
    INIT -> GENERATING -> GENERATED -> COMPUTING_METRICS -> COMPLETE
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ragicamp.agents.base import Agent
from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.datasets.base import QADataset
from ragicamp.execution.phases import (
    ExecutionContext,
    GenerationHandler,
    InitHandler,
    MetricsHandler,
    PhaseHandler,
)
from ragicamp.metrics.base import Metric
from ragicamp.state import (
    ExperimentHealth,
    ExperimentPhase,
    ExperimentState,
    check_health,
    detect_state,
)
from ragicamp.utils.experiment_io import atomic_write_json
from ragicamp.utils.formatting import format_metrics_summary
from ragicamp.utils.resource_manager import ResourceManager

if TYPE_CHECKING:
    from ragicamp.spec import ExperimentSpec

logger = get_logger(__name__)


@dataclass(frozen=True)
class _MinimalSpec:
    """Minimal spec adapter for use with phase handlers."""

    name: str


class _MetricsIncompleteError(Exception):
    """Raised when some metrics fail to compute."""

    def __init__(self, missing_metrics: list[str]):
        self.missing_metrics = missing_metrics
        super().__init__(f"Metrics incomplete: {missing_metrics}")


@dataclass
class ExperimentCallbacks:
    """Callbacks for monitoring experiment progress."""

    on_phase_start: Callable[[ExperimentPhase], None] | None = None
    on_phase_end: Callable[[ExperimentPhase], None] | None = None
    on_batch_start: Callable[[int, int], None] | None = None
    on_batch_end: Callable[[int, int, list[str]], None] | None = None
    on_checkpoint: Callable[[int, Path], None] | None = None
    on_complete: Callable[["ExperimentResult"], None] | None = None


@dataclass
class ExperimentResult:
    """Result of running an experiment."""

    name: str
    metrics: dict[str, float]
    num_examples: int
    duration_seconds: float
    output_path: Path | None = None
    predictions_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def f1(self) -> float:
        return self.metrics.get("f1", 0.0)

    @property
    def exact_match(self) -> float:
        return self.metrics.get("exact_match", 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metrics": self.metrics,
            "num_examples": self.num_examples,
            "duration_seconds": self.duration_seconds,
            "output_path": str(self.output_path) if self.output_path else None,
            "predictions_path": str(self.predictions_path) if self.predictions_path else None,
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        """Save result to JSON file (atomic write)."""
        atomic_write_json(self.to_dict(), path)

    def __repr__(self) -> str:
        return f"ExperimentResult(name={self.name!r}, f1={self.f1:.3f}, em={self.exact_match:.3f})"


@dataclass
class Experiment:
    """Unified experiment abstraction with phased execution.

    Combines agent, dataset, and metrics into a single runnable unit.
    Handles checkpointing, resource management, and result saving.

    Phases:
        1. INIT: Save metadata, export questions
        2. GENERATING: Generate predictions with checkpointing
        3. GENERATED: All predictions complete
        4. COMPUTING_METRICS: Compute all metrics
        5. COMPLETE: Save final results
    """

    name: str
    agent: Agent
    dataset: QADataset
    metrics: list[Metric] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    # Optional references
    _spec: Optional["ExperimentSpec"] = field(default=None, repr=False)

    # Runtime state (not serialized)
    _state: ExperimentState | None = field(default=None, repr=False)
    _callbacks: ExperimentCallbacks | None = field(default=None, repr=False)
    _batch_size: int = field(default=8, repr=False)
    _checkpoint_every: int = field(default=50, repr=False)
    _start_time: float = field(default=0.0, repr=False)
    _phase_timings: dict[str, float] = field(default_factory=dict, repr=False)

    # Phase handlers (lazy initialized)
    _handlers: dict[ExperimentPhase, PhaseHandler] | None = field(default=None, repr=False)

    # =========================================================================
    # Public API
    # =========================================================================

    def check_health(self) -> ExperimentHealth:
        """Check experiment state and detect issues."""
        metric_names = [m.name for m in self.metrics]
        return check_health(self.output_path, metric_names)

    @property
    def output_path(self) -> Path:
        """Path to experiment output directory."""
        return self.output_dir / self.name

    @property
    def state_path(self) -> Path:
        return self.output_path / "state.json"

    @property
    def questions_path(self) -> Path:
        return self.output_path / "questions.json"

    @property
    def predictions_path(self) -> Path:
        return self.output_path / "predictions.json"

    @property
    def results_path(self) -> Path:
        return self.output_path / "results.json"

    @property
    def metadata_path(self) -> Path:
        return self.output_path / "metadata.json"

    def run(
        self,
        batch_size: int = 8,
        checkpoint_every: int = 50,
        resume: bool = True,
        callbacks: ExperimentCallbacks | None = None,
        phases: list[str] | None = None,
        **kwargs: Any,
    ) -> ExperimentResult:
        """Run the experiment with phase-aware execution.

        Args:
            batch_size: Number of examples to process in parallel
            checkpoint_every: Save checkpoint every N examples
            resume: Resume from checkpoint if available
            callbacks: Optional callbacks for progress monitoring
            phases: Optional list of phases to run (default: all remaining)
            **kwargs: Additional arguments passed to agent.run()

        Returns:
            ExperimentResult with metrics and metadata
        """
        self._callbacks = callbacks or ExperimentCallbacks()
        self._batch_size = batch_size
        self._checkpoint_every = checkpoint_every
        self._start_time = time.time()
        self._kwargs = kwargs

        self.output_path.mkdir(parents=True, exist_ok=True)

        # Check current state
        health = self.check_health()

        if health.is_complete and resume:
            logger.info("Experiment %s already complete. Loading results...", self.name)
            return self._load_result()

        # Determine which phases to run
        if phases:
            target_phases = [ExperimentPhase(p) for p in phases]
        elif resume and health.can_resume:
            logger.info("Resuming %s from phase: %s", self.name, health.resume_phase.value)
            target_phases = self._phases_from(health.resume_phase)
        else:
            target_phases = list(ExperimentPhase)[:5]  # All except FAILED

        # Load or create state
        metric_names = [m.name for m in self.metrics]
        if self.state_path.exists() and resume:
            self._state = detect_state(self.output_path, metric_names)
        else:
            self._state = ExperimentState.new(metrics=metric_names)

        logger.info("Running: %s", self.name)

        try:
            for phase in target_phases:
                if phase == ExperimentPhase.FAILED:
                    continue
                if self._state.is_past(phase) and phase != ExperimentPhase.COMPUTING_METRICS:
                    continue

                self._run_phase(phase)

            return self._load_result()

        except _MetricsIncompleteError as e:
            logger.warning("Experiment incomplete: %s", e)
            self._save_partial_result()
            raise

        except Exception as e:
            logger.error("Experiment failed: %s", e)
            self._state.set_error(str(e))
            self._state.save(self.state_path)
            raise

    def run_phase(self, phase: str | ExperimentPhase) -> None:
        """Run a specific phase only."""
        if isinstance(phase, str):
            phase = ExperimentPhase(phase)

        self._callbacks = ExperimentCallbacks()
        self._start_time = time.time()
        self.output_path.mkdir(parents=True, exist_ok=True)

        if self.state_path.exists():
            self._state = ExperimentState.load(self.state_path)
        else:
            metric_names = [m.name for m in self.metrics]
            self._state = ExperimentState.new(metrics=metric_names)

        self._run_phase(phase)

    # =========================================================================
    # Phase Implementations
    # =========================================================================

    def _get_handlers(self) -> dict[ExperimentPhase, PhaseHandler]:
        """Get or create phase handlers."""
        if self._handlers is None:
            self._handlers = {
                ExperimentPhase.INIT: InitHandler(),
                ExperimentPhase.GENERATING: GenerationHandler(),
                ExperimentPhase.COMPUTING_METRICS: MetricsHandler(),
            }
        return self._handlers

    def _create_context(self) -> ExecutionContext:
        """Create execution context for phase handlers."""
        return ExecutionContext(
            output_path=self.output_path,
            agent=self.agent,
            dataset=self.dataset,
            metrics=self.metrics,
            callbacks=self._callbacks,
            batch_size=self._batch_size,
            checkpoint_every=self._checkpoint_every,
            kwargs=getattr(self, "_kwargs", {}),
        )

    def _run_phase(self, phase: ExperimentPhase) -> None:
        """Execute a single phase."""
        _phase_t0 = time.perf_counter()
        logger.info("--- Phase [%s] starting ---", phase.value)

        if self._callbacks.on_phase_start:
            self._callbacks.on_phase_start(phase)

        self._state.advance_to(phase)
        self._state.save(self.state_path)

        handlers = self._get_handlers()
        if phase in handlers:
            spec = self._spec if self._spec is not None else _MinimalSpec(name=self.name)
            context = self._create_context()
            self._state = handlers[phase].execute(spec, self._state, context)

            if phase == ExperimentPhase.COMPUTING_METRICS:
                missing = set(self._state.metrics_requested) - set(self._state.metrics_computed)
                if missing:
                    logger.warning(
                        "Some metrics failed to compute: %s",
                        list(missing),
                    )
                    raise _MetricsIncompleteError(list(missing))
        elif phase == ExperimentPhase.GENERATED:
            self._phase_generated()
        elif phase == ExperimentPhase.COMPLETE:
            self._phase_complete()

        self._state.save(self.state_path)

        _phase_s = time.perf_counter() - _phase_t0
        self._phase_timings[phase.value] = round(_phase_s, 2)
        logger.info("--- Phase [%s] completed in %.1fs ---", phase.value, _phase_s)

        if self._callbacks.on_phase_end:
            self._callbacks.on_phase_end(phase)

    def _phase_generated(self) -> None:
        """Phase 3: Mark generation as complete."""
        logger.info("Phase: GENERATED - cleaning up resources")
        ResourceManager.clear_gpu_memory()

    def _phase_complete(self) -> None:
        """Phase 5: Create final summary and results."""
        logger.info("Phase: COMPLETE - saving results")

        with open(self.predictions_path) as f:
            data = json.load(f)

        metrics = data.get("aggregate_metrics", {})
        duration = time.time() - self._start_time

        # Build metadata from spec (authoritative) + runtime info
        result_metadata = {
            "agent": self.agent.name,
            "dataset": self.dataset.name,
            "batch_size": self._batch_size,
            "timestamp": datetime.now().isoformat(),
        }
        if self._spec is not None:
            spec_data = self._spec.to_dict()
            # Spec fields as base, runtime fields override
            result_metadata = {**spec_data, **result_metadata}

        # Collect timing profile
        timing_profile = {"phases": dict(self._phase_timings)}
        metric_timings = data.get("metric_timings")
        if metric_timings:
            timing_profile["metrics"] = metric_timings
        result_metadata["timing"] = timing_profile

        result = ExperimentResult(
            name=self.name,
            metrics=metrics,
            num_examples=len(data.get("predictions", [])),
            duration_seconds=duration,
            output_path=self.output_path,
            predictions_path=self.predictions_path,
            metadata=result_metadata,
        )

        atomic_write_json(result.to_dict(), self.results_path)

        metrics_str = format_metrics_summary(result.metrics)
        logger.info("Done! %s (%.1fs)", metrics_str, duration)

        if self._callbacks.on_complete:
            self._callbacks.on_complete(result)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _save_predictions(self, data: dict[str, Any]) -> None:
        """Save predictions atomically."""
        atomic_write_json(data, self.predictions_path)

        if self._callbacks.on_checkpoint:
            n = len(data.get("predictions", []))
            self._callbacks.on_checkpoint(n, self.predictions_path)

    def _load_result(self) -> ExperimentResult:
        """Load result from results.json."""
        with open(self.results_path) as f:
            data = json.load(f)
        return ExperimentResult(
            name=self.name,
            metrics=data.get("metrics", {}),
            num_examples=data.get("num_examples", 0),
            duration_seconds=data.get("duration_seconds", 0),
            output_path=self.output_path,
            predictions_path=self.predictions_path if self.predictions_path.exists() else None,
            metadata=data.get("metadata", {}),
        )

    def _save_partial_result(self) -> None:
        """Save partial results when metrics are incomplete."""
        if not self.predictions_path.exists():
            return

        with open(self.predictions_path) as f:
            data = json.load(f)

        metrics = data.get("aggregate_metrics", {})
        duration = time.time() - self._start_time

        # Build metadata from spec when available
        result_metadata = {}
        if self._spec is not None:
            result_metadata = self._spec.to_dict()
        result_metadata.update(
            {
                "agent": self.agent.name,
                "dataset": self.dataset.name,
                "batch_size": self._batch_size,
            }
        )

        result_data = {
            "name": self.name,
            "metrics": metrics,
            "num_examples": len(data.get("predictions", [])),
            "duration_seconds": duration,
            "completed_at": datetime.now().isoformat(),
            "partial": True,
            "metrics_computed": self._state.metrics_computed,
            "metrics_missing": list(
                set(self._state.metrics_requested) - set(self._state.metrics_computed)
            ),
            "metadata": result_metadata,
        }

        atomic_write_json(result_data, self.results_path)

    def _phases_from(self, phase: ExperimentPhase) -> list[ExperimentPhase]:
        """Get list of phases starting from the given phase."""
        all_phases = [
            ExperimentPhase.INIT,
            ExperimentPhase.GENERATING,
            ExperimentPhase.GENERATED,
            ExperimentPhase.COMPUTING_METRICS,
            ExperimentPhase.COMPLETE,
        ]
        try:
            idx = all_phases.index(phase)
            return all_phases[idx:]
        except ValueError:
            return all_phases

    @staticmethod
    def _build_index_loader(
        retriever_type: str,
        index_name: str,
        index_path: Path,
        retriever_config: dict | None = None,
        sparse_index_override: str | None = None,
        rrf_k: int | None = None,
        alpha_override: float | None = None,
    ) -> "Callable[[], Any]":
        """Return a zero-arg callable that loads the search backend from disk.

        Used by :meth:`from_spec` to defer expensive index loading via
        :class:`~ragicamp.retrievers.lazy.LazySearchBackend`.

        Args:
            sparse_index_override: If provided, use this sparse index name
                instead of the one in retriever_config.
            rrf_k: RRF fusion constant for hybrid retrievers.
            alpha_override: If provided, override the retriever config's alpha
                for hybrid dense/sparse blending.
        """

        def _load():
            _t0 = time.perf_counter()
            logger.info(
                "Loading index: %s (type=%s, path=%s)", index_name, retriever_type, index_path
            )

            if retriever_type == "hierarchical":
                from ragicamp.indexes.hierarchical import HierarchicalIndex
                from ragicamp.retrievers.hierarchical import HierarchicalSearcher

                backend = HierarchicalSearcher(
                    HierarchicalIndex.load(name=index_name, path=index_path),
                )
            elif retriever_type == "hybrid":
                from ragicamp.indexes.sparse import SparseIndex
                from ragicamp.indexes.vector_index import VectorIndex
                from ragicamp.retrievers.hybrid import HybridSearcher
                from ragicamp.utils.artifacts import get_artifact_manager

                vector_index = VectorIndex.load(index_path)

                cfg = retriever_config or {}
                # Prefer explicit override (from spec) over on-disk config
                sparse_name = sparse_index_override or cfg.get(
                    "sparse_index", f"{index_name}_sparse_bm25"
                )
                alpha = cfg.get("alpha", 0.5)
                # Override with spec-level alpha if provided
                if alpha_override is not None:
                    alpha = alpha_override

                mgr = get_artifact_manager()
                sparse_path = mgr.get_sparse_index_path(sparse_name)

                logger.info(
                    "Loading sparse index: %s (alpha=%.2f, path=%s)",
                    sparse_name,
                    alpha,
                    sparse_path,
                )
                sparse_index = SparseIndex.load(
                    name=sparse_name,
                    path=sparse_path,
                    documents=vector_index.documents,
                )

                hybrid_kwargs: dict[str, Any] = {
                    "vector_index": vector_index,
                    "sparse_index": sparse_index,
                    "alpha": alpha,
                }
                if rrf_k is not None:
                    hybrid_kwargs["rrf_k"] = rrf_k
                backend = HybridSearcher(**hybrid_kwargs)
            else:
                from ragicamp.indexes.vector_index import VectorIndex

                backend = VectorIndex.load(index_path)

            logger.info("Index loaded in %.1fs: %s", time.perf_counter() - _t0, index_name)
            return backend

        return _load

    @classmethod
    def from_spec(
        cls,
        spec: "ExperimentSpec",
        output_dir: str | Path,
        limit: int | None = None,
        judge_model: Any = None,
    ) -> "Experiment":
        """Create a fully-configured Experiment from an ExperimentSpec.

        Uses the clean architecture with providers and indexes:
        1. Creates EmbedderProvider and GeneratorProvider
        2. Loads VectorIndex for the retriever
        3. Creates agent with providers + index
        4. Agent manages its own GPU lifecycle

        Args:
            spec: Experiment specification with all configuration
            output_dir: Base output directory
            limit: Optional limit on dataset examples
            judge_model: Optional LLM judge model for metrics

        Returns:
            Fully configured Experiment ready to run
        """
        from ragicamp.factory import AgentFactory, DatasetFactory, MetricFactory, ProviderFactory
        from ragicamp.utils.artifacts import get_artifact_manager

        _setup_t0 = time.perf_counter()
        logger.info("Setting up experiment from spec: %s", spec.name)

        # Create providers (lazy loading)
        generator_provider = ProviderFactory.create_generator(spec.model)

        # Create embedder provider if RAG experiment
        embedder_provider = None
        index = None

        if spec.exp_type == "rag" and spec.retriever:
            # Load retriever config from artifacts/retrievers/{name}/config.json
            manager = get_artifact_manager()
            retriever_path = manager.get_retriever_path(spec.retriever)
            config_path = retriever_path / "config.json"

            if config_path.exists():
                with open(config_path) as f:
                    retriever_config = json.load(f)

                embedding_model = retriever_config.get("embedding_model", Defaults.EMBEDDING_MODEL)
                embedding_backend = retriever_config.get(
                    "embedding_backend", Defaults.EMBEDDING_BACKEND
                )

                embedder_provider = ProviderFactory.create_embedder(
                    embedding_model,
                    backend=embedding_backend,
                )

                # Lazy-load index — deferred until first batch_search call.
                # When retrieval cache gives 100% hits the index is never
                # touched, saving ~3 min of pickle/FAISS deserialization.
                retriever_type = retriever_config.get("type", "dense")
                # Prefer spec values (set at spec-build time from YAML) over
                # on-disk config to avoid silent divergence.
                index_name = spec.embedding_index or retriever_config.get(
                    "embedding_index", spec.retriever
                )
                index_path = manager.get_embedding_index_path(index_name)

                from ragicamp.retrievers.lazy import LazySearchBackend

                index = LazySearchBackend(
                    loader=cls._build_index_loader(
                        retriever_type,
                        index_name,
                        index_path,
                        retriever_config=retriever_config,
                        sparse_index_override=spec.sparse_index,
                        rrf_k=spec.rrf_k,
                        alpha_override=spec.alpha,
                    ),
                    is_hybrid=(retriever_type == "hybrid"),
                )
                logger.info(
                    "Index deferred (lazy): %s (type=%s)",
                    index_name,
                    retriever_type,
                )

        # Create dataset
        dataset_config = DatasetFactory.parse_spec(spec.dataset, limit=limit)
        dataset = DatasetFactory.create(dataset_config)

        # Create reranker provider if configured
        reranker_provider = None
        if spec.exp_type == "rag" and spec.reranker and spec.reranker != "none":
            # Use reranker_model (full HF path) if provided, else fall back to short name
            reranker_spec = spec.reranker_model or spec.reranker
            reranker_provider = ProviderFactory.create_reranker(reranker_spec)
            logger.info(
                "Reranker enabled: %s (fetch_k=%s, top_k=%d)",
                reranker_spec,
                spec.fetch_k,
                spec.top_k,
            )

        # Create agent
        if spec.exp_type == "rag":
            if embedder_provider is None or index is None:
                raise ValueError(f"Could not load retriever config for: {spec.retriever}")

            agent = AgentFactory.from_spec(
                spec=spec,
                embedder_provider=embedder_provider,
                generator_provider=generator_provider,
                index=index,
                reranker_provider=reranker_provider,
            )
        else:
            agent = AgentFactory.from_spec(
                spec=spec,
                embedder_provider=embedder_provider,  # May be None for direct
                generator_provider=generator_provider,
                index=None,
            )

        # Create metrics
        metrics = MetricFactory.create(spec.metrics, judge_model=judge_model)

        _setup_s = time.perf_counter() - _setup_t0
        logger.info("Experiment setup completed in %.1fs: %s", _setup_s, spec.name)

        return cls(
            name=spec.name,
            agent=agent,
            dataset=dataset,
            metrics=metrics,
            output_dir=Path(output_dir),
            _spec=spec,
            _batch_size=spec.batch_size,
        )

    @classmethod
    def from_components(
        cls,
        name: str,
        agent: Agent,
        dataset: QADataset,
        metrics: list[Metric],
        output_dir: str | Path = "outputs",
    ) -> "Experiment":
        """Create experiment from pre-built components."""
        return cls(
            name=name,
            agent=agent,
            dataset=dataset,
            metrics=metrics,
            output_dir=Path(output_dir),
        )


def run_experiments(
    experiments: list[Experiment],
    skip_existing: bool = True,
    **kwargs: Any,
) -> list[ExperimentResult]:
    """Run multiple experiments with proper resource management.

    Args:
        experiments: List of experiments to run
        skip_existing: Skip experiments with existing results
        **kwargs: Arguments passed to each experiment's run()

    Returns:
        List of ExperimentResults
    """
    results = []

    for i, exp in enumerate(experiments, 1):
        logger.info("[%d/%d] %s", i, len(experiments), exp.name)

        health = exp.check_health()

        if skip_existing and health.is_complete:
            logger.info("Skipping %s (complete)", exp.name)
            results.append(exp._load_result())
            continue

        try:
            result = exp.run(**kwargs)
            results.append(result)
        except Exception as e:
            logger.error("Experiment %s failed: %s", exp.name, e)
            results.append(
                ExperimentResult(
                    name=exp.name,
                    metrics={},
                    num_examples=0,
                    duration_seconds=0,
                    metadata={"error": str(e)},
                )
            )

        ResourceManager.clear_gpu_memory()

    return results
