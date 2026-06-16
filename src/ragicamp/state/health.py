"""Health checking for experiments.

This module provides health detection and diagnosis:
- detect_state: Infer state from files on disk
- check_health: Comprehensive health check with actionable results
- ExperimentHealth: Detailed status with resume capabilities
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ragicamp.core.logging import get_logger
from ragicamp.state.experiment_state import ExperimentPhase, ExperimentState

logger = get_logger(__name__)


@dataclass
class ExperimentHealth:
    """Health check result for an experiment."""

    phase: ExperimentPhase
    is_complete: bool
    is_healthy: bool  # No errors, consistent state
    total_questions: int
    predictions_complete: int
    missing_predictions: list[int]  # Question indices missing predictions
    metrics_computed: list[str]
    metrics_missing: list[str]
    error: str | None = None

    @property
    def can_resume(self) -> bool:
        """Check if experiment can be resumed (has partial work to continue).

        Returns False for fresh experiments (INIT with no predictions) - those
        should be 'started', not 'resumed'.
        """
        if self.is_complete or self.phase == ExperimentPhase.FAILED:
            return False
        # Fresh experiment - nothing to resume, start fresh
        if self.phase == ExperimentPhase.INIT and self.predictions_complete == 0:
            return False
        return True

    @property
    def resume_phase(self) -> ExperimentPhase | None:
        """Determine which phase to resume from."""
        if self.is_complete:
            return None
        if self.phase == ExperimentPhase.FAILED:
            return None

        # If we haven't started yet (INIT phase with no predictions), start from beginning
        if self.phase == ExperimentPhase.INIT:
            return ExperimentPhase.INIT

        # If there are missing predictions, we need to generate
        if self.missing_predictions:
            return ExperimentPhase.GENERATING

        # If no predictions were made at all, we need to generate (not compute metrics)
        if self.predictions_complete == 0:
            return ExperimentPhase.GENERATING

        # If we have predictions but missing metrics, compute them
        if self.metrics_missing:
            return ExperimentPhase.COMPUTING_METRICS

        return self.phase

    @property
    def needs_generation(self) -> bool:
        """Check if predictions are incomplete."""
        return len(self.missing_predictions) > 0

    @property
    def needs_metrics(self) -> bool:
        """Check if metrics need to be computed."""
        return len(self.metrics_missing) > 0

    def summary(self) -> str:
        """Human-readable summary."""
        if self.is_complete:
            return f"✓ Complete ({self.predictions_complete} predictions, {len(self.metrics_computed)} metrics)"
        if self.phase == ExperimentPhase.FAILED:
            return f"✗ Failed: {self.error}"
        parts = []
        if self.needs_generation:
            parts.append(f"predictions: {self.predictions_complete}/{self.total_questions}")
        if self.needs_metrics:
            parts.append(f"missing metrics: {', '.join(self.metrics_missing)}")
        return f"○ {self.phase.value} - {'; '.join(parts)}"


def _validate_state_artifacts(state: ExperimentState, exp_dir: Path) -> ExperimentState:
    """Validate that required artifacts exist for the claimed phase.

    If artifacts are missing, adjust the phase to match actual state on disk.
    This handles cases where state.json claims a phase but files are missing.
    """
    predictions_path = exp_dir / "predictions.json"
    questions_path = exp_dir / "questions.json"
    results_path = exp_dir / "results.json"

    original_phase = state.phase

    # COMPLETE phase requires results.json
    if state.phase == ExperimentPhase.COMPLETE:
        if not results_path.exists():
            # Downgrade to COMPUTING_METRICS if predictions exist
            if predictions_path.exists():
                state.phase = ExperimentPhase.COMPUTING_METRICS
            elif questions_path.exists():
                state.phase = ExperimentPhase.GENERATING
            else:
                state.phase = ExperimentPhase.INIT

    # COMPUTING_METRICS phase requires predictions.json
    if state.phase == ExperimentPhase.COMPUTING_METRICS:
        if not predictions_path.exists():
            # Downgrade to GENERATING if questions exist
            if questions_path.exists():
                state.phase = ExperimentPhase.GENERATING
                state.predictions_complete = 0
            else:
                state.phase = ExperimentPhase.INIT
                state.predictions_complete = 0

    # GENERATED phase requires predictions.json
    if state.phase == ExperimentPhase.GENERATED:
        if not predictions_path.exists():
            if questions_path.exists():
                state.phase = ExperimentPhase.GENERATING
                state.predictions_complete = 0
            else:
                state.phase = ExperimentPhase.INIT
                state.predictions_complete = 0

    # GENERATING phase requires questions.json
    if state.phase == ExperimentPhase.GENERATING:
        if not questions_path.exists():
            state.phase = ExperimentPhase.INIT
            state.predictions_complete = 0

    # Validate that metrics_computed actually exist in predictions file
    # This catches cases where state.json claims metrics but predictions file doesn't have them
    if predictions_path.exists() and state.metrics_computed:
        try:
            with open(predictions_path) as f:
                data = json.load(f)

            # Collect all metric keys from aggregate_metrics
            actual_keys = set(data.get("aggregate_metrics", {}).keys())

            # Also check per-item metrics (first prediction that has metrics)
            predictions = data.get("predictions", [])
            for pred in predictions:
                if "metrics" in pred and pred["metrics"]:
                    actual_keys.update(pred["metrics"].keys())
                    break

            # Map metric names to their expected keys
            # Some metrics produce multiple keys (e.g., bertscore -> bertscore_f1, bertscore_precision, etc.)
            def metric_exists(metric_name: str) -> bool:
                if metric_name in actual_keys:
                    return True
                # Check for prefixed keys (e.g., bertscore -> bertscore_f1)
                prefix = f"{metric_name}_"
                return any(k.startswith(prefix) for k in actual_keys)

            # Only keep metrics that actually exist in the predictions file
            validated_metrics = [m for m in state.metrics_computed if metric_exists(m)]
            if len(validated_metrics) != len(state.metrics_computed):
                removed = set(state.metrics_computed) - set(validated_metrics)
                logger.warning(
                    "Metrics mismatch: state.json claims %s but predictions file missing: %s",
                    state.metrics_computed,
                    list(removed),
                )
                state.metrics_computed = validated_metrics
        except Exception:
            # If we can't read predictions, assume no metrics computed
            state.metrics_computed = []

    # Log if we adjusted the phase
    if state.phase != original_phase:
        logger.warning(
            "State mismatch detected: state.json claims %s but artifacts missing. Adjusted to %s",
            original_phase.value,
            state.phase.value,
        )

    return state


def detect_state(exp_dir: Path, requested_metrics: list[str] | None = None) -> ExperimentState:
    """Detect experiment state from files on disk.

    Args:
        exp_dir: Experiment output directory
        requested_metrics: Metrics that should be computed

    Returns:
        ExperimentState inferred from files
    """
    requested_metrics = requested_metrics or []
    state_path = exp_dir / "state.json"
    results_path = exp_dir / "results.json"
    predictions_path = exp_dir / "predictions.json"
    questions_path = exp_dir / "questions.json"
    metadata_path = exp_dir / "metadata.json"

    # If state.json exists, load it but VERIFY artifacts match the claimed phase
    if state_path.exists():
        state = ExperimentState.load(state_path)
        # Update requested metrics if provided
        if requested_metrics:
            state.metrics_requested = requested_metrics

        # Validate that required artifacts exist for the claimed phase
        # If artifacts are missing, adjust phase accordingly
        state = _validate_state_artifacts(state, exp_dir)
        return state

    # Otherwise, detect from files
    now = datetime.now().isoformat()

    # Check for completion first
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        metrics = list(data.get("metrics", {}).keys())
        num = data.get("num_examples", 0)
        return ExperimentState(
            phase=ExperimentPhase.COMPLETE,
            started_at=now,
            updated_at=now,
            total_questions=num,
            predictions_complete=num,
            metrics_computed=metrics,
            metrics_requested=requested_metrics or metrics,
        )

    # Check for predictions
    if predictions_path.exists():
        with open(predictions_path) as f:
            data = json.load(f)
        preds = data.get("predictions", [])
        total = len(preds)

        # Determine if generation is complete
        # (heuristic: if we have aggregate_metrics, generation is done)
        if data.get("aggregate_metrics"):
            # Check which metrics are computed
            computed = list(data.get("aggregate_metrics", {}).keys())
            return ExperimentState(
                phase=ExperimentPhase.COMPUTING_METRICS,
                started_at=now,
                updated_at=now,
                total_questions=total,
                predictions_complete=total,
                metrics_computed=computed,
                metrics_requested=requested_metrics,
            )
        else:
            return ExperimentState(
                phase=ExperimentPhase.GENERATED,
                started_at=now,
                updated_at=now,
                total_questions=total,
                predictions_complete=total,
                metrics_requested=requested_metrics,
            )

    # Check for questions
    if questions_path.exists():
        with open(questions_path) as f:
            data = json.load(f)
        total = data.get("count", len(data.get("questions", [])))
        return ExperimentState(
            phase=ExperimentPhase.INIT,
            started_at=now,
            updated_at=now,
            total_questions=total,
            metrics_requested=requested_metrics,
        )

    # Check for metadata only
    if metadata_path.exists():
        return ExperimentState(
            phase=ExperimentPhase.INIT,
            started_at=now,
            updated_at=now,
            metrics_requested=requested_metrics,
        )

    # Nothing exists - new experiment
    return ExperimentState.new(metrics=requested_metrics)


def _load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None if it doesn't exist or is invalid."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def check_health(
    exp_dir: Path,
    requested_metrics: list[str] | None = None,
) -> ExperimentHealth:
    """Check health of an experiment.

    Args:
        exp_dir: Experiment output directory
        requested_metrics: Metrics that should be computed

    Returns:
        ExperimentHealth with detailed status
    """
    requested_metrics = requested_metrics or []

    # Pre-load files once — shared between detect_state and health checking
    predictions_path = exp_dir / "predictions.json"
    questions_path = exp_dir / "questions.json"
    q_data = _load_json(questions_path)
    p_data = _load_json(predictions_path)

    state = detect_state(exp_dir, requested_metrics)

    # Determine missing predictions using pre-loaded data
    missing_predictions = []
    total_questions = state.total_questions
    predictions_complete = state.predictions_complete

    if q_data is not None:
        q_indices = {q["idx"] for q in q_data.get("questions", [])}
        total_questions = len(q_indices)

        if p_data is not None:
            p_indices = {p.get("idx", i) for i, p in enumerate(p_data.get("predictions", []))}
            missing_predictions = sorted(q_indices - p_indices)
            predictions_complete = len(p_indices)
        else:
            missing_predictions = sorted(q_indices)
            predictions_complete = 0

    # Determine missing metrics
    metrics_computed = state.metrics_computed
    metrics_missing = [m for m in requested_metrics if m not in metrics_computed]

    # Experiment is only complete if:
    # 1. Phase is COMPLETE
    # 2. No missing predictions
    # 3. All requested metrics are computed
    is_complete = (
        state.phase == ExperimentPhase.COMPLETE
        and len(missing_predictions) == 0
        and len(metrics_missing) == 0
    )

    return ExperimentHealth(
        phase=state.phase,
        is_complete=is_complete,
        is_healthy=state.error is None and state.phase != ExperimentPhase.FAILED,
        total_questions=total_questions,
        predictions_complete=predictions_complete,
        missing_predictions=missing_predictions,
        metrics_computed=metrics_computed,
        metrics_missing=metrics_missing,
        error=state.error,
    )
