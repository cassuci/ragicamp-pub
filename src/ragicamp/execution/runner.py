"""Experiment execution runner.

This module handles experiment execution with phase-aware dispatching:
- run_spec: Dispatch to appropriate runner based on phase
- run_generation: Run generation phase (needs model, GPU)
- run_metrics_only: Run metrics phase (no model needed)

The phase-aware design ensures resources are only loaded when needed:
- GENERATING phase: Loads model, agent, retriever
- COMPUTING_METRICS phase: Only loads predictions file + metrics

For experiment specification and building, see the spec/ package.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from ragicamp.core.logging import get_logger
from ragicamp.spec import ExperimentSpec

logger = get_logger(__name__)


def _extract_contexts(preds: list[dict]) -> list[list[str]] | None:
    """Extract retrieved contexts from predictions for context-aware metrics.

    Returns None if no predictions contain retrieved_docs (e.g., direct_llm experiments).
    """
    contexts: list[list[str]] = []
    has_any = False
    for p in preds:
        docs = p.get("retrieved_docs", [])
        ctx = [doc["content"] for doc in docs if "content" in doc]
        if ctx:
            has_any = True
        contexts.append(ctx)
    if not has_any:
        return None
    return contexts


# =============================================================================
# Phase-Aware Runners
# =============================================================================


def run_metrics_only(
    exp_name: str,
    output_path: Path,
    metrics: list[str],
    judge_model: Any = None,
    spec: "ExperimentSpec | None" = None,
    force: bool = False,
) -> str:
    """Run metrics computation only - no model needed.

    This is the lightweight path when predictions already exist.
    Uses minimal GPU memory (only for GPU-based metrics like BERTScore).

    Args:
        exp_name: Experiment name
        output_path: Path to experiment output directory
        metrics: List of metric names to compute
        judge_model: Optional LLM judge model for llm_judge metric
        spec: Optional ExperimentSpec for enriching results metadata
        force: Recompute metrics even if already present

    Returns:
        Status string
    """
    from ragicamp.factory import MetricFactory
    from ragicamp.metrics import compute_metrics_batched
    from ragicamp.state import ExperimentPhase, detect_state
    from ragicamp.utils.experiment_io import ExperimentIO

    logger.info("Metrics-only mode (no model loaded)")

    io = ExperimentIO(output_path)

    if not io.predictions_exist():
        logger.error("No predictions file found at %s", io.predictions_path)
        return "failed"

    # Load predictions using ExperimentIO
    data = io.load_predictions()
    preds = data["predictions"]
    predictions = [p["prediction"] for p in preds]
    references = [p["expected"] for p in preds]
    questions = [p["question"] for p in preds]

    # Extract retrieved contexts for context-aware metrics
    contexts = _extract_contexts(preds)

    # Load state to track computed metrics
    state = detect_state(output_path, metrics)

    # Create metric objects
    metric_objs = MetricFactory.create(metrics, judge_model=judge_model)

    # Callback to save state after each metric
    def on_metric_complete(metric_name: str) -> None:
        if metric_name not in state.metrics_computed:
            state.metrics_computed.append(metric_name)
            state.save(io.state_path)

    # Use shared metrics computation
    already_computed = [] if force else state.metrics_computed
    aggregate_results, per_item_metrics, computed, failed, _timings = compute_metrics_batched(
        metrics=metric_objs,
        predictions=predictions,
        references=references,
        questions=questions,
        contexts=contexts,
        already_computed=already_computed,
        on_metric_complete=on_metric_complete,
    )

    # Update predictions with per-item metrics
    for i, pred in enumerate(preds):
        if "metrics" not in pred:
            pred["metrics"] = {}
        for metric_name, scores in per_item_metrics.items():
            if i < len(scores):
                pred["metrics"][metric_name] = scores[i]

    # Merge with existing aggregate metrics
    existing_metrics = data.get("aggregate_metrics", {})
    existing_metrics.update(aggregate_results)
    data["aggregate_metrics"] = existing_metrics

    # Save updated predictions using atomic write
    io.save_predictions(data)

    # Export LLM judge traces if available
    for metric_obj in metric_objs:
        if hasattr(metric_obj, "get_traces"):
            traces = metric_obj.get_traces()
            if traces:
                trace_path = output_path / "llm_judge_traces.jsonl"
                with open(trace_path, "w") as f:
                    for trace in traces:
                        f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                logger.info("Exported %d LLM judge traces to %s", len(traces), trace_path)

    # Check if all metrics are done
    missing = set(metrics) - set(state.metrics_computed)
    if missing:
        logger.warning("Missing metrics: %s", list(missing))
        return "incomplete"

    # Mark complete
    state.phase = ExperimentPhase.COMPLETE
    state.save(io.state_path)

    # Save results using ExperimentIO — include spec metadata when available
    result_data = {
        "name": exp_name,
        "metrics": existing_metrics,
        "num_examples": len(preds),
        "timestamp": datetime.now().isoformat(),
    }
    if spec is not None:
        result_data["metadata"] = spec.to_dict()
    io.save_result_dict(result_data)

    # Print metric summary
    from ragicamp.utils.formatting import format_metrics_summary

    metrics_str = format_metrics_summary(existing_metrics)
    logger.info("All metrics computed: %s", metrics_str)
    return "ran"


def run_generation(
    spec: ExperimentSpec,
    limit: int | None,
    metrics: list[str],
    out: Path,
    judge_model: Any = None,
) -> str:
    """Run full experiment including generation.

    This is the simplified path that uses Experiment.from_spec() to create
    all components automatically from the ExperimentSpec.

    Args:
        spec: Experiment specification
        limit: Max examples to process
        metrics: List of metric names
        out: Output directory
        judge_model: Optional LLM judge model

    Returns:
        Status string
    """
    from ragicamp import Experiment
    from ragicamp.utils.experiment_io import ExperimentIO

    exp_out = out / spec.name
    start = time.time()

    # Note: GPU memory is cleared by parent before subprocess spawn
    # No need to clear again here - subprocess starts with clean GPU state

    logger.info("Loading model: %s", spec.model)

    # Create experiment from spec - handles all component creation
    exp = Experiment.from_spec(
        spec=spec,
        output_dir=out,
        limit=limit,
        judge_model=judge_model,
    )

    logger.info("Dataset: %d examples", len(exp.dataset))

    # Run the experiment
    result = exp.run(
        batch_size=spec.batch_size,
        checkpoint_every=50,
        resume=True,
    )

    # Save metadata using spec.to_dict() as base to avoid field drift,
    # then layer on runtime-specific fields
    io = ExperimentIO(exp_out)
    metadata = spec.to_dict()
    metadata.update(
        {
            "type": spec.exp_type,  # Keep "type" key (vs "exp_type" in spec)
            "metrics": result.metrics,
            "duration": time.time() - start,
        }
    )
    io.save_metadata(metadata)

    # Check if there were aborted predictions (model failures)
    predictions_path = exp_out / "predictions.json"
    if predictions_path.exists():
        with open(predictions_path) as f:
            preds_data = json.load(f)
        predictions = preds_data.get("predictions", [])
        aborted = sum(1 for p in predictions if "[ABORTED" in str(p.get("prediction", "")))
        errored = sum(1 for p in predictions if p.get("error"))

        if aborted > 0 or errored > len(predictions) * 0.5:
            # B3 fix: update state to FAILED so re-run doesn't skip this experiment
            from ragicamp.state import ExperimentState

            state_path = exp_out / "state.json"
            if state_path.exists():
                st = ExperimentState.load(state_path)
                reason = (
                    f"{aborted} aborted predictions"
                    if aborted > 0
                    else f"{errored}/{len(predictions)} errors"
                )
                st.set_error(reason)
                st.save(state_path)

            if aborted > 0:
                logger.warning("%d predictions were aborted due to model failures", aborted)
            else:
                logger.warning("%d/%d predictions had errors", errored, len(predictions))
            return "failed"

    return "ran"


# =============================================================================
# Main Entry Points
# =============================================================================


def _kill_process_group(pid: int) -> None:
    """Kill all processes in the process group of the given PID.

    Used to clean up orphaned grandchild processes (e.g., vLLM EngineCore)
    after a subprocess exits. Safe to call even if processes are already dead.
    """
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    # Give processes a moment to exit gracefully, then force kill
    try:
        time.sleep(0.5)
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def run_spec_subprocess(
    spec: ExperimentSpec,
    limit: int | None,
    metrics: list[str],
    out: Path,
    llm_judge_config: dict[str, Any] | None = None,
    timeout: int = 7200,
) -> str:
    """Run experiment in subprocess for CUDA crash isolation.

    Args:
        spec: Experiment specification
        limit: Max examples to process
        metrics: List of metric names
        out: Output directory
        llm_judge_config: LLM judge configuration
        timeout: Timeout in seconds

    Returns:
        Status string
    """
    import os

    # Ensure TensorFlow doesn't grab all GPU memory in subprocess
    # These are inherited by the subprocess
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # CRITICAL: Clear GPU memory in parent process before spawning subprocess
    # The parent may have loaded embedding models for index checking.
    # If we don't clear, the subprocess will see a full GPU and OOM.
    from ragicamp.utils.resource_manager import ResourceManager

    ResourceManager.clear_gpu_memory()

    from ragicamp.state import ExperimentPhase, check_health

    exp_out = out / spec.name
    exp_out.mkdir(parents=True, exist_ok=True)

    # Use spec's metrics if available, otherwise fall back to parameter
    check_metrics = list(spec.metrics) if spec.metrics else metrics

    # Check health before running
    health = check_health(exp_out, check_metrics)

    if health.is_complete:
        logger.info("%s (complete)", spec.name)
        return "complete"

    if health.phase == ExperimentPhase.FAILED:
        logger.warning("%s (previously failed, retrying)", spec.name)

    # Note: Don't print header here - the subprocess will print it
    # This avoids duplicate output

    from ragicamp.utils.paths import get_project_root

    script_path = get_project_root() / "scripts" / "experiments" / "run_single_experiment.py"

    # Use spec.to_dict() as the single source of truth to avoid field drift
    spec_dict = spec.to_dict()
    # Override metrics with the merged list (spec metrics + CLI metrics)
    if not spec_dict.get("metrics"):
        spec_dict["metrics"] = metrics

    # Use the spec's own metrics (already merged above) as the single
    # source of truth to avoid divergence between spec and CLI arg.
    merged_metrics = spec_dict.get("metrics", metrics)

    cmd = [
        sys.executable,
        str(script_path),
        "--spec-json",
        json.dumps(spec_dict),
        "--output-dir",
        str(out),
        "--metrics",
        ",".join(merged_metrics),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    if llm_judge_config:
        cmd.extend(["--llm-judge-config", json.dumps(llm_judge_config)])

    log_file = exp_out / "experiment.log"

    try:
        with open(log_file, "a", buffering=1, encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,  # Isolate in own process group
            )

            def _tee(stream: IO[str], console: IO[str], log: IO[str]) -> None:
                for line in stream:
                    console.write(line)
                    console.flush()
                    log.write(line)

            tee_thread = threading.Thread(
                target=_tee, args=(proc.stdout, sys.stdout, lf), daemon=True
            )
            tee_thread.start()
            proc.wait(timeout=timeout)
            tee_thread.join(timeout=5)

        # Kill any orphaned children (e.g., vLLM EngineCore) that survived
        # the subprocess exit. The process group ID equals proc.pid because
        # we used start_new_session=True.
        _kill_process_group(proc.pid)

        if proc.returncode == 0:
            return "ran"
        else:
            return "failed"

    except subprocess.TimeoutExpired:
        _kill_process_group(proc.pid)
        proc.wait()
        tee_thread.join(timeout=5)
        logger.warning("Timeout after %ds", timeout)
        return "timeout"


def run_spec(
    spec: ExperimentSpec,
    limit: int | None,
    metrics: list[str],
    out: Path,
    judge_model: Any = None,
    llm_judge_config: dict[str, Any] | None = None,
    force: bool = False,
    use_subprocess: bool = True,
    timeout: int = 7200,
) -> str:
    """Run a single experiment with phase-aware dispatching.

    Dispatches to the appropriate runner based on experiment phase:
    - COMPUTING_METRICS: Uses run_metrics_only() (no model loaded)
    - Other phases: Uses run_generation() (full model load)

    Args:
        spec: Experiment specification
        limit: Max examples to process
        metrics: List of metric names
        out: Output directory
        judge_model: LLM judge model instance
        llm_judge_config: LLM judge configuration
        force: Force re-run even if complete/failed
        use_subprocess: Run in subprocess for isolation
        timeout: Subprocess timeout in seconds (default: 7200)

    Returns:
        Status string
    """
    if use_subprocess:
        return run_spec_subprocess(
            spec,
            limit,
            metrics,
            out,
            llm_judge_config=llm_judge_config,
            timeout=timeout,
        )

    # In-process execution with phase-aware dispatching
    from ragicamp.state import ExperimentPhase, check_health

    exp_out = out / spec.name
    exp_out.mkdir(parents=True, exist_ok=True)

    # Use spec's metrics if available, otherwise fall back to parameter
    check_metrics = list(spec.metrics) if spec.metrics else metrics

    health = check_health(exp_out, check_metrics)

    if health.is_complete and not force:
        logger.info("%s (complete)", spec.name)
        return "complete"

    if health.phase == ExperimentPhase.FAILED and not force:
        logger.error("%s (failed: %s)", spec.name, health.error)
        logger.info("  Use --force to retry")
        return "skipped"

    # Determine which phase we're resuming from
    if health.can_resume:
        action = f"Resuming from {health.resume_phase.value}"
    else:
        action = "Starting"

    logger.info("=" * 60)
    logger.info("%s: %s", spec.exp_type.upper(), spec.name)
    logger.info("%s", action)
    logger.info("=" * 60)

    try:
        # Note: GPU memory cleared by parent before subprocess spawn
        # Phase-aware dispatch: metrics-only vs full generation
        if health.can_resume and health.resume_phase == ExperimentPhase.COMPUTING_METRICS:
            # Metrics-only path - no model needed
            return run_metrics_only(
                exp_name=spec.name,
                output_path=exp_out,
                metrics=metrics,
                judge_model=judge_model,
                spec=spec,
                force=force,
            )
        else:
            # Full path - load model, run generation + metrics
            return run_generation(
                spec=spec,
                limit=limit,
                metrics=metrics,
                out=out,
                judge_model=judge_model,
            )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return "interrupted"

    except Exception as e:
        logger.error("Error: %s", e)
        # Save error info with full spec for debugging
        with open(exp_out / "error.log", "w") as f:
            import traceback

            f.write(f"Error: {e}\n\n")
            f.write(f"Experiment: {spec.name}\n")
            f.write(f"Model: {spec.model}\n")
            f.write(f"Dataset: {spec.dataset}\n")
            f.write(f"Type: {spec.exp_type}\n")
            if spec.retriever:
                f.write(f"Retriever: {spec.retriever}\n")
                f.write(f"Top-K: {spec.top_k}\n")
            if spec.agent_type:
                f.write(f"Agent type: {spec.agent_type}\n")
            f.write(f"\nTraceback:\n{traceback.format_exc()}")
        return "failed"
