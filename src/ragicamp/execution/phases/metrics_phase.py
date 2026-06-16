"""Metrics phase handler - computes evaluation metrics."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragicamp.core.logging import get_logger
from ragicamp.execution.phases.base import ExecutionContext, PhaseHandler
from ragicamp.state import ExperimentPhase, ExperimentState

if TYPE_CHECKING:
    from ragicamp.spec import ExperimentSpec

logger = get_logger(__name__)


class MetricsHandler(PhaseHandler):
    """Handler for the COMPUTING_METRICS phase.

    Computes all requested metrics on the generated predictions.
    Uses the shared compute_metrics_batched function for proper GPU management.
    """

    def can_handle(self, phase: ExperimentPhase) -> bool:
        """Check if this handler processes COMPUTING_METRICS phase."""
        return phase == ExperimentPhase.COMPUTING_METRICS

    def execute(
        self,
        spec: "ExperimentSpec",
        state: ExperimentState,
        context: ExecutionContext,
    ) -> ExperimentState:
        """Compute all metrics on predictions.

        Loads predictions, computes each metric, and saves results.
        Tracks which metrics are computed to enable resume on partial completion.
        """
        from ragicamp.metrics import compute_metrics_batched

        logger.info("Phase: COMPUTING_METRICS")

        predictions_path = context.output_path / "predictions.json"
        state_path = context.output_path / "state.json"

        # Load predictions
        with open(predictions_path) as f:
            data = json.load(f)

        preds = data["predictions"]
        predictions = [p["prediction"] for p in preds]
        references = [p["expected"] for p in preds]
        questions = [p["question"] for p in preds]

        # Extract retrieved contexts for context-aware metrics
        contexts = self._extract_contexts(preds)

        # Callback to save state after each metric
        def on_metric_complete(metric_name: str) -> None:
            if metric_name not in state.metrics_computed:
                state.metrics_computed.append(metric_name)
                state.save(state_path)

        # Use shared metrics computation (handles batching, GPU cleanup, etc.)
        aggregate_results, per_item_metrics, computed, failed, metric_timings = (
            compute_metrics_batched(
                metrics=context.metrics,
                predictions=predictions,
                references=references,
                questions=questions,
                contexts=contexts,
                already_computed=state.metrics_computed,
                on_metric_complete=on_metric_complete,
            )
        )

        # Update predictions with per-item metrics
        for i, pred in enumerate(preds):
            if "metrics" not in pred:
                pred["metrics"] = {}
            for metric_name, scores in per_item_metrics.items():
                if i < len(scores):
                    pred["metrics"][metric_name] = scores[i]

        # Merge with existing aggregate metrics
        existing_agg = data.get("aggregate_metrics", {})
        existing_agg.update(aggregate_results)
        data["aggregate_metrics"] = existing_agg
        data["predictions"] = preds

        # Store per-metric timing for profiling
        if metric_timings:
            existing_timings = data.get("metric_timings", {})
            existing_timings.update(metric_timings)
            data["metric_timings"] = existing_timings

        self._save_predictions(data, predictions_path)
        self._export_traces(context.metrics, context.output_path)
        logger.info("Computed metrics: %s", list(aggregate_results.keys()))

        return state

    @staticmethod
    def _extract_contexts(preds: list[dict]) -> list[list[str]] | None:
        """Extract retrieved contexts from predictions for context-aware metrics.

        Returns None if no predictions contain retrieved_docs.
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

    @staticmethod
    def _export_traces(metrics: list[Any], output_path: Path) -> None:
        """Export LLM judge traces to JSONL if available."""
        for metric in metrics:
            if hasattr(metric, "get_traces"):
                traces = metric.get_traces()
                if traces:
                    trace_path = output_path / "llm_judge_traces.jsonl"
                    with open(trace_path, "w") as f:
                        for trace in traces:
                            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    logger.info("Exported %d LLM judge traces to %s", len(traces), trace_path)

    def _save_predictions(self, data: dict, path: Path) -> None:
        """Save predictions atomically."""
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(path)
