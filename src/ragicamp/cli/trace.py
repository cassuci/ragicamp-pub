"""Pipeline trace verification tool.

Reads predictions.json and verifies that expected pipeline stages executed.
Useful for detecting silent failures where a configured stage (e.g. query
transform) never actually ran.
"""

import json
from collections import Counter
from pathlib import Path

from ragicamp.core.step_types import (
    BATCH_ENCODE,
    BATCH_GENERATE,
    BATCH_SEARCH,
    GENERATE,
    QUERY_TRANSFORM,
    RERANK,
)


def trace_experiment(predictions_path: Path | str) -> dict:
    """Analyze pipeline steps from a predictions file.

    Args:
        predictions_path: Path to predictions.json

    Returns:
        Dict with keys: step_counts, timing_breakdown_ms, warnings
    """
    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        return {"error": f"File not found: {predictions_path}"}

    with open(predictions_path) as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    if not predictions:
        return {"error": "No predictions found in file"}

    # Collect step types across all predictions
    step_counts: Counter[str] = Counter()
    timing_totals: dict[str, float] = {}
    timing_counts: dict[str, int] = {}
    warnings: list[str] = []

    # Check metadata for expected configuration
    metadata = data.get("metadata", {})
    expected_qt = metadata.get("query_transform")
    expected_reranker = metadata.get("reranker")
    exp_type = metadata.get("exp_type") or metadata.get("type")

    for pred in predictions:
        steps = pred.get("steps", [])
        for step in steps:
            step_type = step.get("type", "unknown")
            step_counts[step_type] += 1

            timing = step.get("timing_ms", 0.0)
            if timing:
                timing_totals[step_type] = timing_totals.get(step_type, 0.0) + timing
                timing_counts[step_type] = timing_counts.get(step_type, 0) + 1

    # Build timing breakdown (average per call)
    timing_breakdown = {}
    for step_type, total_ms in timing_totals.items():
        count = timing_counts[step_type]
        timing_breakdown[step_type] = {
            "total_ms": round(total_ms, 1),
            "avg_ms": round(total_ms / count, 1),
            "count": count,
        }

    # Detect missing expected steps
    is_rag = exp_type == "rag" or any(s in step_counts for s in [BATCH_ENCODE, BATCH_SEARCH])

    if is_rag:
        if BATCH_ENCODE not in step_counts and BATCH_SEARCH not in step_counts:
            warnings.append(
                "RAG experiment has no retrieval steps (batch_encode/batch_search). "
                "Retrieval may not have executed."
            )

    if expected_qt and expected_qt != "none":
        if QUERY_TRANSFORM not in step_counts:
            warnings.append(
                f"query_transform='{expected_qt}' is configured but no "
                f"'{QUERY_TRANSFORM}' step was recorded. "
                f"Query transform may not have executed."
            )

    if expected_reranker:
        if RERANK not in step_counts:
            warnings.append(
                f"reranker='{expected_reranker}' is configured but no '{RERANK}' step was recorded."
            )

    if GENERATE not in step_counts and BATCH_GENERATE not in step_counts:
        warnings.append("No generation steps found. Model may not have produced answers.")

    return {
        "num_predictions": len(predictions),
        "step_counts": dict(step_counts),
        "timing_breakdown_ms": timing_breakdown,
        "warnings": warnings,
    }


def format_trace_report(trace: dict) -> str:
    """Format trace results as a human-readable report."""
    if "error" in trace:
        return f"Error: {trace['error']}"

    lines = [
        f"Predictions: {trace['num_predictions']}",
        "",
        "Step Counts:",
    ]

    for step_type, count in sorted(trace["step_counts"].items()):
        lines.append(f"  {step_type}: {count}")

    if trace["timing_breakdown_ms"]:
        lines.append("")
        lines.append("Timing Breakdown:")
        for step_type, info in sorted(
            trace["timing_breakdown_ms"].items(),
            key=lambda x: x[1]["total_ms"],
            reverse=True,
        ):
            lines.append(
                f"  {step_type}: {info['total_ms']:.0f}ms total "
                f"({info['avg_ms']:.0f}ms avg, {info['count']} calls)"
            )

    if trace["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for warning in trace["warnings"]:
            lines.append(f"  - {warning}")

    return "\n".join(lines)
