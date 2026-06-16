"""Compare and analyze experiment results.

Provides utilities for:
- Grouping results by different dimensions (model, dataset, prompt, etc.)
- Finding best configurations
- Creating pivot tables
- Generating summary statistics
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from ragicamp.analysis.loader import ExperimentResult


def compare_results(
    results: list[ExperimentResult],
    group_by: str = "model",
    metric: str = "f1",
    sort: bool = True,
) -> dict[str, dict[str, Any]]:
    """Compare results grouped by a dimension.

    Args:
        results: List of experiment results
        group_by: Dimension to group by. Options:
            - model, dataset, prompt, type, quantization (basic)
            - retriever (full retriever name)
            - embedding_model, corpus, chunk_size, chunk_strategy (parsed from retriever)
            - top_k (number of retrieved docs)
        metric: Metric to compare. Options:
            - f1, exact_match, bertscore_f1, bleurt (always available)
            - bertscore_precision, bertscore_recall
            - llm_judge (if computed)
            - throughput_qps, duration
        sort: Sort by metric value descending

    Returns:
        Dict mapping group values to stats: {group: {mean, min, max, count, std}}

    Example:
        >>> compare_results(results, group_by="embedding_model", metric="f1")
        >>> compare_results(results, group_by="chunk_size", metric="bleurt")
    """
    groups: dict[str, list[float]] = defaultdict(list)

    for r in results:
        # Get group key based on dimension
        if group_by == "model":
            key = r.model_short
        elif group_by == "retriever":
            key = r.retriever_short if r.retriever else "none"
        elif group_by == "model_full":
            key = r.model
        elif group_by == "embedding_model":
            key = r.embedding_model if r.embedding_model != "unknown" else "none"
        elif group_by == "chunk_size":
            key = str(r.chunk_size) if r.chunk_size > 0 else "none"
        elif group_by == "corpus":
            key = r.corpus if r.corpus != "unknown" else "none"
        elif group_by == "top_k":
            key = str(r.top_k) if r.top_k else "none"
        else:
            key = str(getattr(r, group_by, "unknown"))

        # Get metric value
        value = getattr(r, metric, None)
        if value is not None:
            groups[key].append(value)

    # Compute statistics
    stats = {}
    for key, values in groups.items():
        if values:
            stats[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
                "std": _std(values),
            }

    # Sort by mean descending
    if sort:
        stats = dict(sorted(stats.items(), key=lambda x: x[1]["mean"], reverse=True))

    return stats


def best_by(
    results: list[ExperimentResult],
    metric: str = "f1",
    n: int = 10,
    filter_fn: Callable[[ExperimentResult], bool] | None = None,
) -> list[ExperimentResult]:
    """Get the top N results by a metric.

    Args:
        results: List of experiment results
        metric: Metric to rank by
        n: Number of top results to return
        filter_fn: Optional filter function

    Returns:
        List of top N results sorted by metric descending

    Example:
        >>> best = best_by(results, metric="exact_match", n=5)
        >>> for r in best:
        ...     print(f"{r.name}: {r.exact_match:.2%}")
    """
    filtered = results
    if filter_fn:
        filtered = [r for r in results if filter_fn(r)]

    sorted_results = sorted(filtered, key=lambda r: getattr(r, metric, 0), reverse=True)
    return sorted_results[:n]


def pivot_results(
    results: list[ExperimentResult],
    rows: str = "model",
    cols: str = "dataset",
    metric: str = "f1",
    agg: str = "mean",
) -> Any:
    """Create a pivot table of results.

    Args:
        results: List of experiment results
        rows: Dimension for rows (model, prompt, quantization, embedding_model, etc.)
        cols: Dimension for columns (dataset, retriever, chunk_size, top_k, etc.)
        metric: Metric to aggregate
        agg: Aggregation function (mean, max, min)

    Returns:
        pandas DataFrame if pandas available, else nested dict

    Example:
        >>> pivot = pivot_results(results, rows="model", cols="dataset", metric="f1")
        >>> print(pivot)  # Nice DataFrame output
    """
    # Collect values per cell
    cells: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        # Get row key
        if rows == "model":
            row_key = r.model_short
        elif rows == "embedding_model":
            row_key = r.embedding_model if r.embedding_model != "unknown" else "none"
        elif rows == "chunk_size":
            row_key = str(r.chunk_size) if r.chunk_size > 0 else "none"
        else:
            row_key = str(getattr(r, rows, "unknown"))

        # Get col key
        if cols == "model":
            col_key = r.model_short
        elif cols == "embedding_model":
            col_key = r.embedding_model if r.embedding_model != "unknown" else "none"
        elif cols == "chunk_size":
            col_key = str(r.chunk_size) if r.chunk_size > 0 else "none"
        elif cols == "top_k":
            col_key = str(r.top_k) if r.top_k else "none"
        else:
            col_key = str(getattr(r, cols, "unknown"))

        value = getattr(r, metric, None)
        if value is not None:
            cells[row_key][col_key].append(value)

    # Aggregate
    agg_fn = {"mean": lambda x: sum(x) / len(x), "max": max, "min": min}.get(agg, lambda x: x[0])

    pivot_dict = {}
    for row_key, col_dict in cells.items():
        pivot_dict[row_key] = {}
        for col_key, values in col_dict.items():
            if values:
                pivot_dict[row_key][col_key] = agg_fn(values)

    # Try to return DataFrame if pandas available
    try:
        import pandas as pd

        df = pd.DataFrame(pivot_dict).T

        # Sort columns - try numeric first, then string
        def sort_key(x):
            try:
                return (0, int(x))  # Numeric columns first
            except (ValueError, TypeError):
                return (1, str(x))  # Then string columns

        sorted_cols = sorted(df.columns, key=sort_key)
        df = df.reindex(sorted_cols, axis=1)
        return df
    except ImportError:
        return pivot_dict


def summarize_results(results: list[ExperimentResult]) -> dict[str, Any]:
    """Generate summary statistics for a set of results.

    Args:
        results: List of experiment results

    Returns:
        Summary dict with counts, best configs, and averages

    Example:
        >>> summary = summarize_results(results)
        >>> print(f"Total: {summary['count']} experiments")
        >>> print(f"Best F1: {summary['best_f1']['name']} ({summary['best_f1']['value']:.2%})")
    """
    if not results:
        return {"count": 0}

    # Basic counts
    summary = {
        "count": len(results),
        "models": list({r.model_short for r in results}),
        "datasets": list({r.dataset for r in results}),
        "prompts": list({r.prompt for r in results}),
        "types": list({r.type for r in results}),
    }

    # RAG-specific counts
    rag_results = [r for r in results if r.type == "rag"]
    if rag_results:
        summary["embedding_models"] = list(
            {r.embedding_model for r in rag_results if r.embedding_model != "unknown"}
        )
        summary["chunk_sizes"] = list({r.chunk_size for r in rag_results if r.chunk_size > 0})
        summary["corpora"] = list({r.corpus for r in rag_results if r.corpus != "unknown"})

    # Best by each metric
    all_metrics = [
        "f1",
        "exact_match",
        "bertscore_f1",
        "bertscore_precision",
        "bertscore_recall",
        "bleurt",
    ]

    # Add llm_judge if any results have it
    if any(r.llm_judge is not None for r in results):
        all_metrics.append("llm_judge")

    for metric in all_metrics:
        valid_results = [r for r in results if getattr(r, metric, None) is not None]
        if valid_results:
            best = max(valid_results, key=lambda r: getattr(r, metric, 0))
            summary[f"best_{metric}"] = {
                "name": best.name,
                "value": getattr(best, metric, 0),
                "model": best.model_short,
                "dataset": best.dataset,
                "prompt": best.prompt,
            }

    # Averages
    for metric in all_metrics + ["throughput_qps"]:
        values = [getattr(r, metric, 0) for r in results if getattr(r, metric, None) is not None]
        summary[f"avg_{metric}"] = sum(values) / len(values) if values else 0

    # Total duration
    summary["total_duration_hours"] = sum(r.duration for r in results) / 3600

    return summary


def format_comparison_table(
    stats: dict[str, dict[str, Any]],
    title: str = "Comparison",
    metric: str = "f1",
) -> str:
    """Format comparison stats as a text table.

    Args:
        stats: Output from compare_results()
        title: Table title
        metric: Metric name for header

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f" {title}")
    lines.append("=" * 70)
    lines.append(f"{'Group':<30} {'Mean':>10} {'Min':>10} {'Max':>10} {'N':>6}")
    lines.append("-" * 70)

    for group, s in stats.items():
        lines.append(
            f"{group[:29]:<30} {s['mean']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f} {s['count']:>6}"
        )

    lines.append("=" * 70)
    return "\n".join(lines)


def _std(values: list[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance**0.5
