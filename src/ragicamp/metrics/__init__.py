"""Evaluation metrics for RAG systems."""

from collections.abc import Callable
from typing import Any

from ragicamp.core.logging import get_logger
from ragicamp.metrics.async_base import AsyncAPIMetric
from ragicamp.metrics.base import Metric

logger = get_logger(__name__)


def _expand_for_multi_reference(
    predictions: list[str],
    references: list[str | list[str]],
    questions: list[str] | None = None,
) -> tuple[list[str], list[str], list[str] | None, list[int], list[int]]:
    """Expand predictions and references for multi-reference evaluation.

    For each prediction with multiple references, creates one (pred, ref) pair
    per reference. Returns mapping indices to aggregate results later.

    Args:
        predictions: List of model predictions
        references: List of references (can be str or List[str] per item)
        questions: Optional list of questions

    Returns:
        Tuple of:
        - expanded_predictions: Flattened predictions (repeated for each ref)
        - expanded_references: Flattened references (one per prediction)
        - expanded_questions: Flattened questions (or None)
        - pred_indices: Original prediction index for each expanded item
        - ref_indices: Reference index within original list for each expanded item
    """
    expanded_preds = []
    expanded_refs = []
    expanded_questions = [] if questions else None
    pred_indices = []
    ref_indices = []

    for i, (pred, ref) in enumerate(zip(predictions, references, strict=True)):
        # Normalize to list of references
        refs_list = ref if isinstance(ref, list) else [ref]

        for j, single_ref in enumerate(refs_list):
            expanded_preds.append(pred)
            expanded_refs.append(single_ref)
            pred_indices.append(i)
            ref_indices.append(j)
            if questions:
                expanded_questions.append(questions[i])

    return expanded_preds, expanded_refs, expanded_questions, pred_indices, ref_indices


def _aggregate_multi_reference_scores(
    scores: list[float],
    pred_indices: list[int],
    num_predictions: int,
) -> list[float]:
    """Aggregate expanded scores back to original predictions using max.

    For each original prediction, takes the maximum score across all its
    reference comparisons.

    Args:
        scores: Per-item scores from expanded computation
        pred_indices: Original prediction index for each score
        num_predictions: Number of original predictions

    Returns:
        Aggregated scores (one per original prediction)
    """
    # Initialize with -inf so any score will be higher
    best_scores = [float("-inf")] * num_predictions

    for score, idx in zip(scores, pred_indices, strict=True):
        if score > best_scores[idx]:
            best_scores[idx] = score

    # Replace any remaining -inf (predictions with no references) with 0.0
    return [s if s != float("-inf") else 0.0 for s in best_scores]


def _has_multi_references(references: list[Any]) -> bool:
    """Check if references contain multi-reference items."""
    return any(isinstance(ref, list) for ref in references)


def compute_metrics_batched(
    metrics: list[Metric],
    predictions: list[str],
    references: list[Any],
    questions: list[str] | None = None,
    contexts: list[list[str]] | None = None,
    already_computed: list[str] | None = None,
    on_metric_complete: Callable | None = None,
) -> tuple[dict[str, float], dict[str, list[float]], list[str], list[str], dict[str, float]]:
    """Compute metrics with proper GPU memory management and multi-reference support.

    This is the shared implementation used by both Experiment._phase_compute_metrics()
    and run_metrics_only() to avoid code duplication.

    Multi-reference handling:
        When references contain lists (multiple correct answers), this function
        expands the computation to evaluate against all references and aggregates
        using max strategy (best score across all references per prediction).

    Context-aware metrics (faithfulness, hallucination, answer_in_context,
    context_recall) receive the original (non-expanded) data with ``contexts``
    kwarg and handle multi-reference internally.

    Args:
        metrics: List of Metric objects to compute
        predictions: List of model predictions
        references: List of reference answers (can be str or List[str] per item)
        questions: Optional list of questions (needed for llm_judge)
        contexts: Optional list of retrieved doc lists per prediction
        already_computed: List of metric names already computed (to skip)
        on_metric_complete: Callback(metric_name) called after each metric

    Returns:
        Tuple of:
        - aggregate_results: Dict of aggregated metric scores
        - per_item_metrics: Dict of per-item scores for each metric
        - computed: List of successfully computed metric names
        - failed: List of failed metric names
        - timings: Dict of metric_name -> seconds for each computed metric
    """
    import time as _time

    from ragicamp.core.constants import is_error_prediction
    from ragicamp.utils.resource_manager import ResourceManager

    already_computed = already_computed or []
    aggregate_results: dict[str, float] = {}
    per_item_metrics: dict[str, list[float]] = {}
    computed: list[str] = []
    failed: list[str] = []
    timings: dict[str, float] = {}

    # Filter out error predictions (from failed API calls) before scoring
    valid_mask = [not is_error_prediction(p) for p in predictions]
    n_errors = sum(1 for v in valid_mask if not v)
    if n_errors > 0:
        logger.warning(
            "Excluding %d/%d error predictions from metric computation",
            n_errors,
            len(predictions),
        )
        predictions = [p for p, v in zip(predictions, valid_mask, strict=True) if v]
        references = [r for r, v in zip(references, valid_mask, strict=True) if v]
        if questions is not None:
            questions = [q for q, v in zip(questions, valid_mask, strict=True) if v]
        if contexts is not None:
            contexts = [c for c, v in zip(contexts, valid_mask, strict=True) if v]

    if not predictions:
        logger.warning("No valid predictions to score after filtering errors")
        return aggregate_results, per_item_metrics, computed, failed, timings

    # Check if we need multi-reference handling
    has_multi = _has_multi_references(references)

    if has_multi:
        # Expand for multi-reference evaluation (used by non-LLM metrics)
        expanded_preds, expanded_refs, _, pred_indices, _ = _expand_for_multi_reference(
            predictions, references, questions
        )
        logger.info(
            "Multi-reference detected: %d predictions -> %d pairs",
            len(predictions),
            len(expanded_preds),
        )
    else:
        # Simple case: 1-to-1
        expanded_preds = predictions
        expanded_refs = references
        pred_indices = list(range(len(predictions)))

    # Context-aware metrics receive original (non-expanded) data with contexts kwarg
    _context_metric_names = {
        "faithfulness",
        "hallucination",
        "answer_in_context",
        "context_recall",
    }

    for metric in metrics:
        if metric.name in already_computed:
            logger.info("%s already computed", metric.name)
            continue

        try:
            logger.info("Computing %s...", metric.name)
            ResourceManager.clear_gpu_memory()
            _metric_t0 = _time.perf_counter()

            # Call metric with compute_with_details() to get per-item scores
            # directly from the return value, avoiding the stateful
            # _last_per_item side-channel (4.13 fix).
            is_llm_judge = metric.name in ("llm_judge", "llm_judge_qa")
            is_context_metric = metric.name in _context_metric_names

            if is_llm_judge:
                if questions is None:
                    logger.warning("%s requires questions, skipping", metric.name)
                    failed.append(metric.name)
                    continue
                # LLM judge handles multi-reference internally: all valid
                # answers are listed in a single prompt, avoiding N separate
                # API calls per question. Pass original (non-expanded) data.
                metric_result = metric.compute_with_details(
                    predictions=predictions,
                    references=references,
                    questions=questions,
                )
            elif is_context_metric:
                # Context metrics handle multi-reference internally and need
                # the original (non-expanded) data with contexts.
                if contexts is None:
                    logger.warning(
                        "%s requires retrieved contexts but none found in predictions "
                        "(no retrieved_docs). Skipping.",
                        metric.name,
                    )
                    failed.append(metric.name)
                    continue
                metric_result = metric.compute_with_details(
                    predictions=predictions,
                    references=references,
                    contexts=contexts,
                )
            else:
                metric_result = metric.compute_with_details(
                    predictions=expanded_preds, references=expanded_refs
                )

            scores = {metric_result.name: metric_result.aggregate}
            expanded_per_item = metric_result.per_item

            # Aggregate if multi-reference (not needed for LLM judge or context metrics)
            if has_multi and expanded_per_item and not is_llm_judge and not is_context_metric:
                aggregated_per_item = _aggregate_multi_reference_scores(
                    expanded_per_item, pred_indices, len(predictions)
                )
                per_item_metrics[metric.name] = aggregated_per_item

                # Recompute aggregate from aggregated per-item scores
                avg_score = sum(aggregated_per_item) / len(aggregated_per_item)
                # Update the main metric score with properly aggregated value
                if metric.name in scores:
                    scores[metric.name] = avg_score
                else:
                    logger.warning(
                        "Could not find key '%s' in metric results %s; "
                        "multi-reference aggregation skipped for this metric",
                        metric.name,
                        list(scores.keys()),
                    )
            elif expanded_per_item:
                per_item_metrics[metric.name] = expanded_per_item

            _metric_s = _time.perf_counter() - _metric_t0
            timings[metric.name] = round(_metric_s, 2)
            logger.info("  %s done in %.1fs", metric.name, _metric_s)

            aggregate_results.update(scores)
            computed.append(metric.name)

            if on_metric_complete:
                on_metric_complete(metric.name)

            ResourceManager.clear_gpu_memory()

        except Exception as e:
            logger.warning("%s failed: %s", metric.name, e)
            failed.append(metric.name)
            ResourceManager.clear_gpu_memory()

    return aggregate_results, per_item_metrics, computed, failed, timings


# Import specific metrics (but handle import errors gracefully)
try:
    from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric  # noqa: F401

    _has_exact_match = True
except ImportError:
    _has_exact_match = False

try:
    from ragicamp.metrics.bertscore import BERTScoreMetric  # noqa: F401

    _has_bertscore = True
except ImportError:
    _has_bertscore = False

try:
    from ragicamp.metrics.bleurt import BLEURTMetric  # noqa: F401

    _has_bleurt = True
except ImportError:
    _has_bleurt = False

try:
    from ragicamp.metrics.llm_judge_qa import LLMJudgeQAMetric  # noqa: F401

    _has_llm_judge = True
except ImportError:
    _has_llm_judge = False

try:
    from ragicamp.metrics.faithfulness import FaithfulnessMetric  # noqa: F401

    _has_faithfulness = True
except ImportError:
    _has_faithfulness = False

try:
    from ragicamp.metrics.hallucination import HallucinationMetric  # noqa: F401

    _has_hallucination = True
except ImportError:
    _has_hallucination = False

try:
    from ragicamp.metrics.answer_in_context import AnswerInContextMetric  # noqa: F401

    _has_answer_in_context = True
except ImportError:
    _has_answer_in_context = False

try:
    from ragicamp.metrics.context_recall import ContextRecallMetric  # noqa: F401

    _has_context_recall = True
except ImportError:
    _has_context_recall = False

__all__ = ["Metric", "AsyncAPIMetric", "compute_metrics_batched"]

# Add available metrics to __all__
if _has_exact_match:
    __all__.extend(["ExactMatchMetric", "F1Metric"])
if _has_bertscore:
    __all__.append("BERTScoreMetric")
if _has_bleurt:
    __all__.append("BLEURTMetric")
if _has_llm_judge:
    __all__.append("LLMJudgeQAMetric")
if _has_faithfulness:
    __all__.append("FaithfulnessMetric")
if _has_hallucination:
    __all__.append("HallucinationMetric")
if _has_answer_in_context:
    __all__.append("AnswerInContextMetric")
if _has_context_recall:
    __all__.append("ContextRecallMetric")
