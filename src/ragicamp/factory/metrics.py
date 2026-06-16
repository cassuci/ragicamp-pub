"""Metric factory for creating evaluation metrics from configuration.

Uses a registry pattern for clean extensibility. Adding a new metric requires
only a single entry in ``_METRIC_REGISTRY``.
"""

from __future__ import annotations

from typing import Any

from ragicamp.core.logging import get_logger
from ragicamp.metrics import Metric
from ragicamp.models.base import LanguageModel

logger = get_logger(__name__)


# Registry: metric name → (module_path, class_name)
# Lazy-imported on first use so optional dependencies don't block startup.
_METRIC_REGISTRY: dict[str, tuple[str, str]] = {
    "exact_match": ("ragicamp.metrics.exact_match", "ExactMatchMetric"),
    "f1": ("ragicamp.metrics.exact_match", "F1Metric"),
    "bertscore": ("ragicamp.metrics.bertscore", "BERTScoreMetric"),
    "bleurt": ("ragicamp.metrics.bleurt", "BLEURTMetric"),
    "llm_judge": ("ragicamp.metrics.llm_judge_qa", "LLMJudgeQAMetric"),
    "llm_judge_qa": ("ragicamp.metrics.llm_judge_qa", "LLMJudgeQAMetric"),
    "faithfulness": ("ragicamp.metrics.faithfulness", "FaithfulnessMetric"),
    "hallucination": ("ragicamp.metrics.hallucination", "HallucinationMetric"),
    "answer_in_context": ("ragicamp.metrics.answer_in_context", "AnswerInContextMetric"),
    "context_recall": ("ragicamp.metrics.context_recall", "ContextRecallMetric"),
}

# Metrics that require a judge_model to be provided.
_JUDGE_METRICS = {"llm_judge", "llm_judge_qa"}


def _import_metric_class(module_path: str, class_name: str) -> type | None:
    """Lazy-import a metric class, returning None if the dependency is missing."""
    import importlib

    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        return None


class MetricFactory:
    """Factory for creating evaluation metrics from configuration."""

    # Custom metric registry (user-registered at runtime)
    _custom_metrics: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Register a custom metric type."""

        def decorator(metric_class: type) -> type:
            cls._custom_metrics[name] = metric_class
            return metric_class

        return decorator

    @classmethod
    def create(
        cls,
        config: list[str | dict[str, Any]],
        judge_model: LanguageModel | None = None,
    ) -> list[Metric]:
        """Create metrics from configuration.

        Args:
            config: List of metric names or dicts with name and params
            judge_model: Optional judge model for LLM-based metrics

        Returns:
            List of instantiated Metric objects

        Example:
            >>> config = ["exact_match", "f1", {"name": "bertscore", "params": {...}}]
            >>> metrics = MetricFactory.create(config)
        """
        metrics: list[Metric] = []

        for metric_config in config:
            if isinstance(metric_config, str):
                metric_name = metric_config
                metric_params: dict[str, Any] = {}
            else:
                metric_name = metric_config["name"]
                metric_params = metric_config.get("params", {})

            # Check custom registry first
            if metric_name in cls._custom_metrics:
                metrics.append(cls._custom_metrics[metric_name](**metric_params))
                continue

            # Check built-in registry
            if metric_name not in _METRIC_REGISTRY:
                logger.warning("Unknown metric: %s, skipping", metric_name)
                continue

            # Judge metrics need special handling
            if metric_name in _JUDGE_METRICS:
                if not judge_model:
                    logger.warning("Skipping %s (judge_model not configured)", metric_name)
                    continue
                judgment_type = metric_params.get("judgment_type", "binary")
                max_concurrent = metric_params.get(
                    "max_concurrent", metric_params.get("batch_size", 20)
                )
                module_path, class_name = _METRIC_REGISTRY[metric_name]
                metric_cls = _import_metric_class(module_path, class_name)
                if metric_cls is None:
                    logger.warning("Skipping %s (not installed)", metric_name)
                    continue
                metrics.append(
                    metric_cls(
                        judge_model=judge_model,
                        judgment_type=judgment_type,
                        max_concurrent=max_concurrent,
                    )
                )
                continue

            # Standard metrics: lazy import and instantiate
            module_path, class_name = _METRIC_REGISTRY[metric_name]
            metric_cls = _import_metric_class(module_path, class_name)
            if metric_cls is None:
                logger.warning("Skipping %s (not installed)", metric_name)
                continue
            metrics.append(metric_cls(**metric_params))

        return metrics
