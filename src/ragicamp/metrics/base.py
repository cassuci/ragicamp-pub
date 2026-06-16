"""Base class for evaluation metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """Result of computing a metric.

    Attributes:
        name: Metric name (e.g., "f1", "exact_match")
        aggregate: Aggregate score across all examples
        per_item: Per-item scores (one per prediction)
        metadata: Additional metric-specific data
    """

    name: str
    aggregate: float
    per_item: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "aggregate": self.aggregate,
            "per_item": self.per_item,
            "metadata": self.metadata,
        }


class Metric(ABC):
    """Base class for all evaluation metrics.

    Metrics evaluate the quality of generated answers against
    reference answers. Each metric computes a 1-to-1 comparison between
    a prediction and a single reference.

    Multi-reference aggregation (e.g., taking max score across multiple
    correct answers) is handled externally by compute_metrics_batched().
    """

    def __init__(self, name: str, **kwargs: Any):
        """Initialize the metric.

        Args:
            name: Metric identifier
            **kwargs: Metric-specific configuration
        """
        self.name = name
        self.config = kwargs
        self._last_per_item: list[float] = []

    @abstractmethod
    def compute(
        self, predictions: list[str], references: list[str], **kwargs: Any
    ) -> dict[str, float]:
        """Compute the metric (1-to-1 prediction vs reference).

        Args:
            predictions: List of predicted answers
            references: List of reference answers (one per prediction)
            **kwargs: Additional computation parameters

        Returns:
            Dict of metric scores (e.g., {"exact_match": 0.85})

        Note:
            Implementations should also populate self._last_per_item with per-item scores.
        """
        pass

    def compute_with_details(
        self, predictions: list[str], references: list[str], **kwargs: Any
    ) -> MetricResult:
        """Compute the metric and return detailed results.

        Args:
            predictions: List of predicted answers
            references: List of reference answers (one per prediction)
            **kwargs: Additional computation parameters

        Returns:
            MetricResult with aggregate and per-item scores
        """
        result = self.compute(predictions, references, **kwargs)
        aggregate = result.get(self.name, 0.0)
        return MetricResult(
            name=self.name,
            aggregate=aggregate,
            per_item=self._last_per_item.copy(),
        )

    def get_per_item_scores(self) -> list[float]:
        """Get per-item scores from the last compute() call.

        Returns:
            List of scores, one per prediction
        """
        return self._last_per_item.copy()

    def compute_single(self, prediction: str, reference: str, **kwargs: Any) -> dict[str, float]:
        """Compute metric for a single prediction-reference pair.

        Args:
            prediction: Predicted answer
            reference: Single reference answer
            **kwargs: Additional parameters

        Returns:
            Dict of metric scores
        """
        return self.compute([prediction], [reference], **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
