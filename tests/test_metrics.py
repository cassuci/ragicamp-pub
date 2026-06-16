"""Tests for metrics computation."""

import pytest

from ragicamp.metrics.base import Metric
from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric


class MockMetric(Metric):
    """Mock metric for testing base functionality."""

    def __init__(self, name="mock", **kwargs):
        super().__init__(name, **kwargs)
        self.compute_called = False
        self.compute_single_called = False

    def compute(self, predictions, references, **kwargs):
        self.compute_called = True
        return {self.name: 0.75, f"{self.name}_count": len(predictions)}


class TestMetricBase:
    """Test base metric functionality."""

    def test_metric_initialization(self):
        """Test metric initialization."""
        metric = MockMetric("test_metric")
        assert metric.name == "test_metric"
        assert metric.config == {}

    def test_metric_with_config(self):
        """Test metric with configuration."""
        metric = MockMetric("test_metric", param1="value1", param2=42)
        assert metric.config["param1"] == "value1"
        assert metric.config["param2"] == 42

    def test_compute_returns_dict(self):
        """Test that compute returns a dictionary."""
        metric = MockMetric("test")
        result = metric.compute(predictions=["pred1", "pred2"], references=["ref1", "ref2"])

        assert isinstance(result, dict)
        assert "test" in result
        assert result["test"] == 0.75


class TestExactMatch:
    """Test exact match metric."""

    def test_exact_match_perfect(self):
        """Test exact match with perfect predictions."""
        metric = ExactMatchMetric()

        predictions = ["Paris", "London", "Berlin"]
        references = ["Paris", "London", "Berlin"]

        result = metric.compute(predictions, references)

        assert "exact_match" in result
        assert result["exact_match"] == 1.0  # 100% match

    def test_exact_match_partial(self):
        """Test exact match with partial matches."""
        metric = ExactMatchMetric()

        predictions = ["Paris", "Madrid", "Berlin"]
        references = ["Paris", "London", "Berlin"]

        result = metric.compute(predictions, references)

        assert "exact_match" in result
        assert result["exact_match"] == pytest.approx(2 / 3, rel=0.01)  # 2 out of 3

    def test_exact_match_none(self):
        """Test exact match with no matches."""
        metric = ExactMatchMetric()

        predictions = ["Wrong1", "Wrong2", "Wrong3"]
        references = ["Right1", "Right2", "Right3"]

        result = metric.compute(predictions, references)

        assert result["exact_match"] == 0.0

    def test_exact_match_case_insensitive(self):
        """Test exact match is case insensitive (after normalization)."""
        metric = ExactMatchMetric()

        predictions = ["PARIS", "london", "BeRlIn"]
        references = ["Paris", "London", "Berlin"]

        result = metric.compute(predictions, references)

        # Should match after normalization
        assert result["exact_match"] == 1.0

    def test_exact_match_single(self):
        """Test exact match on single prediction."""
        metric = ExactMatchMetric()

        score = metric.compute_single("Paris", "Paris")

        assert score["exact_match"] == 1.0

    def test_exact_match_single_case_insensitive(self):
        """Test exact match on single prediction with different case."""
        metric = ExactMatchMetric()

        score = metric.compute_single("paris", "PARIS")

        assert score["exact_match"] == 1.0


class TestF1Score:
    """Test F1 score metric."""

    def test_f1_perfect(self):
        """Test F1 with perfect predictions."""
        metric = F1Metric()

        predictions = ["The capital of France is Paris"]
        references = ["The capital of France is Paris"]

        result = metric.compute(predictions, references)

        assert "f1" in result
        assert result["f1"] == 1.0

    def test_f1_partial_overlap(self):
        """Test F1 with partial token overlap."""
        metric = F1Metric()

        predictions = ["Paris is nice"]
        references = ["The capital is Paris"]

        result = metric.compute(predictions, references)

        # Should have some overlap (only "Paris" and "is" match)
        assert 0.0 < result["f1"] < 1.0

    def test_f1_no_overlap(self):
        """Test F1 with no overlap."""
        metric = F1Metric()

        predictions = ["Berlin"]
        references = ["Paris"]

        result = metric.compute(predictions, references)

        assert result["f1"] == 0.0

    def test_f1_multiple_predictions(self):
        """Test F1 with multiple predictions."""
        metric = F1Metric()

        predictions = ["Paris", "London", "Berlin"]
        references = ["Paris", "London", "Berlin"]

        result = metric.compute(predictions, references)

        assert result["f1"] == 1.0

    def test_f1_single(self):
        """Test F1 on single prediction."""
        metric = F1Metric()

        score = metric.compute_single("Paris", "Paris")

        assert score["f1"] == 1.0


class TestMetricEdgeCases:
    """Test metric edge cases."""

    def test_empty_predictions(self):
        """Test metrics with empty predictions."""
        em_metric = ExactMatchMetric()
        f1_metric = F1Metric()

        predictions = []
        references = []

        em_result = em_metric.compute(predictions, references)
        f1_result = f1_metric.compute(predictions, references)

        # Should handle empty gracefully
        assert em_result["exact_match"] == 0.0
        assert f1_result["f1"] == 0.0

    def test_empty_string_prediction(self):
        """Test metrics with empty string predictions."""
        em_metric = ExactMatchMetric()
        f1_metric = F1Metric()

        predictions = ["", "Paris"]
        references = ["Paris", "Paris"]

        em_result = em_metric.compute(predictions, references)
        f1_result = f1_metric.compute(predictions, references)

        # First should fail, second should match
        assert em_result["exact_match"] == 0.5
        assert f1_result["f1"] == 0.5

    def test_whitespace_handling(self):
        """Test that metrics handle whitespace correctly."""
        em_metric = ExactMatchMetric()

        predictions = ["  Paris  ", "London\n", "\tBerlin"]
        references = ["Paris", "London", "Berlin"]

        result = em_metric.compute(predictions, references)

        # Should match after whitespace normalization
        assert result["exact_match"] == 1.0

    def test_punctuation_handling(self):
        """Test that metrics handle punctuation."""
        f1_metric = F1Metric()

        predictions = ["Paris."]
        references = ["Paris"]

        result = f1_metric.compute(predictions, references)

        # Should still have high F1 (punctuation is minor)
        assert result["f1"] > 0.5


class TestMetricConsistency:
    """Test metric consistency."""

    def test_compute_vs_compute_single(self):
        """Test that compute and compute_single give consistent results."""
        em_metric = ExactMatchMetric()

        prediction = "Paris"
        reference = "Paris"

        # Compute on single item
        batch_result = em_metric.compute([prediction], [reference])
        single_result = em_metric.compute_single(prediction, reference)

        # Should be the same
        assert batch_result["exact_match"] == single_result["exact_match"]

    def test_metric_deterministic(self):
        """Test that metrics are deterministic."""
        em_metric = ExactMatchMetric()
        f1_metric = F1Metric()

        predictions = ["Paris", "London", "Berlin"]
        references = ["Paris", "London", "Berlin"]

        # Compute twice
        em_result1 = em_metric.compute(predictions, references)
        em_result2 = em_metric.compute(predictions, references)
        f1_result1 = f1_metric.compute(predictions, references)
        f1_result2 = f1_metric.compute(predictions, references)

        # Should be exactly the same
        assert em_result1 == em_result2
        assert f1_result1 == f1_result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
