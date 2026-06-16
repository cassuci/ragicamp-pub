"""Integration tests for context-aware metrics in the pipeline.

Covers:
- contexts passed correctly to context metrics
- contexts filtered by valid_mask (error predictions)
- context metrics use non-expanded data with multi-ref
- non-context metrics still get expanded data with multi-ref
- empty contexts (direct_llm) handled gracefully
- MetricsHandler extracts contexts from retrieved_docs
- AsyncAPIMetric._last_per_item fix
"""

import json
from unittest.mock import MagicMock, patch

from ragicamp.metrics import compute_metrics_batched
from ragicamp.metrics.answer_in_context import AnswerInContextMetric
from ragicamp.metrics.base import Metric
from ragicamp.metrics.context_recall import ContextRecallMetric

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SpyMetric(Metric):
    """Captures the kwargs passed to compute() for assertion."""

    def __init__(self, name: str = "spy"):
        super().__init__(name=name)
        self.last_kwargs: dict = {}
        self.last_predictions: list = []
        self.last_references: list = []

    def compute(self, predictions, references, **kwargs):
        self.last_predictions = list(predictions)
        self.last_references = list(references)
        self.last_kwargs = dict(kwargs)
        self._last_per_item = [1.0] * len(predictions)
        return {self.name: 1.0}


class SpyContextMetric(SpyMetric):
    """A spy metric whose name is in the context-metric set."""

    def __init__(self):
        super().__init__(name="faithfulness")


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestContextsPassedToContextMetrics:
    def test_contexts_forwarded(self):
        """Context metrics receive contexts kwarg."""
        spy = SpyContextMetric()
        compute_metrics_batched(
            metrics=[spy],
            predictions=["answer"],
            references=["ref"],
            contexts=[["doc1", "doc2"]],
        )
        assert spy.last_kwargs.get("contexts") == [["doc1", "doc2"]]

    def test_contexts_not_forwarded_to_non_context_metric(self):
        """Non-context metrics do NOT receive contexts."""
        spy = SpyMetric(name="exact_match")
        compute_metrics_batched(
            metrics=[spy],
            predictions=["answer"],
            references=["ref"],
            contexts=[["doc1"]],
        )
        assert "contexts" not in spy.last_kwargs


class TestContextsFilteredByValidMask:
    def test_error_predictions_filter_contexts(self):
        """Contexts are filtered alongside predictions when errors are excluded."""
        spy = SpyContextMetric()
        compute_metrics_batched(
            metrics=[spy],
            predictions=["good answer", "[ERROR: timeout]"],
            references=["ref1", "ref2"],
            contexts=[["good_doc"], ["error_doc"]],
        )
        # Only the first (valid) item should remain
        assert spy.last_predictions == ["good answer"]
        assert spy.last_kwargs["contexts"] == [["good_doc"]]


class TestContextMetricsNonExpanded:
    def test_multi_ref_context_metric_gets_original_data(self):
        """Context metrics get original (non-expanded) data, not expanded pairs."""
        spy = SpyContextMetric()
        compute_metrics_batched(
            metrics=[spy],
            predictions=["answer1", "answer2"],
            references=[["ref1a", "ref1b"], "ref2"],
            contexts=[["doc1"], ["doc2"]],
        )
        # Should get original 2-item lists, not expanded 3-item lists
        assert len(spy.last_predictions) == 2
        assert len(spy.last_references) == 2

    def test_multi_ref_non_context_metric_gets_expanded(self):
        """Non-context metrics get expanded data for multi-reference."""
        spy = SpyMetric(name="f1")
        compute_metrics_batched(
            metrics=[spy],
            predictions=["answer1", "answer2"],
            references=[["ref1a", "ref1b"], "ref2"],
        )
        # Should be expanded: 3 pairs (2 for first, 1 for second)
        assert len(spy.last_predictions) == 3


class TestEmptyContextsGraceful:
    def test_direct_llm_no_contexts(self):
        """When contexts=None (direct_llm), context metrics are skipped gracefully."""
        metric = AnswerInContextMetric()
        agg, per_item, computed, failed, timings = compute_metrics_batched(
            metrics=[metric],
            predictions=["answer"],
            references=["answer"],
            contexts=None,
        )
        assert "answer_in_context" not in agg
        assert "answer_in_context" in failed

    def test_empty_context_lists(self):
        """When all context lists are empty, metrics handle gracefully."""
        metric = ContextRecallMetric()
        agg, per_item, computed, failed, timings = compute_metrics_batched(
            metrics=[metric],
            predictions=["answer"],
            references=["The war ended in 1945."],
            contexts=[[]],
        )
        assert agg["context_recall"] == 0.0

    def test_real_answer_in_context(self):
        """End-to-end: AnswerInContextMetric through the pipeline."""
        metric = AnswerInContextMetric()
        agg, per_item, computed, failed, timings = compute_metrics_batched(
            metrics=[metric],
            predictions=["1945"],
            references=["1945"],
            contexts=[["World War II ended in 1945."]],
        )
        assert agg["answer_in_context"] == 1.0
        assert per_item["answer_in_context"] == [1.0]

    def test_real_context_recall(self):
        """End-to-end: ContextRecallMetric through the pipeline."""
        metric = ContextRecallMetric()
        agg, per_item, computed, failed, timings = compute_metrics_batched(
            metrics=[metric],
            predictions=["unused"],
            references=["The war ended in 1945. It was devastating."],
            contexts=[["The war ended in 1945. It was devastating and costly."]],
        )
        assert agg["context_recall"] == 1.0


# ---------------------------------------------------------------------------
# MetricsHandler integration
# ---------------------------------------------------------------------------


class TestMetricsHandlerContextExtraction:
    """Verify MetricsHandler extracts contexts from retrieved_docs."""

    @patch("ragicamp.metrics.compute_metrics_batched")
    def test_contexts_extracted_from_retrieved_docs(self, mock_compute, tmp_path):
        from datetime import datetime

        from ragicamp.execution.phases.base import ExecutionContext
        from ragicamp.execution.phases.metrics_phase import MetricsHandler
        from ragicamp.spec.experiment import ExperimentSpec
        from ragicamp.state.experiment_state import ExperimentPhase, ExperimentState

        predictions = [
            {
                "idx": 0,
                "question": "Q1?",
                "prediction": "A1",
                "expected": "A1",
                "retrieved_docs": [
                    {"content": "doc1 content", "score": 0.9},
                    {"content": "doc2 content", "score": 0.8},
                ],
            },
            {
                "idx": 1,
                "question": "Q2?",
                "prediction": "A2",
                "expected": "A2",
                "retrieved_docs": [],
            },
        ]
        data = {"experiment": "test", "predictions": predictions}
        pred_path = tmp_path / "predictions.json"
        with open(pred_path, "w") as f:
            json.dump(data, f)

        mock_compute.return_value = ({}, {}, [], [], {})

        now = datetime.now().isoformat()
        state = ExperimentState(
            phase=ExperimentPhase.COMPUTING_METRICS,
            started_at=now,
            updated_at=now,
            total_questions=2,
            predictions_complete=2,
        )

        spec = ExperimentSpec(name="test", exp_type="direct", model="m", dataset="d", prompt="p")
        mock_metric = MagicMock()
        mock_metric.name = "exact_match"
        ctx = ExecutionContext(output_path=tmp_path, metrics=[mock_metric])

        handler = MetricsHandler()
        handler.execute(spec, state, ctx)

        call_kwargs = mock_compute.call_args[1]
        assert call_kwargs["contexts"] == [
            ["doc1 content", "doc2 content"],
            [],
        ]

    @patch("ragicamp.metrics.compute_metrics_batched")
    def test_no_retrieved_docs_gives_none_contexts(self, mock_compute, tmp_path):
        """Predictions without retrieved_docs key produce contexts=None."""
        from datetime import datetime

        from ragicamp.execution.phases.base import ExecutionContext
        from ragicamp.execution.phases.metrics_phase import MetricsHandler
        from ragicamp.spec.experiment import ExperimentSpec
        from ragicamp.state.experiment_state import ExperimentPhase, ExperimentState

        predictions = [
            {
                "idx": 0,
                "question": "Q?",
                "prediction": "A",
                "expected": "A",
            },
        ]
        data = {"experiment": "test", "predictions": predictions}
        with open(tmp_path / "predictions.json", "w") as f:
            json.dump(data, f)

        mock_compute.return_value = ({}, {}, [], [], {})

        now = datetime.now().isoformat()
        state = ExperimentState(
            phase=ExperimentPhase.COMPUTING_METRICS,
            started_at=now,
            updated_at=now,
            total_questions=1,
            predictions_complete=1,
        )

        spec = ExperimentSpec(name="test", exp_type="direct", model="m", dataset="d", prompt="p")
        mock_metric = MagicMock()
        mock_metric.name = "exact_match"
        ctx = ExecutionContext(output_path=tmp_path, metrics=[mock_metric])

        handler = MetricsHandler()
        handler.execute(spec, state, ctx)

        call_kwargs = mock_compute.call_args[1]
        assert call_kwargs["contexts"] is None


# ---------------------------------------------------------------------------
# AsyncAPIMetric._last_per_item fix
# ---------------------------------------------------------------------------


class TestAsyncAPIMetricPerItemFix:
    """Verify that acompute() now populates _last_per_item."""

    def test_compute_with_details_has_per_item(self):
        """After the fix, compute_with_details() should return non-empty per_item."""

        from ragicamp.metrics.async_base import AsyncAPIMetric

        class FakeAsyncMetric(AsyncAPIMetric):
            async def acompute_single(self, prediction, reference, question=None, **kw):
                return {self.name: 0.75}

        metric = FakeAsyncMetric(name="fake_llm", max_concurrent=1, show_progress=False)
        detail = metric.compute_with_details(
            predictions=["pred1", "pred2"],
            references=["ref1", "ref2"],
        )
        assert detail.per_item == [0.75, 0.75]
        assert detail.aggregate == 0.75
