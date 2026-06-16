"""Tests for execution phase handlers (GenerationHandler, MetricsHandler).

Tests cover:
- Phase routing (can_handle)
- Generation with agent mocking
- Resume from existing predictions
- Dedup guard on duplicate results
- Metrics computation with mocked compute_metrics_batched
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from ragicamp.agents.base import Agent, AgentResult, Query, Step
from ragicamp.execution.phases.base import ExecutionContext
from ragicamp.execution.phases.generation import GenerationHandler
from ragicamp.execution.phases.metrics_phase import MetricsHandler
from ragicamp.spec.experiment import ExperimentSpec
from ragicamp.state.experiment_state import ExperimentPhase, ExperimentState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(name: str = "test_exp") -> ExperimentSpec:
    """Create a minimal ExperimentSpec for testing."""
    return ExperimentSpec(
        name=name,
        exp_type="direct",
        model="mock-model",
        dataset="mock-ds",
        prompt="default",
        metrics=["exact_match"],
    )


def _make_state() -> ExperimentState:
    """Create a fresh ExperimentState in GENERATING phase."""
    now = datetime.now().isoformat()
    return ExperimentState(
        phase=ExperimentPhase.GENERATING,
        started_at=now,
        updated_at=now,
        total_questions=3,
    )


def _write_questions(path: Path, questions: list[dict]) -> None:
    """Write a questions.json file."""
    data = {"experiment": "test_exp", "questions": questions, "count": len(questions)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _default_questions() -> list[dict]:
    return [
        {"idx": 0, "question": "What is 1+1?", "expected": ["2"]},
        {"idx": 1, "question": "What is 2+2?", "expected": ["4"]},
        {"idx": 2, "question": "What is 3+3?", "expected": ["6"]},
    ]


class FakeAgent(Agent):
    """Minimal agent that returns canned answers."""

    def __init__(self, answers: dict[int, str] | None = None):
        super().__init__(name="fake_agent")
        self.answers = answers or {}
        self.received_queries: list[Query] = []

    def run(
        self,
        queries: list[Query],
        *,
        on_result=None,
        checkpoint_path=None,
        show_progress=True,
    ) -> list[AgentResult]:
        self.received_queries.extend(queries)
        results = []
        for q in queries:
            answer = self.answers.get(q.idx, "default_answer")
            result = AgentResult(
                query=q,
                answer=answer,
                steps=[Step(type="generate", timing_ms=1.0)],
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
        return results


# ===========================================================================
# GenerationHandler Tests
# ===========================================================================


class TestGenerationHandlerCanHandle:
    """Tests for GenerationHandler.can_handle()."""

    def test_can_handle_generating_phase(self):
        handler = GenerationHandler()
        assert handler.can_handle(ExperimentPhase.GENERATING) is True

    def test_cannot_handle_other_phases(self):
        handler = GenerationHandler()
        for phase in ExperimentPhase:
            if phase != ExperimentPhase.GENERATING:
                assert handler.can_handle(phase) is False, f"Should not handle {phase}"


class TestGenerationHandlerExecute:
    """Tests for GenerationHandler.execute()."""

    def test_execute_runs_agent_and_saves_predictions(self, tmp_path):
        """Agent runs on all queries and predictions are saved to disk."""
        spec = _make_spec()
        state = _make_state()
        questions = _default_questions()
        _write_questions(tmp_path / "questions.json", questions)

        agent = FakeAgent(answers={0: "2", 1: "4", 2: "6"})
        context = ExecutionContext(output_path=tmp_path, agent=agent)

        handler = GenerationHandler()
        new_state = handler.execute(spec, state, context)

        # All 3 queries should have been sent to the agent
        assert len(agent.received_queries) == 3

        # Predictions file should exist and contain 3 predictions
        predictions_path = tmp_path / "predictions.json"
        assert predictions_path.exists()
        with open(predictions_path) as f:
            data = json.load(f)
        assert len(data["predictions"]) == 3

        # State should track completion count
        assert new_state.predictions_complete == 3

        # Verify prediction content
        preds_by_idx = {p["idx"]: p for p in data["predictions"]}
        assert preds_by_idx[0]["prediction"] == "2"
        assert preds_by_idx[1]["prediction"] == "4"
        assert preds_by_idx[2]["prediction"] == "6"

    def test_execute_resumes_from_existing_predictions(self, tmp_path):
        """When predictions already exist, only pending queries are sent to the agent."""
        spec = _make_spec()
        state = _make_state()
        questions = _default_questions()
        _write_questions(tmp_path / "questions.json", questions)

        # Pre-populate predictions for idx 0
        existing_predictions = {
            "experiment": "test_exp",
            "predictions": [
                {
                    "idx": 0,
                    "question": "What is 1+1?",
                    "prediction": "2",
                    "expected": ["2"],
                    "metrics": {},
                }
            ],
        }
        with open(tmp_path / "predictions.json", "w") as f:
            json.dump(existing_predictions, f)

        agent = FakeAgent(answers={1: "4", 2: "6"})
        context = ExecutionContext(output_path=tmp_path, agent=agent)

        handler = GenerationHandler()
        new_state = handler.execute(spec, state, context)

        # Only idx 1 and 2 should have been sent to the agent
        received_idxs = {q.idx for q in agent.received_queries}
        assert received_idxs == {1, 2}

        # Total predictions should be 3
        with open(tmp_path / "predictions.json") as f:
            data = json.load(f)
        assert len(data["predictions"]) == 3
        assert new_state.predictions_complete == 3

    def test_execute_skips_when_all_complete(self, tmp_path):
        """When all predictions exist, the agent should not be called."""
        spec = _make_spec()
        state = _make_state()
        questions = _default_questions()
        _write_questions(tmp_path / "questions.json", questions)

        # Pre-populate all predictions
        all_preds = {
            "experiment": "test_exp",
            "predictions": [
                {
                    "idx": i,
                    "question": q["question"],
                    "prediction": str(i),
                    "expected": q["expected"],
                    "metrics": {},
                }
                for i, q in enumerate(questions)
            ],
        }
        with open(tmp_path / "predictions.json", "w") as f:
            json.dump(all_preds, f)

        agent = FakeAgent()
        context = ExecutionContext(output_path=tmp_path, agent=agent)

        handler = GenerationHandler()
        handler.execute(spec, state, context)

        # Agent should NOT have been called
        assert len(agent.received_queries) == 0

    def test_on_result_dedup_guard(self, tmp_path):
        """Submitting the same idx twice should only record one prediction."""
        spec = _make_spec()
        state = _make_state()
        questions = [{"idx": 0, "question": "What is 1+1?", "expected": ["2"]}]
        _write_questions(tmp_path / "questions.json", questions)

        # Create an agent that fires on_result twice for the same idx
        class DuplicatingAgent(Agent):
            def __init__(self):
                super().__init__(name="dup_agent")

            def run(self, queries, *, on_result=None, checkpoint_path=None, show_progress=True):
                results = []
                for q in queries:
                    result = AgentResult(
                        query=q,
                        answer="2",
                        steps=[Step(type="generate", timing_ms=1.0)],
                    )
                    results.append(result)
                    if on_result is not None:
                        on_result(result)
                        # Fire a duplicate
                        on_result(result)
                return results

        agent = DuplicatingAgent()
        context = ExecutionContext(output_path=tmp_path, agent=agent)

        handler = GenerationHandler()
        handler.execute(spec, state, context)

        with open(tmp_path / "predictions.json") as f:
            data = json.load(f)

        # Should have exactly 1 prediction, not 2
        assert len(data["predictions"]) == 1

    def test_checkpoint_cleaned_up_after_execution(self, tmp_path):
        """The agent checkpoint file should be removed after successful execution."""
        spec = _make_spec()
        state = _make_state()
        questions = [{"idx": 0, "question": "Q?", "expected": ["A"]}]
        _write_questions(tmp_path / "questions.json", questions)

        # Create a checkpoint file that should be cleaned up
        checkpoint_path = tmp_path / "agent_checkpoint.json"
        checkpoint_path.write_text("{}")

        agent = FakeAgent(answers={0: "A"})
        context = ExecutionContext(output_path=tmp_path, agent=agent)

        handler = GenerationHandler()
        handler.execute(spec, state, context)

        assert not checkpoint_path.exists(), "Checkpoint should be removed after execution"


# ===========================================================================
# MetricsHandler Tests
# ===========================================================================


class TestMetricsHandlerCanHandle:
    """Tests for MetricsHandler.can_handle()."""

    def test_can_handle_metrics_phase(self):
        handler = MetricsHandler()
        assert handler.can_handle(ExperimentPhase.COMPUTING_METRICS) is True

    def test_cannot_handle_other_phases(self):
        handler = MetricsHandler()
        for phase in ExperimentPhase:
            if phase != ExperimentPhase.COMPUTING_METRICS:
                assert handler.can_handle(phase) is False, f"Should not handle {phase}"


class TestMetricsHandlerExecute:
    """Tests for MetricsHandler.execute()."""

    def _write_predictions(self, path: Path, predictions: list[dict]) -> None:
        """Write a predictions.json file."""
        data = {"experiment": "test_exp", "predictions": predictions}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def _default_predictions(self) -> list[dict]:
        return [
            {"idx": 0, "question": "What is 1+1?", "prediction": "2", "expected": "2"},
            {"idx": 1, "question": "What is 2+2?", "prediction": "4", "expected": "4"},
        ]

    @patch("ragicamp.metrics.compute_metrics_batched")
    def test_execute_computes_metrics(self, mock_compute, tmp_path):
        """Metrics are computed and saved to predictions file."""
        predictions = self._default_predictions()
        self._write_predictions(tmp_path / "predictions.json", predictions)

        # Mock compute_metrics_batched return value
        mock_compute.return_value = (
            {"exact_match": 1.0},  # aggregate_results
            {"exact_match": [1.0, 1.0]},  # per_item_metrics
            ["exact_match"],  # computed
            [],  # failed
            {"exact_match": 0.01},  # timings
        )

        now = datetime.now().isoformat()
        state = ExperimentState(
            phase=ExperimentPhase.COMPUTING_METRICS,
            started_at=now,
            updated_at=now,
            total_questions=2,
            predictions_complete=2,
        )

        mock_metric = MagicMock()
        mock_metric.name = "exact_match"
        context = ExecutionContext(
            output_path=tmp_path,
            metrics=[mock_metric],
        )

        handler = MetricsHandler()
        handler.execute(_make_spec(), state, context)

        # Verify compute_metrics_batched was called with correct args
        mock_compute.assert_called_once()
        call_kwargs = mock_compute.call_args
        assert call_kwargs[1]["predictions"] == ["2", "4"]
        assert call_kwargs[1]["references"] == ["2", "4"]
        assert call_kwargs[1]["questions"] == ["What is 1+1?", "What is 2+2?"]

        # Verify predictions file was updated with per-item metrics
        with open(tmp_path / "predictions.json") as f:
            data = json.load(f)
        assert data["predictions"][0]["metrics"]["exact_match"] == 1.0
        assert data["predictions"][1]["metrics"]["exact_match"] == 1.0

        # Verify aggregate metrics
        assert data["aggregate_metrics"]["exact_match"] == 1.0

        # Verify timing info
        assert data["metric_timings"]["exact_match"] == 0.01

    @patch("ragicamp.metrics.compute_metrics_batched")
    def test_execute_skips_already_computed(self, mock_compute, tmp_path):
        """Already-computed metrics are passed to compute_metrics_batched for skipping."""
        predictions = self._default_predictions()
        self._write_predictions(tmp_path / "predictions.json", predictions)

        mock_compute.return_value = (
            {},  # no new aggregate results
            {},  # no new per-item metrics
            [],  # nothing newly computed
            [],  # nothing failed
            {},  # no timings
        )

        now = datetime.now().isoformat()
        state = ExperimentState(
            phase=ExperimentPhase.COMPUTING_METRICS,
            started_at=now,
            updated_at=now,
            total_questions=2,
            predictions_complete=2,
            metrics_computed=["exact_match"],
        )

        mock_metric = MagicMock()
        mock_metric.name = "exact_match"
        context = ExecutionContext(
            output_path=tmp_path,
            metrics=[mock_metric],
        )

        handler = MetricsHandler()
        handler.execute(_make_spec(), state, context)

        # Verify already_computed was passed through
        call_kwargs = mock_compute.call_args
        assert call_kwargs[1]["already_computed"] == ["exact_match"]

    @patch("ragicamp.metrics.compute_metrics_batched")
    def test_execute_calls_on_metric_complete_callback(self, mock_compute, tmp_path):
        """The on_metric_complete callback updates state.metrics_computed."""
        predictions = self._default_predictions()
        self._write_predictions(tmp_path / "predictions.json", predictions)

        # Capture the callback so we can invoke it manually
        captured_callback = None

        def side_effect(**kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("on_metric_complete")
            # Simulate calling the callback
            if captured_callback:
                captured_callback("exact_match")
            return (
                {"exact_match": 1.0},
                {"exact_match": [1.0, 1.0]},
                ["exact_match"],
                [],
                {"exact_match": 0.01},
            )

        mock_compute.side_effect = side_effect

        now = datetime.now().isoformat()
        state = ExperimentState(
            phase=ExperimentPhase.COMPUTING_METRICS,
            started_at=now,
            updated_at=now,
            total_questions=2,
            predictions_complete=2,
        )

        # State needs to be saveable, so create state.json path
        state_path = tmp_path / "state.json"
        state.save(state_path)

        mock_metric = MagicMock()
        mock_metric.name = "exact_match"
        context = ExecutionContext(
            output_path=tmp_path,
            metrics=[mock_metric],
        )

        handler = MetricsHandler()
        new_state = handler.execute(_make_spec(), state, context)

        # The callback should have added exact_match to metrics_computed
        assert "exact_match" in new_state.metrics_computed

    @patch("ragicamp.metrics.compute_metrics_batched")
    def test_execute_merges_with_existing_aggregate_metrics(self, mock_compute, tmp_path):
        """New metrics should merge with existing aggregate_metrics."""
        predictions = self._default_predictions()
        # Write predictions with pre-existing aggregate metrics
        data = {
            "experiment": "test_exp",
            "predictions": predictions,
            "aggregate_metrics": {"old_metric": 0.5},
        }
        with open(tmp_path / "predictions.json", "w") as f:
            json.dump(data, f)

        mock_compute.return_value = (
            {"new_metric": 0.9},
            {"new_metric": [0.9, 0.9]},
            ["new_metric"],
            [],
            {"new_metric": 0.02},
        )

        now = datetime.now().isoformat()
        state = ExperimentState(
            phase=ExperimentPhase.COMPUTING_METRICS,
            started_at=now,
            updated_at=now,
            total_questions=2,
            predictions_complete=2,
        )

        mock_metric = MagicMock()
        mock_metric.name = "new_metric"
        context = ExecutionContext(
            output_path=tmp_path,
            metrics=[mock_metric],
        )

        handler = MetricsHandler()
        handler.execute(_make_spec(), state, context)

        with open(tmp_path / "predictions.json") as f:
            result = json.load(f)

        # Both old and new metrics should be present
        assert result["aggregate_metrics"]["old_metric"] == 0.5
        assert result["aggregate_metrics"]["new_metric"] == 0.9
