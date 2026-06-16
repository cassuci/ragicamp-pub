"""Generation phase handler - runs agent with the new clean interface."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragicamp.agents.base import AgentResult, Query
from ragicamp.core.logging import get_logger
from ragicamp.execution.phases.base import ExecutionContext, PhaseHandler
from ragicamp.state import ExperimentPhase, ExperimentState
from ragicamp.utils.experiment_io import ExperimentIO

if TYPE_CHECKING:
    from ragicamp.spec import ExperimentSpec

logger = get_logger(__name__)


class GenerationHandler(PhaseHandler):
    """Handler for the GENERATING phase.

    Simply calls agent.run(queries) - the agent manages everything:
    - Model loading/unloading
    - Batching strategy
    - GPU optimization
    - Checkpointing
    """

    def can_handle(self, phase: ExperimentPhase) -> bool:
        return phase == ExperimentPhase.GENERATING

    def execute(
        self,
        spec: "ExperimentSpec",
        state: ExperimentState,
        context: ExecutionContext,
    ) -> ExperimentState:
        """Run the agent on all pending questions."""
        import time as _time

        _phase_t0 = _time.perf_counter()
        logger.info("Phase: GENERATING")

        questions_path = context.output_path / "questions.json"
        predictions_path = context.output_path / "predictions.json"
        checkpoint_path = context.output_path / "agent_checkpoint.json"

        # Load questions
        with open(questions_path) as f:
            q_data = json.load(f)

        # Convert to Query objects
        queries = [
            Query(
                idx=q["idx"],
                text=q["question"],
                expected=q.get("expected"),
            )
            for q in q_data["questions"]
        ]

        # Load existing predictions to skip completed
        completed_idx: set[int] = set()
        predictions_data: dict[str, Any] = {"experiment": spec.name, "predictions": []}

        if predictions_path.exists():
            with open(predictions_path) as f:
                predictions_data = json.load(f)
            completed_idx = {p["idx"] for p in predictions_data["predictions"]}
            logger.info("Resuming: %d/%d complete", len(completed_idx), len(queries))

        pending = [q for q in queries if q.idx not in completed_idx]

        if not pending:
            logger.info("All predictions complete")
            return state

        # C1 fix: save every N results instead of every single one
        _save_interval = 10
        _unsaved_count = 0

        # Callback to save results incrementally (with dedup guard)
        def on_result(result: AgentResult) -> None:
            nonlocal _unsaved_count
            if result.query.idx in completed_idx:
                return  # B4 fix: skip duplicate on crash-resume
            completed_idx.add(result.query.idx)
            pred = self._result_to_prediction(result)
            predictions_data["predictions"].append(pred)
            state.predictions_complete = len(predictions_data["predictions"])
            _unsaved_count += 1
            if _unsaved_count >= _save_interval:
                self._save_predictions(predictions_data, predictions_path)
                _unsaved_count = 0

        # Run the agent - it manages its own resources
        logger.info("Running agent on %d queries", len(pending))
        results = context.agent.run(
            pending,
            on_result=on_result,
            checkpoint_path=checkpoint_path,
            show_progress=True,
        )

        # Final save
        state.predictions_complete = len(predictions_data["predictions"])
        self._save_predictions(predictions_data, predictions_path)

        # Clean up checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        _phase_s = _time.perf_counter() - _phase_t0
        logger.info("Generated %d predictions in %.1fs", len(results), _phase_s)
        return state

    def _result_to_prediction(self, result: AgentResult) -> dict[str, Any]:
        """Convert AgentResult to prediction format.

        Uses AgentResult.to_dict() which properly serializes:
        - Steps with timing info
        - Retrieved docs with scores and content
        - Metadata
        """
        d = result.to_dict(include_content=True, max_content_len=500)
        # Rename fields to match PredictionRecord format
        d["prediction"] = d.pop("answer")
        d["metrics"] = {}  # Filled in metrics phase
        return d

    def _save_predictions(self, data: dict[str, Any], path: Path) -> None:
        """Save predictions atomically."""
        io = ExperimentIO(path.parent)
        io.save_predictions(data)
