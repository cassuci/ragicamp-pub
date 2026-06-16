"""Tests for experiment state management.

Tests cover:
- State creation, saving, and loading
- Phase transitions and ordering
- Atomic write behavior
- Error handling
"""

import json
import time
from datetime import datetime

from ragicamp.state.experiment_state import PHASE_ORDER, ExperimentPhase, ExperimentState


class TestExperimentStateCreation:
    """Test state creation methods."""

    def test_new_creates_init_phase_state(self):
        """Test that new() creates a state in INIT phase."""
        state = ExperimentState.new(total_questions=100, metrics=["exact_match", "f1"])

        assert state.phase == ExperimentPhase.INIT
        assert state.total_questions == 100
        assert state.predictions_complete == 0
        assert state.metrics_requested == ["exact_match", "f1"]
        assert state.metrics_computed == []
        assert state.error is None
        assert state.started_at
        assert state.updated_at
        assert state.started_at == state.updated_at  # Should be same at creation

    def test_new_with_defaults(self):
        """Test new() with default parameters."""
        state = ExperimentState.new()

        assert state.phase == ExperimentPhase.INIT
        assert state.total_questions == 0
        assert state.metrics_requested == []

    def test_new_timestamps_are_iso_format(self):
        """Test that timestamps are valid ISO format."""
        state = ExperimentState.new()

        # Should parse without error
        datetime.fromisoformat(state.started_at)
        datetime.fromisoformat(state.updated_at)


class TestExperimentStatePersistence:
    """Test state saving and loading."""

    def test_save_and_load_round_trip(self, tmp_path):
        """Test that save/load preserves all fields."""
        state_path = tmp_path / "state.json"
        original_state = ExperimentState.new(total_questions=50, metrics=["bertscore", "bleurt"])
        original_state.predictions_complete = 25
        original_state.metrics_computed = ["bertscore"]

        original_state.save(state_path)
        loaded_state = ExperimentState.load(state_path)

        assert loaded_state.phase == original_state.phase
        assert loaded_state.started_at == original_state.started_at
        assert loaded_state.total_questions == original_state.total_questions
        assert loaded_state.predictions_complete == original_state.predictions_complete
        assert loaded_state.metrics_requested == original_state.metrics_requested
        assert loaded_state.metrics_computed == original_state.metrics_computed
        assert loaded_state.error == original_state.error

    def test_save_uses_atomic_write(self, tmp_path):
        """Test that save() uses atomic write (temp file doesn't remain)."""
        state_path = tmp_path / "state.json"
        temp_path = state_path.with_suffix(".tmp")
        state = ExperimentState.new()

        state.save(state_path)

        # Main file should exist
        assert state_path.exists()
        # Temp file should not exist (was renamed)
        assert not temp_path.exists()

    def test_save_updates_timestamp(self, tmp_path):
        """Test that save() updates the updated_at timestamp."""
        state_path = tmp_path / "state.json"
        state = ExperimentState.new()
        original_updated_at = state.updated_at

        # Small sleep to ensure timestamp difference
        time.sleep(0.01)

        state.save(state_path)

        # updated_at should have changed
        assert state.updated_at != original_updated_at

    def test_load_handles_missing_optional_fields(self, tmp_path):
        """Test load() gracefully handles missing optional fields."""
        state_path = tmp_path / "state.json"

        # Write minimal JSON (old format without optional fields)
        minimal_data = {
            "phase": "init",
            "started_at": "2026-02-11T10:00:00",
            "updated_at": "2026-02-11T10:00:00",
        }
        with open(state_path, "w") as f:
            json.dump(minimal_data, f)

        loaded_state = ExperimentState.load(state_path)

        assert loaded_state.phase == ExperimentPhase.INIT
        assert loaded_state.total_questions == 0
        assert loaded_state.predictions_complete == 0
        assert loaded_state.metrics_computed == []
        assert loaded_state.metrics_requested == []
        assert loaded_state.error is None

    def test_round_trip_with_metrics_lists(self, tmp_path):
        """Test round-trip with non-empty metrics_computed and metrics_requested."""
        state_path = tmp_path / "state.json"
        state = ExperimentState.new(metrics=["exact_match", "bertscore", "bleurt"])
        state.metrics_computed = ["exact_match", "bertscore"]
        state.advance_to(ExperimentPhase.COMPUTING_METRICS)

        state.save(state_path)
        loaded = ExperimentState.load(state_path)

        assert loaded.metrics_requested == ["exact_match", "bertscore", "bleurt"]
        assert loaded.metrics_computed == ["exact_match", "bertscore"]


class TestPhaseTransitions:
    """Test phase transition methods."""

    def test_advance_to_updates_phase_and_timestamp(self):
        """Test that advance_to() updates phase and updated_at."""
        state = ExperimentState.new()
        original_updated_at = state.updated_at

        time.sleep(0.01)  # Ensure timestamp difference
        state.advance_to(ExperimentPhase.GENERATING)

        assert state.phase == ExperimentPhase.GENERATING
        assert state.updated_at != original_updated_at

    def test_set_error_sets_phase_to_failed(self):
        """Test that set_error() sets phase to FAILED and stores error."""
        state = ExperimentState.new()
        state.advance_to(ExperimentPhase.GENERATING)

        error_msg = "CUDA out of memory"
        state.set_error(error_msg)

        assert state.phase == ExperimentPhase.FAILED
        assert state.error == error_msg

    def test_set_error_updates_timestamp(self):
        """Test that set_error() updates the timestamp."""
        state = ExperimentState.new()
        original_updated_at = state.updated_at

        time.sleep(0.01)
        state.set_error("Test error")

        assert state.updated_at != original_updated_at


class TestPhaseOrdering:
    """Test phase comparison methods."""

    def test_is_at_least_same_phase(self):
        """Test is_at_least() returns True for same phase."""
        state = ExperimentState.new()
        state.advance_to(ExperimentPhase.GENERATING)

        assert state.is_at_least(ExperimentPhase.GENERATING)

    def test_is_at_least_earlier_phase(self):
        """Test is_at_least() returns True when current is later than target."""
        state = ExperimentState.new()
        state.advance_to(ExperimentPhase.GENERATING)

        assert state.is_at_least(ExperimentPhase.INIT)

    def test_is_at_least_later_phase(self):
        """Test is_at_least() returns False when current is earlier than target."""
        state = ExperimentState.new()  # INIT phase

        assert not state.is_at_least(ExperimentPhase.GENERATING)

    def test_is_at_least_complete_is_at_least_all_normal_phases(self):
        """Test that COMPLETE phase is at least all normal phases."""
        state = ExperimentState.new()
        state.advance_to(ExperimentPhase.COMPLETE)

        assert state.is_at_least(ExperimentPhase.INIT)
        assert state.is_at_least(ExperimentPhase.GENERATING)
        assert state.is_at_least(ExperimentPhase.GENERATED)
        assert state.is_at_least(ExperimentPhase.COMPUTING_METRICS)
        assert state.is_at_least(ExperimentPhase.COMPLETE)

    def test_is_past_next_phase(self):
        """Test is_past() returns True when current phase is past target."""
        state = ExperimentState.new()
        state.advance_to(ExperimentPhase.GENERATING)

        assert state.is_past(ExperimentPhase.INIT)

    def test_is_past_same_phase(self):
        """Test is_past() returns False for same phase."""
        state = ExperimentState.new()  # INIT

        assert not state.is_past(ExperimentPhase.INIT)

    def test_is_past_complete_past_computing_metrics(self):
        """Test that COMPLETE is past COMPUTING_METRICS."""
        state = ExperimentState.new()
        state.advance_to(ExperimentPhase.COMPLETE)

        assert state.is_past(ExperimentPhase.COMPUTING_METRICS)


class TestFailedPhaseOrdering:
    """Test ordering behavior of FAILED phase (order -1)."""

    def test_failed_phase_has_negative_order(self):
        """Test that FAILED phase has order -1."""
        assert PHASE_ORDER[ExperimentPhase.FAILED] == -1

    def test_is_at_least_failed_not_at_least_init(self):
        """Test that FAILED is not at least INIT (order -1 < 0)."""
        state = ExperimentState.new()
        state.set_error("Test failure")

        assert state.phase == ExperimentPhase.FAILED
        assert not state.is_at_least(ExperimentPhase.INIT)

    def test_is_past_failed_not_past_any_normal_phase(self):
        """Test that FAILED is not past any normal phase (order -1)."""
        state = ExperimentState.new()
        state.set_error("Test failure")

        assert not state.is_past(ExperimentPhase.INIT)
        assert not state.is_past(ExperimentPhase.GENERATING)
        assert not state.is_past(ExperimentPhase.GENERATED)
        assert not state.is_past(ExperimentPhase.COMPUTING_METRICS)
        assert not state.is_past(ExperimentPhase.COMPLETE)

    def test_generating_is_at_least_init_not_vice_versa(self):
        """Test phase ordering: GENERATING >= INIT but not INIT >= GENERATING."""
        state_generating = ExperimentState.new()
        state_generating.advance_to(ExperimentPhase.GENERATING)

        state_init = ExperimentState.new()

        # GENERATING is at least INIT
        assert state_generating.is_at_least(ExperimentPhase.INIT)
        # INIT is not at least GENERATING
        assert not state_init.is_at_least(ExperimentPhase.GENERATING)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_phase_enum_values(self):
        """Test that phase enum values match expected strings."""
        assert ExperimentPhase.INIT.value == "init"
        assert ExperimentPhase.GENERATING.value == "generating"
        assert ExperimentPhase.GENERATED.value == "generated"
        assert ExperimentPhase.COMPUTING_METRICS.value == "computing_metrics"
        assert ExperimentPhase.COMPLETE.value == "complete"
        assert ExperimentPhase.FAILED.value == "failed"

    def test_save_creates_valid_json(self, tmp_path):
        """Test that saved state is valid JSON with expected structure."""
        state_path = tmp_path / "state.json"
        state = ExperimentState.new(total_questions=10, metrics=["exact_match"])
        state.predictions_complete = 5

        state.save(state_path)

        with open(state_path) as f:
            data = json.load(f)

        assert data["phase"] == "init"
        assert data["total_questions"] == 10
        assert data["predictions_complete"] == 5
        assert data["metrics_requested"] == ["exact_match"]
        assert data["metrics_computed"] == []
        assert data["error"] is None
        assert "started_at" in data
        assert "updated_at" in data
