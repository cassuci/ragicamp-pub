"""Tests for experiment_io.py — atomic writes and centralized I/O."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from ragicamp.utils.experiment_io import ExperimentIO, atomic_write_json


class TestExperimentIO:
    """Test ExperimentIO class for managing experiment artifacts."""

    def test_property_accessors(self, tmp_path: Path):
        """Test that property accessors return correct paths."""
        io = ExperimentIO(tmp_path)
        assert io.output_path == tmp_path
        assert io.predictions_path == tmp_path / "predictions.json"
        assert io.results_path == tmp_path / "results.json"
        assert io.questions_path == tmp_path / "questions.json"
        assert io.metadata_path == tmp_path / "metadata.json"
        assert io.state_path == tmp_path / "state.json"

    def test_atomic_write_creates_file_no_tmp_remains(self, tmp_path: Path):
        """Test atomic write creates final file and removes temp file."""
        io = ExperimentIO(tmp_path)
        io.ensure_dir()
        data = {"key": "value", "count": 42}
        target = tmp_path / "test.json"

        # Call atomic write directly
        io._atomic_write(data, target)

        # File exists, temp does not
        assert target.exists()
        assert not target.with_suffix(".tmp").exists()

        # Content is correct
        with open(target) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_atomic_write_creates_parent_directories(self, tmp_path: Path):
        """Test atomic write creates parent directories."""
        io = ExperimentIO(tmp_path)
        nested_path = tmp_path / "nested" / "deep" / "file.json"
        data = {"nested": True}

        io._atomic_write(data, nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_predictions_save_load_roundtrip(self, tmp_path: Path):
        """Test predictions save/load round-trip preserves data."""
        io = ExperimentIO(tmp_path)
        io.ensure_dir()

        predictions_data = {
            "predictions": [
                {"idx": 0, "question": "Q1", "answer": "A1"},
                {"idx": 1, "question": "Q2", "answer": "A2"},
            ],
            "metadata": {"model": "test-model"},
        }

        io.save_predictions(predictions_data)
        loaded = io.load_predictions()

        assert loaded == predictions_data
        assert len(loaded["predictions"]) == 2

    def test_questions_save_load_roundtrip(self, tmp_path: Path):
        """Test questions save/load round-trip — verify wrapping structure."""
        io = ExperimentIO(tmp_path)
        io.ensure_dir()

        questions = [
            {"idx": 0, "question": "What is RAG?", "expected": "Retrieval-Augmented"},
            {"idx": 1, "question": "What is FAISS?", "expected": "Facebook AI"},
        ]
        experiment_name = "test_experiment"

        io.save_questions(questions, experiment_name)

        # Load raw file to verify wrapping structure
        with open(io.questions_path) as f:
            raw_data = json.load(f)

        assert raw_data["experiment"] == experiment_name
        assert raw_data["questions"] == questions
        assert raw_data["count"] == 2

        # Load via method returns only questions list
        loaded_questions = io.load_questions()
        assert loaded_questions == questions

    def test_result_save_load_roundtrip_dict(self, tmp_path: Path):
        """Test result save/load round-trip using dict method."""
        io = ExperimentIO(tmp_path)
        io.ensure_dir()

        result_data = {
            "name": "test_exp",
            "metrics": {"exact_match": 0.75, "bertscore": 0.82},
            "predictions_count": 100,
        }

        io.save_result_dict(result_data)
        loaded = io.load_result()

        assert loaded == result_data

    def test_metadata_save_adds_timestamp_when_missing(self, tmp_path: Path):
        """Test metadata save adds timestamp when not present."""
        io = ExperimentIO(tmp_path)
        io.ensure_dir()

        metadata = {"model": "test-model", "dataset": "test-dataset"}
        io.save_metadata(metadata)

        loaded = io.load_metadata()

        assert "timestamp" in loaded
        assert loaded["model"] == "test-model"
        assert loaded["dataset"] == "test-dataset"

        # Verify timestamp is valid ISO format
        datetime.fromisoformat(loaded["timestamp"])

    def test_metadata_save_preserves_existing_timestamp(self, tmp_path: Path):
        """Test metadata save preserves existing timestamp."""
        io = ExperimentIO(tmp_path)
        io.ensure_dir()

        original_timestamp = "2026-02-11T10:00:00.123456"
        metadata = {
            "model": "test-model",
            "timestamp": original_timestamp,
        }

        io.save_metadata(metadata)
        loaded = io.load_metadata()

        assert loaded["timestamp"] == original_timestamp

    def test_existence_checks(self, tmp_path: Path):
        """Test existence checks return False when files don't exist, True after save."""
        io = ExperimentIO(tmp_path)
        io.ensure_dir()

        # All should be False initially
        assert not io.predictions_exist()
        assert not io.result_exists()
        assert not io.questions_exist()
        assert not io.metadata_exists()

        # Save predictions
        io.save_predictions({"predictions": []})
        assert io.predictions_exist()
        assert not io.result_exists()  # Others still False

        # Save result
        io.save_result_dict({"name": "test"})
        assert io.result_exists()

        # Save questions
        io.save_questions([{"idx": 0, "question": "Q"}], "test_exp")
        assert io.questions_exist()

        # Save metadata
        io.save_metadata({"model": "test"})
        assert io.metadata_exists()

    def test_get_completed_indices_returns_correct_set(self, tmp_path: Path):
        """Test get_completed_indices returns correct set of indices."""
        io = ExperimentIO(tmp_path)
        io.ensure_dir()

        predictions_data = {
            "predictions": [
                {"idx": 0, "answer": "A1"},
                {"idx": 2, "answer": "A2"},
                {"idx": 5, "answer": "A3"},
            ]
        }

        io.save_predictions(predictions_data)
        completed = io.get_completed_indices()

        assert completed == {0, 2, 5}

    def test_get_completed_indices_returns_empty_when_no_predictions(self, tmp_path: Path):
        """Test get_completed_indices returns empty set when no predictions file."""
        io = ExperimentIO(tmp_path)
        io.ensure_dir()

        completed = io.get_completed_indices()
        assert completed == set()

    def test_get_completed_indices_handles_missing_idx(self, tmp_path: Path):
        """Test get_completed_indices uses enumerate index when idx missing."""
        io = ExperimentIO(tmp_path)
        io.ensure_dir()

        # Predictions without explicit idx field
        predictions_data = {
            "predictions": [
                {"answer": "A1"},  # Should use index 0
                {"answer": "A2"},  # Should use index 1
                {"idx": 10, "answer": "A3"},  # Should use idx 10
            ]
        }

        io.save_predictions(predictions_data)
        completed = io.get_completed_indices()

        assert completed == {0, 1, 10}

    def test_ensure_dir_creates_nested_directories(self, tmp_path: Path):
        """Test ensure_dir creates nested directories."""
        nested_path = tmp_path / "level1" / "level2" / "level3"
        io = ExperimentIO(nested_path)

        assert not nested_path.exists()
        io.ensure_dir()
        assert nested_path.exists()
        assert nested_path.is_dir()

    def test_load_nonexistent_file_raises_file_not_found(self, tmp_path: Path):
        """Test load methods raise FileNotFoundError for nonexistent files."""
        io = ExperimentIO(tmp_path)
        io.ensure_dir()

        with pytest.raises(FileNotFoundError):
            io.load_predictions()

        with pytest.raises(FileNotFoundError):
            io.load_result()

        with pytest.raises(FileNotFoundError):
            io.load_questions()

        with pytest.raises(FileNotFoundError):
            io.load_metadata()


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    def test_atomic_write_json_works_standalone(self, tmp_path: Path):
        """Test module-level atomic_write_json works standalone."""
        target = tmp_path / "standalone.json"
        data = {"standalone": True, "value": 123}

        atomic_write_json(data, target)

        assert target.exists()
        assert not target.with_suffix(".tmp").exists()

        with open(target) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_atomic_write_json_creates_parent_dirs(self, tmp_path: Path):
        """Test atomic_write_json creates parent directories."""
        nested = tmp_path / "deep" / "nested" / "path" / "file.json"
        data = {"deep": True}

        atomic_write_json(data, nested)

        assert nested.exists()
        assert nested.parent.exists()

    def test_atomic_write_json_custom_kwargs(self, tmp_path: Path):
        """Test atomic_write_json respects custom kwargs."""
        target = tmp_path / "custom.json"
        data = {"z": 1, "a": 2}

        # Write with custom formatting
        atomic_write_json(data, target, indent=4, sort_keys=True)

        # Read raw content to check formatting
        content = target.read_text()
        assert "    " in content  # 4-space indent
        # Keys should be sorted (a before z)
        assert content.index('"a"') < content.index('"z"')

    def test_atomic_write_json_default_indent(self, tmp_path: Path):
        """Test atomic_write_json uses indent=2 by default."""
        target = tmp_path / "default_indent.json"
        data = {"key": "value"}

        atomic_write_json(data, target)

        content = target.read_text()
        # Default indent=2 should be present
        assert "  " in content  # 2-space indent
