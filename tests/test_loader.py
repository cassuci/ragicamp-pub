"""Tests for analysis loader.

Tests ExperimentResult parsing and ResultsLoader functionality.
"""

import json

from ragicamp.analysis.loader import ExperimentResult, ResultsLoader


class TestParseRetrieverName:
    """Test ExperimentResult._parse_retriever_name() method."""

    def test_valid_retriever_name(self):
        """Test parsing valid retriever name."""
        result = ExperimentResult._parse_retriever_name("simple_minilm_recursive_1024")

        assert result["corpus"] == "simple"
        assert result["embedding_model"] == "minilm"
        assert result["chunk_strategy"] == "recursive"
        assert result["chunk_size"] == 1024

    def test_valid_retriever_name_with_different_values(self):
        """Test parsing retriever name with different values."""
        result = ExperimentResult._parse_retriever_name("en_e5_fixed_512")

        assert result["corpus"] == "en"
        assert result["embedding_model"] == "e5"
        assert result["chunk_strategy"] == "fixed"
        assert result["chunk_size"] == 512

    def test_retriever_name_with_extra_segments(self):
        """Test parsing retriever name with extra underscored segments."""
        # Should only match the first 4 segments
        result = ExperimentResult._parse_retriever_name(
            "simple_minilm_recursive_1024_extra_segment"
        )

        assert result["corpus"] == "simple"
        assert result["embedding_model"] == "minilm"
        assert result["chunk_strategy"] == "recursive"
        assert result["chunk_size"] == 1024

    def test_invalid_retriever_name_partial(self):
        """Test parsing invalid/partial retriever name."""
        result = ExperimentResult._parse_retriever_name("simple_minilm")

        assert result["corpus"] == "unknown"
        assert result["embedding_model"] == "unknown"
        assert result["chunk_strategy"] == "unknown"
        assert result["chunk_size"] == 0

    def test_invalid_retriever_name_wrong_format(self):
        """Test parsing retriever name with wrong format."""
        result = ExperimentResult._parse_retriever_name("invalid_format")

        assert result["corpus"] == "unknown"
        assert result["embedding_model"] == "unknown"
        assert result["chunk_strategy"] == "unknown"
        assert result["chunk_size"] == 0

    def test_none_input(self):
        """Test parsing None input."""
        result = ExperimentResult._parse_retriever_name(None)

        assert result["corpus"] == "unknown"
        assert result["embedding_model"] == "unknown"
        assert result["chunk_strategy"] == "unknown"
        assert result["chunk_size"] == 0

    def test_empty_string(self):
        """Test parsing empty string."""
        result = ExperimentResult._parse_retriever_name("")

        assert result["corpus"] == "unknown"
        assert result["embedding_model"] == "unknown"
        assert result["chunk_strategy"] == "unknown"
        assert result["chunk_size"] == 0

    def test_retriever_name_with_numbers_in_embedding(self):
        """Test parsing retriever name with numbers in embedding model."""
        result = ExperimentResult._parse_retriever_name("simple_bge2_fixed_256")

        assert result["corpus"] == "simple"
        assert result["embedding_model"] == "bge2"
        assert result["chunk_strategy"] == "fixed"
        assert result["chunk_size"] == 256


class TestExperimentResultFromDict:
    """Test ExperimentResult.from_dict() method."""

    def test_from_dict_complete_data(self):
        """Test creating from complete dictionary."""
        data = {
            "name": "test_experiment",
            "type": "rag",
            "model": "hf:meta-llama/Llama-3.2-3B",
            "dataset": "nq",
            "prompt": "test_prompt",
            "quantization": "fp16",
            "retriever": "simple_minilm_recursive_1024",
            "top_k": 5,
            "batch_size": 8,
            "num_questions": 100,
            "results": {
                "f1": 0.85,
                "exact_match": 0.75,
                "bertscore_f1": 0.82,
                "bertscore_precision": 0.80,
                "bertscore_recall": 0.84,
                "bleurt": 0.78,
                "llm_judge": 0.88,
            },
            "duration": 120.5,
            "throughput_qps": 0.83,
            "timestamp": "2024-01-01T00:00:00",
        }

        result = ExperimentResult.from_dict(data)

        assert result.name == "test_experiment"
        assert result.type == "rag"
        assert result.model == "hf:meta-llama/Llama-3.2-3B"
        assert result.dataset == "nq"
        assert result.prompt == "test_prompt"
        assert result.quantization == "fp16"
        assert result.retriever == "simple_minilm_recursive_1024"
        assert result.top_k == 5
        assert result.batch_size == 8
        assert result.num_questions == 100
        assert result.f1 == 0.85
        assert result.exact_match == 0.75
        assert result.bertscore_f1 == 0.82
        assert result.bertscore_precision == 0.80
        assert result.bertscore_recall == 0.84
        assert result.bleurt == 0.78
        assert result.llm_judge == 0.88
        assert result.duration == 120.5
        assert result.throughput_qps == 0.83
        assert result.timestamp == "2024-01-01T00:00:00"
        assert result.raw == data
        # Check parsed retriever components
        assert result.corpus == "simple"
        assert result.embedding_model == "minilm"
        assert result.chunk_strategy == "recursive"
        assert result.chunk_size == 1024

    def test_from_dict_with_metrics_key(self):
        """Test creating from dictionary with 'metrics' key instead of 'results'."""
        data = {
            "name": "test_experiment",
            "type": "direct",
            "model": "hf:google/gemma-2-2b-it",
            "metrics": {
                "f1": 0.90,
                "exact_match": 0.85,
            },
        }

        result = ExperimentResult.from_dict(data)

        assert result.name == "test_experiment"
        assert result.type == "direct"
        assert result.f1 == 0.90
        assert result.exact_match == 0.85

    def test_from_dict_partial_data(self):
        """Test creating from partial dictionary with defaults."""
        data = {
            "name": "minimal_experiment",
        }

        result = ExperimentResult.from_dict(data)

        assert result.name == "minimal_experiment"
        assert result.type == "unknown"
        assert result.model == "unknown"
        assert result.dataset == "unknown"
        assert result.prompt == "unknown"
        assert result.quantization == "unknown"
        assert result.retriever is None
        assert result.top_k is None
        assert result.batch_size == 1
        assert result.num_questions == 0
        assert result.f1 == 0.0
        assert result.exact_match == 0.0
        assert result.duration == 0.0
        assert result.throughput_qps == 0.0
        assert result.timestamp == ""
        assert result.corpus == "unknown"
        assert result.embedding_model == "unknown"
        assert result.chunk_strategy == "unknown"
        assert result.chunk_size == 0

    def test_from_dict_with_num_examples(self):
        """Test creating from dictionary with num_examples in results."""
        data = {
            "name": "test",
            "results": {
                "num_examples": 50,
            },
        }

        result = ExperimentResult.from_dict(data)

        assert result.num_questions == 50

    def test_from_dict_llm_judge_variants(self):
        """Test creating from dictionary with different llm_judge key names."""
        # Test llm_judge_qa
        data1 = {
            "name": "test1",
            "results": {
                "llm_judge_qa": 0.95,
            },
        }
        result1 = ExperimentResult.from_dict(data1)
        assert result1.llm_judge == 0.95

        # Test llm_judge
        data2 = {
            "name": "test2",
            "results": {
                "llm_judge": 0.92,
            },
        }
        result2 = ExperimentResult.from_dict(data2)
        assert result2.llm_judge == 0.92

        # Test preference for llm_judge_qa
        data3 = {
            "name": "test3",
            "results": {
                "llm_judge": 0.90,
                "llm_judge_qa": 0.95,
            },
        }
        result3 = ExperimentResult.from_dict(data3)
        assert result3.llm_judge == 0.95

    def test_from_dict_empty_results(self):
        """Test creating from dictionary with empty results."""
        data = {
            "name": "test",
            "results": {},
        }

        result = ExperimentResult.from_dict(data)

        assert result.name == "test"
        assert result.f1 == 0.0
        assert result.exact_match == 0.0
        assert result.llm_judge is None


class TestExperimentResultFromMetadata:
    """Test ExperimentResult.from_metadata() method."""

    def test_from_metadata_complete_data(self):
        """Test creating from complete metadata and summary."""
        metadata = {
            "name": "test_experiment",
            "type": "rag",
            "model": "hf:meta-llama/Llama-3.2-3B",
            "dataset": "nq",
            "prompt": "test_prompt",
            "quantization": "fp16",
            "retriever": "simple_minilm_recursive_1024",
            "top_k": 5,
            "batch_size": 8,
            "num_questions": 100,
            "duration": 120.5,
            "throughput_qps": 0.83,
            "timestamp": "2024-01-01T00:00:00",
        }
        summary = {
            "overall_metrics": {
                "f1": 0.85,
                "exact_match": 0.75,
                "bertscore_f1": 0.82,
                "bertscore_precision": 0.80,
                "bertscore_recall": 0.84,
                "bleurt": 0.78,
                "llm_judge": 0.88,
            },
            "num_examples": 100,
        }

        result = ExperimentResult.from_metadata(metadata, summary)

        assert result.name == "test_experiment"
        assert result.type == "rag"
        assert result.model == "hf:meta-llama/Llama-3.2-3B"
        assert result.dataset == "nq"
        assert result.prompt == "test_prompt"
        assert result.quantization == "fp16"
        assert result.retriever == "simple_minilm_recursive_1024"
        assert result.top_k == 5
        assert result.batch_size == 8
        assert result.num_questions == 100
        assert result.f1 == 0.85
        assert result.exact_match == 0.75
        assert result.bertscore_f1 == 0.82
        assert result.bertscore_precision == 0.80
        assert result.bertscore_recall == 0.84
        assert result.bleurt == 0.78
        assert result.llm_judge == 0.88
        assert result.duration == 120.5
        assert result.throughput_qps == 0.83
        assert result.timestamp == "2024-01-01T00:00:00"
        assert "summary" in result.raw
        assert result.corpus == "simple"
        assert result.embedding_model == "minilm"
        assert result.chunk_strategy == "recursive"
        assert result.chunk_size == 1024

    def test_from_metadata_without_overall_metrics(self):
        """Test creating from metadata with summary without overall_metrics."""
        metadata = {"name": "test"}
        summary = {
            "f1": 0.85,
            "exact_match": 0.75,
            "num_examples": 50,
        }

        result = ExperimentResult.from_metadata(metadata, summary)

        assert result.name == "test"
        assert result.f1 == 0.85
        assert result.exact_match == 0.75
        assert result.num_questions == 50

    def test_from_metadata_partial_data(self):
        """Test creating from partial metadata and summary."""
        metadata = {"name": "minimal"}
        summary = {}

        result = ExperimentResult.from_metadata(metadata, summary)

        assert result.name == "minimal"
        assert result.type == "unknown"
        assert result.model == "unknown"
        assert result.dataset == "unknown"
        assert result.f1 == 0.0
        assert result.exact_match == 0.0

    def test_from_metadata_with_agent_name_fallback(self):
        """Test creating from metadata with agent_name in summary."""
        metadata = {}
        summary = {
            "agent_name": "fallback_name",
            "overall_metrics": {},
        }

        result = ExperimentResult.from_metadata(metadata, summary)

        assert result.name == "fallback_name"

    def test_from_metadata_with_dataset_name_fallback(self):
        """Test creating from metadata with dataset_name in summary."""
        metadata = {}
        summary = {
            "dataset_name": "fallback_dataset",
            "overall_metrics": {},
        }

        result = ExperimentResult.from_metadata(metadata, summary)

        assert result.dataset == "fallback_dataset"

    def test_from_metadata_with_timestamp_fallback(self):
        """Test creating from metadata with timestamp in summary."""
        metadata = {}
        summary = {
            "timestamp": "2024-01-01T00:00:00",
            "overall_metrics": {},
        }

        result = ExperimentResult.from_metadata(metadata, summary)

        assert result.timestamp == "2024-01-01T00:00:00"

    def test_from_metadata_llm_judge_variants(self):
        """Test creating from metadata with different llm_judge key names."""
        metadata = {"name": "test"}
        summary1 = {
            "overall_metrics": {
                "llm_judge_qa": 0.95,
            },
        }
        result1 = ExperimentResult.from_metadata(metadata, summary1)
        assert result1.llm_judge == 0.95

        summary2 = {
            "overall_metrics": {
                "llm_judge": 0.92,
            },
        }
        result2 = ExperimentResult.from_metadata(metadata, summary2)
        assert result2.llm_judge == 0.92


class TestExperimentResultProperties:
    """Test ExperimentResult properties."""

    def test_model_short_with_hf_prefix(self):
        """Test model_short property with hf: prefix."""
        result = ExperimentResult(
            name="test",
            model="hf:meta-llama/Llama-3.2-3B",
        )

        assert result.model_short == "Llama-3.2-3B"

    def test_model_short_without_prefix(self):
        """Test model_short property without prefix."""
        result = ExperimentResult(
            name="test",
            model="google/gemma-2-2b-it",
        )

        assert result.model_short == "gemma-2-2b-it"

    def test_model_short_simple_name(self):
        """Test model_short property with simple model name."""
        result = ExperimentResult(
            name="test",
            model="gpt-4",
        )

        assert result.model_short == "gpt-4"

    def test_model_short_nested_path(self):
        """Test model_short property with nested path."""
        result = ExperimentResult(
            name="test",
            model="hf:org/suborg/model-name",
        )

        assert result.model_short == "model-name"

    def test_retriever_short_with_simple_prefix(self):
        """Test retriever_short property with simple_ prefix."""
        result = ExperimentResult(
            name="test",
            retriever="simple_minilm_recursive_1024",
        )

        assert result.retriever_short == "minilm_1024"

    def test_retriever_short_with_recursive_prefix(self):
        """Test retriever_short property with recursive_ prefix."""
        result = ExperimentResult(
            name="test",
            retriever="en_recursive_512",
        )

        assert result.retriever_short == "en_512"

    def test_retriever_short_with_both_prefixes(self):
        """Test retriever_short property with both simple_ and recursive_ prefixes."""
        result = ExperimentResult(
            name="test",
            retriever="simple_minilm_recursive_1024",
        )

        assert result.retriever_short == "minilm_1024"

    def test_retriever_short_without_prefixes(self):
        """Test retriever_short property without prefixes."""
        result = ExperimentResult(
            name="test",
            retriever="en_e5_fixed_512",
        )

        assert result.retriever_short == "en_e5_fixed_512"

    def test_retriever_short_none(self):
        """Test retriever_short property with None retriever."""
        result = ExperimentResult(
            name="test",
            retriever=None,
        )

        assert result.retriever_short == "none"

    def test_retriever_short_empty_string(self):
        """Test retriever_short property with empty string retriever."""
        result = ExperimentResult(
            name="test",
            retriever="",
        )

        # Empty string is falsy, so should return "none"
        assert result.retriever_short == "none"


class TestExperimentResultToDict:
    """Test ExperimentResult.to_dict() method."""

    def test_to_dict_complete(self):
        """Test converting complete result to dictionary."""
        result = ExperimentResult(
            name="test_experiment",
            type="rag",
            model="hf:meta-llama/Llama-3.2-3B",
            dataset="nq",
            prompt="test_prompt",
            quantization="fp16",
            retriever="simple_minilm_recursive_1024",
            top_k=5,
            batch_size=8,
            num_questions=100,
            f1=0.85,
            exact_match=0.75,
            bertscore_f1=0.82,
            bertscore_precision=0.80,
            bertscore_recall=0.84,
            bleurt=0.78,
            llm_judge=0.88,
            duration=120.5,
            throughput_qps=0.83,
            timestamp="2024-01-01T00:00:00",
            corpus="simple",
            embedding_model="minilm",
            chunk_size=1024,
            chunk_strategy="recursive",
        )

        data = result.to_dict()

        assert data["name"] == "test_experiment"
        assert data["type"] == "rag"
        assert data["model"] == "hf:meta-llama/Llama-3.2-3B"
        assert data["model_short"] == "Llama-3.2-3B"
        assert data["dataset"] == "nq"
        assert data["prompt"] == "test_prompt"
        assert data["quantization"] == "fp16"
        assert data["retriever"] == "simple_minilm_recursive_1024"
        assert data["top_k"] == 5
        assert data["batch_size"] == 8
        assert data["num_questions"] == 100
        assert data["f1"] == 0.85
        assert data["exact_match"] == 0.75
        assert data["bertscore_f1"] == 0.82
        assert data["bertscore_precision"] == 0.80
        assert data["bertscore_recall"] == 0.84
        assert data["bleurt"] == 0.78
        assert data["llm_judge"] == 0.88
        assert data["duration"] == 120.5
        assert data["throughput_qps"] == 0.83
        assert data["timestamp"] == "2024-01-01T00:00:00"
        assert data["corpus"] == "simple"
        assert data["embedding_model"] == "minilm"
        assert data["chunk_size"] == 1024
        assert data["chunk_strategy"] == "recursive"

    def test_to_dict_without_llm_judge(self):
        """Test converting result without llm_judge to dictionary."""
        result = ExperimentResult(
            name="test",
            llm_judge=None,
        )

        data = result.to_dict()

        assert "llm_judge" not in data

    def test_to_dict_round_trip(self):
        """Test round-trip conversion: from_dict -> to_dict."""
        original_data = {
            "name": "test_experiment",
            "type": "rag",
            "model": "hf:meta-llama/Llama-3.2-3B",
            "dataset": "nq",
            "prompt": "test_prompt",
            "quantization": "fp16",
            "retriever": "simple_minilm_recursive_1024",
            "top_k": 5,
            "batch_size": 8,
            "num_questions": 100,
            "results": {
                "f1": 0.85,
                "exact_match": 0.75,
                "bertscore_f1": 0.82,
                "bertscore_precision": 0.80,
                "bertscore_recall": 0.84,
                "bleurt": 0.78,
                "llm_judge": 0.88,
            },
            "duration": 120.5,
            "throughput_qps": 0.83,
            "timestamp": "2024-01-01T00:00:00",
        }

        result = ExperimentResult.from_dict(original_data)
        round_trip_data = result.to_dict()

        # Check that essential fields are preserved
        assert round_trip_data["name"] == original_data["name"]
        assert round_trip_data["type"] == original_data["type"]
        assert round_trip_data["model"] == original_data["model"]
        assert round_trip_data["dataset"] == original_data["dataset"]
        assert round_trip_data["f1"] == original_data["results"]["f1"]
        assert round_trip_data["exact_match"] == original_data["results"]["exact_match"]
        assert round_trip_data["llm_judge"] == original_data["results"]["llm_judge"]
        # Check that parsed fields are present
        assert round_trip_data["corpus"] == "simple"
        assert round_trip_data["embedding_model"] == "minilm"
        assert round_trip_data["chunk_strategy"] == "recursive"
        assert round_trip_data["chunk_size"] == 1024


class TestIsDeprecatedPath:
    """Test ResultsLoader._is_deprecated_path() method."""

    def test_skip_collisions_dir(self, tmp_path):
        loader = ResultsLoader(tmp_path)
        assert loader._is_deprecated_path(tmp_path / "_collisions" / "exp1") is True

    def test_skip_incomplete_dir(self, tmp_path):
        loader = ResultsLoader(tmp_path)
        assert loader._is_deprecated_path(tmp_path / "_incomplete" / "exp1") is True

    def test_skip_tainted_dir(self, tmp_path):
        loader = ResultsLoader(tmp_path)
        assert loader._is_deprecated_path(tmp_path / "_tainted" / "exp1") is True

    def test_skip_archived_fake_reranked_dir(self, tmp_path):
        loader = ResultsLoader(tmp_path)
        assert loader._is_deprecated_path(tmp_path / "_archived_fake_reranked" / "exp1") is True

    def test_skip_underscore_prefixed_dir(self, tmp_path):
        loader = ResultsLoader(tmp_path)
        assert loader._is_deprecated_path(tmp_path / "_whatever" / "exp1") is True

    def test_skip_dot_prefixed_dir(self, tmp_path):
        loader = ResultsLoader(tmp_path)
        assert loader._is_deprecated_path(tmp_path / ".hidden" / "exp1") is True

    def test_allow_normal_experiment(self, tmp_path):
        loader = ResultsLoader(tmp_path)
        assert loader._is_deprecated_path(tmp_path / "rag_vllm_llama_k5_nq") is False

    def test_skip_analysis_dir(self, tmp_path):
        loader = ResultsLoader(tmp_path)
        assert loader._is_deprecated_path(tmp_path / "analysis" / "some_file") is True


class TestResultsLoaderFiltering:
    """Test that ResultsLoader skips deprecated and incomplete experiments."""

    def _make_complete_experiment(self, exp_dir):
        """Create a complete experiment directory with metadata, results, and state."""
        exp_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"name": exp_dir.name, "type": "rag", "model": "test"}
        results = {"metrics": {"f1": 0.85}, "overall_metrics": {"f1": 0.85}}
        state = {"phase": "complete"}
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f)
        with open(exp_dir / "state.json", "w") as f:
            json.dump(state, f)

    def test_skips_collisions_directory(self, tmp_path):
        self._make_complete_experiment(tmp_path / "good_exp")
        self._make_complete_experiment(tmp_path / "_collisions" / "bad_exp")

        loader = ResultsLoader(tmp_path)
        results = loader.load_all()
        names = [r.name for r in results]

        assert "good_exp" in names
        assert "bad_exp" not in names

    def test_skips_incomplete_directory(self, tmp_path):
        self._make_complete_experiment(tmp_path / "good_exp")
        self._make_complete_experiment(tmp_path / "_incomplete" / "stale_exp")

        loader = ResultsLoader(tmp_path)
        results = loader.load_all()
        names = [r.name for r in results]

        assert "good_exp" in names
        assert "stale_exp" not in names

    def test_skips_failed_experiments(self, tmp_path):
        self._make_complete_experiment(tmp_path / "good_exp")

        # Create a failed experiment
        failed_dir = tmp_path / "failed_exp"
        failed_dir.mkdir()
        with open(failed_dir / "metadata.json", "w") as f:
            json.dump({"name": "failed_exp", "type": "rag"}, f)
        with open(failed_dir / "results.json", "w") as f:
            json.dump({"metrics": {"f1": 0.0}, "overall_metrics": {"f1": 0.0}}, f)
        with open(failed_dir / "state.json", "w") as f:
            json.dump({"phase": "failed", "error": "OOM"}, f)

        loader = ResultsLoader(tmp_path)
        results = loader.load_all()
        names = [r.name for r in results]

        assert "good_exp" in names
        assert "failed_exp" not in names

    def test_skips_generating_experiments(self, tmp_path):
        self._make_complete_experiment(tmp_path / "good_exp")

        in_progress_dir = tmp_path / "generating_exp"
        in_progress_dir.mkdir()
        with open(in_progress_dir / "metadata.json", "w") as f:
            json.dump({"name": "generating_exp", "type": "rag"}, f)
        with open(in_progress_dir / "results.json", "w") as f:
            json.dump({"metrics": {}, "overall_metrics": {}}, f)
        with open(in_progress_dir / "state.json", "w") as f:
            json.dump({"phase": "generating"}, f)

        loader = ResultsLoader(tmp_path)
        results = loader.load_all()
        names = [r.name for r in results]

        assert "good_exp" in names
        assert "generating_exp" not in names

    def test_loads_experiment_without_state_file(self, tmp_path):
        """Experiments without state.json should still be loaded (legacy data)."""
        exp_dir = tmp_path / "legacy_exp"
        exp_dir.mkdir()
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump({"name": "legacy_exp", "type": "rag", "model": "test"}, f)
        with open(exp_dir / "results.json", "w") as f:
            json.dump({"metrics": {"f1": 0.80}, "overall_metrics": {"f1": 0.80}}, f)

        loader = ResultsLoader(tmp_path)
        results = loader.load_all()
        names = [r.name for r in results]

        assert "legacy_exp" in names


class TestResultsLoader:
    """Test ResultsLoader class."""

    def test_load_from_comparison_json(self, tmp_path):
        """Test loading from comparison.json file."""
        comparison_data = {
            "experiments": [
                {
                    "name": "exp1",
                    "type": "rag",
                    "model": "hf:meta-llama/Llama-3.2-3B",
                    "dataset": "nq",
                    "results": {
                        "f1": 0.85,
                        "exact_match": 0.75,
                    },
                },
                {
                    "name": "exp2",
                    "type": "direct",
                    "model": "hf:google/gemma-2-2b-it",
                    "dataset": "nq",
                    "results": {
                        "f1": 0.90,
                        "exact_match": 0.80,
                    },
                },
            ],
        }

        comparison_file = tmp_path / "comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison_data, f)

        loader = ResultsLoader(tmp_path)
        results = loader.load_all()

        assert len(results) == 2
        assert results[0].name == "exp1"
        assert results[0].type == "rag"
        assert results[0].f1 == 0.85
        assert results[1].name == "exp2"
        assert results[1].type == "direct"
        assert results[1].f1 == 0.90

    def test_load_from_study_summary_json(self, tmp_path):
        """Test loading from study_summary.json file (preferred format)."""
        summary_data = {
            "experiments": [
                {
                    "name": "exp1",
                    "type": "rag",
                    "model": "hf:meta-llama/Llama-3.2-3B",
                    "results": {
                        "f1": 0.85,
                    },
                },
            ],
        }

        summary_file = tmp_path / "study_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary_data, f)

        loader = ResultsLoader(tmp_path)
        results = loader.load_all()

        assert len(results) == 1
        assert results[0].name == "exp1"
        assert results[0].f1 == 0.85

    def test_load_from_directories_metadata_results(self, tmp_path):
        """Test loading from individual directories with metadata.json and results.json."""
        exp_dir = tmp_path / "exp1"
        exp_dir.mkdir()

        metadata = {
            "name": "exp1",
            "type": "rag",
            "model": "hf:meta-llama/Llama-3.2-3B",
            "dataset": "nq",
            "retriever": "simple_minilm_recursive_1024",
        }
        results = {
            "metrics": {
                "f1": 0.85,
                "exact_match": 0.75,
            },
        }

        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f)

        loader = ResultsLoader(tmp_path)
        results_list = loader.load_all()

        assert len(results_list) == 1
        assert results_list[0].name == "exp1"
        assert results_list[0].f1 == 0.85
        assert results_list[0].exact_match == 0.75

    def test_load_from_directories_metadata_summary(self, tmp_path):
        """Test loading from individual directories with metadata.json and *_summary.json."""
        exp_dir = tmp_path / "exp1"
        exp_dir.mkdir()

        metadata = {
            "name": "exp1",
            "type": "rag",
            "model": "hf:meta-llama/Llama-3.2-3B",
        }
        summary = {
            "overall_metrics": {
                "f1": 0.85,
                "exact_match": 0.75,
            },
        }

        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        with open(exp_dir / "exp1_summary.json", "w") as f:
            json.dump(summary, f)

        loader = ResultsLoader(tmp_path)
        results = loader.load_all()

        assert len(results) == 1
        assert results[0].name == "exp1"
        assert results[0].f1 == 0.85

    def test_load_from_directories_nested(self, tmp_path):
        """Test loading from nested directories."""
        nested_dir = tmp_path / "subdir" / "exp1"
        nested_dir.mkdir(parents=True)

        metadata = {
            "name": "exp1",
            "type": "rag",
        }
        summary = {
            "overall_metrics": {
                "f1": 0.85,
            },
        }

        with open(nested_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        with open(nested_dir / "exp1_summary.json", "w") as f:
            json.dump(summary, f)

        loader = ResultsLoader(tmp_path)
        results = loader.load_all()

        assert len(results) == 1
        assert results[0].name == "exp1"

    def test_load_from_comparison_with_llm_judge_from_summary(self, tmp_path):
        """Test loading from comparison.json with llm_judge enriched from summary file."""
        comparison_data = {
            "experiments": [
                {
                    "name": "exp1",
                    "type": "rag",
                    "results": {
                        "f1": 0.85,
                    },
                },
            ],
        }

        comparison_file = tmp_path / "comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison_data, f)

        exp_dir = tmp_path / "exp1"
        exp_dir.mkdir()
        summary = {
            "overall_metrics": {
                "llm_judge_qa": 0.95,
            },
        }
        with open(exp_dir / "exp1_summary.json", "w") as f:
            json.dump(summary, f)

        loader = ResultsLoader(tmp_path)
        results = loader.load_all()

        assert len(results) == 1
        assert results[0].llm_judge == 0.95

    def test_load_predictions_new_format(self, tmp_path):
        """Test loading predictions from predictions.json."""
        exp_dir = tmp_path / "exp1"
        exp_dir.mkdir()

        predictions_data = {
            "predictions": [
                {
                    "question": "Q1?",
                    "prediction": "A1",
                    "references": ["A1"],
                    "retrieved_docs": [
                        {
                            "rank": 1,
                            "doc_id": "doc1",
                            "content": "Content 1",
                            "score": 0.9,
                        },
                    ],
                },
            ],
        }

        with open(exp_dir / "predictions.json", "w") as f:
            json.dump(predictions_data, f)

        loader = ResultsLoader(tmp_path)
        predictions = loader.load_predictions("exp1")

        assert predictions is not None
        assert len(predictions) == 1
        assert predictions[0]["question"] == "Q1?"
        assert predictions[0]["prediction"] == "A1"
        assert "retrieved_docs" in predictions[0]

    def test_load_predictions_legacy_format(self, tmp_path):
        """Test loading predictions from *_predictions.json."""
        exp_dir = tmp_path / "exp1"
        exp_dir.mkdir()

        predictions_data = {
            "predictions": [
                {
                    "question": "Q1?",
                    "prediction": "A1",
                    "references": ["A1"],
                },
            ],
        }

        with open(exp_dir / "exp1_predictions.json", "w") as f:
            json.dump(predictions_data, f)

        loader = ResultsLoader(tmp_path)
        predictions = loader.load_predictions("exp1")

        assert predictions is not None
        assert len(predictions) == 1

    def test_load_predictions_with_normalization(self, tmp_path):
        """Test loading predictions with normalization from old format."""
        exp_dir = tmp_path / "exp1"
        exp_dir.mkdir()

        predictions_data = {
            "predictions": [
                {
                    "question": "Q1?",
                    "prediction": "A1",
                    "metadata": {
                        "doc_scores": [0.9, 0.8, 0.7],
                    },
                },
            ],
        }

        with open(exp_dir / "exp1_predictions.json", "w") as f:
            json.dump(predictions_data, f)

        loader = ResultsLoader(tmp_path)
        predictions = loader.load_predictions("exp1", normalize=True)

        assert predictions is not None
        assert len(predictions) == 1
        assert "retrieved_docs" in predictions[0]
        assert len(predictions[0]["retrieved_docs"]) == 3
        assert predictions[0]["retrieved_docs"][0]["rank"] == 1
        assert predictions[0]["retrieved_docs"][0]["score"] == 0.9

    def test_load_predictions_not_found(self, tmp_path):
        """Test loading predictions when file doesn't exist."""
        loader = ResultsLoader(tmp_path)
        predictions = loader.load_predictions("nonexistent")

        assert predictions is None

    def test_load_all_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        loader = ResultsLoader(tmp_path)
        results = loader.load_all()

        assert len(results) == 0

    def test_load_all_prefers_study_summary_over_comparison(self, tmp_path):
        """Test that study_summary.json is preferred over comparison.json."""
        study_summary_data = {
            "experiments": [
                {
                    "name": "from_summary",
                    "results": {"f1": 0.85},
                },
            ],
        }
        comparison_data = {
            "experiments": [
                {
                    "name": "from_comparison",
                    "results": {"f1": 0.90},
                },
            ],
        }

        with open(tmp_path / "study_summary.json", "w") as f:
            json.dump(study_summary_data, f)
        with open(tmp_path / "comparison.json", "w") as f:
            json.dump(comparison_data, f)

        loader = ResultsLoader(tmp_path)
        results = loader.load_all()

        assert len(results) == 1
        assert results[0].name == "from_summary"
        assert results[0].f1 == 0.85

    def test_load_from_directories_skips_missing_summary(self, tmp_path):
        """Test that directories without summary files are skipped."""
        exp_dir = tmp_path / "exp1"
        exp_dir.mkdir()

        metadata = {
            "name": "exp1",
        }

        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        loader = ResultsLoader(tmp_path)
        results = loader.load_all()

        # Should skip exp1 because it has no summary file
        assert len(results) == 0

    def test_load_from_directories_handles_invalid_json(self, tmp_path):
        """Test that invalid JSON files are handled gracefully."""
        exp_dir = tmp_path / "exp1"
        exp_dir.mkdir()

        with open(exp_dir / "metadata.json", "w") as f:
            f.write("invalid json{")

        loader = ResultsLoader(tmp_path)
        results = loader.load_all()

        # Should handle error gracefully and return empty list or skip
        assert isinstance(results, list)
