"""Tests for the Optuna-powered experiment search module.

Covers search space extraction, conditional agent parameters, experiment
naming, trial-to-spec conversion, warm-starting from disk, and sampler
factory.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock

import optuna
import pytest

from ragicamp.optimization.optuna_search import (
    _check_context_feasibility,
    _create_sampler,
    _extract_search_space,
    _get_experiment_metric,
    _seed_from_existing,
    _suggest_agent_params,
    _trial_to_spec,
    run_optuna_study,
)
from ragicamp.spec.experiment import ExperimentSpec

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config():
    """Minimal study config with one model, retriever, etc."""
    return {
        "datasets": ["nq"],
        "metrics": ["f1", "exact_match"],
        "batch_size": 8,
        "rag": {
            "enabled": True,
            "models": ["vllm:test/model-a"],
            "retriever_names": ["dense_bge"],
            "retrievers": [
                {
                    "name": "dense_bge",
                    "type": "dense",
                    "embedding_index": "idx_bge",
                },
            ],
            "top_k_values": [5],
            "prompts": ["concise"],
            "query_transform": ["none"],
            "reranker": {
                "configs": [
                    {"enabled": False, "name": "none"},
                ],
            },
        },
    }


@pytest.fixture
def agent_config(base_config):
    """Config with agent_types and agent_params in sampling section."""
    base_config["rag"]["sampling"] = {
        "mode": "tpe",
        "n_experiments": 50,
        "optimize_metric": "f1",
        "seed": 42,
        "agent_types": ["fixed_rag", "iterative_rag", "self_rag"],
        "agent_params": {
            "iterative_rag": {
                "max_iterations": [1, 2, 3],
                "stop_on_sufficient": [True],
            },
        },
    }
    return base_config


@pytest.fixture
def single_agent_config(base_config):
    """Config with only one agent type (should NOT add agent_type dim)."""
    base_config["rag"]["sampling"] = {
        "mode": "tpe",
        "agent_types": ["fixed_rag"],
    }
    return base_config


# ---------------------------------------------------------------------------
# _extract_search_space
# ---------------------------------------------------------------------------


class TestExtractSearchSpace:
    """Tests for search space extraction from config."""

    def test_base_dimensions(self, base_config):
        """Test that standard RAG dimensions are extracted."""
        space = _extract_search_space(base_config)

        assert space["model"] == ["vllm:test/model-a"]
        assert space["retriever"] == ["dense_bge"]
        assert space["top_k"] == [5]
        assert space["prompt"] == ["concise"]
        assert space["query_transform"] == ["none"]
        assert space["dataset"] == ["nq"]
        assert space["reranker"] == ["none"]

    def test_no_agent_type_without_sampling(self, base_config):
        """Test that agent_type is not in space when sampling is absent."""
        space = _extract_search_space(base_config)
        assert "agent_type" not in space

    def test_no_agent_type_with_single_agent(self, single_agent_config):
        """Test that a single agent_type does NOT create a dimension."""
        space = _extract_search_space(single_agent_config)
        assert "agent_type" not in space

    def test_agent_type_dimension_added(self, agent_config):
        """Test that multiple agent_types create a search space dimension."""
        space = _extract_search_space(agent_config)

        assert "agent_type" in space
        assert space["agent_type"] == ["fixed_rag", "iterative_rag", "self_rag"]

    def test_multiple_rerankers(self, base_config):
        """Test that multiple reranker configs produce correct values."""
        base_config["rag"]["reranker"]["configs"] = [
            {"enabled": False, "name": "none"},
            {"enabled": True, "name": "bge"},
            {"enabled": True, "name": "bge-v2"},
        ]
        space = _extract_search_space(base_config)

        assert space["reranker"] == ["none", "bge", "bge-v2"]


# ---------------------------------------------------------------------------
# _suggest_agent_params
# ---------------------------------------------------------------------------


class TestSuggestAgentParams:
    """Tests for conditional agent parameter suggestion."""

    def test_fixed_rag_returns_empty(self, agent_config):
        """fixed_rag should return no extra params."""
        trial = MagicMock(spec=optuna.Trial)
        params = _suggest_agent_params(trial, "fixed_rag", agent_config)

        assert params == {}
        trial.suggest_categorical.assert_not_called()

    def test_iterative_rag_suggests_params(self, agent_config):
        """iterative_rag should suggest max_iterations and stop_on_sufficient."""
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]

        params = _suggest_agent_params(trial, "iterative_rag", agent_config)

        assert "max_iterations" in params
        assert params["max_iterations"] == 1
        assert "stop_on_sufficient" in params
        assert params["stop_on_sufficient"] is True
        assert trial.suggest_categorical.call_count == 2

    def test_self_rag_no_extra_params(self, agent_config):
        """self_rag has no agent_params configured, should return empty."""
        trial = MagicMock(spec=optuna.Trial)
        params = _suggest_agent_params(trial, "self_rag", agent_config)

        assert params == {}
        trial.suggest_categorical.assert_not_called()

    def test_unknown_agent_returns_empty(self, agent_config):
        """Unknown agent with no config entry returns empty params."""
        trial = MagicMock(spec=optuna.Trial)
        params = _suggest_agent_params(trial, "some_new_agent", agent_config)

        assert params == {}


# ---------------------------------------------------------------------------
# _trial_to_spec
# ---------------------------------------------------------------------------


class TestTrialToSpec:
    """Tests for converting Optuna trials to ExperimentSpec."""

    def _make_trial(self, param_map):
        """Create a mock trial that returns values from a dict."""
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_categorical.side_effect = lambda name, choices: param_map[name]
        return trial

    def test_basic_rag_spec(self, base_config):
        """Test basic RAG spec without agent dimension."""
        space = _extract_search_space(base_config)
        from ragicamp.optimization.optuna_search import (
            _build_reranker_lookup,
            _build_retriever_lookup,
        )

        ret_lookup = _build_retriever_lookup(base_config)
        rr_lookup = _build_reranker_lookup(base_config)

        trial = self._make_trial(
            {
                "model": "vllm:test/model-a",
                "retriever": "dense_bge",
                "top_k": 5,
                "prompt": "concise",
                "query_transform": "none",
                "dataset": "nq",
                "reranker": "none",
            }
        )

        spec = _trial_to_spec(trial, space, ret_lookup, rr_lookup, base_config)

        assert isinstance(spec, ExperimentSpec)
        assert spec.exp_type == "rag"
        assert spec.model == "vllm:test/model-a"
        assert spec.retriever == "dense_bge"
        assert spec.embedding_index == "idx_bge"
        assert spec.top_k == 5
        assert spec.agent_type is None  # fixed_rag → None
        assert spec.agent_params == ()
        assert spec.name.startswith("rag_")

    def test_spec_with_agent_type(self, agent_config):
        """Test spec generation with agent_type dimension."""
        space = _extract_search_space(agent_config)
        from ragicamp.optimization.optuna_search import (
            _build_reranker_lookup,
            _build_retriever_lookup,
        )

        ret_lookup = _build_retriever_lookup(agent_config)
        rr_lookup = _build_reranker_lookup(agent_config)

        trial = self._make_trial(
            {
                "model": "vllm:test/model-a",
                "retriever": "dense_bge",
                "top_k": 5,
                "prompt": "concise",
                "query_transform": "none",
                "dataset": "nq",
                "reranker": "none",
                "agent_type": "iterative_rag",
                "max_iterations": 2,
                "stop_on_sufficient": True,
            }
        )

        spec = _trial_to_spec(trial, space, ret_lookup, rr_lookup, agent_config)

        assert spec.agent_type == "iterative_rag"
        assert dict(spec.agent_params) == {
            "max_iterations": 2,
            "stop_on_sufficient": True,
        }
        assert spec.name.startswith("iterative_")

    def test_spec_with_self_rag(self, agent_config):
        """Test self_rag agent has no agent_params."""
        space = _extract_search_space(agent_config)
        from ragicamp.optimization.optuna_search import (
            _build_reranker_lookup,
            _build_retriever_lookup,
        )

        ret_lookup = _build_retriever_lookup(agent_config)
        rr_lookup = _build_reranker_lookup(agent_config)

        trial = self._make_trial(
            {
                "model": "vllm:test/model-a",
                "retriever": "dense_bge",
                "top_k": 5,
                "prompt": "concise",
                "query_transform": "none",
                "dataset": "nq",
                "reranker": "none",
                "agent_type": "self_rag",
            }
        )

        spec = _trial_to_spec(trial, space, ret_lookup, rr_lookup, agent_config)

        assert spec.agent_type == "self_rag"
        assert spec.agent_params == ()
        assert spec.name.startswith("self_")

    def test_fixed_rag_agent_type_is_none(self, agent_config):
        """Test that fixed_rag maps to agent_type=None in spec."""
        space = _extract_search_space(agent_config)
        from ragicamp.optimization.optuna_search import (
            _build_reranker_lookup,
            _build_retriever_lookup,
        )

        ret_lookup = _build_retriever_lookup(agent_config)
        rr_lookup = _build_reranker_lookup(agent_config)

        trial = self._make_trial(
            {
                "model": "vllm:test/model-a",
                "retriever": "dense_bge",
                "top_k": 5,
                "prompt": "concise",
                "query_transform": "none",
                "dataset": "nq",
                "reranker": "none",
                "agent_type": "fixed_rag",
            }
        )

        spec = _trial_to_spec(trial, space, ret_lookup, rr_lookup, agent_config)

        assert spec.agent_type is None
        assert spec.name.startswith("rag_")

    def test_fetch_k_with_reranker(self, base_config):
        """Test fetch_k is computed when reranker is enabled."""
        base_config["rag"]["reranker"]["configs"] = [
            {"enabled": True, "name": "bge", "model": "BAAI/bge-reranker-large"},
        ]
        base_config["rag"]["fetch_k_multiplier"] = 4

        space = _extract_search_space(base_config)
        from ragicamp.optimization.optuna_search import (
            _build_reranker_lookup,
            _build_retriever_lookup,
        )

        ret_lookup = _build_retriever_lookup(base_config)
        rr_lookup = _build_reranker_lookup(base_config)

        trial = self._make_trial(
            {
                "model": "vllm:test/model-a",
                "retriever": "dense_bge",
                "top_k": 5,
                "prompt": "concise",
                "query_transform": "none",
                "dataset": "nq",
                "reranker": "bge",
            }
        )

        spec = _trial_to_spec(trial, space, ret_lookup, rr_lookup, base_config)

        assert spec.reranker == "bge"
        assert spec.fetch_k == 20  # 5 * 4


# ---------------------------------------------------------------------------
# _get_experiment_metric
# ---------------------------------------------------------------------------


class TestGetExperimentMetric:
    """Tests for metric extraction from experiment output files."""

    def test_reads_from_results_json(self, tmp_path):
        """Test reading metric from results.json."""
        exp_dir = tmp_path / "exp1"
        exp_dir.mkdir()
        (exp_dir / "results.json").write_text(
            json.dumps({"metrics": {"f1": 0.85, "exact_match": 0.70}})
        )

        value = _get_experiment_metric("exp1", tmp_path, "f1")
        assert value == 0.85

    def test_falls_back_to_predictions_json(self, tmp_path):
        """Test fallback to predictions.json when results.json missing."""
        exp_dir = tmp_path / "exp2"
        exp_dir.mkdir()
        (exp_dir / "predictions.json").write_text(json.dumps({"aggregate_metrics": {"f1": 0.72}}))

        value = _get_experiment_metric("exp2", tmp_path, "f1")
        assert value == 0.72

    def test_returns_none_when_missing(self, tmp_path):
        """Test returns None when no metric files exist."""
        exp_dir = tmp_path / "exp3"
        exp_dir.mkdir()

        value = _get_experiment_metric("exp3", tmp_path, "f1")
        assert value is None

    def test_returns_none_for_missing_metric_key(self, tmp_path):
        """Test returns None when the requested metric is not in the file."""
        exp_dir = tmp_path / "exp4"
        exp_dir.mkdir()
        (exp_dir / "results.json").write_text(json.dumps({"metrics": {"exact_match": 0.70}}))

        value = _get_experiment_metric("exp4", tmp_path, "f1")
        assert value is None


# ---------------------------------------------------------------------------
# _create_sampler
# ---------------------------------------------------------------------------


class TestCreateSampler:
    """Tests for sampler factory."""

    def test_random_sampler(self):
        """Test that 'random' creates a RandomSampler."""
        sampler = _create_sampler("random", seed=42)
        assert isinstance(sampler, optuna.samplers.RandomSampler)

    def test_tpe_sampler(self):
        """Test that 'tpe' creates a TPESampler."""
        sampler = _create_sampler("tpe", seed=42)
        assert isinstance(sampler, optuna.samplers.TPESampler)

    def test_unknown_mode_defaults_to_tpe(self):
        """Test that unknown modes fall back to TPE."""
        sampler = _create_sampler("stratified", seed=42)
        assert isinstance(sampler, optuna.samplers.TPESampler)


# ---------------------------------------------------------------------------
# _seed_from_existing
# ---------------------------------------------------------------------------


class TestSeedFromExisting:
    """Tests for warm-starting Optuna from experiments on disk."""

    def _make_study(self):
        """Create an in-memory Optuna study."""
        return optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(seed=0),
        )

    def _write_experiment(
        self,
        output_dir,
        name,
        metadata,
        metric_value=0.5,
    ):
        """Write a fake experiment directory with metadata and results."""
        exp_dir = output_dir / name
        exp_dir.mkdir(parents=True)
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))
        (exp_dir / "results.json").write_text(json.dumps({"metrics": {"f1": metric_value}}))

    def test_seeds_basic_rag_experiment(self, tmp_path, base_config):
        """Test seeding a standard RAG experiment from disk."""
        study = self._make_study()
        space = _extract_search_space(base_config)

        self._write_experiment(
            tmp_path,
            "rag_exp_1",
            {
                "type": "rag",
                "model": "vllm:test/model-a",
                "retriever": "dense_bge",
                "top_k": 5,
                "prompt": "concise",
                "query_transform": "none",
                "dataset": "nq",
                "reranker": "none",
            },
            metric_value=0.85,
        )

        seeded = _seed_from_existing(study, space, tmp_path, "f1", base_config)

        assert seeded == 1
        assert len(study.trials) == 1
        assert study.trials[0].value == 0.85
        assert study.trials[0].params["model"] == "vllm:test/model-a"

    def test_skips_direct_experiments(self, tmp_path, base_config):
        """Test that direct (non-RAG) experiments are skipped."""
        study = self._make_study()
        space = _extract_search_space(base_config)

        self._write_experiment(
            tmp_path,
            "direct_exp",
            {
                "type": "direct",
                "model": "vllm:test/model-a",
                "dataset": "nq",
                "prompt": "concise",
            },
        )

        seeded = _seed_from_existing(study, space, tmp_path, "f1", base_config)
        assert seeded == 0

    def test_skips_out_of_space_experiments(self, tmp_path, base_config):
        """Test that experiments outside the search space are skipped."""
        study = self._make_study()
        space = _extract_search_space(base_config)

        self._write_experiment(
            tmp_path,
            "rag_other",
            {
                "type": "rag",
                "model": "vllm:other/model",  # Not in search space
                "retriever": "dense_bge",
                "top_k": 5,
                "prompt": "concise",
                "query_transform": "none",
                "dataset": "nq",
                "reranker": "none",
            },
        )

        seeded = _seed_from_existing(study, space, tmp_path, "f1", base_config)
        assert seeded == 0

    def test_no_duplicate_seeding(self, tmp_path, base_config):
        """Test that the same experiment is not seeded twice."""
        study = self._make_study()
        space = _extract_search_space(base_config)

        meta = {
            "type": "rag",
            "model": "vllm:test/model-a",
            "retriever": "dense_bge",
            "top_k": 5,
            "prompt": "concise",
            "query_transform": "none",
            "dataset": "nq",
            "reranker": "none",
        }
        self._write_experiment(tmp_path, "rag_exp_1", meta, 0.80)

        seeded1 = _seed_from_existing(study, space, tmp_path, "f1", base_config)
        seeded2 = _seed_from_existing(study, space, tmp_path, "f1", base_config)

        assert seeded1 == 1
        assert seeded2 == 0
        assert len(study.trials) == 1

    def test_seeds_agent_type_experiment(self, tmp_path, agent_config):
        """Test seeding an iterative_rag experiment with agent params."""
        study = self._make_study()
        space = _extract_search_space(agent_config)

        self._write_experiment(
            tmp_path,
            "iterative_exp",
            {
                "type": "rag",
                "model": "vllm:test/model-a",
                "retriever": "dense_bge",
                "top_k": 5,
                "prompt": "concise",
                "query_transform": "none",
                "dataset": "nq",
                "reranker": "none",
                "agent_type": "iterative_rag",
                "agent_params": {"max_iterations": 2, "stop_on_sufficient": True},
            },
            metric_value=0.90,
        )

        seeded = _seed_from_existing(study, space, tmp_path, "f1", agent_config)

        assert seeded == 1
        trial = study.trials[0]
        assert trial.params["agent_type"] == "iterative_rag"
        assert trial.params["max_iterations"] == 2
        assert trial.params["stop_on_sufficient"] is True
        assert trial.value == 0.90

    def test_skips_agent_with_invalid_params(self, tmp_path, agent_config):
        """Test that agent experiments with out-of-range params are skipped."""
        study = self._make_study()
        space = _extract_search_space(agent_config)

        self._write_experiment(
            tmp_path,
            "iterative_bad",
            {
                "type": "rag",
                "model": "vllm:test/model-a",
                "retriever": "dense_bge",
                "top_k": 5,
                "prompt": "concise",
                "query_transform": "none",
                "dataset": "nq",
                "reranker": "none",
                "agent_type": "iterative_rag",
                "agent_params": {"max_iterations": 99},  # Not in [1, 2, 3]
            },
        )

        seeded = _seed_from_existing(study, space, tmp_path, "f1", agent_config)
        assert seeded == 0

    def test_skips_agent_type_not_in_space(self, tmp_path, agent_config):
        """Test that experiments with unknown agent_type are skipped."""
        study = self._make_study()
        space = _extract_search_space(agent_config)

        self._write_experiment(
            tmp_path,
            "unknown_agent_exp",
            {
                "type": "rag",
                "model": "vllm:test/model-a",
                "retriever": "dense_bge",
                "top_k": 5,
                "prompt": "concise",
                "query_transform": "none",
                "dataset": "nq",
                "reranker": "none",
                "agent_type": "unknown_agent",
            },
        )

        seeded = _seed_from_existing(study, space, tmp_path, "f1", agent_config)
        assert seeded == 0

    def test_seeds_fixed_rag_in_agent_space(self, tmp_path, agent_config):
        """Test seeding a fixed_rag experiment when agent_type is a dimension."""
        study = self._make_study()
        space = _extract_search_space(agent_config)

        self._write_experiment(
            tmp_path,
            "rag_fixed",
            {
                "type": "rag",
                "model": "vllm:test/model-a",
                "retriever": "dense_bge",
                "top_k": 5,
                "prompt": "concise",
                "query_transform": "none",
                "dataset": "nq",
                "reranker": "none",
                # agent_type absent → defaults to fixed_rag
            },
            metric_value=0.75,
        )

        seeded = _seed_from_existing(study, space, tmp_path, "f1", agent_config)

        assert seeded == 1
        assert study.trials[0].params["agent_type"] == "fixed_rag"

    def test_skips_experiments_without_metric(self, tmp_path, base_config):
        """Test that experiments without metric results are skipped."""
        study = self._make_study()
        space = _extract_search_space(base_config)

        exp_dir = tmp_path / "no_metric"
        exp_dir.mkdir()
        (exp_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "type": "rag",
                    "model": "vllm:test/model-a",
                    "retriever": "dense_bge",
                    "top_k": 5,
                    "prompt": "concise",
                    "query_transform": "none",
                    "dataset": "nq",
                    "reranker": "none",
                }
            )
        )
        # No results.json or predictions.json

        seeded = _seed_from_existing(study, space, tmp_path, "f1", base_config)
        assert seeded == 0


# ---------------------------------------------------------------------------
# _check_context_feasibility
# ---------------------------------------------------------------------------


class TestCheckContextFeasibility:
    """Tests for context window feasibility checks."""

    @pytest.fixture
    def retriever_lookup(self):
        """Retriever configs with different chunk sizes."""
        return {
            "dense_bge_512": {"chunk_size": 512, "type": "dense"},
            "hier_bge_2048p_448c": {
                "parent_chunk_size": 2048,
                "child_chunk_size": 448,
                "type": "hierarchical",
            },
        }

    @pytest.fixture
    def constraints(self):
        """Typical constraints config."""
        return {
            "chars_per_token": 4,
            "generation_buffer": 256,
            "default_context_length": 32768,
            "model_context_lengths": {
                "vllm:google/gemma-2-2b-it": 8192,
                "vllm:microsoft/Phi-3-mini-4k-instruct": 4096,
            },
            "prompt_overhead_tokens": {
                "default": 500,
                "fewshot_3": 1500,
                "concise": 400,
            },
        }

    def test_feasible_dense_large_context(self, retriever_lookup, constraints):
        """Dense retriever + large context model should always fit."""
        feasible, reason = _check_context_feasibility(
            model="vllm:Qwen/Qwen2.5-7B-Instruct",  # default 32768
            retriever_name="dense_bge_512",
            top_k=20,
            prompt="concise",
            retriever_lookup=retriever_lookup,
            constraints=constraints,
        )
        assert feasible is True
        assert reason == ""

    def test_infeasible_hier_small_context(self, retriever_lookup, constraints):
        """Hierarchical + Gemma2-2B (8K) + top_k=20 should be pruned."""
        # 2048 chars / 4 = 512 tok/chunk × 20 = 10240 + 400 = 10640 > 8192-256=7936
        feasible, reason = _check_context_feasibility(
            model="vllm:google/gemma-2-2b-it",
            retriever_name="hier_bge_2048p_448c",
            top_k=20,
            prompt="concise",
            retriever_lookup=retriever_lookup,
            constraints=constraints,
        )
        assert feasible is False
        assert "estimated prompt" in reason

    def test_infeasible_phi3_hier_k5_fewshot(self, retriever_lookup, constraints):
        """Phi-3-mini (4K) + hierarchical + k=5 + fewshot should be pruned."""
        # 512 tok/chunk × 5 = 2560 + 1500 = 4060 > 4096-256=3840
        feasible, reason = _check_context_feasibility(
            model="vllm:microsoft/Phi-3-mini-4k-instruct",
            retriever_name="hier_bge_2048p_448c",
            top_k=5,
            prompt="fewshot_3",
            retriever_lookup=retriever_lookup,
            constraints=constraints,
        )
        assert feasible is False

    def test_feasible_phi3_dense_k3_concise(self, retriever_lookup, constraints):
        """Phi-3-mini (4K) + dense (512 chars) + k=3 + concise should fit."""
        # 128 tok/chunk × 3 = 384 + 400 = 784 < 4096-256=3840
        feasible, _ = _check_context_feasibility(
            model="vllm:microsoft/Phi-3-mini-4k-instruct",
            retriever_name="dense_bge_512",
            top_k=3,
            prompt="concise",
            retriever_lookup=retriever_lookup,
            constraints=constraints,
        )
        assert feasible is True

    def test_no_constraints_always_feasible(self, retriever_lookup):
        """Empty constraints dict disables the check entirely."""
        feasible, _ = _check_context_feasibility(
            model="vllm:google/gemma-2-2b-it",
            retriever_name="hier_bge_2048p_448c",
            top_k=20,
            prompt="fewshot_3",
            retriever_lookup=retriever_lookup,
            constraints={},
        )
        assert feasible is True

    def test_unknown_model_uses_default_context(self, retriever_lookup, constraints):
        """Models not in the map use default_context_length (32768)."""
        feasible, _ = _check_context_feasibility(
            model="vllm:unknown/new-model",
            retriever_name="hier_bge_2048p_448c",
            top_k=20,
            prompt="fewshot_3",
            retriever_lookup=retriever_lookup,
            constraints=constraints,
        )
        # 512 × 20 + 1500 = 11740 < 32768 - 256 = 32512
        assert feasible is True

    def test_unknown_prompt_uses_default_overhead(self, retriever_lookup, constraints):
        """Unknown prompt names fall back to the 'default' overhead."""
        feasible, _ = _check_context_feasibility(
            model="vllm:google/gemma-2-2b-it",
            retriever_name="dense_bge_512",
            top_k=5,
            prompt="some_new_prompt",
            retriever_lookup=retriever_lookup,
            constraints=constraints,
        )
        # 128 × 5 + 500 = 1140 < 7936 → feasible
        assert feasible is True

    def test_scalar_prompt_overhead(self, retriever_lookup):
        """prompt_overhead_tokens can be a single number instead of dict."""
        constraints = {
            "model_context_lengths": {"vllm:small": 2048},
            "prompt_overhead_tokens": 500,
            "chars_per_token": 4,
            "generation_buffer": 256,
        }
        feasible, _ = _check_context_feasibility(
            model="vllm:small",
            retriever_name="hier_bge_2048p_448c",
            top_k=3,
            prompt="concise",
            retriever_lookup=retriever_lookup,
            constraints=constraints,
        )
        # 512 × 3 + 500 = 2036 > 2048-256=1792 → infeasible
        assert feasible is False

    def test_boundary_exact_fit(self, retriever_lookup):
        """Edge case: estimated exactly equals max_input → feasible."""
        # We want chunk_tokens * top_k + overhead == model_ctx - gen_buffer
        # 128 tok/chunk * 5 = 640 + 500 = 1140, need max_input = 1140
        # model_ctx = 1140 + 256 = 1396
        constraints = {
            "model_context_lengths": {"vllm:exact": 1396},
            "prompt_overhead_tokens": {"default": 500},
            "chars_per_token": 4,
            "generation_buffer": 256,
        }
        feasible, _ = _check_context_feasibility(
            model="vllm:exact",
            retriever_name="dense_bge_512",
            top_k=5,
            prompt="anything",
            retriever_lookup=retriever_lookup,
            constraints=constraints,
        )
        assert feasible is True


# ---------------------------------------------------------------------------
# Integration: run_optuna_study (lightweight, no real experiment execution)
# ---------------------------------------------------------------------------


class TestStratifiedSearch:
    """Tests for the stratified (fixed_dims) round-robin schedule."""

    def test_fixed_dims_default(self, base_config):
        """Default fixed_dims is ['dataset', 'model'] from config without sampling."""
        # base_config has no rag.sampling, so defaults apply.
        # But each dim has only 1 value, so they're filtered out.
        sampling_cfg = base_config.get("rag", {}).get("sampling", {})
        fixed_dims = sampling_cfg.get("fixed_dims", ["dataset", "model"])
        space = _extract_search_space(base_config)
        fixed_dims = [d for d in fixed_dims if d in space and len(space[d]) > 1]
        assert fixed_dims == []  # Single-value dims are filtered out

    def test_fixed_dims_multi_value(self):
        """fixed_dims keeps dimensions with >1 value."""
        config = {
            "datasets": ["nq", "triviaqa", "hotpotqa"],
            "rag": {
                "models": ["vllm:model-a", "vllm:model-b"],
                "retriever_names": ["dense"],
                "top_k_values": [5],
                "prompts": ["concise"],
                "query_transform": ["none"],
                "reranker": {"configs": [{"enabled": False, "name": "none"}]},
            },
        }
        space = _extract_search_space(config)
        fixed_dims = ["dataset", "model"]
        fixed_dims = [d for d in fixed_dims if d in space and len(space[d]) > 1]
        assert fixed_dims == ["dataset", "model"]

    def test_schedule_uniform_coverage(self):
        """Every (dataset, model) combo gets equal (±1) trial count."""
        import random
        from itertools import product as itertools_product

        datasets = ["nq", "triviaqa", "hotpotqa"]
        models = ["model-a", "model-b"]
        fixed_dims = ["dataset", "model"]
        search_space = {"dataset": datasets, "model": models}

        fixed_combos = list(itertools_product(*(search_space[d] for d in fixed_dims)))
        assert len(fixed_combos) == 6  # 3 datasets × 2 models

        remaining = 25  # Not evenly divisible by 6
        rng = random.Random(42)
        schedule: list[dict] = []
        while len(schedule) < remaining:
            epoch = [dict(zip(fixed_dims, combo, strict=True)) for combo in fixed_combos]
            rng.shuffle(epoch)
            schedule.extend(epoch)
        schedule = schedule[:remaining]

        assert len(schedule) == 25

        # Count trials per combo
        from collections import Counter

        counts = Counter(tuple(sorted(fp.items())) for fp in schedule)

        # 25 / 6 = 4 remainder 1 → each combo gets 4 or 5
        for combo, count in counts.items():
            assert count in (4, 5), f"Combo {combo} got {count} trials (expected 4-5)"

        # All 6 combos are present
        assert len(counts) == 6

    def test_schedule_shuffled_across_epochs(self):
        """Pairs are shuffled each epoch (not always same order)."""
        import random
        from itertools import product as itertools_product

        datasets = ["nq", "triviaqa"]
        models = ["model-a", "model-b"]
        fixed_dims = ["dataset", "model"]
        search_space = {"dataset": datasets, "model": models}

        fixed_combos = list(itertools_product(*(search_space[d] for d in fixed_dims)))

        remaining = 8  # Exactly 2 epochs of 4 combos
        rng = random.Random(42)
        schedule: list[dict] = []
        while len(schedule) < remaining:
            epoch = [dict(zip(fixed_dims, combo, strict=True)) for combo in fixed_combos]
            rng.shuffle(epoch)
            schedule.extend(epoch)
        schedule = schedule[:remaining]

        epoch1 = schedule[:4]
        epoch2 = schedule[4:]

        # Both epochs cover all 4 combos
        keys1 = {tuple(sorted(d.items())) for d in epoch1}
        keys2 = {tuple(sorted(d.items())) for d in epoch2}
        assert keys1 == keys2  # Same set of combos

        # But order should differ (with high probability given seed=42)
        assert epoch1 != epoch2, "Epochs should be shuffled differently"

    def test_fixed_dims_from_config(self):
        """fixed_dims can be customized via rag.sampling.fixed_dims."""
        config = {
            "datasets": ["nq", "triviaqa"],
            "rag": {
                "models": ["model-a", "model-b"],
                "retriever_names": ["dense", "hybrid"],
                "top_k_values": [5],
                "prompts": ["concise"],
                "query_transform": ["none"],
                "reranker": {"configs": [{"enabled": False, "name": "none"}]},
                "sampling": {
                    "fixed_dims": ["dataset"],  # Only fix dataset, not model
                },
            },
        }
        space = _extract_search_space(config)
        sampling_cfg = config.get("rag", {}).get("sampling", {})
        fixed_dims = sampling_cfg.get("fixed_dims", ["dataset", "model"])
        fixed_dims = [d for d in fixed_dims if d in space and len(space[d]) > 1]

        assert fixed_dims == ["dataset"]  # Only dataset, model is optimized

    def test_empty_fixed_dims_disables_stratification(self):
        """Setting fixed_dims: [] disables stratification entirely."""
        config = {
            "datasets": ["nq", "triviaqa"],
            "rag": {
                "models": ["model-a"],
                "retriever_names": ["dense"],
                "top_k_values": [5],
                "prompts": ["concise"],
                "query_transform": ["none"],
                "reranker": {"configs": [{"enabled": False, "name": "none"}]},
                "sampling": {
                    "fixed_dims": [],  # Explicitly empty
                },
            },
        }
        sampling_cfg = config.get("rag", {}).get("sampling", {})
        fixed_dims = sampling_cfg.get("fixed_dims", ["dataset", "model"])

        assert fixed_dims == []


class TestRunOptunaStudy:
    """Smoke tests for the run_optuna_study entry point."""

    def test_study_returns_with_zero_trials(self, tmp_path, base_config):
        """Test that running with 0 remaining trials returns immediately."""
        # Pre-seed a completed trial so remaining == 0
        storage = f"sqlite:///{tmp_path / 'optuna_study.db'}"
        study = optuna.create_study(
            study_name="test_0",
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        from optuna.distributions import CategoricalDistribution

        _now = datetime.now()
        trial = optuna.trial.FrozenTrial(
            number=0,
            state=optuna.trial.TrialState.COMPLETE,
            value=0.5,
            datetime_start=_now,
            datetime_complete=_now,
            params={
                "model": "vllm:test/model-a",
                "retriever": "dense_bge",
                "top_k": 5,
                "prompt": "concise",
                "query_transform": "none",
                "dataset": "nq",
                "reranker": "none",
            },
            distributions={
                "model": CategoricalDistribution(["vllm:test/model-a"]),
                "retriever": CategoricalDistribution(["dense_bge"]),
                "top_k": CategoricalDistribution([5]),
                "prompt": CategoricalDistribution(["concise"]),
                "query_transform": CategoricalDistribution(["none"]),
                "dataset": CategoricalDistribution(["nq"]),
                "reranker": CategoricalDistribution(["none"]),
            },
            user_attrs={},
            system_attrs={},
            intermediate_values={},
            trial_id=0,
        )
        study.add_trial(trial)

        # Run with n_trials=1 — already 1 completed → 0 remaining
        result = run_optuna_study(
            config=base_config,
            n_trials=1,
            output_dir=tmp_path,
            sampler_mode="random",
            optimize_metric="f1",
            study_name="test_0",
            seed=42,
        )

        assert isinstance(result, optuna.Study)
        assert len(result.trials) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
