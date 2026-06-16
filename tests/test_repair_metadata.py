"""Tests for the repair_metadata script.

Verifies that the repair logic correctly:
- Infers model specs from directory names
- Normalises dataset aliases (natural_questions -> nq)
- Fixes name mismatches (trailing _none)
- Creates metadata.json for experiments missing it
"""

# Import repair helpers — the script is not a package, so we add its dir to
# sys.path and import the module.
import json
import sys
from pathlib import Path

import pytest

_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import repair_metadata as rm  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def study_dir(tmp_path: Path) -> Path:
    """Create a temporary study directory with experiment subdirectories."""
    study = tmp_path / "test_study"
    study.mkdir()
    return study


def _create_experiment(
    study: Path,
    dir_name: str,
    metadata: dict | None = None,
    results: dict | None = None,
) -> Path:
    """Helper to create an experiment directory with optional JSON files."""
    exp_dir = study / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    if metadata is not None:
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    if results is not None:
        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    return exp_dir


# ---------------------------------------------------------------------------
# Model inference tests
# ---------------------------------------------------------------------------


class TestInferModelSpec:
    """Test _infer_model_spec from directory names."""

    def test_gemma_2b(self):
        name = "direct_vllm_google_gemma22bit_concise_nq"
        assert rm._infer_model_spec(name) == "vllm:google/gemma-2-2b-it"

    def test_gemma_9b(self):
        name = "rag_vllm_google_gemma29bit_dense_bge_large_512_k5_concise_nq"
        assert rm._infer_model_spec(name) == "vllm:google/gemma-2-9b-it"

    def test_phi3(self):
        name = "rag_vllm_microsoft_Phi3mini4kinstruct_dense_bge_large_512_k3_hyde_concise_triviaqa"
        assert rm._infer_model_spec(name) == "vllm:microsoft/Phi-3-mini-4k-instruct"

    def test_llama(self):
        name = "direct_vllm_metallama_Llama3.23BInstruct_fewshot_3_nq"
        assert rm._infer_model_spec(name) == "vllm:meta-llama/Llama-3.2-3B-Instruct"

    def test_mistral(self):
        name = "iterative_rag_iter3_stopok_vllm_mistralai_Mistral7BInstructv0.3_hier_bge_large_2048p_448c_k5_fewshot_3_nq"
        assert rm._infer_model_spec(name) == "vllm:mistralai/Mistral-7B-Instruct-v0.3"

    def test_qwen_3b(self):
        name = "rag_vllm_Qwen_Qwen2.53BInstruct_dense_bge_large_512_k5_concise_nq"
        assert rm._infer_model_spec(name) == "vllm:Qwen/Qwen2.5-3B-Instruct"

    def test_qwen_1_5b(self):
        name = "direct_vllm_Qwen_Qwen2.51.5BInstruct_concise_hotpotqa"
        assert rm._infer_model_spec(name) == "vllm:Qwen/Qwen2.5-1.5B-Instruct"

    def test_qwen_7b(self):
        name = "rag_vllm_Qwen_Qwen2.57BInstruct_dense_bge_large_512_k5_concise_nq"
        assert rm._infer_model_spec(name) == "vllm:Qwen/Qwen2.5-7B-Instruct"

    def test_unknown_model_returns_none(self):
        name = "direct_vllm_unknownmodel_concise_nq"
        assert rm._infer_model_spec(name) is None


# ---------------------------------------------------------------------------
# Experiment type inference tests
# ---------------------------------------------------------------------------


class TestInferExpType:
    """Test _infer_exp_type from directory names."""

    def test_direct(self):
        assert rm._infer_exp_type("direct_vllm_test_concise_nq") == "direct"

    def test_rag(self):
        assert rm._infer_exp_type("rag_vllm_test_dense_k5_concise_nq") == "rag"

    def test_iterative_rag(self):
        assert rm._infer_exp_type("iterative_rag_iter1_stopok_vllm_test_k5_nq") == "rag"

    def test_self_rag(self):
        assert rm._infer_exp_type("self_rag_vllm_test_k5_nq") == "rag"


# ---------------------------------------------------------------------------
# Dataset inference tests
# ---------------------------------------------------------------------------


class TestInferDataset:
    """Test _infer_dataset from directory names."""

    def test_nq(self):
        assert rm._infer_dataset("direct_vllm_test_concise_nq") == "nq"

    def test_hotpotqa(self):
        assert rm._infer_dataset("direct_vllm_test_concise_hotpotqa") == "hotpotqa"

    def test_triviaqa(self):
        assert rm._infer_dataset("direct_vllm_test_concise_triviaqa") == "triviaqa"

    def test_no_dataset(self):
        assert rm._infer_dataset("something_random") is None


# ---------------------------------------------------------------------------
# Prompt inference tests
# ---------------------------------------------------------------------------


class TestInferPrompt:
    """Test _infer_prompt from directory names."""

    def test_concise(self):
        assert rm._infer_prompt("direct_vllm_test_concise_nq", "nq") == "concise"

    def test_fewshot_3(self):
        assert rm._infer_prompt("direct_vllm_test_fewshot_3_nq", "nq") == "fewshot_3"

    def test_concise_strict(self):
        assert rm._infer_prompt("rag_vllm_test_k5_concise_strict_nq", "nq") == "concise_strict"

    def test_extractive_quoted(self):
        assert (
            rm._infer_prompt("rag_vllm_test_k5_extractive_quoted_nq", "nq") == "extractive_quoted"
        )


# ---------------------------------------------------------------------------
# Top-K inference tests
# ---------------------------------------------------------------------------


class TestInferTopK:
    """Test _infer_top_k from directory names."""

    def test_k5(self):
        assert rm._infer_top_k("rag_vllm_test_dense_k5_concise_nq") == 5

    def test_k10(self):
        assert rm._infer_top_k("rag_vllm_test_dense_k10_concise_nq") == 10

    def test_k20(self):
        assert rm._infer_top_k("rag_vllm_test_hier_k20_concise_nq") == 20

    def test_no_k(self):
        assert rm._infer_top_k("direct_vllm_test_concise_nq") is None


# ---------------------------------------------------------------------------
# Integration: plan_repairs
# ---------------------------------------------------------------------------


class TestPlanRepairs:
    """Test plan_repairs on realistic directory structures."""

    def test_empty_model_gets_backfilled(self, study_dir):
        """Experiments with empty model should be repaired."""
        _create_experiment(
            study_dir,
            "rag_vllm_google_gemma29bit_dense_bge_m3_512_k5_multiquery_fewshot_1_hotpotqa",
            metadata={
                "name": "rag_vllm_google_gemma29bit_dense_bge_m3_512_k5_multiquery_fewshot_1_hotpotqa",
                "model": "",
            },
        )

        actions = rm.plan_repairs(study_dir)

        assert len(actions) == 1
        repair_fields = {r[0] for r in actions[0]["repairs"]}
        assert "model" in repair_fields

        model_repair = next(r for r in actions[0]["repairs"] if r[0] == "model")
        assert model_repair[2] == "vllm:google/gemma-2-9b-it"

    def test_dataset_alias_normalised(self, study_dir):
        """natural_questions should be normalised to nq."""
        _create_experiment(
            study_dir,
            "iterative_rag_test_nq",
            metadata={
                "name": "iterative_rag_test_nq",
                "model": "vllm:test/model",
                "type": "rag",
                "prompt": "concise",
                "dataset": "natural_questions",
            },
        )

        actions = rm.plan_repairs(study_dir)

        assert len(actions) == 1
        ds_repair = next(r for r in actions[0]["repairs"] if r[0] == "dataset")
        assert ds_repair[1] == "natural_questions"
        assert ds_repair[2] == "nq"

    def test_name_mismatch_fixed(self, study_dir):
        """metadata name with trailing _none should be corrected to match dir."""
        _create_experiment(
            study_dir,
            "direct_vllm_test_concise_nq",
            metadata={
                "name": "direct_vllm_test_concise_nq_none",
                "model": "vllm:test/model",
                "type": "direct",
                "prompt": "concise",
                "dataset": "nq",
            },
        )

        actions = rm.plan_repairs(study_dir)

        assert len(actions) == 1
        name_repair = next(r for r in actions[0]["repairs"] if r[0] == "name")
        assert name_repair[1] == "direct_vllm_test_concise_nq_none"
        assert name_repair[2] == "direct_vllm_test_concise_nq"

    def test_missing_metadata_creates_new(self, study_dir):
        """Experiment with no metadata.json should get one created."""
        exp_dir = (
            study_dir
            / "rag_vllm_metallama_Llama3.23BInstruct_dense_bge_large_512_k10_concise_triviaqa"
        )
        exp_dir.mkdir()

        actions = rm.plan_repairs(study_dir)

        assert len(actions) == 1
        assert actions[0]["create_new"] is True
        repair_fields = {r[0] for r in actions[0]["repairs"]}
        assert "name" in repair_fields
        assert "model" in repair_fields
        assert "type" in repair_fields
        assert "dataset" in repair_fields

    def test_clean_experiment_no_repairs(self, study_dir):
        """Fully populated experiment should require no repairs."""
        _create_experiment(
            study_dir,
            "direct_vllm_google_gemma22bit_concise_nq",
            metadata={
                "name": "direct_vllm_google_gemma22bit_concise_nq",
                "model": "vllm:google/gemma-2-2b-it",
                "type": "direct",
                "prompt": "concise",
                "dataset": "nq",
            },
        )

        actions = rm.plan_repairs(study_dir)
        assert len(actions) == 0

    def test_skip_dirs_are_ignored(self, study_dir):
        """Directories in SKIP_DIRS should be ignored."""
        (study_dir / "_archived_fake_reranked").mkdir()
        (study_dir / "__pycache__").mkdir()

        actions = rm.plan_repairs(study_dir)
        assert len(actions) == 0


# ---------------------------------------------------------------------------
# Integration: execute_repairs
# ---------------------------------------------------------------------------


class TestExecuteRepairs:
    """Test execute_repairs actually writes correct files."""

    def test_backfills_empty_model(self, study_dir):
        """Execute should write inferred model into metadata.json."""
        _create_experiment(
            study_dir,
            "rag_vllm_microsoft_Phi3mini4kinstruct_dense_bge_large_512_k3_hyde_concise_triviaqa",
            metadata={
                "name": "rag_vllm_microsoft_Phi3mini4kinstruct_dense_bge_large_512_k3_hyde_concise_triviaqa",
                "model": "",
                "dataset": "triviaqa",
            },
        )

        actions = rm.plan_repairs(study_dir)
        rm.execute_repairs(actions)

        meta_path = (
            study_dir
            / "rag_vllm_microsoft_Phi3mini4kinstruct_dense_bge_large_512_k3_hyde_concise_triviaqa"
            / "metadata.json"
        )
        metadata = json.loads(meta_path.read_text())
        assert metadata["model"] == "vllm:microsoft/Phi-3-mini-4k-instruct"

    def test_normalises_dataset_in_metadata(self, study_dir):
        """Execute should replace natural_questions with nq."""
        _create_experiment(
            study_dir,
            "test_exp_nq",
            metadata={
                "name": "test_exp_nq",
                "model": "vllm:test/model",
                "type": "rag",
                "prompt": "concise",
                "dataset": "natural_questions",
            },
        )

        actions = rm.plan_repairs(study_dir)
        rm.execute_repairs(actions)

        metadata = json.loads((study_dir / "test_exp_nq" / "metadata.json").read_text())
        assert metadata["dataset"] == "nq"

    def test_normalises_dataset_in_results(self, study_dir):
        """Execute should also fix dataset in results.json if present."""
        _create_experiment(
            study_dir,
            "test_exp_nq",
            metadata={
                "name": "test_exp_nq",
                "model": "vllm:test/model",
                "type": "rag",
                "prompt": "concise",
                "dataset": "natural_questions",
            },
            results={
                "name": "test_exp_nq",
                "dataset": "natural_questions",
                "metrics": {"f1": 0.5},
            },
        )

        actions = rm.plan_repairs(study_dir)
        rm.execute_repairs(actions)

        results = json.loads((study_dir / "test_exp_nq" / "results.json").read_text())
        assert results["dataset"] == "nq"

    def test_fixes_name_mismatch_in_results(self, study_dir):
        """Execute should fix name in results.json when dir/metadata name differ."""
        _create_experiment(
            study_dir,
            "direct_vllm_test_concise_nq",
            metadata={
                "name": "direct_vllm_test_concise_nq_none",
                "model": "vllm:test/model",
                "type": "direct",
                "prompt": "concise",
                "dataset": "nq",
            },
            results={
                "name": "direct_vllm_test_concise_nq_none",
                "metrics": {"f1": 0.3},
            },
        )

        actions = rm.plan_repairs(study_dir)
        rm.execute_repairs(actions)

        metadata = json.loads(
            (study_dir / "direct_vllm_test_concise_nq" / "metadata.json").read_text()
        )
        results = json.loads(
            (study_dir / "direct_vllm_test_concise_nq" / "results.json").read_text()
        )
        assert metadata["name"] == "direct_vllm_test_concise_nq"
        assert results["name"] == "direct_vllm_test_concise_nq"

    def test_creates_metadata_for_missing(self, study_dir):
        """Execute should create metadata.json when none exists."""
        exp_dir = study_dir / "rag_vllm_google_gemma22bit_dense_bge_large_512_k10_concise_triviaqa"
        exp_dir.mkdir()

        actions = rm.plan_repairs(study_dir)
        rm.execute_repairs(actions)

        meta_path = exp_dir / "metadata.json"
        assert meta_path.exists()
        metadata = json.loads(meta_path.read_text())
        assert (
            metadata["name"]
            == "rag_vllm_google_gemma22bit_dense_bge_large_512_k10_concise_triviaqa"
        )
        assert metadata["model"] == "vllm:google/gemma-2-2b-it"
        assert metadata["type"] == "rag"
        assert metadata["dataset"] == "triviaqa"
        assert metadata["top_k"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
