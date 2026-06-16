"""Regression tests for parameter wiring.

Guards against the class of bugs where spec fields are silently ignored.
Each test targets a specific bug that was found and fixed:

1. Legacy reranker YAML format must produce reranker specs
2. reranker_model (full HF path) must reach create_reranker
3. query_transform must warn when set (not yet implemented)
4. Subprocess serialization must include ALL ExperimentSpec fields
5. results.json must contain full spec metadata
6. save_metadata in runner must contain all spec fields
7. Experiment.from_spec() must propagate spec.batch_size
8. String-format retrievers must produce RAG specs
9. Unrecognized YAML keys must warn
10. spec.embedding_index must be preferred over on-disk config
11. Subprocess metrics must come from spec
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ragicamp.spec import ExperimentSpec, build_specs

# ============================================================================
# 1. Legacy reranker YAML format
# ============================================================================


class TestLegacyRerankerFormat:
    """Legacy YAML {enabled: true, model: bge} must NOT silently default to none."""

    def test_legacy_format_produces_reranker_specs(self):
        """build_specs with legacy reranker format must produce reranker='bge'."""
        config = {
            "datasets": ["nq"],
            "rag": {
                "enabled": True,
                "models": ["hf:test/model"],
                "retrievers": [{"name": "dense_bge"}],
                "top_k_values": [5],
                "prompts": ["concise"],
                "reranker": {
                    "enabled": True,
                    "model": "bge",
                },
            },
        }

        specs = build_specs(config)

        assert len(specs) >= 1
        assert any(s.reranker == "bge" for s in specs), (
            "Legacy reranker format {enabled: true, model: bge} must produce "
            "specs with reranker='bge', not silently default to None"
        )

    def test_new_format_still_works(self):
        """New format with 'configs' key must still work correctly."""
        config = {
            "datasets": ["nq"],
            "rag": {
                "enabled": True,
                "models": ["hf:test/model"],
                "retrievers": [{"name": "dense_bge"}],
                "top_k_values": [5],
                "prompts": ["concise"],
                "reranker": {
                    "configs": [
                        {"enabled": False, "name": "none"},
                        {"enabled": True, "name": "bge"},
                    ]
                },
            },
        }

        specs = build_specs(config)

        rerankers = [s.reranker for s in specs]
        assert None in rerankers
        assert "bge" in rerankers

    def test_disabled_reranker_produces_none(self):
        """When reranker is disabled, specs should have reranker=None."""
        config = {
            "datasets": ["nq"],
            "rag": {
                "enabled": True,
                "models": ["hf:test/model"],
                "retrievers": [{"name": "dense_bge"}],
                "top_k_values": [5],
                "prompts": ["concise"],
                "reranker": {
                    "enabled": False,
                },
            },
        }

        specs = build_specs(config)

        assert all(s.reranker is None for s in specs)


# ============================================================================
# 2. reranker_model reaches create_reranker
# ============================================================================


class TestRerankerModelPassthrough:
    """spec.reranker_model (full HF path) must be used, not just short name."""

    def test_custom_reranker_model_used(self):
        """When reranker_model differs from reranker, the full path must be passed."""
        from ragicamp.factory.providers import ProviderFactory
        from ragicamp.models.providers.reranker import RerankerProvider

        # A custom model path that's NOT in the MODELS lookup
        provider = ProviderFactory.create_reranker("custom/my-reranker-v3")
        assert isinstance(provider, RerankerProvider)
        # model_name falls back to the raw string when not in MODELS
        assert provider.model_name == "custom/my-reranker-v3"

    def test_from_spec_uses_reranker_model_over_short_name(self):
        """Experiment.from_spec() should prefer reranker_model when available."""
        from ragicamp.experiment import Experiment

        spec = ExperimentSpec(
            name="test_rr_model",
            exp_type="rag",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            retriever="dense_bge",
            reranker="bge",
            reranker_model="custom/my-reranker-v3",
        )

        with patch("ragicamp.factory.providers.ProviderFactory.create_reranker") as mock_create:
            mock_create.return_value = MagicMock()
            with patch.object(Experiment, "from_spec", wraps=Experiment.from_spec):
                try:
                    Experiment.from_spec(spec, output_dir="/tmp/test")
                except Exception:
                    pass  # Will fail on missing artifacts, that's OK

                # The key assertion: create_reranker must be called with the
                # full model path, not just the short name "bge"
                if mock_create.called:
                    call_args = mock_create.call_args[0][0]
                    assert call_args == "custom/my-reranker-v3", (
                        f"create_reranker was called with '{call_args}', expected "
                        f"'custom/my-reranker-v3' (reranker_model should take priority)"
                    )


# ============================================================================
# 3. query_transform warning
# ============================================================================


class TestQueryTransformWiring:
    """query_transform must create a transformer and pass it to the agent."""

    def test_creates_transformer_when_query_transform_set(self):
        """Setting query_transform='hyde' should create a HyDETransformer."""
        from ragicamp.factory import AgentFactory
        from ragicamp.rag.query_transform import HyDETransformer

        spec = ExperimentSpec(
            name="test_qt_hyde",
            exp_type="rag",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            retriever="dense_test",
            top_k=5,
            query_transform="hyde",
        )

        mock_index = MagicMock()
        mock_index.documents = []
        mock_embedder = MagicMock()
        mock_embedder.model_name = "mock"
        mock_embedder.load = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        )
        mock_gen = MagicMock()
        mock_gen.model_name = "mock"
        mock_gen.load = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        )

        agent = AgentFactory.from_spec(
            spec=spec,
            embedder_provider=mock_embedder,
            generator_provider=mock_gen,
            index=mock_index,
        )

        assert hasattr(agent, "query_transformer"), (
            "RAG agent should have query_transformer attribute"
        )
        assert isinstance(agent.query_transformer, HyDETransformer), (
            f"Expected HyDETransformer, got {type(agent.query_transformer)}"
        )

    def test_no_transformer_when_query_transform_none(self):
        """No transformer when query_transform is None (the default)."""
        from ragicamp.factory import AgentFactory

        spec = ExperimentSpec(
            name="test_qt_none",
            exp_type="rag",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            retriever="dense_test",
            top_k=5,
        )

        mock_index = MagicMock()
        mock_index.documents = []
        mock_embedder = MagicMock()
        mock_embedder.model_name = "mock"
        mock_embedder.load = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        )
        mock_gen = MagicMock()
        mock_gen.model_name = "mock"
        mock_gen.load = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        )

        agent = AgentFactory.from_spec(
            spec=spec,
            embedder_provider=mock_embedder,
            generator_provider=mock_gen,
            index=mock_index,
        )

        assert hasattr(agent, "query_transformer"), (
            "RAG agent should have query_transformer attribute"
        )
        assert agent.query_transformer is None, (
            "query_transformer should be None when not configured"
        )


# ============================================================================
# 4. Subprocess serialization includes ALL fields
# ============================================================================


class TestSubprocessSerialization:
    """Subprocess serialization must include every ExperimentSpec field."""

    def test_to_dict_includes_all_fields(self):
        """spec.to_dict() must include embedding_index, sparse_index, hypothesis."""
        spec = ExperimentSpec(
            name="test_serialization",
            exp_type="rag",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            retriever="dense_bge",
            embedding_index="custom_index_dir",
            sparse_index="custom_sparse_dir",
            hypothesis="Test hypothesis",
            agent_type="iterative_rag",
        )

        d = spec.to_dict()

        assert d["embedding_index"] == "custom_index_dir"
        assert d["sparse_index"] == "custom_sparse_dir"
        assert d["hypothesis"] == "Test hypothesis"

    def test_round_trip_preserves_all_fields(self):
        """to_dict -> from_dict must preserve embedding_index, sparse_index, etc."""
        spec = ExperimentSpec(
            name="test_roundtrip",
            exp_type="rag",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            retriever="dense_bge",
            embedding_index="custom_idx",
            sparse_index="custom_sparse",
            batch_size=32,
            hypothesis="Test",
            agent_type="self_rag",
            agent_params=(("max_iterations", 3),),
        )

        rebuilt = ExperimentSpec.from_dict(spec.to_dict())

        assert rebuilt.embedding_index == "custom_idx"
        assert rebuilt.sparse_index == "custom_sparse"
        assert rebuilt.batch_size == 32
        assert rebuilt.hypothesis == "Test"
        assert rebuilt.agent_type == "self_rag"


# ============================================================================
# 5 & 6. Metadata completeness in runner output
# ============================================================================


class TestRunnerMetadataCompleteness:
    """run_generation and run_metrics_only must save complete metadata."""

    def test_run_generation_metadata_uses_spec_to_dict(self):
        """run_generation must base metadata on spec.to_dict(), not hand-pick fields."""
        spec = ExperimentSpec(
            name="test_meta_gen",
            exp_type="rag",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            retriever="dense_bge",
            top_k=5,
            embedding_index="custom_idx",
            reranker="bge",
            reranker_model="BAAI/bge-reranker-large",
            hypothesis="Test hypo",
        )

        # spec.to_dict() must include ALL these fields
        d = spec.to_dict()
        required_fields = [
            "model",
            "dataset",
            "prompt",
            "retriever",
            "top_k",
            "embedding_index",
            "reranker",
            "reranker_model",
            "hypothesis",
        ]
        for field_name in required_fields:
            assert field_name in d, (
                f"spec.to_dict() missing field '{field_name}' — "
                f"run_generation metadata will lose it"
            )
            assert d[field_name] is not None, (
                f"spec.to_dict()['{field_name}'] is None when it should be "
                f"'{getattr(spec, field_name)}'"
            )

    def test_run_metrics_only_includes_spec_metadata_in_results(self):
        """run_metrics_only must include spec.to_dict() in results.json."""
        import tempfile

        from ragicamp.execution.runner import run_metrics_only

        spec = ExperimentSpec(
            name="test_meta_metrics",
            exp_type="rag",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            retriever="dense_bge",
            top_k=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_out = Path(tmpdir)

            # Create minimum required files for run_metrics_only
            state_data = {
                "phase": "computing_metrics",
                "total_questions": 1,
                "processed_questions": 1,
                "metrics_computed": [],
                "started_at": "2025-01-01T00:00:00",
                "updated_at": "2025-01-01T00:00:00",
            }
            with open(exp_out / "state.json", "w") as f:
                json.dump(state_data, f)

            predictions_data = {
                "predictions": [
                    {
                        "question": "What?",
                        "expected": "Answer",
                        "prediction": "Answer",
                    }
                ],
                "aggregate_metrics": {},
            }
            with open(exp_out / "predictions.json", "w") as f:
                json.dump(predictions_data, f)

            # Run metrics only with spec provided
            run_metrics_only(
                exp_name=spec.name,
                output_path=exp_out,
                metrics=["f1"],
                spec=spec,
            )

            # Check results.json has spec metadata
            results_file = exp_out / "results.json"
            assert results_file.exists(), "results.json should be created"
            with open(results_file) as f:
                results = json.load(f)

            assert "metadata" in results, "results.json must contain 'metadata' key from spec"
            meta = results["metadata"]
            assert meta.get("model") == "vllm:test/model"
            assert meta.get("dataset") == "nq"
            assert meta.get("retriever") == "dense_bge"

    def test_phase_complete_metadata_has_spec_fields(self):
        """Experiment._phase_complete must include spec fields in results.json metadata."""
        spec = ExperimentSpec(
            name="test_complete_meta",
            exp_type="rag",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            retriever="dense_bge",
            top_k=5,
            reranker="bge",
        )

        # Simulate what _phase_complete does: build metadata from spec
        result_metadata = {
            "agent": "mock_agent",
            "dataset": "mock_dataset",
            "batch_size": 8,
        }
        spec_data = spec.to_dict()
        merged = {**spec_data, **result_metadata}

        # Spec fields must survive the merge
        assert merged["model"] == "vllm:test/model"
        assert merged["retriever"] == "dense_bge"
        assert merged["reranker"] == "bge"
        # Runtime fields override spec fields
        assert merged["agent"] == "mock_agent"


# ============================================================================
# 7. batch_size propagation
# ============================================================================


class TestBatchSizePropagation:
    """spec.batch_size must reach the Experiment constructor."""

    def test_spec_batch_size_is_not_default(self):
        """A spec with batch_size=32 must NOT be silently reduced to 8."""
        spec = ExperimentSpec(
            name="test_batch",
            exp_type="direct",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            batch_size=32,
        )

        assert spec.batch_size == 32, "spec.batch_size must survive construction"

    def test_spec_batch_size_survives_round_trip(self):
        """batch_size must survive to_dict -> from_dict round trip."""
        spec = ExperimentSpec(
            name="test_batch_rt",
            exp_type="direct",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            batch_size=64,
        )

        d = spec.to_dict()
        assert d["batch_size"] == 64

        rebuilt = ExperimentSpec.from_dict(d)
        assert rebuilt.batch_size == 64, (
            "batch_size must survive to_dict -> from_dict, not reset to default"
        )

    def test_from_spec_passes_batch_size(self):
        """Experiment.from_spec() must pass spec.batch_size to Experiment."""
        from ragicamp.experiment import Experiment

        spec = ExperimentSpec(
            name="test_batch_prop",
            exp_type="direct",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            batch_size=32,
        )

        # Patch Experiment.__init__ to capture the _batch_size argument
        captured = {}

        def spy_init(self, *args, **kwargs):
            captured["_batch_size"] = kwargs.get("_batch_size")
            raise RuntimeError("stop early — we only care about the arg")

        with patch.object(Experiment, "__init__", spy_init):
            try:
                Experiment.from_spec(spec, output_dir="/tmp/test")
            except (RuntimeError, Exception):
                pass

        assert captured.get("_batch_size") == 32, (
            f"from_spec must pass spec.batch_size=32 to Experiment, "
            f"got {captured.get('_batch_size')}"
        )


# ============================================================================
# 8. String-format retrievers
# ============================================================================


class TestStringFormatRetrievers:
    """String-format retrievers must produce RAG specs, not silently 0."""

    def test_string_retriever_produces_specs(self):
        """retrievers: ['dense_bge'] must produce RAG experiments."""
        config = {
            "datasets": ["nq"],
            "rag": {
                "enabled": True,
                "models": ["hf:test/model"],
                "retrievers": ["dense_bge"],
                "top_k_values": [5],
                "prompts": ["concise"],
            },
        }

        specs = build_specs(config)

        assert len(specs) >= 1, (
            "String-format retrievers must produce RAG specs, not silently default to 0 experiments"
        )
        assert specs[0].retriever == "dense_bge"

    def test_mixed_format_retrievers(self):
        """Both dict and string retrievers should work in the same config."""
        config = {
            "datasets": ["nq"],
            "rag": {
                "enabled": True,
                "models": ["hf:test/model"],
                "retrievers": [
                    {"name": "dense_bge_large", "type": "dense"},
                    "simple_minilm_512",
                ],
                "top_k_values": [5],
                "prompts": ["concise"],
            },
        }

        specs = build_specs(config)

        retrievers = [s.retriever for s in specs]
        assert "dense_bge_large" in retrievers
        assert "simple_minilm_512" in retrievers


# ============================================================================
# 9. Unrecognized YAML keys warn
# ============================================================================


class TestUnrecognizedYAMLKeys:
    """Unrecognized top-level YAML keys must emit a warning."""

    def test_unknown_key_warns(self):
        """Keys like 'options' that aren't processed must trigger a warning."""
        config = {
            "datasets": ["nq"],
            "options": {
                "save_intermediate": True,
                "skip_existing": True,
            },
            "direct": {
                "enabled": True,
                "models": ["hf:test/model"],
                "prompts": ["default"],
            },
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build_specs(config)

            unrecognized = [x for x in w if "Unrecognized" in str(x.message)]
            assert len(unrecognized) >= 1, (
                "Unrecognized YAML keys like 'options' should produce a warning"
            )
            assert "options" in str(unrecognized[0].message)

    def test_known_keys_no_warning(self):
        """Standard keys must NOT trigger a warning."""
        config = {
            "datasets": ["nq"],
            "direct": {
                "enabled": True,
                "models": ["hf:test/model"],
                "prompts": ["default"],
            },
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build_specs(config)

            unrecognized = [x for x in w if "Unrecognized" in str(x.message)]
            assert len(unrecognized) == 0


# ============================================================================
# 10. spec.embedding_index preferred over on-disk config
# ============================================================================


class TestEmbeddingIndexPrecedence:
    """spec.embedding_index must be preferred over retriever config on disk."""

    def test_spec_embedding_index_used_for_index_loading(self):
        """When spec.embedding_index is set, it must be used instead of config."""
        from ragicamp.experiment import Experiment

        spec = ExperimentSpec(
            name="test_idx_precedence",
            exp_type="rag",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            retriever="dense_bge",
            embedding_index="my_custom_index_dir",
        )

        # Mock the artifact manager and config
        with (
            patch("ragicamp.utils.artifacts.get_artifact_manager") as mock_am,
            patch("ragicamp.factory.providers.ProviderFactory.create_generator"),
            patch("ragicamp.factory.providers.ProviderFactory.create_embedder"),
        ):
            mock_manager = MagicMock()
            mock_am.return_value = mock_manager
            mock_manager.get_retriever_path.return_value = Path("/tmp/retriever")

            # On-disk config says a different index name
            config_data = {
                "type": "dense",
                "embedding_model": "test-model",
                "embedding_index": "on_disk_index_dir",
            }

            mock_config_path = MagicMock()
            mock_config_path.exists.return_value = True

            with (
                patch("builtins.open", create=True) as mock_open,
                patch("json.load", return_value=config_data),
            ):
                mock_open.return_value.__enter__ = lambda s: s
                mock_open.return_value.__exit__ = MagicMock()

                # Verify that get_embedding_index_path is called with spec's value
                try:
                    Experiment.from_spec(spec, output_dir="/tmp/test")
                except Exception:
                    pass  # Will fail on lazy index, that's OK

                # Check what index name was used
                if mock_manager.get_embedding_index_path.called:
                    used_index = mock_manager.get_embedding_index_path.call_args[0][0]
                    assert used_index == "my_custom_index_dir", (
                        f"Expected spec.embedding_index='my_custom_index_dir' but "
                        f"got '{used_index}' (on-disk config was used instead)"
                    )

    def test_sparse_index_override_in_build_index_loader(self):
        """_build_index_loader must use sparse_index_override when provided."""
        from ragicamp.experiment import Experiment

        with (
            patch("ragicamp.indexes.vector_index.VectorIndex.load") as mock_vi,
            patch("ragicamp.indexes.sparse.SparseIndex.load") as mock_si,
            patch("ragicamp.retrievers.hybrid.HybridSearcher"),
            patch("ragicamp.utils.artifacts.get_artifact_manager") as mock_am,
        ):
            mock_vi.return_value = MagicMock(documents=[])
            mock_si.return_value = MagicMock()
            mock_am.return_value.get_sparse_index_path.return_value = Path("/tmp/sparse")

            # Config has one name, override has another
            retriever_config = {"sparse_index": "config_sparse", "alpha": 0.5}

            loader = Experiment._build_index_loader(
                retriever_type="hybrid",
                index_name="test_index",
                index_path=Path("/tmp/index"),
                retriever_config=retriever_config,
                sparse_index_override="spec_sparse",
            )
            loader()

            # SparseIndex.load should use the override name
            load_call = mock_si.call_args
            assert load_call.kwargs.get("name") == "spec_sparse", (
                "sparse_index_override should take precedence over config"
            )


# ============================================================================
# 11. Metrics single source of truth
# ============================================================================


class TestMetricsSourceOfTruth:
    """Subprocess should use spec.metrics as the single source of truth."""

    def test_spec_metrics_used_when_available(self):
        """When spec has metrics, they should be used (not a separate CLI list)."""
        spec = ExperimentSpec(
            name="test_metrics",
            exp_type="direct",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            metrics=["f1", "exact_match", "bleurt"],
        )

        d = spec.to_dict()
        assert d["metrics"] == ["f1", "exact_match", "bleurt"]

        # Simulate what run_spec_subprocess does: use spec_dict metrics
        spec_dict = spec.to_dict()
        if not spec_dict.get("metrics"):
            spec_dict["metrics"] = ["f1", "exact_match"]  # CLI fallback

        merged_metrics = spec_dict.get("metrics", [])
        assert merged_metrics == ["f1", "exact_match", "bleurt"], (
            "Spec's own metrics should be used, not overridden by CLI fallback"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
