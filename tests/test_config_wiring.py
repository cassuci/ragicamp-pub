"""Tests for config-to-component wiring.

Verifies that every configuration value in ExperimentSpec actually reaches
the component that should use it.  These tests guard against "silent bugs"
where a config field is defined, named, and serialised but the code path
that should consume it has no branch for it.

Each test follows the pattern:
    1. Build an ExperimentSpec with a specific config value set.
    2. Push it through the factory / builder / loader.
    3. Assert the downstream component received the value.
"""

from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ragicamp.agents import FixedRAGAgent, IterativeRAGAgent, SelfRAGAgent
from ragicamp.agents.base import Query
from ragicamp.core.types import Document, SearchResult
from ragicamp.factory import AgentFactory
from ragicamp.factory.providers import ProviderFactory
from ragicamp.spec.experiment import ExperimentSpec

# ---------------------------------------------------------------------------
# Shared mocks
# ---------------------------------------------------------------------------


class _MockEmbedder:
    def batch_encode(self, texts):
        import numpy as np

        return np.random.randn(len(texts), 64).astype("float32")


class _MockGenerator:
    model_name = "mock-gen"

    def batch_generate(self, prompts, **kw):
        return ["answer"] * len(prompts)


class _MockReranker:
    """Mock cross-encoder that records calls for assertion."""

    def __init__(self):
        self.calls: list[dict] = []

    def batch_rerank(self, queries, docs_list, top_k):
        self.calls.append({"queries": queries, "top_k": top_k})
        # Return docs trimmed to top_k (reversed to prove reranking happened)
        result = []
        for docs in docs_list:
            reranked = list(reversed(docs[:top_k]))
            for i, doc in enumerate(reranked):
                doc.score = 1.0 - i * 0.1
            result.append(reranked)
        return result


@contextmanager
def _mock_provider(obj):
    """Context-manager provider that yields *obj* directly."""
    yield obj


def _make_provider(obj, model_name="mock"):
    """Build a mock provider wrapping *obj*."""
    p = MagicMock()
    p.model_name = model_name
    p.load = lambda: _mock_provider(obj)
    return p


SAMPLE_DOCS = [Document(id=f"d{i}", text=f"Document {i} text.", metadata={}) for i in range(20)]


class _MockIndex:
    """Mock search backend that returns deterministic results."""

    documents = SAMPLE_DOCS

    def batch_search(self, embeddings, top_k=5, **kw):
        results = []
        for _ in embeddings:
            results.append(
                [
                    SearchResult(document=SAMPLE_DOCS[j], score=1.0 - j * 0.05, rank=j)
                    for j in range(top_k)
                ]
            )
        return results


# ---------------------------------------------------------------------------
# Bug 1 – Hybrid retriever loading
# ---------------------------------------------------------------------------


class TestHybridRetrieverWiring:
    """Ensure hybrid retriever config is not silently treated as dense."""

    def test_build_index_loader_creates_hybrid_searcher(self):
        """_build_index_loader must create HybridSearcher for type='hybrid'."""
        from ragicamp.experiment import Experiment

        retriever_config = {
            "sparse_index": "test_sparse_bm25",
            "alpha": 0.7,
        }

        # We can't actually load indexes from disk in unit tests,
        # but we CAN verify the loader calls the right classes.
        with (
            patch("ragicamp.indexes.vector_index.VectorIndex.load") as mock_vi,
            patch("ragicamp.indexes.sparse.SparseIndex.load") as mock_si,
            patch("ragicamp.retrievers.hybrid.HybridSearcher") as mock_hs,
            patch("ragicamp.utils.artifacts.get_artifact_manager") as mock_am,
        ):
            mock_vi.return_value = MagicMock(documents=[])
            mock_si.return_value = MagicMock()
            mock_hs.return_value = MagicMock()
            mock_am.return_value.get_sparse_index_path.return_value = Path("/tmp/sparse")

            loader = Experiment._build_index_loader(
                retriever_type="hybrid",
                index_name="test_index",
                index_path=Path("/tmp/index"),
                retriever_config=retriever_config,
            )
            loader()  # Execute the loader

            mock_vi.assert_called_once()
            mock_si.assert_called_once()
            mock_hs.assert_called_once()

            # Verify alpha was passed through
            call_kwargs = mock_hs.call_args
            assert call_kwargs.kwargs.get("alpha") == 0.7 or (
                call_kwargs.args and len(call_kwargs.args) >= 3
            )

    def test_build_index_loader_hierarchical_unchanged(self):
        """_build_index_loader still works for hierarchical."""
        from ragicamp.experiment import Experiment

        with (
            patch("ragicamp.indexes.hierarchical.HierarchicalIndex.load") as mock_hi,
            patch("ragicamp.retrievers.hierarchical.HierarchicalSearcher") as mock_hs,
        ):
            mock_hi.return_value = MagicMock()
            mock_hs.return_value = MagicMock()

            loader = Experiment._build_index_loader(
                retriever_type="hierarchical",
                index_name="hier_test",
                index_path=Path("/tmp/hier"),
            )
            loader()

            mock_hi.assert_called_once()
            mock_hs.assert_called_once()

    def test_build_index_loader_dense_unchanged(self):
        """_build_index_loader still works for dense."""
        from ragicamp.experiment import Experiment

        with patch("ragicamp.indexes.vector_index.VectorIndex.load") as mock_vi:
            mock_vi.return_value = MagicMock()

            loader = Experiment._build_index_loader(
                retriever_type="dense",
                index_name="dense_test",
                index_path=Path("/tmp/dense"),
            )
            loader()

            mock_vi.assert_called_once()


# ---------------------------------------------------------------------------
# Bug 2 – Reranker wiring
# ---------------------------------------------------------------------------


class TestRerankerWiring:
    """Ensure reranker config reaches the agent and is actually called."""

    def test_provider_factory_creates_reranker(self):
        """ProviderFactory.create_reranker must return a RerankerProvider."""
        from ragicamp.models.providers.reranker import RerankerProvider

        provider = ProviderFactory.create_reranker("bge")
        assert isinstance(provider, RerankerProvider)
        assert provider.model_name == "BAAI/bge-reranker-large"

    def test_provider_factory_reranker_model_mapping(self):
        """All reranker short names must resolve to valid HF models."""
        known = {"bge", "bge-base", "bge-v2", "ms-marco", "ms-marco-large"}
        for name in known:
            provider = ProviderFactory.create_reranker(name)
            assert provider.model_name.startswith(("BAAI/", "cross-encoder/"))

    def test_agent_factory_forwards_reranker_to_fixed_rag(self):
        """AgentFactory.from_spec must pass reranker_provider to FixedRAGAgent."""
        spec = ExperimentSpec(
            name="test_reranker",
            exp_type="rag",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            retriever="dense_test",
            top_k=5,
            fetch_k=20,
            reranker="bge",
            reranker_model="BAAI/bge-reranker-large",
        )

        mock_reranker_provider = MagicMock()
        mock_reranker_provider.model_name = "mock-reranker"

        agent = AgentFactory.from_spec(
            spec=spec,
            embedder_provider=_make_provider(_MockEmbedder(), "mock-emb"),
            generator_provider=_make_provider(_MockGenerator(), "mock-gen"),
            index=_MockIndex(),
            reranker_provider=mock_reranker_provider,
        )

        assert isinstance(agent, FixedRAGAgent)
        assert agent.reranker_provider is mock_reranker_provider
        assert agent.fetch_k == 20

    def test_agent_factory_forwards_reranker_to_iterative_rag(self):
        """AgentFactory.from_spec must pass reranker to IterativeRAGAgent."""
        spec = ExperimentSpec(
            name="test_iter_reranker",
            exp_type="rag",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            retriever="dense_test",
            top_k=5,
            fetch_k=20,
            reranker="bge",
            reranker_model="BAAI/bge-reranker-large",
            agent_type="iterative_rag",
            agent_params=(("max_iterations", 1), ("stop_on_sufficient", True)),
        )

        mock_reranker_provider = MagicMock()
        mock_reranker_provider.model_name = "mock-reranker"

        agent = AgentFactory.from_spec(
            spec=spec,
            embedder_provider=_make_provider(_MockEmbedder(), "mock-emb"),
            generator_provider=_make_provider(_MockGenerator(), "mock-gen"),
            index=_MockIndex(),
            reranker_provider=mock_reranker_provider,
        )

        assert isinstance(agent, IterativeRAGAgent)
        assert agent.reranker_provider is mock_reranker_provider
        assert agent.fetch_k == 20

    def test_agent_factory_forwards_reranker_to_self_rag(self):
        """AgentFactory.from_spec must pass reranker to SelfRAGAgent."""
        spec = ExperimentSpec(
            name="test_self_reranker",
            exp_type="rag",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            retriever="dense_test",
            top_k=5,
            fetch_k=20,
            reranker="bge",
            reranker_model="BAAI/bge-reranker-large",
            agent_type="self_rag",
        )

        mock_reranker_provider = MagicMock()
        mock_reranker_provider.model_name = "mock-reranker"

        agent = AgentFactory.from_spec(
            spec=spec,
            embedder_provider=_make_provider(_MockEmbedder(), "mock-emb"),
            generator_provider=_make_provider(_MockGenerator(), "mock-gen"),
            index=_MockIndex(),
            reranker_provider=mock_reranker_provider,
        )

        assert isinstance(agent, SelfRAGAgent)
        assert agent.reranker_provider is mock_reranker_provider
        assert agent.fetch_k == 20

    def test_no_reranker_when_spec_is_none(self):
        """When spec.reranker is None, agent must NOT have reranker_provider."""
        spec = ExperimentSpec(
            name="test_no_reranker",
            exp_type="rag",
            model="vllm:test/model",
            dataset="nq",
            prompt="concise",
            retriever="dense_test",
            top_k=5,
        )

        agent = AgentFactory.from_spec(
            spec=spec,
            embedder_provider=_make_provider(_MockEmbedder(), "mock-emb"),
            generator_provider=_make_provider(_MockGenerator(), "mock-gen"),
            index=_MockIndex(),
        )

        assert isinstance(agent, FixedRAGAgent)
        assert agent.reranker_provider is None
        assert agent.fetch_k == 5  # Same as top_k when no reranker

    def test_fixed_rag_calls_reranker_during_retrieve(self):
        """FixedRAGAgent._phase_retrieve must call batch_rerank when configured."""
        mock_reranker = _MockReranker()
        mock_reranker_provider = _make_provider(mock_reranker, "mock-reranker")

        agent = FixedRAGAgent(
            name="test_rerank_run",
            embedder_provider=_make_provider(_MockEmbedder(), "mock-emb"),
            generator_provider=_make_provider(_MockGenerator(), "mock-gen"),
            index=_MockIndex(),
            top_k=3,
            fetch_k=10,
            reranker_provider=mock_reranker_provider,
        )

        queries = [Query(idx=0, text="test query")]
        results = agent.run(queries)

        assert len(results) == 1
        # Reranker must have been called
        assert len(mock_reranker.calls) == 1
        assert mock_reranker.calls[0]["top_k"] == 3
        # The result steps should include a rerank step
        step_types = [s.type for s in results[0].steps]
        assert "rerank" in step_types

    def test_fixed_rag_retrieves_fetch_k_docs_before_reranking(self):
        """When reranking, agent should retrieve fetch_k (not top_k) docs."""
        mock_reranker = _MockReranker()
        mock_reranker_provider = _make_provider(mock_reranker, "mock-reranker")

        # Use a spy index to verify the top_k passed to batch_search
        spy_index = _MockIndex()
        original_search = spy_index.batch_search
        search_calls = []

        def tracking_search(embeddings, top_k=5, **kw):
            search_calls.append({"top_k": top_k})
            return original_search(embeddings, top_k=top_k, **kw)

        spy_index.batch_search = tracking_search

        agent = FixedRAGAgent(
            name="test_fetch_k",
            embedder_provider=_make_provider(_MockEmbedder(), "mock-emb"),
            generator_provider=_make_provider(_MockGenerator(), "mock-gen"),
            index=spy_index,
            top_k=3,
            fetch_k=10,
            reranker_provider=mock_reranker_provider,
        )

        agent.run([Query(idx=0, text="test")])

        # batch_search should have been called with fetch_k=10, not top_k=3
        assert len(search_calls) == 1
        assert search_calls[0]["top_k"] == 10

    def test_no_rerank_step_without_reranker(self):
        """Without a reranker, there should be no rerank step."""
        agent = FixedRAGAgent(
            name="test_no_rerank",
            embedder_provider=_make_provider(_MockEmbedder(), "mock-emb"),
            generator_provider=_make_provider(_MockGenerator(), "mock-gen"),
            index=_MockIndex(),
            top_k=3,
        )

        results = agent.run([Query(idx=0, text="test")])
        step_types = [s.type for s in results[0].steps]
        assert "rerank" not in step_types


# ---------------------------------------------------------------------------
# Bug 3 – Config fields land in ExperimentSpec correctly
# ---------------------------------------------------------------------------


class TestExperimentSpecCompleteness:
    """Ensure all config fields survive serialisation round-trip."""

    def _make_full_spec(self, **overrides) -> ExperimentSpec:
        defaults = {
            "name": "test_full",
            "exp_type": "rag",
            "model": "vllm:test/model",
            "dataset": "nq",
            "prompt": "concise",
            "retriever": "dense_test",
            "top_k": 5,
            "fetch_k": 20,
            "query_transform": "hyde",
            "reranker": "bge",
            "reranker_model": "BAAI/bge-reranker-large",
            "agent_type": "iterative_rag",
            "agent_params": (("max_iterations", 2),),
        }
        defaults.update(overrides)
        return ExperimentSpec(**defaults)

    def test_reranker_survives_round_trip(self):
        """reranker and reranker_model must survive to_dict -> from_dict."""
        spec = self._make_full_spec()
        rebuilt = ExperimentSpec.from_dict(spec.to_dict())

        assert rebuilt.reranker == "bge"
        assert rebuilt.reranker_model == "BAAI/bge-reranker-large"

    def test_fetch_k_survives_round_trip(self):
        """fetch_k must survive to_dict -> from_dict."""
        spec = self._make_full_spec()
        rebuilt = ExperimentSpec.from_dict(spec.to_dict())

        assert rebuilt.fetch_k == 20

    def test_query_transform_survives_round_trip(self):
        """query_transform must survive to_dict -> from_dict."""
        spec = self._make_full_spec()
        rebuilt = ExperimentSpec.from_dict(spec.to_dict())

        assert rebuilt.query_transform == "hyde"

    def test_agent_params_survive_round_trip(self):
        """agent_params must survive to_dict -> from_dict."""
        spec = self._make_full_spec()
        rebuilt = ExperimentSpec.from_dict(spec.to_dict())

        params_dict = rebuilt.get_agent_params_dict()
        assert params_dict["max_iterations"] == 2


# ---------------------------------------------------------------------------
# Naming consistency
# ---------------------------------------------------------------------------


class TestNamingReflectsConfig:
    """Ensure experiment names encode the config values they should."""

    def test_reranker_changes_hash(self):
        """When reranker is set, it must produce a different hash."""
        from ragicamp.spec.naming import name_rag

        name_with = name_rag(
            model="vllm:test/model",
            prompt="concise",
            dataset="nq",
            retriever="dense_test",
            top_k=5,
            reranker="bge",
        )
        name_without = name_rag(
            model="vllm:test/model",
            prompt="concise",
            dataset="nq",
            retriever="dense_test",
            top_k=5,
            reranker="none",
        )

        # Same prefix/dataset but different hash
        assert name_with != name_without
        assert name_with.startswith("rag_")
        assert name_without.startswith("rag_")

    def test_query_transform_changes_hash(self):
        """When query_transform is set, it must produce a different hash."""
        from ragicamp.spec.naming import name_rag

        name_with = name_rag(
            model="vllm:test/model",
            prompt="concise",
            dataset="nq",
            retriever="dense_test",
            top_k=5,
            query_transform="hyde",
        )
        name_without = name_rag(
            model="vllm:test/model",
            prompt="concise",
            dataset="nq",
            retriever="dense_test",
            top_k=5,
            query_transform="none",
        )

        assert name_with != name_without


# ---------------------------------------------------------------------------
# Analysis utils – metadata enrichment
# ---------------------------------------------------------------------------


class TestAnalysisMetadataEnrichment:
    """Ensure analysis_utils reads structured metadata, not just names."""

    def test_enrich_from_metadata_overrides_name_parsing(self):
        """Structured metadata fields must override name-parsed values."""
        from analysis_utils import _enrich_from_metadata, parse_experiment_name

        # Name with wrong/ambiguous parsing
        name = "rag_vllm_google_gemma29bit_hier_bge_large_2048p_448c_k5_bge_fewshot_3_nq"
        row = parse_experiment_name(name)

        metadata = {
            "model": "vllm:google/gemma-2-9b-it",
            "dataset": "nq",
            "prompt": "fewshot_3",
            "retriever": "hier_bge_large_2048p_448c",
            "top_k": 5,
            "reranker": "bge",
            "reranker_model": "BAAI/bge-reranker-large",
            "query_transform": "none",
            "agent_type": "fixed_rag",
        }

        _enrich_from_metadata(row, metadata)

        assert row["model_short"] == "Gemma2-9B"
        assert row["dataset"] == "nq"
        assert row["prompt"] == "fewshot_3"
        assert row["retriever"] == "hier_bge_large_2048p_448c"
        assert row["retriever_type"] == "hierarchical"
        assert row["top_k"] == 5
        assert row["reranker"] == "bge"
        assert row["agent_type"] == "fixed_rag"
        assert row["embedding_model"] == "BGE-large"
        assert row["chunk_size"] == "2048p/448c"

    def test_enrich_from_metadata_no_reranker(self):
        """When metadata says reranker=none, row should reflect that."""
        from analysis_utils import _enrich_from_metadata

        row = {"reranker": "bge"}  # Wrong initial value
        metadata = {"reranker": "none"}

        _enrich_from_metadata(row, metadata)
        assert row["reranker"] == "none"


# ---------------------------------------------------------------------------
# Migration script – name stripping
# ---------------------------------------------------------------------------


class TestMigrationNameStripping:
    """Ensure the migration script correctly identifies and strips reranker."""

    @pytest.fixture(autouse=True)
    def _add_scripts_to_path(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        yield

    def test_strips_bge_reranker(self):
        from migrate_fake_reranked import _strip_reranker_from_name

        name = "rag_vllm_google_gemma29bit_hier_bge_large_2048p_448c_k5_bge_fewshot_3_nq"
        result = _strip_reranker_from_name(name)
        assert result == "rag_vllm_google_gemma29bit_hier_bge_large_2048p_448c_k5_fewshot_3_nq"

    def test_strips_bgev2_reranker(self):
        from migrate_fake_reranked import _strip_reranker_from_name

        name = "rag_vllm_google_gemma29bit_dense_bge_large_512_k10_hyde_bgev2_concise_hotpotqa"
        result = _strip_reranker_from_name(name)
        assert result == "rag_vllm_google_gemma29bit_dense_bge_large_512_k10_hyde_concise_hotpotqa"

    def test_no_change_when_no_reranker(self):
        from migrate_fake_reranked import _strip_reranker_from_name

        name = "rag_vllm_google_gemma29bit_dense_bge_large_512_k5_concise_nq"
        result = _strip_reranker_from_name(name)
        assert result is None

    def test_no_change_for_direct(self):
        from migrate_fake_reranked import _strip_reranker_from_name

        name = "direct_vllm_google_gemma29bit_concise_nq"
        result = _strip_reranker_from_name(name)
        assert result is None

    def test_strips_from_iterative_rag(self):
        from migrate_fake_reranked import _strip_reranker_from_name

        name = "iterative_rag_iter1_stopok_vllm_google_gemma29bit_hier_bge_large_2048p_448c_k5_bge_fewshot_3_nq"
        result = _strip_reranker_from_name(name)
        assert (
            result
            == "iterative_rag_iter1_stopok_vllm_google_gemma29bit_hier_bge_large_2048p_448c_k5_fewshot_3_nq"
        )

    def test_strips_msmarco_reranker(self):
        from migrate_fake_reranked import _strip_reranker_from_name

        name = "rag_vllm_google_gemma29bit_dense_bge_large_512_k5_msmarco_concise_nq"
        result = _strip_reranker_from_name(name)
        assert result == "rag_vllm_google_gemma29bit_dense_bge_large_512_k5_concise_nq"

    def test_two_rerankers_same_corrected_name(self):
        """When _bge_ and _bgev2_ both map to the same name, scan must not
        produce two 'rename' actions — one should be archived."""
        from migrate_fake_reranked import _save_json, scan_study

        with tempfile.TemporaryDirectory() as tmpdir:
            study = Path(tmpdir)

            # Create two fake-reranked dirs that map to the same corrected name
            base = "rag_vllm_google_gemma29bit_hier_bge_large_2048p_448c_k5_multiquery"
            bge_dir = study / f"{base}_bge_fewshot_3_triviaqa"
            bgev2_dir = study / f"{base}_bgev2_fewshot_3_triviaqa"
            bge_dir.mkdir()
            bgev2_dir.mkdir()

            # Give bgev2 a better F1
            _save_json(bge_dir / "results.json", {"name": bge_dir.name, "metrics": {"f1": 0.3}})
            _save_json(bgev2_dir / "results.json", {"name": bgev2_dir.name, "metrics": {"f1": 0.5}})

            actions = scan_study(study)

            # Both should be in the action list
            assert len(actions) == 2

            # Only one should be 'rename', the other should be 'merge_keep_new' (archived)
            action_types = [a["action"] for a in actions]
            assert action_types.count("rename") == 1
            assert action_types.count("merge_keep_new") == 1

            # The rename winner should be the one with higher F1 (bgev2)
            rename_action = [a for a in actions if a["action"] == "rename"][0]
            assert "bgev2" in rename_action["old_name"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
