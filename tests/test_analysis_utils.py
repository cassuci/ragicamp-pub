"""Tests for analysis_utils deduplication and fake-token stripping."""

import pandas as pd
from analysis_utils import (
    _deduplicate_experiments,
    _effective_config_key,
    _strip_fake_tokens,
    filter_to_search_space,
    get_outside_search_space_summary,
)


class TestStripFakeTokens:
    """Test _strip_fake_tokens()."""

    def test_strips_hyde_token(self):
        name = "rag_vllm_llama_dense_bge_large_512_k5_hyde_concise_nq"
        result = _strip_fake_tokens(name)
        assert result is not None
        assert "hyde" not in result
        assert result.endswith("_k5_concise_nq")

    def test_strips_multiquery_token(self):
        name = "rag_vllm_llama_dense_bge_large_512_k5_multiquery_concise_nq"
        result = _strip_fake_tokens(name)
        assert result is not None
        assert "multiquery" not in result

    def test_strips_bge_reranker_token(self):
        name = "rag_vllm_llama_dense_bge_large_512_k5_bge_concise_nq"
        result = _strip_fake_tokens(name)
        assert result is not None
        assert result.endswith("_k5_concise_nq")

    def test_strips_bgev2_reranker_token(self):
        name = "rag_vllm_llama_dense_bge_large_512_k5_bgev2_concise_nq"
        result = _strip_fake_tokens(name)
        assert result is not None
        assert "bgev2" not in result

    def test_strips_multiple_fake_tokens(self):
        name = "rag_vllm_llama_dense_bge_large_512_k5_hyde_bge_concise_nq"
        result = _strip_fake_tokens(name)
        assert result is not None
        assert "hyde" not in result
        assert result.endswith("_k5_concise_nq")

    def test_no_strip_on_clean_name(self):
        name = "rag_vllm_llama_dense_bge_large_512_k5_concise_nq"
        result = _strip_fake_tokens(name)
        assert result is None

    def test_no_strip_on_direct_experiment(self):
        name = "direct_vllm_llama_concise_nq"
        result = _strip_fake_tokens(name)
        assert result is None  # no k{N} segment

    def test_no_strip_short_tail(self):
        name = "rag_vllm_llama_k5_nq"  # only 1 token after k5
        result = _strip_fake_tokens(name)
        assert result is None

    def test_preserves_prefix_and_retriever(self):
        name = "rag_vllm_metallamaLlama3_dense_bge_large_512_k5_hyde_concise_nq"
        result = _strip_fake_tokens(name)
        assert result is not None
        assert result.startswith("rag_vllm_metallamaLlama3_dense_bge_large_512_k5_")


class TestEffectiveConfigKey:
    """Test _effective_config_key()."""

    def test_same_key_for_different_reranker(self):
        row_a = {
            "model_short": "Llama-3.2-3B",
            "dataset": "nq",
            "exp_type": "rag",
            "retriever_type": "dense",
            "embedding_model": "BGE-large",
            "top_k": 5,
            "prompt": "concise",
            "agent_type": "fixed_rag",
            "reranker": "bge",
            "query_transform": "none",
        }
        row_b = row_a.copy()
        row_b["reranker"] = "none"
        # reranker is NOT part of the key (since it was never wired)
        assert _effective_config_key(row_a) == _effective_config_key(row_b)

    def test_different_key_for_different_model(self):
        row_a = {
            "model_short": "Llama-3.2-3B",
            "dataset": "nq",
            "exp_type": "rag",
            "retriever_type": "dense",
            "embedding_model": "BGE-large",
            "top_k": 5,
            "prompt": "concise",
            "agent_type": "fixed_rag",
        }
        row_b = row_a.copy()
        row_b["model_short"] = "Phi-3-mini"
        assert _effective_config_key(row_a) != _effective_config_key(row_b)


class TestDeduplicateExperiments:
    """Test _deduplicate_experiments()."""

    def _make_df(self, rows):
        return pd.DataFrame(rows)

    def test_keeps_best_f1_among_duplicates(self):
        rows = [
            {
                "name": "rag_vllm_llama_dense_bge_large_512_k5_hyde_bge_concise_nq",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "exp_type": "rag",
                "retriever_type": "dense",
                "embedding_model": "BGE-large",
                "top_k": 5,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "reranker": "bge",
                "query_transform": "hyde",
                "f1": 0.40,
            },
            {
                "name": "rag_vllm_llama_dense_bge_large_512_k5_concise_nq",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "exp_type": "rag",
                "retriever_type": "dense",
                "embedding_model": "BGE-large",
                "top_k": 5,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "reranker": "none",
                "query_transform": "none",
                "f1": 0.42,
            },
        ]
        df = self._make_df(rows)
        result = _deduplicate_experiments(df)

        assert len(result) == 1
        assert result.iloc[0]["f1"] == 0.42

    def test_normalises_reranker_and_qt_for_fake_tokens(self):
        rows = [
            {
                "name": "rag_vllm_llama_dense_bge_large_512_k5_hyde_concise_nq",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "exp_type": "rag",
                "retriever_type": "dense",
                "embedding_model": "BGE-large",
                "top_k": 5,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "reranker": "none",
                "query_transform": "hyde",
                "f1": 0.45,
            },
        ]
        df = self._make_df(rows)
        result = _deduplicate_experiments(df)

        # query_transform should be overridden to 'none'
        assert result.iloc[0]["query_transform"] == "none"

    def test_keeps_distinct_configs(self):
        rows = [
            {
                "name": "rag_vllm_llama_dense_bge_large_512_k5_concise_nq",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "exp_type": "rag",
                "retriever_type": "dense",
                "embedding_model": "BGE-large",
                "top_k": 5,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "reranker": "none",
                "query_transform": "none",
                "f1": 0.42,
            },
            {
                "name": "rag_vllm_llama_dense_bge_large_512_k10_concise_nq",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "exp_type": "rag",
                "retriever_type": "dense",
                "embedding_model": "BGE-large",
                "top_k": 10,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "reranker": "none",
                "query_transform": "none",
                "f1": 0.50,
            },
        ]
        df = self._make_df(rows)
        result = _deduplicate_experiments(df)

        assert len(result) == 2  # different top_k → distinct configs

    def test_handles_three_way_duplicate(self):
        """Three experiments with same effective config → keep best."""
        base = {
            "model_short": "Llama-3.2-3B",
            "dataset": "nq",
            "exp_type": "rag",
            "retriever_type": "dense",
            "embedding_model": "BGE-large",
            "top_k": 5,
            "prompt": "concise",
            "agent_type": "fixed_rag",
        }
        rows = [
            {
                **base,
                "name": "rag_vllm_llama_dense_bge_large_512_k5_hyde_bge_concise_nq",
                "reranker": "bge",
                "query_transform": "hyde",
                "f1": 0.38,
            },
            {
                **base,
                "name": "rag_vllm_llama_dense_bge_large_512_k5_hyde_concise_nq",
                "reranker": "none",
                "query_transform": "hyde",
                "f1": 0.40,
            },
            {
                **base,
                "name": "rag_vllm_llama_dense_bge_large_512_k5_concise_nq",
                "reranker": "none",
                "query_transform": "none",
                "f1": 0.42,
            },
        ]
        df = self._make_df(rows)
        result = _deduplicate_experiments(df)

        assert len(result) == 1
        assert result.iloc[0]["f1"] == 0.42

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = _deduplicate_experiments(df)
        assert result.empty

    def test_direct_experiments_not_stripped(self):
        """Direct experiments have no k{N} → no stripping → each is unique."""
        rows = [
            {
                "name": "direct_vllm_llama_concise_nq",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "exp_type": "direct",
                "retriever_type": None,
                "embedding_model": None,
                "top_k": None,
                "prompt": "concise",
                "agent_type": "direct_llm",
                "reranker": "none",
                "query_transform": "none",
                "f1": 0.30,
            },
            {
                "name": "direct_vllm_llama_cot_nq",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "exp_type": "direct",
                "retriever_type": None,
                "embedding_model": None,
                "top_k": None,
                "prompt": "cot",
                "agent_type": "direct_llm",
                "reranker": "none",
                "query_transform": "none",
                "f1": 0.35,
            },
        ]
        df = self._make_df(rows)
        result = _deduplicate_experiments(df)

        assert len(result) == 2  # different prompts → kept


class TestFilterToSearchSpace:
    """Test filter_to_search_space()."""

    _SPACE = {
        "model": {"vllm:meta-llama/Llama-3.2-3B-Instruct"},
        "retriever": {"dense_bge_large_512"},
        "top_k": {5, 10},
        "prompt": {"concise", "cot"},
        "direct_prompt": {"concise", "fewshot_3"},
        "dataset": {"nq", "triviaqa"},
        "query_transform": {"none", "hyde"},
        "reranker": {"none", "bge"},
        "agent_type": {"fixed_rag"},
    }

    def _make_df(self, rows):
        return pd.DataFrame(rows)

    def test_keeps_experiment_within_space(self):
        rows = [
            {
                "name": "rag_test",
                "exp_type": "rag",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "retriever": "dense_bge_large_512",
                "top_k": 5,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "query_transform": "none",
                "reranker": "none",
                "f1": 0.5,
            }
        ]
        df = self._make_df(rows)
        result = filter_to_search_space(df, search_space=self._SPACE)
        assert len(result) == 1

    def test_drops_experiment_with_unknown_retriever(self):
        rows = [
            {
                "name": "rag_test",
                "exp_type": "rag",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "retriever": "dense_e5_mistral_512",  # not in space
                "top_k": 5,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "query_transform": "none",
                "reranker": "none",
                "f1": 0.5,
            }
        ]
        df = self._make_df(rows)
        result = filter_to_search_space(df, search_space=self._SPACE)
        assert len(result) == 0

    def test_drops_experiment_with_unknown_model(self):
        rows = [
            {
                "name": "rag_test",
                "exp_type": "rag",
                "model_short": "Gemma2-9B",  # not in space (only Llama)
                "dataset": "nq",
                "retriever": "dense_bge_large_512",
                "top_k": 5,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "query_transform": "none",
                "reranker": "none",
                "f1": 0.5,
            }
        ]
        df = self._make_df(rows)
        result = filter_to_search_space(df, search_space=self._SPACE)
        assert len(result) == 0

    def test_drops_experiment_with_top_k_outside_space(self):
        rows = [
            {
                "name": "rag_test",
                "exp_type": "rag",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "retriever": "dense_bge_large_512",
                "top_k": 20,  # not in {5, 10}
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "query_transform": "none",
                "reranker": "none",
                "f1": 0.5,
            }
        ]
        df = self._make_df(rows)
        result = filter_to_search_space(df, search_space=self._SPACE)
        assert len(result) == 0

    def test_direct_uses_direct_prompt_set(self):
        rows = [
            {
                "name": "direct_test",
                "exp_type": "direct",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "retriever": None,
                "top_k": None,
                "prompt": "fewshot_3",  # in direct_prompt but not rag prompt
                "agent_type": "direct_llm",
                "f1": 0.3,
            }
        ]
        df = self._make_df(rows)
        result = filter_to_search_space(df, search_space=self._SPACE)
        assert len(result) == 1

    def test_direct_dropped_for_unknown_prompt(self):
        rows = [
            {
                "name": "direct_test",
                "exp_type": "direct",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "retriever": None,
                "top_k": None,
                "prompt": "structured",  # not in direct_prompt
                "agent_type": "direct_llm",
                "f1": 0.3,
            }
        ]
        df = self._make_df(rows)
        result = filter_to_search_space(df, search_space=self._SPACE)
        assert len(result) == 0

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = filter_to_search_space(df, search_space=self._SPACE)
        assert result.empty

    def test_drops_experiment_with_qt_outside_space(self):
        rows = [
            {
                "name": "rag_test",
                "exp_type": "rag",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "retriever": "dense_bge_large_512",
                "top_k": 5,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "query_transform": "iterative_2",  # not in {none, hyde, multiquery}
                "reranker": "none",
                "f1": 0.5,
            }
        ]
        df = self._make_df(rows)
        result = filter_to_search_space(df, search_space=self._SPACE)
        assert len(result) == 0

    def test_drops_experiment_with_rr_outside_space(self):
        rows = [
            {
                "name": "rag_test",
                "exp_type": "rag",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "retriever": "dense_bge_large_512",
                "top_k": 5,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "query_transform": "none",
                "reranker": "msmarco",  # not in {none, bge}
                "f1": 0.5,
            }
        ]
        df = self._make_df(rows)
        result = filter_to_search_space(df, search_space=self._SPACE)
        assert len(result) == 0

    def test_keeps_valid_qt_and_rr(self):
        rows = [
            {
                "name": "rag_test",
                "exp_type": "rag",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "retriever": "dense_bge_large_512",
                "top_k": 5,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "query_transform": "hyde",  # in space
                "reranker": "bge",  # in space
                "f1": 0.5,
            }
        ]
        df = self._make_df(rows)
        result = filter_to_search_space(df, search_space=self._SPACE)
        assert len(result) == 1

    def test_mixed_keep_and_drop(self):
        rows = [
            {  # Keep: within space
                "name": "rag_good",
                "exp_type": "rag",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "retriever": "dense_bge_large_512",
                "top_k": 5,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "query_transform": "none",
                "reranker": "none",
                "f1": 0.5,
            },
            {  # Drop: retriever outside space
                "name": "rag_old",
                "exp_type": "rag",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "retriever": "hier_bge_large_2048p_448c",
                "top_k": 5,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "query_transform": "none",
                "reranker": "none",
                "f1": 0.4,
            },
            {  # Keep: direct within space
                "name": "direct_good",
                "exp_type": "direct",
                "model_short": "Llama-3.2-3B",
                "dataset": "triviaqa",
                "retriever": None,
                "top_k": None,
                "prompt": "concise",
                "agent_type": "direct_llm",
                "query_transform": "none",
                "reranker": "none",
                "f1": 0.3,
            },
        ]
        df = self._make_df(rows)
        result = filter_to_search_space(df, search_space=self._SPACE)
        assert len(result) == 2
        assert set(result["name"]) == {"rag_good", "direct_good"}


class TestGetOutsideSearchSpaceSummary:
    """Test get_outside_search_space_summary()."""

    _SPACE = TestFilterToSearchSpace._SPACE

    def test_all_in_space(self):
        rows = [
            {
                "name": "rag_1",
                "exp_type": "rag",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "retriever": "dense_bge_large_512",
                "top_k": 5,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "query_transform": "none",
                "reranker": "none",
                "f1": 0.5,
            }
        ]
        df = pd.DataFrame(rows)
        out = get_outside_search_space_summary(df, search_space=self._SPACE)
        assert out["n_in_space"] == 1
        assert out["n_outside"] == 0
        assert not out["outside_values"]
        assert out["df_outside"].empty

    def test_reports_outside_retriever_and_values(self):
        rows = [
            {
                "name": "rag_in",
                "exp_type": "rag",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "retriever": "dense_bge_large_512",
                "top_k": 5,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "query_transform": "none",
                "reranker": "none",
                "f1": 0.5,
            },
            {
                "name": "rag_out",
                "exp_type": "rag",
                "model_short": "Llama-3.2-3B",
                "dataset": "nq",
                "retriever": "dense_e5_mistral_512",
                "top_k": 5,
                "prompt": "concise",
                "agent_type": "fixed_rag",
                "query_transform": "none",
                "reranker": "none",
                "f1": 0.4,
            },
        ]
        df = pd.DataFrame(rows)
        out = get_outside_search_space_summary(df, search_space=self._SPACE)
        assert out["n_in_space"] == 1
        assert out["n_outside"] == 1
        assert out["outside_by_dimension"]["retriever"] == 1
        assert "dense_e5_mistral_512" in out["outside_values"]["retriever"]
        assert len(out["df_outside"]) == 1
        assert out["df_outside"].iloc[0]["name"] == "rag_out"
