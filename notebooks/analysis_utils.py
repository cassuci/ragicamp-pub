"""
Shared utilities for Smart Retrieval SLM Analysis notebooks.

This module provides common functions for loading experiment data,
computing statistics, and generating visualizations.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import yaml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default paths
DEFAULT_STUDY_PATH = Path("../outputs/smart_retrieval_slm")

# Metrics to analyze
METRICS = [
    'f1', 'exact_match',
    'bertscore', 'bertscore_f1', 'bertscore_precision', 'bertscore_recall',
    'bleurt', 'llm_judge',
    'faithfulness', 'hallucination', 'answer_in_context', 'context_recall',
]
PRIMARY_METRIC = 'f1'

# Metric groupings for multi-metric analysis
CORRECTNESS_METRICS = ['f1', 'exact_match']
SEMANTIC_METRICS = ['bertscore_f1', 'bleurt']
CONTEXT_METRICS = ['answer_in_context', 'context_recall']
GROUNDEDNESS_METRICS = ['faithfulness', 'hallucination']

# All metrics suitable for multi-metric comparison (common, interpretable 0-1 scale)
MULTI_METRIC_SET = ['f1', 'exact_match', 'bertscore_f1', 'faithfulness',
                    'answer_in_context', 'context_recall']

# Model name mappings
# Keys are substrings that appear in experiment names after normalization:
# model.replace(":", "_").replace("/", "_").replace("-", "")
# Order matters: more specific patterns should come first!
MODEL_MAP = {
    # XL models (70B) - most specific first
    'llama3.370binstruct': 'Llama-3.3-70B',
    # Large models (14-27B)
    'qwen2.514binstruct': 'Qwen2.5-14B',
    'gemma227bit': 'Gemma2-27B',
    # Medium models (7-9B)
    'mistral7binstructv0.3': 'Mistral-7B',
    'qwen2.57binstruct': 'Qwen2.5-7B',
    'gemma29bit': 'Gemma2-9B',
    # Tiny models (1-2B)
    'qwen2.51.5binstruct': 'Qwen2.5-1.5B',
    'gemma22bit': 'Gemma2-2B',
    # Small models (3-4B)
    'llama3.23binstruct': 'Llama-3.2-3B',
    'phi4miniinstruct': 'Phi-4-mini',
    'phi3.5miniinstruct': 'Phi-3.5-mini',
    'phi3mini4kinstruct': 'Phi-3-mini',
    'qwen2.53binstruct': 'Qwen2.5-3B',
    'qwen34b': 'Qwen3-4B',
    # Medium (new generation)
    'qwen38b': 'Qwen3-8B',
    # Legacy fallbacks (shorter patterns - must be last)
    'llama': 'Llama-3.2-3B',
    'phi': 'Phi-4-mini',
    'qwen': 'Qwen2.5-3B',
}

# Model parameter counts (billions)
MODEL_PARAMS = {
    'Qwen2.5-1.5B': 1.5, 'Gemma2-2B': 2.0,
    'Llama-3.2-3B': 3.0, 'Phi-3-mini': 3.8, 'Phi-3.5-mini': 3.8, 'Phi-4-mini': 3.8,
    'Qwen2.5-3B': 3.0, 'Qwen3-4B': 4.0,
    'Mistral-7B': 7.0, 'Qwen2.5-7B': 7.0, 'Gemma2-9B': 9.0, 'Qwen3-8B': 8.0,
    'Qwen2.5-14B': 14.0, 'Gemma2-27B': 27.0,
    'Llama-3.3-70B': 70.0,
}

# Model size tiers
MODEL_TIER = {
    'Qwen2.5-1.5B': 'tiny', 'Gemma2-2B': 'tiny',
    'Llama-3.2-3B': 'small', 'Phi-3-mini': 'small', 'Phi-3.5-mini': 'small', 'Phi-4-mini': 'small',
    'Qwen2.5-3B': 'small', 'Qwen3-4B': 'small',
    'Mistral-7B': 'medium', 'Qwen2.5-7B': 'medium', 'Gemma2-9B': 'medium', 'Qwen3-8B': 'medium',
    'Qwen2.5-14B': 'large', 'Gemma2-27B': 'large',
    'Llama-3.3-70B': 'xl',
}

# Models with known issues (near-zero F1 across all conditions)
BROKEN_MODELS = {'Phi-3-mini'}

# Retriever type detection
RETRIEVER_TYPES = {
    'dense': ['dense_bge', 'dense_gte', 'dense_e5', 'en_bge', 'en_gte', 'en_e5'],
    'hybrid': ['hybrid_'],
    'hierarchical': ['hier_', 'hierarchical_'],
}

# Embedding model detection
EMBEDDING_MAP = {
    'bge_large': 'BGE-large',
    'bge_m3': 'BGE-M3',
    'gte_qwen2': 'GTE-Qwen2-1.5B',
    'e5_mistral': 'E5-Mistral-7B',
}

# Dataset aliases — normalise alternative names to canonical short forms
DATASET_ALIASES = {
    'natural_questions': 'nq',
}

# Per-dataset styling — consistent colours/markers/labels across all notebooks
DATASET_COLORS = {'nq': '#1f77b4', 'triviaqa': '#ff7f0e', 'hotpotqa': '#2ca02c'}
DATASET_MARKERS = {'nq': 'o', 'triviaqa': 's', 'hotpotqa': '^'}
DATASET_LABELS = {'nq': 'Natural Questions', 'triviaqa': 'TriviaQA', 'hotpotqa': 'HotpotQA'}

# Directories to skip when scanning experiment outputs
SKIP_DIRS = {
    '_archived_fake_reranked', '_tainted', '_collisions', '_incomplete',
    'analysis', '__pycache__', '.ipynb_checkpoints',
}

# Tokens for parameters that were configured but NEVER WIRED in the old code.
# These appear in experiment names but had no effect on results.
_FAKE_RERANKER_TOKENS = {
    'bge', 'bgev2', 'bgev2m3', 'bgebase',
    'msmarco', 'msmarcolarge',
    'bge-v2', 'bge-base', 'ms-marco', 'ms-marco-large',
}
_FAKE_RERANKER_NORMALISED = {t.replace('-', '') for t in _FAKE_RERANKER_TOKENS}
_FAKE_QT_TOKENS = {'hyde', 'multiquery'}

# Default config path (relative to notebooks/)
DEFAULT_CONFIG_PATH = Path("../conf/study/smart_retrieval_slm.yaml")


def load_study_search_space(config_path: Path = None) -> Dict[str, set]:
    """Load the current study's search space from the YAML config.

    Returns a dict mapping dimension names to sets of valid values::

        {
            'model': {'vllm:meta-llama/Llama-3.2-3B-Instruct', ...},
            'retriever': {'dense_bge_large_512', ...},
            'top_k': {3, 5, 10, 15, 20},
            'prompt': {'concise', 'concise_strict', ...},
            'dataset': {'nq', 'triviaqa', 'hotpotqa'},
            'query_transform': {'none', 'hyde', 'multiquery'},
            'reranker': {'none', 'bge', 'bge-v2'},
            'agent_type': {'fixed_rag', 'iterative_rag', 'self_rag'},
        }
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    rag = cfg.get('rag', {})
    sampling = rag.get('sampling', {})

    space: Dict[str, set] = {}

    # Models (shared for direct + rag via YAML anchors)
    models = cfg.get('models') or rag.get('models', [])
    space['model'] = set(models)

    # Retriever names
    space['retriever'] = set(rag.get('retriever_names', []))

    # Top-K
    space['top_k'] = set(rag.get('top_k_values', []))

    # Prompts (RAG)
    space['prompt'] = set(rag.get('prompts', []))

    # Direct prompts (may differ from RAG)
    direct_cfg = cfg.get('direct', {})
    space['direct_prompt'] = set(direct_cfg.get('prompts', []))

    # Datasets
    space['dataset'] = set(cfg.get('datasets', []))

    # Query transform
    qt = rag.get('query_transform', [])
    space['query_transform'] = set(qt) if qt else {'none'}

    # Reranker
    rr_configs = rag.get('reranker', {}).get('configs', [])
    space['reranker'] = {c.get('name', 'none') for c in rr_configs}

    # Agent types
    space['agent_type'] = set(sampling.get('agent_types', ['fixed_rag']))

    return space


def filter_to_search_space(
    df: pd.DataFrame,
    search_space: Dict[str, set] = None,
    config_path: Path = None,
) -> pd.DataFrame:
    """Filter a DataFrame to only experiments within the study's search space.

    This mirrors Optuna's ``_seed_from_existing`` logic: an experiment is
    included only if **every** parameter is in the configured search space.

    Dimensions checked for RAG: model (full spec), dataset, retriever,
    top_k, prompt, query_transform, reranker, agent_type.

    Dimensions checked for Direct: model (full spec), dataset, prompt.

    Args:
        df: DataFrame from ``load_all_results()``.
        search_space: Pre-loaded search space dict, or None to auto-load.
        config_path: Path to the YAML config (used if search_space is None).

    Returns:
        Filtered DataFrame.
    """
    if df.empty:
        return df

    if search_space is None:
        search_space = load_study_search_space(config_path)

    # Two-level model check: prefer exact full-spec match, fallback to model_short
    model_full_set = search_space.get('model', set())
    model_short_set = {_model_short_from_spec(m) for m in model_full_set}

    masks = []
    for idx, row in df.iterrows():
        keep = True

        # Model check — exact full spec first, fallback to model_short
        full_model = row.get('model')
        if model_full_set:
            if full_model and full_model in model_full_set:
                pass  # exact match
            elif row.get('model_short', 'unknown') not in model_short_set:
                keep = False

        # Dataset check
        ds_set = search_space.get('dataset', set())
        if ds_set and row.get('dataset', 'unknown') not in ds_set:
            keep = False

        if row.get('exp_type') == 'direct':
            # Direct: check prompt against direct_prompt set
            dp = search_space.get('direct_prompt', set())
            if dp and row.get('prompt', 'unknown') not in dp:
                keep = False
        else:
            # RAG: check ALL dimensions that Optuna validates

            ret_set = search_space.get('retriever', set())
            if ret_set and row.get('retriever') not in ret_set:
                keep = False

            tk_set = search_space.get('top_k', set())
            if tk_set and row.get('top_k') not in tk_set:
                keep = False

            p_set = search_space.get('prompt', set())
            if p_set and row.get('prompt', 'unknown') not in p_set:
                keep = False

            qt_set = search_space.get('query_transform', set())
            qt_val = row.get('query_transform') or 'none'
            if qt_set and qt_val not in qt_set:
                keep = False

            rr_set = search_space.get('reranker', set())
            rr_val = row.get('reranker') or 'none'
            if rr_set and rr_val not in rr_set:
                keep = False

            at_set = search_space.get('agent_type', set())
            if at_set and row.get('agent_type', 'fixed_rag') not in at_set:
                keep = False

        masks.append(keep)

    filtered = df.loc[masks].copy().reset_index(drop=True)

    n_removed = len(df) - len(filtered)
    if n_removed > 0:
        print(f"  Search-space filter: dropped {n_removed} experiments "
              f"outside current YAML config")

    return filtered


def get_outside_search_space_summary(
    df: pd.DataFrame,
    search_space: Dict[str, set] = None,
    config_path: Path = None,
) -> Dict[str, Any]:
    """Identify experiments outside the current YAML search space and summarize.

    Use this to keep all experiments in the notebook but see which configs
    are "outside" the active study, and get suggestions for what to add to
    the YAML if you want to explore them.

    Returns:
        Dict with:
        - n_in_space: number of experiments with all params in the config
        - n_outside: number with at least one param outside
        - outside_by_dimension: { 'retriever': 45, 'prompt': 30, ... }  (count
          of experiments that have this dimension outside)
        - outside_values: { 'retriever': ['dense_e5_mistral_512', ...],
          'prompt': ['structured', ...], ... }  (unique values seen outside)
        - df_outside: DataFrame of experiments that are outside (for inspection)
    """
    if df.empty:
        return {
            'n_in_space': 0,
            'n_outside': 0,
            'outside_by_dimension': {},
            'outside_values': {},
            'df_outside': pd.DataFrame(),
        }

    if search_space is None:
        search_space = load_study_search_space(config_path)

    model_full_set = search_space.get('model', set())
    model_short_set = {_model_short_from_spec(m) for m in model_full_set}

    # For each row, record which dimensions are outside
    dims_checked = [
        'model', 'dataset', 'retriever', 'top_k', 'prompt',
        'query_transform', 'reranker', 'agent_type', 'direct_prompt',
    ]
    outside_by_dimension: Dict[str, int] = defaultdict(int)
    outside_values: Dict[str, set] = defaultdict(set)

    outside_idx = []

    for idx, row in df.iterrows():
        row_outside_dims = []

        # Model
        full_model = row.get('model')
        if model_full_set:
            if not (full_model and full_model in model_full_set) and row.get('model_short', 'unknown') not in model_short_set:
                row_outside_dims.append('model')
                outside_values['model'].add(row.get('model_short', 'unknown'))

        # Dataset
        ds_set = search_space.get('dataset', set())
        if ds_set and row.get('dataset', 'unknown') not in ds_set:
            row_outside_dims.append('dataset')
            outside_values['dataset'].add(row.get('dataset', 'unknown'))

        if row.get('exp_type') == 'direct':
            dp = search_space.get('direct_prompt', set())
            if dp and row.get('prompt', 'unknown') not in dp:
                row_outside_dims.append('direct_prompt')
                outside_values['direct_prompt'].add(row.get('prompt', 'unknown'))
        else:
            ret_set = search_space.get('retriever', set())
            rv = row.get('retriever')
            if ret_set and rv not in ret_set:
                row_outside_dims.append('retriever')
                if rv is not None:
                    outside_values['retriever'].add(rv)

            tk_set = search_space.get('top_k', set())
            tv = row.get('top_k')
            if tk_set and tv not in tk_set:
                row_outside_dims.append('top_k')
                if tv is not None:
                    outside_values['top_k'].add(tv)

            p_set = search_space.get('prompt', set())
            if p_set and row.get('prompt', 'unknown') not in p_set:
                row_outside_dims.append('prompt')
                outside_values['prompt'].add(row.get('prompt', 'unknown'))

            qt_set = search_space.get('query_transform', set())
            qt_val = row.get('query_transform') or 'none'
            if qt_set and qt_val not in qt_set:
                row_outside_dims.append('query_transform')
                outside_values['query_transform'].add(qt_val)

            rr_set = search_space.get('reranker', set())
            rr_val = row.get('reranker') or 'none'
            if rr_set and rr_val not in rr_set:
                row_outside_dims.append('reranker')
                outside_values['reranker'].add(rr_val)

            at_set = search_space.get('agent_type', set())
            if at_set and row.get('agent_type', 'fixed_rag') not in at_set:
                row_outside_dims.append('agent_type')
                outside_values['agent_type'].add(row.get('agent_type', 'fixed_rag'))

        for d in row_outside_dims:
            outside_by_dimension[d] += 1
        if row_outside_dims:
            outside_idx.append(idx)

    n_outside = len(outside_idx)
    n_in_space = len(df) - n_outside

    # Convert sets to sorted lists for stable output
    outside_values_sorted = {
        k: sorted(v, key=str) for k, v in outside_values.items() if v
    }

    return {
        'n_in_space': n_in_space,
        'n_outside': n_outside,
        'outside_by_dimension': dict(outside_by_dimension),
        'outside_values': outside_values_sorted,
        'df_outside': df.loc[outside_idx].copy() if outside_idx else pd.DataFrame(),
    }


def print_outside_search_space_suggestion(
    df: pd.DataFrame,
    config_path: Path = None,
    max_values_per_dim: int = 10,
) -> None:
    """Print a short summary of experiments outside the current YAML and suggest exploring them.

    Call this after load_all_results() to see what configs are not in the
    current study and might be worth adding to the YAML.
    """
    cfg = config_path or DEFAULT_CONFIG_PATH
    if not Path(cfg).exists():
        return

    summary = get_outside_search_space_summary(df, config_path=cfg)

    if summary['n_outside'] == 0:
        return

    n = summary['n_outside']
    by_dim = summary['outside_by_dimension']
    vals = summary['outside_values']

    print(f"\n  {n} experiment(s) use configs outside the current YAML.")
    print("  Consider adding these to your study config to explore them:")
    for dim in ['retriever', 'prompt', 'model', 'top_k', 'query_transform', 'reranker', 'agent_type', 'dataset', 'direct_prompt']:
        if dim not in vals or not vals[dim]:
            continue
        count = by_dim.get(dim, 0)
        items = vals[dim][:max_values_per_dim]
        extra = f", ... (+{len(vals[dim]) - max_values_per_dim} more)" if len(vals[dim]) > max_values_per_dim else ""
        print(f"    • {dim}: {count} exps — {', '.join(str(x) for x in items)}{extra}")


# =============================================================================
# STYLING
# =============================================================================

def setup_plotting():
    """Configure matplotlib and seaborn for consistent styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11


# =============================================================================
# DATA LOADING
# =============================================================================

def _model_short_from_spec(model_spec: str) -> str:
    """Derive short model name from a full model spec string.

    Args:
        model_spec: Full model specification (e.g. 'vllm:google/gemma-2-9b-it')

    Returns:
        Human-readable short name (e.g. 'Gemma2-9B').
    """
    # Normalize: strip provider prefix and collapse separators
    normalized = model_spec.replace(":", "_").replace("/", "_").replace("-", "").lower()
    for key, display in MODEL_MAP.items():
        if key.lower() in normalized:
            return display
    return model_spec  # Fallback: return raw spec


def _retriever_type_from_name(retriever_name: str) -> Optional[str]:
    """Detect retriever type from its name."""
    if not retriever_name:
        return None
    name_lower = retriever_name.lower()
    for rtype, patterns in RETRIEVER_TYPES.items():
        for pattern in patterns:
            if pattern in name_lower:
                return rtype
    return 'dense'  # default


def _embedding_model_from_name(retriever_name: str) -> Optional[str]:
    """Detect embedding model from retriever name."""
    if not retriever_name:
        return None
    name_lower = retriever_name.lower()
    for key, display in EMBEDDING_MAP.items():
        if key in name_lower:
            return display
    return None


def _chunk_size_from_name(retriever_name: str) -> Optional[str]:
    """Extract chunk size info from retriever name."""
    if not retriever_name:
        return None
    # Hierarchical: 2048p_448c
    m = re.search(r'(\d+)p_(\d+)c', retriever_name)
    if m:
        return f"{m.group(1)}p/{m.group(2)}c"
    # Dense: _512_, _1024_
    m = re.search(r'_(\d{3,4})(?:_|$)', retriever_name)
    if m:
        return m.group(1)
    return None


def _agent_type_from_name(name: str) -> str:
    """Detect agent type from experiment name."""
    if name.startswith('iterative_rag') or name.startswith('iterative_'):
        return 'iterative_rag'
    elif name.startswith('self_rag') or name.startswith('selfrag_'):
        return 'self_rag'
    elif name.startswith('direct_'):
        return 'direct_llm'
    else:
        return 'fixed_rag'


def parse_experiment_name(name: str) -> Dict[str, Any]:
    """
    Parse experiment name into structured components.

    Handles formats:
    - direct_vllm_metallamaLlama3.23BInstruct_concise_nq
    - rag_vllm_metallamaLlama3.23BInstruct_dense_bge_large_512_k5_hyde_bge_concise_nq
    - iterative_rag_iter1_stopok_vllm_*_*_k5_*_nq
    - self_rag_vllm_*_*_k5_*_nq
    - Singleton experiments (iterative_*, selfrag_*, premium_*)
    """
    config = {
        'name': name,
        'exp_type': 'unknown',
        'model': 'unknown',
        'model_short': 'unknown',
        'dataset': 'unknown',
        'prompt': 'unknown',
        'retriever': None,
        'retriever_type': None,
        'embedding_model': None,
        'top_k': None,
        'query_transform': 'none',
        'reranker': 'none',
        'agent_type': _agent_type_from_name(name),
        'chunk_size': None,
        'is_singleton': False,
    }

    # Detect dataset (at end of name)
    for ds in ['nq', 'triviaqa', 'hotpotqa']:
        if name.endswith(f'_{ds}'):
            config['dataset'] = ds
            break

    # Fallback: look for dataset anywhere in the name
    if config['dataset'] == 'unknown':
        for ds in ['triviaqa', 'hotpotqa', 'nq']:
            if f'_{ds}_' in name or f'_{ds}' in name:
                config['dataset'] = ds
                break

    # Handle singleton experiments
    if name.startswith('iterative_') or name.startswith('selfrag_') or name.startswith('premium_'):
        config['is_singleton'] = True
        config['exp_type'] = 'rag'

        # Detect model from singleton name - order matters (most specific first)
        name_lower = name.lower()
        if 'gemma9b' in name_lower or 'gemma2_9b' in name_lower:
            config['model_short'] = 'Gemma2-9B'
        elif 'gemma2_2b' in name_lower or 'gemma2b' in name_lower:
            config['model_short'] = 'Gemma2-2B'
        elif 'mistral7b' in name_lower or 'mistral' in name_lower:
            config['model_short'] = 'Mistral-7B'
        elif 'qwen7b' in name_lower:
            config['model_short'] = 'Qwen2.5-7B'
        elif 'qwen1.5b' in name_lower or 'qwen1_5b' in name_lower:
            config['model_short'] = 'Qwen2.5-1.5B'
        elif 'qwen3b' in name_lower:
            config['model_short'] = 'Qwen2.5-3B'
        elif 'qwen' in name_lower:
            config['model_short'] = 'Qwen2.5-3B'  # fallback for older naming
        elif 'llama' in name_lower:
            config['model_short'] = 'Llama-3.2-3B'
        elif 'phi' in name_lower:
            config['model_short'] = 'Phi-3-mini'

        if name.startswith('iterative_'):
            # iterative_rag is an *agent strategy*, not a retriever type.
            # Detect the underlying retriever from the name; leave None if unknown
            # so that metadata.json can override it.
            config['retriever_type'] = _retriever_type_from_name(name)
            iter_match = re.search(r'(\d+)iter', name)
            config['query_transform'] = f"iterative_{iter_match.group(1)}" if iter_match else 'iterative'
        elif name.startswith('selfrag_'):
            # self_rag is an *agent strategy*, not a retriever type.
            config['retriever_type'] = _retriever_type_from_name(name)
            config['query_transform'] = 'self_rag'
        elif name.startswith('premium_'):
            config['retriever_type'] = 'hybrid'
            config['query_transform'] = 'hyde'
            config['reranker'] = 'bge-v2'
        return config

    # Direct experiments
    if name.startswith('direct_'):
        config['exp_type'] = 'direct'
        for key, display in MODEL_MAP.items():
            if key.lower() in name.lower():
                config['model_short'] = display
                break
        # Order matters: more specific patterns first
        for prompt in ['concise_strict', 'concise_json', 'extractive_quoted', 'cot_final',
                       'fewshot_3', 'fewshot_1', 'fewshot', 'concise', 'structured',
                       'cot', 'extractive', 'cited']:
            if f'_{prompt}_' in name or name.endswith(f'_{prompt}_{config["dataset"]}'):
                config['prompt'] = prompt
                break
        return config

    # RAG experiments (rag_*, iterative_rag_*, self_rag_*)
    if name.startswith('rag_') or name.startswith('self_rag_') or name.startswith('iterative_rag_'):
        config['exp_type'] = 'rag'

        for key, display in MODEL_MAP.items():
            if key.lower() in name.lower():
                config['model_short'] = display
                break

        k_match = re.search(r'_k(\d+)_', name)
        if k_match:
            config['top_k'] = int(k_match.group(1))

        for rtype, patterns in RETRIEVER_TYPES.items():
            for pattern in patterns:
                if pattern in name.lower():
                    config['retriever_type'] = rtype
                    break

        for key, display in EMBEDDING_MAP.items():
            if key in name.lower():
                config['embedding_model'] = display
                break

        if '_hyde_' in name.lower():
            config['query_transform'] = 'hyde'
        elif '_multiquery_' in name.lower():
            config['query_transform'] = 'multiquery'

        if '_bgev2_' in name.lower() or '_bge-v2_' in name.lower():
            config['reranker'] = 'bge-v2'
        elif '_bge_' in name.lower() and config['embedding_model'] is None:
            config['reranker'] = 'bge'

        # Order matters: more specific patterns first
        for prompt in ['concise_strict', 'concise_json', 'extractive_quoted', 'cot_final',
                       'fewshot_3', 'fewshot_1', 'fewshot', 'concise', 'structured',
                       'cot', 'extractive', 'cited']:
            if f'_{prompt}_' in name:
                config['prompt'] = prompt
                break

        if k_match:
            retriever_match = re.search(r'Instruct_(.+?)_k\d+', name)
            if not retriever_match:
                # Try broader match for normalized model names
                retriever_match = re.search(r'(?:bit|ruct)_(.+?)_k\d+', name)
            if retriever_match:
                config['retriever'] = retriever_match.group(1)

    return config


def _enrich_from_metadata(row: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Overlay structured fields from metadata.json onto a parsed row.

    Fields from metadata take priority over name-based parsing because
    they are authoritative (written by the experiment runner).  However,
    empty-string values are treated as missing so that name-parsed values
    are preserved.
    """
    # Direct fields from metadata — only override if value is truthy
    if metadata.get('type'):
        row['exp_type'] = metadata['type']
    if metadata.get('model'):
        row['model'] = metadata['model']
        row['model_short'] = _model_short_from_spec(metadata['model'])
    if metadata.get('dataset'):
        ds = metadata['dataset']
        row['dataset'] = DATASET_ALIASES.get(ds, ds)
    if metadata.get('prompt'):
        row['prompt'] = metadata['prompt']

    # Retriever fields
    if 'retriever' in metadata and metadata['retriever']:
        row['retriever'] = metadata['retriever']
        if row.get('retriever_type') is None or row['retriever_type'] == 'unknown':
            row['retriever_type'] = _retriever_type_from_name(metadata['retriever'])
        if row.get('embedding_model') is None:
            row['embedding_model'] = _embedding_model_from_name(metadata['retriever'])
        if row.get('chunk_size') is None:
            row['chunk_size'] = _chunk_size_from_name(metadata['retriever'])

    if 'top_k' in metadata and metadata['top_k'] is not None:
        row['top_k'] = metadata['top_k']
    if 'fetch_k' in metadata and metadata['fetch_k'] is not None:
        row['fetch_k'] = metadata['fetch_k']

    # Query transform
    qt = metadata.get('query_transform')
    if qt and qt != 'none':
        row['query_transform'] = qt
    elif qt == 'none' or qt is None:
        row.setdefault('query_transform', 'none')

    # Reranker — metadata is authoritative, always override
    if 'reranker' in metadata:
        rr = metadata['reranker']
        row['reranker'] = rr if rr else 'none'

    rr_model = metadata.get('reranker_model')
    if rr_model:
        row['reranker_model'] = rr_model

    # Agent type
    at = metadata.get('agent_type')
    if at:
        row['agent_type'] = at
    elif row.get('agent_type') is None:
        row['agent_type'] = _agent_type_from_name(row.get('name', ''))


def _strip_fake_tokens(name: str) -> Optional[str]:
    """Strip fake reranker/query_transform tokens from an experiment name.

    Due to bugs in the old code, ``query_transform`` (hyde, multiquery) and
    ``reranker`` (bge, bgev2, …) were encoded in names but never wired.
    This returns the corrected name (or None if nothing to strip).
    """
    parts = name.split('_')

    # Find _k{N}_ segment — tokens to strip sit between k{N} and prompt_dataset
    k_idx = None
    for i, p in enumerate(parts):
        if re.match(r'^k\d+$', p):
            k_idx = i
            break

    if k_idx is None:
        return None

    tail = parts[k_idx + 1:]
    if len(tail) < 2:
        return None

    prompt_dataset = tail[-2:]
    middle = tail[:-2]

    cleaned = []
    any_stripped = False
    for token in middle:
        token_norm = token.lower().replace('-', '')
        if token_norm in _FAKE_RERANKER_NORMALISED or token.lower() in _FAKE_QT_TOKENS:
            any_stripped = True
        else:
            cleaned.append(token)

    if not any_stripped:
        return None

    new_parts = parts[: k_idx + 1] + cleaned + prompt_dataset
    return '_'.join(new_parts)


def _effective_config_key(row: Dict[str, Any]) -> tuple:
    """Compute a deduplication key treating unwired reranker/qt as 'none'.

    Two experiments that differ only in fake tokens produce the same key.
    """
    return (
        row.get('model_short', 'unknown'),
        row.get('dataset', 'unknown'),
        row.get('exp_type', 'unknown'),
        row.get('retriever_type'),
        row.get('embedding_model'),
        row.get('top_k'),
        row.get('prompt', 'unknown'),
        row.get('agent_type', 'unknown'),
    )


def _deduplicate_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate experiments that differ only in unwired parameters.

    For each group of experiments that share the same *effective* configuration
    (model, dataset, retriever, top_k, prompt, agent_type — treating reranker
    and query_transform as 'none' because they were never wired), keeps only
    the row with the best F1 score.

    Also normalises the ``query_transform`` and ``reranker`` columns to
    ``'none'`` for experiments whose names contained fake tokens.
    """
    if df.empty:
        return df

    # Detect which rows have fake tokens in their name
    fake_mask = df['name'].apply(lambda n: _strip_fake_tokens(n) is not None)

    # For rows with fake tokens, override reranker/qt to 'none'
    if fake_mask.any():
        df.loc[fake_mask, 'query_transform'] = 'none'
        df.loc[fake_mask, 'reranker'] = 'none'

    # Compute effective key for every row
    keys = df.apply(_effective_config_key, axis=1)

    # Group and keep best F1
    keep_idx = []
    for _key, group in df.groupby(keys):
        if PRIMARY_METRIC in group.columns:
            best = group[PRIMARY_METRIC].fillna(-1).idxmax()
        else:
            best = group.index[0]
        keep_idx.append(best)

    deduped = df.loc[keep_idx].copy()

    n_removed = len(df) - len(deduped)
    if n_removed > 0:
        print(f"  Deduplicated: dropped {n_removed} duplicate experiments "
              f"(same effective config, kept best F1)")

    return deduped.reset_index(drop=True)


def load_all_results(
    study_path: Path = None,
    deduplicate: bool = True,
    filter_to_search_space: bool = False,
    config_path: Path = None,
) -> pd.DataFrame:
    """Load all experiment results into a DataFrame.

    Prefers structured fields from ``metadata.json`` over name parsing.
    Falls back to name parsing when metadata is unavailable.

    Args:
        study_path: Path to the study directory.
        deduplicate: If True (default), remove duplicate experiments that
            differ only in unwired reranker/query_transform tokens, keeping
            the one with the best F1 per effective configuration.
        filter_to_search_space: If True, drop experiments whose parameters
            fall outside the current YAML search space (to align with Optuna).
            Default False — all experiments are kept; use
            ``get_outside_search_space_summary()`` or
            ``print_outside_search_space_suggestion()`` to inspect outside configs.
        config_path: Path to the study YAML config (for filtering or summary).
    """
    if study_path is None:
        study_path = DEFAULT_STUDY_PATH

    results = []
    loading_errors = []

    if not study_path.exists():
        print(f"Warning: Study path does not exist: {study_path}")
        return pd.DataFrame()

    for exp_dir in study_path.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in SKIP_DIRS:
            continue
        # Skip hidden/archive directories (e.g. _tainted, _collisions)
        if exp_dir.name.startswith('.') or exp_dir.name.startswith('_'):
            continue

        # Skip incomplete / failed experiments — they have no useful metrics
        state_file = exp_dir / "state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                phase = state.get('phase', '')
                if phase in ('failed', 'generating', 'init', 'computing_metrics'):
                    continue
            except (json.JSONDecodeError, OSError):
                pass

        results_file = exp_dir / "results.json"
        metadata_file = exp_dir / "metadata.json"

        data = None
        metadata = {}

        # Load results.json (primary: has metrics + metadata)
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)

        # Load metadata.json for structured fields
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

        # Fallback: use metadata as data source if no results.json
        if data is None and metadata:
            data = metadata.copy()
            summary_files = list(exp_dir.glob("*_summary.json"))
            if summary_files:
                with open(summary_files[0]) as f:
                    summary = json.load(f)
                data['metrics'] = summary.get('overall_metrics', summary)

        if data is None:
            continue

        try:
            exp_name = data.get('name', exp_dir.name)

            # Step 1: Parse name (baseline, may be imprecise)
            config = parse_experiment_name(exp_name)

            # Step 2: Overlay authoritative fields from metadata
            _enrich_from_metadata(config, metadata if metadata else data)

            row = config.copy()

            # Extract metrics
            metrics = data.get('metrics', data)
            if isinstance(metrics, list):
                metrics = data.get('overall_metrics', {})
            if not isinstance(metrics, dict):
                metrics = {}

            for metric in METRICS:
                if metric in metrics:
                    row[metric] = metrics[metric]
                elif metric in data:
                    row[metric] = data[metric]

            row['n_samples'] = data.get('n_samples', data.get('num_questions', None))
            row['duration'] = data.get('duration', 0)
            row['throughput'] = data.get('throughput_qps', 0)

            results.append(row)
        except Exception as e:
            loading_errors.append((exp_dir.name, str(e)))

    if loading_errors:
        print(f"Warning: Failed to load {len(loading_errors)} experiments")

    df = pd.DataFrame(results)
    if not df.empty:
        cfg = config_path or DEFAULT_CONFIG_PATH
        if filter_to_search_space and Path(cfg).exists():
            # Filter FIRST (before dedup) so dedup only picks best F1 among in-space exps
            df = filter_to_search_space(df, config_path=cfg)
        elif Path(cfg).exists():
            # Keep all; suggest exploring outside configs
            print_outside_search_space_suggestion(df, config_path=cfg)

        if deduplicate:
            df = _deduplicate_experiments(df)

        df = df.sort_values(['exp_type', 'model_short', 'dataset']).reset_index(drop=True)

    return df


def load_failed_experiments(study_path: Path = None) -> pd.DataFrame:
    """
    Load failed experiments from a study to analyze failure patterns.
    
    Returns a DataFrame with:
    - name: experiment name
    - phase: experiment phase when it failed
    - error: error message
    - model_short: parsed model name
    - retriever_type: parsed retriever type
    - top_k: parsed top_k value
    
    Useful for identifying:
    - Context length issues (e.g., hier retrieval + small context models)
    - OOM errors
    - Other systematic failures
    """
    if study_path is None:
        study_path = DEFAULT_STUDY_PATH
    
    failed = []
    
    if not study_path.exists():
        print(f"Warning: Study path does not exist: {study_path}")
        return pd.DataFrame()
    
    for exp_dir in study_path.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in SKIP_DIRS:
            continue
        if exp_dir.name.startswith('.') or exp_dir.name.startswith('_'):
            continue
        
        state_file = exp_dir / "state.json"
        if not state_file.exists():
            continue
        
        with open(state_file) as f:
            state = json.load(f)
        
        # Check if experiment failed
        if state.get('phase') == 'failed' or state.get('error'):
            config = parse_experiment_name(exp_dir.name)
            
            failed.append({
                'name': exp_dir.name,
                'phase': state.get('phase', 'unknown'),
                'error': state.get('error', 'unknown'),
                'model_short': config.get('model_short', 'unknown'),
                'retriever_type': config.get('retriever_type'),
                'top_k': config.get('top_k'),
                'embedding_model': config.get('embedding_model'),
                'prompt': config.get('prompt'),
                'dataset': config.get('dataset'),
                'predictions_complete': state.get('predictions_complete', 0),
                'total_questions': state.get('total_questions', 0),
            })
    
    df = pd.DataFrame(failed)
    if not df.empty:
        df = df.sort_values(['model_short', 'retriever_type']).reset_index(drop=True)
    
    return df


def analyze_failure_patterns(failed_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze patterns in failed experiments.
    
    Returns a dict with:
    - by_model: failure counts per model
    - by_retriever: failure counts per retriever type
    - by_error_type: categorized error patterns
    - context_length_issues: experiments likely failed due to context length
    """
    if failed_df.empty:
        return {'total_failed': 0}
    
    # Categorize errors
    context_errors = failed_df[
        failed_df['error'].str.contains('context|length|token|truncat', case=False, na=False)
    ]
    oom_errors = failed_df[
        failed_df['error'].str.contains('OOM|out of memory|CUDA|GPU', case=False, na=False)
    ]
    
    return {
        'total_failed': len(failed_df),
        'by_model': failed_df['model_short'].value_counts().to_dict(),
        'by_retriever': failed_df['retriever_type'].value_counts().to_dict(),
        'context_length_issues': len(context_errors),
        'context_length_details': context_errors[['name', 'model_short', 'retriever_type', 'top_k']].to_dict('records') if len(context_errors) > 0 else [],
        'oom_issues': len(oom_errors),
        'oom_details': oom_errors[['name', 'model_short', 'error']].to_dict('records') if len(oom_errors) > 0 else [],
        'partial_completions': failed_df[failed_df['predictions_complete'] > 0][['name', 'predictions_complete', 'total_questions']].to_dict('records'),
    }


def predict_context_length_issues(study_path: Path = None) -> pd.DataFrame:
    """
    Predict which experiment configurations are likely to fail due to context length.
    
    Context length estimation:
    - dense_* retriever: ~512 tokens/doc
    - hier_* retriever: ~2048 tokens/doc (parent chunks)
    - hybrid_* retriever: ~512 tokens/doc
    
    Model context limits (approximate):
    - Phi-3-mini-4k: 4096 tokens
    - Others: 8192+ tokens (typically safe)
    
    Returns a DataFrame of risky configurations.
    """
    # Known context limits
    MODEL_CONTEXT_LIMITS = {
        'Phi-3-mini': 4096,
        'Qwen2.5-1.5B': 32768,
        'Gemma2-2B': 8192,
        'Llama-3.2-3B': 8192,
        'Qwen2.5-3B': 32768,
        'Mistral-7B': 32768,
        'Qwen2.5-7B': 131072,
        'Gemma2-9B': 8192,
    }
    
    # Tokens per doc by retriever type
    TOKENS_PER_DOC = {
        'dense': 512,
        'hybrid': 512,
        'hierarchical': 2048,  # Parent chunks
    }
    
    # Prompt overhead (approximate)
    PROMPT_OVERHEAD = 200  # System + instructions + question
    
    risky = []
    
    if study_path is None:
        study_path = DEFAULT_STUDY_PATH
    
    if not study_path.exists():
        return pd.DataFrame()
    
    for exp_dir in study_path.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in SKIP_DIRS:
            continue
        if exp_dir.name.startswith('.') or exp_dir.name.startswith('_'):
            continue
        
        config = parse_experiment_name(exp_dir.name)
        
        if config['exp_type'] != 'rag':
            continue
        
        model = config.get('model_short', 'unknown')
        retriever_type = config.get('retriever_type')
        top_k = config.get('top_k')
        
        if model == 'unknown' or not retriever_type or not top_k:
            continue
        
        context_limit = MODEL_CONTEXT_LIMITS.get(model, 8192)
        tokens_per_doc = TOKENS_PER_DOC.get(retriever_type, 512)
        
        estimated_context = (top_k * tokens_per_doc) + PROMPT_OVERHEAD
        
        if estimated_context > context_limit * 0.9:  # 90% threshold
            risky.append({
                'name': exp_dir.name,
                'model': model,
                'retriever_type': retriever_type,
                'top_k': top_k,
                'estimated_tokens': estimated_context,
                'context_limit': context_limit,
                'headroom_pct': (context_limit - estimated_context) / context_limit * 100,
            })
    
    return pd.DataFrame(risky)


def get_experiment_health_summary(study_path: Path = None) -> Dict[str, Any]:
    """
    Get a complete health summary of all experiments in a study.
    
    Returns:
    - total_experiments: total experiment directories
    - completed: successfully completed experiments
    - failed: failed experiments
    - in_progress: experiments still running
    - no_state: directories without state.json
    """
    if study_path is None:
        study_path = DEFAULT_STUDY_PATH
    
    if not study_path.exists():
        return {'error': f"Study path does not exist: {study_path}"}
    
    summary = {
        'total_experiments': 0,
        'complete': 0,
        'failed': 0,
        'in_progress': 0,
        'no_state': 0,
    }
    
    for exp_dir in study_path.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in SKIP_DIRS:
            continue
        if exp_dir.name.startswith('.') or exp_dir.name.startswith('_'):
            continue
        
        summary['total_experiments'] += 1
        
        state_file = exp_dir / "state.json"
        if not state_file.exists():
            summary['no_state'] += 1
            continue
        
        with open(state_file) as f:
            state = json.load(f)
        
        phase = state.get('phase', 'unknown')
        if phase == 'complete':
            summary['complete'] += 1
        elif phase == 'failed':
            summary['failed'] += 1
        elif phase in ('generating', 'computing_metrics'):
            summary['in_progress'] += 1
    
    return summary


def print_search_space_summary(df: pd.DataFrame) -> None:
    """Print a summary of all iterable search dimensions in the loaded data.

    Useful to quickly verify that every configuration axis is being
    parsed correctly and to see the full combinatorial space.
    """
    dims = [
        ('exp_type',        'Experiment Type'),
        ('model_short',     'Model'),
        ('dataset',         'Dataset'),
        ('retriever_type',  'Retriever Type'),
        ('embedding_model', 'Embedding Model'),
        ('top_k',           'Top-K'),
        ('query_transform', 'Query Transform'),
        ('reranker',        'Reranker'),
        ('prompt',          'Prompt'),
        ('agent_type',      'Agent Type'),
        ('chunk_size',      'Chunk Size'),
    ]

    print("=" * 60)
    print("Search Space Summary")
    print("=" * 60)
    print(f"Total experiments: {len(df)}")
    print()

    for col, label in dims:
        if col not in df.columns:
            continue
        unique = sorted(df[col].dropna().unique(), key=str)
        n = len(unique)
        vals = ', '.join(str(v) for v in unique)
        print(f"  {label:<18s} ({n:>2d}): {vals}")

    print("=" * 60)


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def weighted_mean_with_ci(
    df: pd.DataFrame,
    group_col: str,
    metric: str = PRIMARY_METRIC,
    weight_by: str = None,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    stratify_by: str = 'dataset',
) -> pd.DataFrame:
    """
    Compute weighted mean with bootstrap confidence intervals.

    When *stratify_by* is set and present in *df*, the mean for each group is
    computed **within** each stratum first, then averaged across strata.  This
    prevents datasets with different difficulty levels from confounding the
    group comparison.  Bootstrap CIs are also stratified.
    """
    if metric not in df.columns:
        return pd.DataFrame()

    use_strata = stratify_by and stratify_by in df.columns and df[stratify_by].nunique() > 1

    results = []

    for group_val, group_df in df.groupby(group_col):
        values = group_df[metric].dropna().values
        if len(values) == 0:
            continue

        if use_strata:
            # Compute mean within each stratum, then average across strata
            stratum_means = group_df.groupby(stratify_by)[metric].mean()
            point_est = stratum_means.mean()

            # Stratified bootstrap
            if len(stratum_means) >= 2 and len(values) >= 3:
                strata_groups = {
                    k: v[metric].dropna().values
                    for k, v in group_df.groupby(stratify_by)
                }
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    boot_stratum = []
                    for vals in strata_groups.values():
                        if len(vals) == 0:
                            continue
                        sample = np.random.choice(vals, size=len(vals), replace=True)
                        boot_stratum.append(np.mean(sample))
                    bootstrap_means.append(np.mean(boot_stratum))
                alpha = (1 - confidence) / 2
                ci_low = np.percentile(bootstrap_means, alpha * 100)
                ci_high = np.percentile(bootstrap_means, (1 - alpha) * 100)
            else:
                ci_low = ci_high = point_est
        else:
            if weight_by and weight_by in group_df.columns:
                weight_counts = group_df[weight_by].value_counts()
                weights = group_df[weight_by].map(lambda x: 1.0 / weight_counts.get(x, 1))
                weights = weights / weights.sum()
                point_est = (group_df[metric] * weights).sum()
            else:
                point_est = np.mean(values)

            if len(values) >= 3:
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    sample = np.random.choice(values, size=len(values), replace=True)
                    bootstrap_means.append(np.mean(sample))
                alpha = (1 - confidence) / 2
                ci_low = np.percentile(bootstrap_means, alpha * 100)
                ci_high = np.percentile(bootstrap_means, (1 - alpha) * 100)
            else:
                ci_low = ci_high = point_est

        results.append({
            group_col: group_val,
            'mean': point_est,
            'std': np.std(values) if len(values) > 1 else 0,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n': len(values),
            'min': np.min(values),
            'max': np.max(values),
        })

    return pd.DataFrame(results).sort_values('mean', ascending=False).reset_index(drop=True)


def normalize_within_dataset(
    df: pd.DataFrame,
    metric: str,
    method: str = 'zscore',
    dataset_col: str = 'dataset',
) -> pd.Series:
    """Normalise a metric within each dataset to remove difficulty confounding.

    Args:
        df: DataFrame with per-experiment rows.
        metric: Column name to normalise.
        method: ``'zscore'`` (mean=0, std=1) or ``'minmax'`` (0–1 range).
        dataset_col: Column identifying the dataset stratum.

    Returns:
        A Series aligned to *df*'s index with the normalised values.
    """
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for _ds, grp in df.groupby(dataset_col):
        vals = grp[metric]
        if method == 'zscore':
            std = vals.std()
            normed = (vals - vals.mean()) / std if std > 0 else vals * 0.0
        elif method == 'minmax':
            vmin, vmax = vals.min(), vals.max()
            span = vmax - vmin
            normed = (vals - vmin) / span if span > 0 else vals * 0.0
        else:
            raise ValueError(f"Unknown method: {method!r}")
        result.loc[grp.index] = normed
    return result


def effect_size(baseline_values: np.ndarray, treatment_values: np.ndarray) -> Tuple[float, float, str]:
    """
    Compute Cohen's d effect size and interpret it.
    
    Returns: (effect_size, p_value, interpretation)
    """
    if len(baseline_values) < 2 or len(treatment_values) < 2:
        return 0, 1, 'insufficient data'
    
    pooled_std = np.sqrt((
        (len(baseline_values) - 1) * np.var(baseline_values, ddof=1) + 
        (len(treatment_values) - 1) * np.var(treatment_values, ddof=1)
    ) / (len(baseline_values) + len(treatment_values) - 2))
    
    if pooled_std == 0:
        return 0, 1, 'no variance'
    
    d = (np.mean(treatment_values) - np.mean(baseline_values)) / pooled_std
    t_stat, p_value = scipy_stats.ttest_ind(treatment_values, baseline_values)
    
    if abs(d) < 0.2:
        interpretation = 'negligible'
    elif abs(d) < 0.5:
        interpretation = 'small'
    elif abs(d) < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'
    
    return d, p_value, interpretation


def compute_marginal_means(
    df: pd.DataFrame,
    factor: str,
    metric: str = PRIMARY_METRIC,
    control_vars: List[str] = ['model_short', 'dataset'],
) -> pd.DataFrame:
    """
    Compute marginal means for a factor, controlling for confounding variables.
    """
    if metric not in df.columns or factor not in df.columns:
        return pd.DataFrame()
    
    needed_cols = [factor, metric] + [c for c in control_vars if c in df.columns]
    work_df = df[needed_cols].dropna()
    
    if len(work_df) < 5:
        return pd.DataFrame()
    
    strata_cols = [c for c in control_vars if c in work_df.columns]
    if strata_cols:
        work_df['_stratum'] = work_df[strata_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    else:
        work_df['_stratum'] = 'all'
    
    results = []
    factor_levels = work_df[factor].dropna().unique()
    
    for level in factor_levels:
        level_df = work_df[work_df[factor] == level]
        stratum_means = level_df.groupby('_stratum')[metric].mean()
        
        if len(stratum_means) == 0:
            continue
        
        marginal_mean = stratum_means.mean()
        
        # Bootstrap the *marginal* mean (mean-of-stratum-means) so the CI
        # matches the point estimate.  Resampling raw values would compute a
        # CI for the raw mean, which can diverge from the marginal mean when
        # strata have unequal sizes, leading to negative yerr in plots.
        strata_groups = {k: v[metric].values for k, v in level_df.groupby('_stratum')}
        n_strata_with_data = sum(1 for v in strata_groups.values() if len(v) > 0)
        if n_strata_with_data >= 2 and len(level_df) >= 3:
            bootstrap_means = []
            for _ in range(500):
                stratum_boot_means = []
                for vals in strata_groups.values():
                    if len(vals) == 0:
                        continue
                    sample = np.random.choice(vals, size=len(vals), replace=True)
                    stratum_boot_means.append(np.mean(sample))
                bootstrap_means.append(np.mean(stratum_boot_means))
            ci_low = np.percentile(bootstrap_means, 2.5)
            ci_high = np.percentile(bootstrap_means, 97.5)
        elif len(level_df) >= 3:
            # Single stratum -- bootstrap the raw values directly
            all_values = level_df[metric].values
            bootstrap_means = []
            for _ in range(500):
                sample = np.random.choice(all_values, size=len(all_values), replace=True)
                bootstrap_means.append(np.mean(sample))
            ci_low = np.percentile(bootstrap_means, 2.5)
            ci_high = np.percentile(bootstrap_means, 97.5)
        else:
            ci_low = ci_high = marginal_mean
        
        results.append({
            factor: level,
            'marginal_mean': marginal_mean,
            'raw_mean': level_df[metric].mean(),
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_experiments': len(level_df),
            'n_strata': len(stratum_means),
        })
    
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values('marginal_mean', ascending=False).reset_index(drop=True)
    return result_df


# =============================================================================
# RAG BENEFIT ANALYSIS
# =============================================================================

def analyze_rag_benefit_distribution(df: pd.DataFrame, metric: str = PRIMARY_METRIC) -> dict:
    """
    Analyze the distribution of RAG benefit across configurations.
    """
    rag_df = df[df['exp_type'] == 'rag'].copy()
    direct_df = df[df['exp_type'] == 'direct'].copy()
    
    if len(rag_df) == 0 or len(direct_df) == 0:
        return {}
    
    group_cols = []
    if 'model_short' in direct_df.columns:
        group_cols.append('model_short')
    if 'dataset' in direct_df.columns:
        group_cols.append('dataset')
    
    if not group_cols:
        direct_baseline = direct_df[metric].mean()
        rag_df['direct_baseline'] = direct_baseline
    else:
        direct_baselines = direct_df.groupby(group_cols)[metric].mean().reset_index()
        direct_baselines.columns = group_cols + ['direct_baseline']
        rag_df = rag_df.merge(direct_baselines, on=group_cols, how='left')
    
    rag_df['rag_benefit'] = rag_df[metric] - rag_df['direct_baseline']
    rag_df['rag_benefit_pct'] = (rag_df['rag_benefit'] / rag_df['direct_baseline'] * 100).replace([np.inf, -np.inf], np.nan)
    
    rag_df['rag_helps'] = rag_df['rag_benefit'] > 0.01
    rag_df['rag_hurts'] = rag_df['rag_benefit'] < -0.01
    rag_df['rag_neutral'] = ~rag_df['rag_helps'] & ~rag_df['rag_hurts']
    
    return {
        'rag_df': rag_df,
        'n_helps': rag_df['rag_helps'].sum(),
        'n_hurts': rag_df['rag_hurts'].sum(),
        'n_neutral': rag_df['rag_neutral'].sum(),
        'pct_helps': rag_df['rag_helps'].mean() * 100,
        'pct_hurts': rag_df['rag_hurts'].mean() * 100,
        'mean_benefit_when_helps': rag_df.loc[rag_df['rag_helps'], 'rag_benefit'].mean(),
        'mean_hurt_when_hurts': rag_df.loc[rag_df['rag_hurts'], 'rag_benefit'].mean(),
        'best_rag_benefit': rag_df['rag_benefit'].max(),
        'worst_rag_benefit': rag_df['rag_benefit'].min(),
    }


def compare_best_rag_vs_direct(df: pd.DataFrame, metric: str = PRIMARY_METRIC, 
                                top_k: int = 5) -> pd.DataFrame:
    """
    Compare only the top-K best RAG configurations against direct baseline.
    """
    rag_df = df[df['exp_type'] == 'rag'].copy()
    direct_df = df[df['exp_type'] == 'direct'].copy()
    
    results = []
    
    group_cols = []
    if 'model_short' in df.columns:
        group_cols.append('model_short')
    if 'dataset' in df.columns:
        group_cols.append('dataset')
    
    if not group_cols:
        direct_mean = direct_df[metric].mean()
        top_rag = rag_df.nlargest(top_k, metric)
        
        results.append({
            'group': 'overall',
            'direct_mean': direct_mean,
            'top_rag_mean': top_rag[metric].mean(),
            'best_rag': top_rag[metric].max(),
            'rag_advantage': top_rag[metric].mean() - direct_mean,
            'rag_advantage_pct': (top_rag[metric].mean() - direct_mean) / direct_mean * 100 if direct_mean > 0 else 0
        })
    else:
        for group_vals, rag_group in rag_df.groupby(group_cols):
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            
            direct_mask = pd.Series(True, index=direct_df.index)
            for col, val in zip(group_cols, group_vals):
                direct_mask &= (direct_df[col] == val)

            direct_group = direct_df[direct_mask]
            if len(direct_group) == 0:
                continue
            
            direct_mean = direct_group[metric].mean()
            top_rag = rag_group.nlargest(min(top_k, len(rag_group)), metric)
            
            results.append({
                'group': ' / '.join(str(v) for v in group_vals),
                **{col: val for col, val in zip(group_cols, group_vals)},
                'direct_mean': direct_mean,
                'top_rag_mean': top_rag[metric].mean(),
                'best_rag': top_rag[metric].max(),
                'rag_advantage': top_rag[metric].mean() - direct_mean,
                'rag_advantage_pct': (top_rag[metric].mean() - direct_mean) / direct_mean * 100 if direct_mean > 0 else 0,
                'n_rag_configs': len(rag_group)
            })
    
    return pd.DataFrame(results)


def identify_rag_success_factors(
    df: pd.DataFrame, metric: str = PRIMARY_METRIC, stratify_by: str = 'dataset',
) -> dict:
    """
    Identify which factors predict RAG success (helping vs hurting).

    When *stratify_by* is set, aggregates within each stratum first and then
    averages across strata so that dataset difficulty doesn't confound results.
    """
    benefit_analysis = analyze_rag_benefit_distribution(df, metric)
    if not benefit_analysis:
        return {}

    rag_df = benefit_analysis['rag_df']

    factors = ['retriever_type', 'embedding_model', 'reranker', 'prompt', 'query_transform', 'top_k']
    available_factors = [f for f in factors if f in rag_df.columns]

    use_strata = stratify_by and stratify_by in rag_df.columns and rag_df[stratify_by].nunique() > 1

    success_factors = {}
    for factor in available_factors:
        if use_strata:
            # Aggregate within each (factor, stratum), then average across strata
            per_stratum = rag_df.groupby([factor, stratify_by]).agg({
                'rag_helps': 'mean',
                'rag_benefit': ['mean', 'std'],
                'rag_benefit_pct': 'mean',
            })
            per_stratum.columns = ['pct_helps', 'mean_benefit', 'std_benefit', 'mean_benefit_pct']
            factor_success = per_stratum.groupby(level=factor).mean().round(3)
        else:
            factor_success = rag_df.groupby(factor).agg({
                'rag_helps': 'mean',
                'rag_benefit': ['mean', 'std'],
                'rag_benefit_pct': 'mean',
            }).round(3)
            factor_success.columns = ['pct_helps', 'mean_benefit', 'std_benefit', 'mean_benefit_pct']
        factor_success = factor_success.sort_values('pct_helps', ascending=False)
        success_factors[factor] = factor_success

    return success_factors


# =============================================================================
# BOTTLENECK ANALYSIS
# =============================================================================

def identify_bottlenecks(
    df: pd.DataFrame, metric: str = PRIMARY_METRIC, stratify_by: str = 'dataset',
) -> dict:
    """
    Identify which RAG components contribute most to performance variance.

    When *stratify_by* is set and the column exists with >1 value, the variance
    decomposition is performed **within** each stratum and then averaged.  This
    removes confounding from datasets of different difficulty levels.
    """
    rag_df = df[df['exp_type'] == 'rag']

    factors = ['model_short', 'retriever_type', 'embedding_model', 'reranker',
               'prompt', 'query_transform', 'top_k']

    use_strata = stratify_by and stratify_by in rag_df.columns and rag_df[stratify_by].nunique() > 1

    if use_strata:
        strata = rag_df[stratify_by].unique()
        per_stratum_results = []
        for stratum_val in strata:
            sdf = rag_df[rag_df[stratify_by] == stratum_val]
            total_var = sdf[metric].var()
            if total_var == 0 or len(sdf) < 3:
                continue
            stratum_dict = {}
            for factor in factors:
                if factor not in sdf.columns or sdf[factor].nunique() <= 1:
                    continue
                group_means = sdf.groupby(factor)[metric].mean()
                overall_mean = sdf[metric].mean()
                between_group_var = np.sum(
                    (group_means - overall_mean) ** 2
                    * sdf.groupby(factor).size() / len(sdf)
                )
                stratum_dict[factor] = (between_group_var / total_var) * 100
            per_stratum_results.append(stratum_dict)

        if not per_stratum_results:
            return {}

        # Average variance-explained across strata
        all_factors = set()
        for d in per_stratum_results:
            all_factors.update(d.keys())
        bottleneck_results = {}
        for f in all_factors:
            vals = [d[f] for d in per_stratum_results if f in d]
            bottleneck_results[f] = np.mean(vals)
    else:
        total_var = rag_df[metric].var()
        if total_var == 0:
            return {}

        bottleneck_results = {}
        for factor in factors:
            if factor not in rag_df.columns or rag_df[factor].nunique() <= 1:
                continue
            group_means = rag_df.groupby(factor)[metric].mean()
            overall_mean = rag_df[metric].mean()
            between_group_var = np.sum(
                (group_means - overall_mean) ** 2
                * rag_df.groupby(factor).size() / len(rag_df)
            )
            bottleneck_results[factor] = (between_group_var / total_var) * 100

    return dict(sorted(bottleneck_results.items(), key=lambda x: x[1], reverse=True))


# =============================================================================
# MULTI-METRIC ANALYSIS
# =============================================================================

def multi_metric_bottlenecks(
    df: pd.DataFrame,
    metrics: List[str] = None,
) -> pd.DataFrame:
    """Run variance decomposition across multiple metrics.

    Returns a DataFrame with factors as rows and metrics as columns,
    values are % variance explained.  Useful for side-by-side comparison
    of which components matter for different quality dimensions.
    """
    if metrics is None:
        metrics = [m for m in MULTI_METRIC_SET if m in df.columns and df[m].notna().sum() >= 10]

    results = {}
    for m in metrics:
        bn = identify_bottlenecks(df, m)
        if bn:
            results[m] = bn

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).fillna(0).sort_values(
        results.get(PRIMARY_METRIC, list(results.keys())[0]).keys().__iter__().__next__(),
        ascending=False,
    ) if results else pd.DataFrame()


def multi_metric_bottlenecks_df(
    df: pd.DataFrame,
    metrics: List[str] = None,
) -> pd.DataFrame:
    """Run variance decomposition across multiple metrics.

    Returns a tidy DataFrame: columns [factor, metric, variance_pct].
    """
    if metrics is None:
        metrics = [m for m in MULTI_METRIC_SET if m in df.columns and df[m].notna().sum() >= 10]

    rows = []
    for m in metrics:
        bn = identify_bottlenecks(df, m)
        for factor, pct in bn.items():
            rows.append({'factor': factor, 'metric': m, 'variance_pct': pct})

    return pd.DataFrame(rows)


def metric_correlation_matrix(
    df: pd.DataFrame,
    metrics: List[str] = None,
    method: str = 'spearman',
) -> pd.DataFrame:
    """Compute pairwise correlation matrix for available metrics."""
    if metrics is None:
        metrics = [m for m in MULTI_METRIC_SET if m in df.columns and df[m].notna().sum() >= 10]

    return df[metrics].dropna().corr(method=method)


# =============================================================================
# INTERACTION ANALYSIS
# =============================================================================

def analyze_interactions(
    df: pd.DataFrame, factor1: str, factor2: str,
    metric: str = PRIMARY_METRIC, stratify_by: str = 'dataset',
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Analyze interaction effects between two RAG components.

    When *stratify_by* is set, computes the interaction heatmap within each
    stratum and then averages across strata to control for dataset difficulty.
    """
    rag_df = df[df['exp_type'] == 'rag'].copy()

    if factor1 not in rag_df.columns or factor2 not in rag_df.columns:
        return None

    use_strata = stratify_by and stratify_by in rag_df.columns and rag_df[stratify_by].nunique() > 1

    if use_strata:
        # Compute mean within each (factor1, factor2, stratum), then average
        per_stratum = rag_df.groupby([factor1, factor2, stratify_by])[metric].agg(
            ['mean', 'std', 'count']
        ).reset_index()
        interaction = per_stratum.groupby([factor1, factor2]).agg(
            mean=('mean', 'mean'),
            std=('std', 'mean'),
            count=('count', 'sum'),
        ).reset_index()
    else:
        interaction = rag_df.groupby([factor1, factor2])[metric].agg(
            ['mean', 'std', 'count']
        ).reset_index()
        interaction.columns = [factor1, factor2, 'mean', 'std', 'count']

    pivot = interaction.pivot(index=factor1, columns=factor2, values='mean')

    return pivot, interaction


def find_synergistic_combinations(
    df: pd.DataFrame, factor1: str, factor2: str,
    metric: str = PRIMARY_METRIC, stratify_by: str = 'dataset',
) -> List[dict]:
    """
    Identify synergistic and redundant component combinations.

    When *stratify_by* is set, interaction effects are computed within each
    stratum and averaged, preventing dataset difficulty from biasing the
    synergy/redundancy classification.
    """
    rag_df = df[df['exp_type'] == 'rag'].copy()

    if factor1 not in rag_df.columns or factor2 not in rag_df.columns:
        return []

    use_strata = stratify_by and stratify_by in rag_df.columns and rag_df[stratify_by].nunique() > 1

    if use_strata:
        strata = rag_df[stratify_by].unique()
        # Accumulate interaction effects per combo across strata
        combo_effects: Dict[Tuple, List[dict]] = defaultdict(list)
        for stratum_val in strata:
            sdf = rag_df[rag_df[stratify_by] == stratum_val]
            if len(sdf) < 3:
                continue
            s_overall = sdf[metric].mean()
            s_f1 = sdf.groupby(factor1)[metric].mean()
            s_f2 = sdf.groupby(factor2)[metric].mean()
            s_combo = sdf.groupby([factor1, factor2])[metric].mean()
            for (v1, v2), actual in s_combo.items():
                if v1 not in s_f1.index or v2 not in s_f2.index:
                    continue
                expected = s_overall + (s_f1[v1] - s_overall) + (s_f2[v2] - s_overall)
                combo_effects[(v1, v2)].append({
                    'actual': actual, 'expected': expected,
                    'interaction_effect': actual - expected,
                })

        results = []
        for (v1, v2), effects in combo_effects.items():
            avg_actual = np.mean([e['actual'] for e in effects])
            avg_expected = np.mean([e['expected'] for e in effects])
            avg_ie = np.mean([e['interaction_effect'] for e in effects])
            results.append({
                factor1: v1, factor2: v2,
                'actual': avg_actual, 'expected': avg_expected,
                'interaction_effect': avg_ie,
                'synergy': 'Synergistic' if avg_ie > 0.01 else
                           'Redundant' if avg_ie < -0.01 else 'Neutral',
            })
    else:
        overall_mean = rag_df[metric].mean()
        f1_means = rag_df.groupby(factor1)[metric].mean()
        f2_means = rag_df.groupby(factor2)[metric].mean()
        combo_means = rag_df.groupby([factor1, factor2])[metric].mean()

        results = []
        for (v1, v2), actual in combo_means.items():
            expected = overall_mean + (f1_means[v1] - overall_mean) + (f2_means[v2] - overall_mean)
            interaction_effect = actual - expected
            results.append({
                factor1: v1, factor2: v2,
                'actual': actual, 'expected': expected,
                'interaction_effect': interaction_effect,
                'synergy': 'Synergistic' if interaction_effect > 0.01 else
                           'Redundant' if interaction_effect < -0.01 else 'Neutral',
            })

    return sorted(results, key=lambda x: x['interaction_effect'], reverse=True)


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def plot_rag_vs_direct(df: pd.DataFrame, metric: str = PRIMARY_METRIC, ax=None):
    """Plot RAG vs Direct comparison with confidence intervals."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    rag_df = df[df['exp_type'] == 'rag']
    direct_df = df[df['exp_type'] == 'direct']
    
    rag_mean = rag_df[metric].mean()
    direct_mean = direct_df[metric].mean()
    rag_std = rag_df[metric].std()
    direct_std = direct_df[metric].std()
    
    x = [0, 1]
    means = [direct_mean, rag_mean]
    stds = [direct_std, rag_std]
    labels = ['Direct', 'RAG']
    colors = ['steelblue', 'coral']
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric.upper())
    ax.set_title(f'Direct vs RAG Performance ({metric})')
    ax.grid(axis='y', alpha=0.3)
    
    return ax


def plot_component_effects(df: pd.DataFrame, factor: str, metric: str = PRIMARY_METRIC, ax=None):
    """Plot component effects with confidence intervals."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    marginal = compute_marginal_means(df[df['exp_type'] == 'rag'], factor, metric)
    
    if marginal.empty:
        return ax
    
    x = range(len(marginal))
    # Clamp to non-negative to guard against any residual mismatch
    yerr_low = np.maximum(marginal['marginal_mean'] - marginal['ci_low'], 0)
    yerr_high = np.maximum(marginal['ci_high'] - marginal['marginal_mean'], 0)
    ax.bar(x, marginal['marginal_mean'],
           yerr=[yerr_low, yerr_high],
           capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(marginal[factor], rotation=45, ha='right')
    ax.set_ylabel(f'Marginal Mean ({metric})')
    ax.set_title(f'{factor} Effect on {metric}')
    ax.grid(axis='y', alpha=0.3)
    
    return ax


def plot_rag_benefit_distribution(benefit_data: dict, ax=None):
    """Plot the distribution of RAG benefit."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    rag_benefits = benefit_data['rag_df']['rag_benefit'].dropna()
    
    ax.hist(rag_benefits, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax.axvline(x=rag_benefits.mean(), color='orange', linestyle='-', linewidth=2, 
               label=f'Mean: {rag_benefits.mean():.3f}')
    ax.axvline(x=rag_benefits.median(), color='green', linestyle='-', linewidth=2, 
               label=f'Median: {rag_benefits.median():.3f}')
    ax.set_xlabel('RAG Benefit')
    ax.set_ylabel('Number of Configurations')
    ax.set_title('Distribution of RAG Benefit vs Direct Baseline')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return ax


def plot_interaction_heatmap(df: pd.DataFrame, factor1: str, factor2: str, 
                              metric: str = PRIMARY_METRIC, ax=None):
    """Plot a heatmap showing interaction between two factors."""
    result = analyze_interactions(df, factor1, factor2, metric)
    if result is None:
        return ax
    
    pivot, _ = result
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=pivot.values.mean(), ax=ax)
    ax.set_title(f'{factor1} × {factor2} Interaction\n(Mean {metric})')
    ax.set_xlabel(factor2)
    ax.set_ylabel(factor1)

    return ax


def facet_by_dataset(
    df: pd.DataFrame,
    plot_fn,
    figsize_per_panel: Tuple[float, float] = (6, 5),
    sharey: bool = True,
    suptitle: str = None,
    dataset_col: str = 'dataset',
):
    """Create a 1×N subplot grid with one panel per dataset.

    Args:
        df: DataFrame containing a *dataset_col* column.
        plot_fn: Callable ``(ax, dataset_df, dataset_name, **kwargs) -> None``.
        figsize_per_panel: ``(width, height)`` per subplot panel.
        sharey: Whether to share the y-axis across panels.
        suptitle: Optional super-title above all panels.
        dataset_col: Column identifying dataset strata.

    Returns:
        ``(fig, axes)`` — the matplotlib Figure and array of Axes.
    """
    datasets = sorted(df[dataset_col].dropna().unique())
    n = len(datasets)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize_per_panel)
        return fig, [ax]

    fig, axes = plt.subplots(
        1, n,
        figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
        sharey=sharey,
    )
    if n == 1:
        axes = [axes]
    else:
        axes = list(axes)

    for ax, ds in zip(axes, datasets):
        ds_df = df[df[dataset_col] == ds]
        label = DATASET_LABELS.get(ds, ds)
        plot_fn(ax, ds_df, ds)
        ax.set_title(label)

    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=14)

    fig.tight_layout()
    return fig, axes


# =============================================================================
# QUESTION-LEVEL DATA LOADING
# =============================================================================

def load_predictions(exp_dir: Path, include_docs: bool = False) -> Optional[List[Dict]]:
    """Load predictions from a single experiment directory.

    Args:
        exp_dir: Path to the experiment directory.
        include_docs: If True, include ``retrieved_docs`` in each prediction.
            Defaults to False to reduce memory usage.

    Returns:
        List of prediction dicts, or None if predictions.json is missing.
    """
    pred_file = exp_dir / "predictions.json"
    if not pred_file.exists():
        return None

    with open(pred_file) as f:
        data = json.load(f)

    preds = data.get("predictions", [])

    if not include_docs:
        for p in preds:
            p.pop("retrieved_docs", None)
            p.pop("prompt", None)

    return preds


def load_per_item_scores(
    study_path: Path = None,
    experiments: List[str] = None,
    metric: str = PRIMARY_METRIC,
    include_text: bool = False,
) -> pd.DataFrame:
    """Load per-item metric scores across experiments.

    This reads the ``metrics`` dict inside each prediction entry of
    ``predictions.json``.  It is designed to be memory-efficient: only
    the requested metric (plus optionally question/answer text) is kept.

    Args:
        study_path: Root study directory.
        experiments: List of experiment directory names to load.
            If None, loads all completed experiments.
        metric: Which metric to extract (default ``f1``).
        include_text: If True, include ``question``, ``prediction``,
            and ``expected`` columns.

    Returns:
        DataFrame with columns: experiment, idx, {metric}, and optionally
        question, prediction, expected.  Also includes ``model_short``,
        ``dataset``, ``exp_type``, ``agent_type`` parsed from the experiment.
    """
    if study_path is None:
        study_path = DEFAULT_STUDY_PATH

    rows = []

    exp_dirs = []
    if experiments is not None:
        exp_dirs = [study_path / name for name in experiments]
    else:
        if not study_path.exists():
            return pd.DataFrame()
        for d in study_path.iterdir():
            if not d.is_dir() or d.name in SKIP_DIRS:
                continue
            if d.name.startswith('.') or d.name.startswith('_'):
                continue
            exp_dirs.append(d)

    for exp_dir in exp_dirs:
        preds = load_predictions(exp_dir, include_docs=False)
        if preds is None:
            continue

        config = parse_experiment_name(exp_dir.name)
        # Enrich from metadata if available
        meta_file = exp_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                _enrich_from_metadata(config, json.load(f))

        for p in preds:
            metrics = p.get("metrics", {})
            score = metrics.get(metric)
            if score is None:
                continue

            row = {
                "experiment": exp_dir.name,
                "idx": p.get("idx"),
                metric: score,
                "model_short": config.get("model_short", "unknown"),
                "dataset": config.get("dataset", "unknown"),
                "exp_type": config.get("exp_type", "unknown"),
                "agent_type": config.get("agent_type", "unknown"),
            }
            if include_text:
                row["question"] = p.get("question", "")
                row["prediction"] = p.get("prediction", "")
                row["expected"] = p.get("expected", [])
            rows.append(row)

    return pd.DataFrame(rows)


def sample_predictions_with_docs(
    study_path: Path,
    experiment_name: str,
    indices: List[int] = None,
    n: int = 10,
) -> List[Dict]:
    """Load sample predictions *with* retrieved documents for qualitative inspection.

    Args:
        study_path: Root study directory.
        experiment_name: Name of the experiment directory.
        indices: Specific question indices to load.  If None, samples
            ``n`` random predictions.
        n: Number of predictions to sample (ignored if ``indices`` given).

    Returns:
        List of prediction dicts including ``retrieved_docs``.
    """
    exp_dir = study_path / experiment_name
    preds = load_predictions(exp_dir, include_docs=True)
    if preds is None:
        return []

    if indices is not None:
        idx_set = set(indices)
        return [p for p in preds if p.get("idx") in idx_set]

    if len(preds) <= n:
        return preds

    rng = np.random.default_rng(42)
    chosen = rng.choice(len(preds), size=n, replace=False)
    return [preds[i] for i in sorted(chosen)]


def compute_question_difficulty(
    scores_df: pd.DataFrame,
    metric: str = PRIMARY_METRIC,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute per-question difficulty from per-item metric scores.

    For each unique ``(idx, question)`` pair across experiments, computes:
    - ``mean_score``: average metric across all configs
    - ``pct_correct``: fraction of configs scoring above ``threshold``
    - ``difficulty``: categorical label (easy / discriminating / hard)

    Args:
        scores_df: Output of :func:`load_per_item_scores` with
            ``include_text=True``.
        metric: Metric column to use.
        threshold: Score above which a prediction is "correct".

    Returns:
        DataFrame sorted by ``mean_score`` descending.
    """
    scores_df = scores_df.dropna(subset=[metric])
    scores_df["correct"] = scores_df[metric] >= threshold

    group_cols = ["idx"]
    if "question" in scores_df.columns:
        group_cols.append("question")
    if "dataset" in scores_df.columns:
        group_cols.append("dataset")

    agg = scores_df.groupby(group_cols).agg(
        mean_score=(metric, "mean"),
        std_score=(metric, "std"),
        pct_correct=("correct", "mean"),
        n_configs=(metric, "count"),
    ).reset_index()

    def _label(pct):
        if pct >= 0.8:
            return "easy"
        elif pct <= 0.2:
            return "hard"
        else:
            return "discriminating"

    agg["difficulty"] = agg["pct_correct"].apply(_label)
    return agg.sort_values("mean_score", ascending=False).reset_index(drop=True)


def compute_per_question_rag_delta(
    scores_df: pd.DataFrame,
    metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """Compute per-question RAG vs Direct delta.

    For each question (idx) and each (model, dataset) group, computes the
    mean direct score and each RAG config's delta from that baseline.

    Returns:
        DataFrame with columns: idx, model_short, dataset, direct_mean,
        rag_score, rag_delta, experiment, agent_type.
    """
    direct = scores_df[scores_df["exp_type"] == "direct"]
    rag = scores_df[scores_df["exp_type"] != "direct"]

    if direct.empty or rag.empty:
        return pd.DataFrame()

    # Mean direct score per (idx, model, dataset)
    direct_means = (
        direct.groupby(["idx", "model_short", "dataset"])[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "direct_mean"})
    )

    merged = rag.merge(direct_means, on=["idx", "model_short", "dataset"], how="inner")
    merged["rag_delta"] = merged[metric] - merged["direct_mean"]
    merged = merged.rename(columns={metric: "rag_score"})

    return merged
