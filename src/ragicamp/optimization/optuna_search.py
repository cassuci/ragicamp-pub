"""Optuna-powered experiment search for RAG configurations.

Provides a unified trial-by-trial loop that works with any Optuna sampler:

- **random**: ``RandomSampler`` -- equivalent to random search, but with
  SQLite persistence and resume capability.
- **tpe**: ``TPESampler`` -- Bayesian optimization that learns from previous
  trials to focus on high-performing regions of the search space.

Each trial:
1. The sampler suggests a parameter combination (model, retriever, top_k, ...)
2. An ExperimentSpec is built from those parameters
3. The experiment is run via the standard runner (subprocess isolation)
4. The metric is extracted from results and reported back to Optuna

Study state is persisted to ``{output_dir}/optuna_study.db`` (SQLite),
enabling transparent resume across runs.

By default, **dataset** and **model** are treated as *benchmark axes* and
explored uniformly via round-robin (``PartialFixedSampler``), while TPE
optimizes the remaining dimensions (retriever, top_k, prompt, …).  This
prevents the optimizer from biasing towards easy datasets or strong models.
Control which dimensions are fixed via ``rag.sampling.fixed_dims``.

Usage (YAML config):
    rag:
      sampling:
        mode: tpe            # or "random"
        n_experiments: 100
        optimize_metric: f1  # only matters for tpe
        seed: 42
        fixed_dims:          # dims explored uniformly (not optimized)
          - dataset
          - model

Usage (CLI override):
    ragicamp run config.yaml --sample 100 --sample-mode tpe
"""

import json
import random
from datetime import datetime
from itertools import product as itertools_product
from pathlib import Path
from typing import Any

import optuna
from optuna.samplers import PartialFixedSampler

from ragicamp.core.logging import get_logger
from ragicamp.execution.runner import run_spec
from ragicamp.spec.experiment import ExperimentSpec
from ragicamp.spec.naming import name_rag

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Search space extraction
# ---------------------------------------------------------------------------


def _extract_search_space(config: dict[str, Any]) -> dict[str, list[Any]]:
    """Extract categorical search space dimensions from the study config.

    Reads the same YAML fields that grid search uses, so the search space
    is always consistent with what the user configured.

    When ``rag.sampling.agent_types`` is set, an ``agent_type`` dimension is
    added so the optimizer can explore different agent strategies (e.g.
    ``iterative_rag``, ``self_rag``) alongside the standard RAG grid.

    Returns:
        Dict mapping dimension names to their possible values.
    """
    rag = config.get("rag", {})

    space: dict[str, list[Any]] = {
        "model": rag.get("models", []),
        "retriever": rag.get("retriever_names", []),
        "top_k": rag.get("top_k_values", [5]),
        "prompt": rag.get("prompts", ["concise"]),
        "query_transform": rag.get("query_transform", ["none"]),
        "dataset": config.get("datasets", ["nq"]),
    }

    # rrf_k and alpha are now conditional dimensions (hybrid-only),
    # handled by _suggest_hybrid_params() instead of being in the global space.

    # Build unique reranker names from reranker configs
    reranker_config = rag.get("reranker", {})
    if isinstance(reranker_config, dict):
        reranker_cfgs = reranker_config.get("configs", [{"enabled": False, "name": "none"}])
    else:
        reranker_cfgs = [{"enabled": False, "name": "none"}]

    reranker_names: list[str] = []
    for cfg in reranker_cfgs:
        name = cfg.get("name", "none") if cfg.get("enabled") else "none"
        if name not in reranker_names:
            reranker_names.append(name)

    if not reranker_names:
        reranker_names = ["none"]

    space["reranker"] = reranker_names

    # Optional: agent_type dimension (from rag.sampling.agent_types)
    sampling_config = rag.get("sampling", {})
    agent_types = sampling_config.get("agent_types")
    if agent_types and len(agent_types) > 1:
        space["agent_type"] = agent_types

    return space


# ---------------------------------------------------------------------------
# Feasibility constraints (prune impossible combinations early)
# ---------------------------------------------------------------------------


def _check_context_feasibility(
    model: str,
    retriever_name: str,
    top_k: int,
    prompt: str,
    retriever_lookup: dict[str, dict[str, Any]],
    constraints: dict[str, Any],
) -> tuple[bool, str]:
    """Check whether a combination will fit in the model's context window.

    Uses a conservative token estimate to decide if the prompt (retrieved
    chunks + template + question) would exceed the model's maximum input
    length.  This prevents wasting GPU time on combinations that always
    crash with "prompt length exceeds model length" errors.

    The check relies on ``rag.sampling.constraints`` in the study config:

    .. code-block:: yaml

        constraints:
          model_context_lengths:
            "vllm:google/gemma-2-2b-it": 8192
            "vllm:microsoft/Phi-3-mini-4k-instruct": 4096
          default_context_length: 32768
          chars_per_token: 4
          generation_buffer: 256
          prompt_overhead_tokens:
            default: 500
            fewshot_3: 1500
            fewshot_1: 800

    Returns:
        ``(True, "")`` if feasible, ``(False, reason)`` if not.
    """
    if not constraints:
        return True, ""

    # --- Model context length ---
    model_ctx_map = constraints.get("model_context_lengths", {})
    default_ctx = constraints.get("default_context_length", 32768)
    model_ctx = model_ctx_map.get(model, default_ctx)

    # --- Retriever effective chunk size (characters) ---
    ret_cfg = retriever_lookup.get(retriever_name, {})
    # Hierarchical retrievers return parent chunks; dense/hybrid return chunk_size
    chunk_chars = ret_cfg.get("parent_chunk_size") or ret_cfg.get("chunk_size", 512)

    # --- Convert to tokens ---
    chars_per_token = constraints.get("chars_per_token", 4)
    chunk_tokens = chunk_chars / chars_per_token

    # --- Prompt overhead (template + question + examples) ---
    overhead_cfg = constraints.get("prompt_overhead_tokens", {})
    if isinstance(overhead_cfg, dict):
        overhead = overhead_cfg.get(prompt, overhead_cfg.get("default", 500))
    else:
        overhead = int(overhead_cfg)

    # --- Generation buffer (tokens reserved for output) ---
    gen_buffer = constraints.get("generation_buffer", 256)

    # --- Feasibility check ---
    estimated_input = chunk_tokens * top_k + overhead
    max_input = model_ctx - gen_buffer

    if estimated_input > max_input:
        reason = (
            f"estimated prompt ~{int(estimated_input)} tokens "
            f"(~{int(chunk_tokens)}tok/chunk × {top_k} + {overhead} overhead) "
            f"> model context {model_ctx} - {gen_buffer} buffer = {int(max_input)}"
        )
        return False, reason

    return True, ""


# ---------------------------------------------------------------------------
# Config lookups (reuse retriever/reranker YAML definitions)
# ---------------------------------------------------------------------------


def _build_retriever_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build retriever name -> config dict lookup from the YAML."""
    rag = config.get("rag", {})
    lookup: dict[str, dict[str, Any]] = {}
    for ret_config in rag.get("retrievers", []):
        if isinstance(ret_config, dict):
            lookup[ret_config["name"]] = ret_config
    return lookup


def _build_reranker_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build reranker name -> config dict lookup from the YAML."""
    rag = config.get("rag", {})
    reranker_config = rag.get("reranker", {})
    if isinstance(reranker_config, dict):
        reranker_cfgs = reranker_config.get("configs", [])
    else:
        reranker_cfgs = []

    lookup: dict[str, dict[str, Any]] = {}
    for cfg in reranker_cfgs:
        name = cfg.get("name", "none") if cfg.get("enabled") else "none"
        lookup[name] = cfg
    return lookup


# ---------------------------------------------------------------------------
# Trial -> ExperimentSpec conversion
# ---------------------------------------------------------------------------


def _suggest_agent_params(
    trial: optuna.Trial,
    agent_type: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Conditionally suggest agent-specific parameters based on *agent_type*.

    Reads ``rag.sampling.agent_params.<agent_type>`` from the config to
    discover which extra dimensions to add for a given agent strategy.

    Returns:
        Dict of agent param name → suggested value (empty for ``fixed_rag``).
    """
    if agent_type == "fixed_rag":
        return {}

    agent_params_config = (
        config.get("rag", {}).get("sampling", {}).get("agent_params", {}).get(agent_type, {})
    )

    params: dict[str, Any] = {}
    for param_name, choices in agent_params_config.items():
        if isinstance(choices, list) and len(choices) > 0:
            params[param_name] = trial.suggest_categorical(param_name, choices)
        else:
            # Scalar value → fixed (no suggestion needed)
            params[param_name] = choices

    return params


def _suggest_hybrid_params(
    trial: optuna.Trial,
    retriever_name: str,
    retriever_lookup: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[int | None, float | None]:
    """Conditionally suggest rrf_k and alpha when retriever is hybrid.

    These parameters only matter for hybrid retrievers (dense + sparse
    fusion). For non-hybrid retrievers, returns (None, None) without
    adding any dimensions to the Optuna trial.

    Returns:
        (rrf_k, alpha) — both None if retriever is not hybrid.
    """
    ret_cfg = retriever_lookup.get(retriever_name, {})
    if ret_cfg.get("type") != "hybrid":
        return None, None

    rag = config.get("rag", {})

    rrf_k = None
    rrf_k_values = rag.get("rrf_k_values")
    if rrf_k_values:
        rrf_k = trial.suggest_categorical("rrf_k", rrf_k_values)

    alpha = None
    alpha_values = rag.get("alpha_values")
    if alpha_values:
        alpha = trial.suggest_categorical("alpha", alpha_values)

    return rrf_k, alpha


def _trial_to_spec(
    trial: optuna.Trial,
    search_space: dict[str, list[Any]],
    retriever_lookup: dict[str, dict[str, Any]],
    reranker_lookup: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> ExperimentSpec:
    """Convert an Optuna trial's suggested parameters into an ExperimentSpec.

    Uses the same logic as ``_build_rag_specs`` for fetch_k computation,
    naming, and retriever/reranker resolution so that Optuna-generated
    experiments are fully compatible with the rest of the framework.

    When the search space includes an ``agent_type`` dimension, the trial
    will additionally suggest agent-specific parameters (e.g.
    ``max_iterations`` for ``iterative_rag``) and encode them in both the
    experiment name and the ``ExperimentSpec``.
    """
    rag = config.get("rag", {})

    # --- Suggest one value per dimension ---
    model = trial.suggest_categorical("model", search_space["model"])
    retriever = trial.suggest_categorical("retriever", search_space["retriever"])
    top_k = trial.suggest_categorical("top_k", search_space["top_k"])
    prompt = trial.suggest_categorical("prompt", search_space["prompt"])
    qt = trial.suggest_categorical("query_transform", search_space["query_transform"])
    dataset = trial.suggest_categorical("dataset", search_space["dataset"])
    reranker = trial.suggest_categorical("reranker", search_space["reranker"])

    # --- Hybrid-only params: rrf_k + alpha (conditional dimensions) ---
    rrf_k, alpha = _suggest_hybrid_params(trial, retriever, retriever_lookup, config)

    # --- Agent type (conditional dimension) ---
    if "agent_type" in search_space:
        agent_type = trial.suggest_categorical("agent_type", search_space["agent_type"])
    else:
        # Single agent type from config, or default to fixed_rag
        sampling_cfg = rag.get("sampling", {})
        agent_types = sampling_cfg.get("agent_types", ["fixed_rag"])
        agent_type = agent_types[0] if agent_types else "fixed_rag"

    agent_params = _suggest_agent_params(trial, agent_type, config)

    # --- Resolve retriever config ---
    ret_cfg = retriever_lookup.get(retriever, {})
    embedding_index = ret_cfg.get("embedding_index")
    sparse_index = ret_cfg.get("sparse_index")

    # --- Resolve reranker config ---
    rr_cfg = reranker_lookup.get(reranker, {})
    has_reranker = reranker != "none" and rr_cfg.get("enabled", False)
    rr_model = rr_cfg.get("model") if has_reranker else None
    rr_name = reranker if has_reranker else None

    # --- Compute fetch_k (same logic as builder.py) ---
    fetch_k_config = rag.get("fetch_k")
    fetch_k_multiplier = rag.get("fetch_k_multiplier", 4)

    if fetch_k_config is not None:
        fetch_k = fetch_k_config
    elif has_reranker:
        fetch_k = top_k * fetch_k_multiplier
    else:
        fetch_k = None

    # --- Build canonical name ---
    name = name_rag(
        model,
        prompt,
        dataset,
        retriever,
        top_k,
        qt,
        reranker if has_reranker else "none",
        rrf_k=rrf_k,
        alpha=alpha,
        agent_type=agent_type,
        agent_params=agent_params or None,
    )

    batch_size = config.get("batch_size", 8)
    metrics = config.get("metrics", ["f1", "exact_match"])

    # Spec agent_type: None means default (fixed_rag) — keep compat
    spec_agent_type = agent_type if agent_type != "fixed_rag" else None

    return ExperimentSpec(
        name=name,
        exp_type="rag",
        model=model,
        dataset=dataset,
        prompt=prompt,
        retriever=retriever,
        embedding_index=embedding_index,
        sparse_index=sparse_index,
        top_k=top_k,
        fetch_k=fetch_k,
        query_transform=qt if qt != "none" else None,
        reranker=rr_name,
        reranker_model=rr_model,
        rrf_k=rrf_k,
        alpha=alpha,
        batch_size=batch_size,
        metrics=metrics,
        agent_type=spec_agent_type,
        agent_params=tuple(agent_params.items()) if agent_params else (),
    )


# ---------------------------------------------------------------------------
# Metric extraction from completed experiments
# ---------------------------------------------------------------------------


def _get_experiment_metric(
    exp_name: str,
    output_dir: Path,
    metric_name: str = "f1",
) -> float | None:
    """Extract a metric value from a completed experiment's output files.

    Tries ``results.json`` first, then ``predictions.json``.

    Returns:
        The metric value, or None if not found.
    """
    exp_dir = output_dir / exp_name

    # Try results.json first (written on COMPLETE phase)
    results_path = exp_dir / "results.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                data = json.load(f)
            metrics = data.get("metrics", {})
            if metric_name in metrics:
                return float(metrics[metric_name])
        except Exception as e:
            logger.warning("Failed to read results.json for %s: %s", exp_name, e)

    # Fallback: predictions.json (aggregate_metrics section)
    predictions_path = exp_dir / "predictions.json"
    if predictions_path.exists():
        try:
            with open(predictions_path) as f:
                data = json.load(f)
            metrics = data.get("aggregate_metrics", {})
            if metric_name in metrics:
                return float(metrics[metric_name])
        except Exception as e:
            logger.warning("Failed to read predictions.json for %s: %s", exp_name, e)

    return None


# ---------------------------------------------------------------------------
# Sampler factory
# ---------------------------------------------------------------------------


def _create_sampler(mode: str, seed: int | None = None) -> optuna.samplers.BaseSampler:
    """Map a sampling mode name to an Optuna sampler instance.

    Args:
        mode: ``"random"``, ``"tpe"``, or any legacy alias.
        seed: Optional random seed for reproducibility.
    """
    if mode == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    # "tpe", "optuna", "stratified", or anything else → TPE
    return optuna.samplers.TPESampler(seed=seed)


# ---------------------------------------------------------------------------
# Warm-start from existing experiments on disk
# ---------------------------------------------------------------------------


def _seed_from_existing(
    study: optuna.Study,
    search_space: dict[str, list[Any]],
    output_dir: Path,
    optimize_metric: str,
    config: dict[str, Any],
) -> int:
    """Register completed experiments on disk as Optuna trials (warm-start).

    Scans ``output_dir`` for completed RAG experiments that are NOT already
    in the Optuna study, reads their parameters from ``metadata.json`` and
    their metric from ``results.json`` / ``predictions.json``, and adds
    them as completed trials.

    This lets TPE learn from prior runs immediately instead of starting
    blind, and lets RandomSampler skip already-explored combinations.

    Only experiments whose parameters fall within the current search space
    are registered (experiments from a different study config are skipped).
    For experiments with an ``agent_type``, the corresponding conditional
    agent parameters are also read from metadata and validated.

    Returns:
        Number of trials seeded.
    """
    from optuna.distributions import CategoricalDistribution

    # Read agent_params config so we know valid choices for conditional dims
    agent_params_config = config.get("rag", {}).get("sampling", {}).get("agent_params", {})
    rag = config.get("rag", {})
    retriever_lookup = _build_retriever_lookup(config)

    # Build a set of param fingerprints already in the study to avoid dupes
    existing_params: set[tuple] = set()
    for trial in study.trials:
        key = tuple(sorted(trial.params.items()))
        existing_params.add(key)

    seeded = 0

    for exp_dir in sorted(output_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        metadata_path = exp_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            with open(metadata_path) as f:
                meta = json.load(f)
        except Exception:
            continue

        # Only seed RAG experiments
        if meta.get("type") != "rag":
            continue

        # Map metadata fields to Optuna param names
        params: dict[str, Any] = {
            "model": meta.get("model"),
            "retriever": meta.get("retriever"),
            "top_k": meta.get("top_k"),
            "prompt": meta.get("prompt"),
            "query_transform": meta.get("query_transform") or "none",
            "dataset": meta.get("dataset"),
            "reranker": meta.get("reranker") or "none",
        }

        # Handle conditional hybrid params (rrf_k, alpha)
        ret_cfg = retriever_lookup.get(params.get("retriever", ""), {})
        is_hybrid = ret_cfg.get("type") == "hybrid"

        if is_hybrid:
            # rrf_k: include if present in metadata and config has rrf_k_values
            rrf_k_values = rag.get("rrf_k_values")
            meta_rrf_k = meta.get("rrf_k")
            if rrf_k_values and meta_rrf_k is not None:
                if meta_rrf_k in rrf_k_values:
                    params["rrf_k"] = meta_rrf_k
                else:
                    continue  # rrf_k value not in configured choices

            # alpha: include if present in metadata and config has alpha_values
            alpha_values = rag.get("alpha_values")
            meta_alpha = meta.get("alpha")
            if alpha_values and meta_alpha is not None:
                if meta_alpha in alpha_values:
                    params["alpha"] = meta_alpha
                else:
                    continue  # alpha value not in configured choices

        # Handle agent_type dimension
        meta_agent_type = meta.get("agent_type") or "fixed_rag"
        if "agent_type" in search_space:
            if meta_agent_type not in search_space["agent_type"]:
                continue
            params["agent_type"] = meta_agent_type

        # Handle conditional agent params from metadata
        meta_agent_params = meta.get("agent_params", {})
        agent_cfg = agent_params_config.get(meta_agent_type, {})
        agent_params_valid = True
        for param_name, valid_choices in agent_cfg.items():
            if isinstance(valid_choices, list) and len(valid_choices) > 0:
                value = meta_agent_params.get(param_name)
                if value is not None and value in valid_choices:
                    params[param_name] = value
                elif meta_agent_type != "fixed_rag":
                    # Non-default agent missing a required param → skip
                    agent_params_valid = False
                    break

        if not agent_params_valid:
            continue

        # Validate every base param is within the current search space
        valid = True
        for dim, value in params.items():
            if dim in search_space and value not in search_space[dim]:
                valid = False
                break

        if not valid:
            continue

        # Skip if already registered
        key = tuple(sorted(params.items()))
        if key in existing_params:
            continue

        # Extract metric value
        metric_value = _get_experiment_metric(exp_dir.name, output_dir, optimize_metric)
        if metric_value is None:
            continue

        # Build distributions matching this trial's params
        distributions: dict[str, CategoricalDistribution] = {}
        for dim, value in params.items():
            if dim in search_space:
                distributions[dim] = CategoricalDistribution(choices=search_space[dim])
            elif dim == "rrf_k":
                # Conditional hybrid param
                rrf_choices = rag.get("rrf_k_values", [value])
                distributions[dim] = CategoricalDistribution(choices=rrf_choices)
            elif dim == "alpha":
                # Conditional hybrid param
                alpha_choices = rag.get("alpha_values", [value])
                distributions[dim] = CategoricalDistribution(choices=alpha_choices)
            else:
                # Conditional agent param — distribution from agent_params config
                choices = agent_cfg.get(dim, [value])
                if not isinstance(choices, list):
                    choices = [choices]
                distributions[dim] = CategoricalDistribution(choices=choices)

        # Create and register the trial
        _now = datetime.now()
        trial = optuna.trial.FrozenTrial(
            number=0,  # reassigned by storage
            state=optuna.trial.TrialState.COMPLETE,
            value=metric_value,
            datetime_start=_now,
            datetime_complete=_now,
            params=params,
            distributions=distributions,
            user_attrs={"seeded_from": exp_dir.name},
            system_attrs={},
            intermediate_values={},
            trial_id=0,  # reassigned by storage
        )
        study.add_trial(trial)
        existing_params.add(key)
        seeded += 1

    return seeded


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_optuna_study(
    config: dict[str, Any],
    n_trials: int,
    output_dir: Path,
    sampler_mode: str = "tpe",
    optimize_metric: str = "f1",
    limit: int | None = None,
    judge_model: Any = None,
    llm_judge_config: dict[str, Any] | None = None,
    study_name: str | None = None,
    seed: int | None = None,
) -> optuna.Study:
    """Run an Optuna-driven experiment search over the RAG search space.

    Works with any supported sampler mode:

    - ``"random"``: Uniform random exploration (``RandomSampler``).
    - ``"tpe"``: Bayesian optimization via TPE (``TPESampler``).

    The study is persisted to ``{output_dir}/optuna_study.db`` (SQLite),
    so it can be resumed transparently by re-running the same command.

    Args:
        config: Full study YAML config dict.
        n_trials: Total number of trials to run.
        output_dir: Directory for experiment outputs.
        sampler_mode: Sampler strategy -- ``"random"`` or ``"tpe"``.
        optimize_metric: Metric to maximize (default: ``"f1"``).
        limit: Max questions per experiment (overrides config).
        judge_model: LLM judge model instance.
        llm_judge_config: LLM judge config dict.
        study_name: Optuna study name (defaults to ``{config.name}_{mode}``).
        seed: Random seed for the sampler.

    Returns:
        The :class:`optuna.Study` object containing all trial results.
    """
    config_name = config.get("name", "ragicamp")
    if study_name is None:
        study_name = f"{config_name}_{sampler_mode}"

    # Persist to SQLite for resume capability
    storage_path = output_dir / "optuna_study.db"
    storage = f"sqlite:///{storage_path}"

    sampler = _create_sampler(sampler_mode, seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    # Extract search space and config lookups
    search_space = _extract_search_space(config)
    retriever_lookup = _build_retriever_lookup(config)
    reranker_lookup = _build_reranker_lookup(config)
    metrics = config.get("metrics", ["f1", "exact_match"])

    # --- Stratified search: fixed dimensions get uniform coverage ---
    # By default, dataset and model are "benchmark axes" that should be
    # explored uniformly rather than optimized (TPE would bias towards
    # easy datasets and strong models, skewing the benchmark).
    sampling_cfg = config.get("rag", {}).get("sampling", {})
    fixed_dims: list[str] = sampling_cfg.get("fixed_dims", ["dataset", "model"])
    # Keep only dims that exist in the search space with >1 value
    fixed_dims = [d for d in fixed_dims if d in search_space and len(search_space[d]) > 1]

    if fixed_dims:
        fixed_combos = list(itertools_product(*(search_space[d] for d in fixed_dims)))
    else:
        fixed_combos = []

    # --- Warm-start: seed from existing experiments on disk ---
    seeded = _seed_from_existing(study, search_space, output_dir, optimize_metric, config)
    if seeded > 0:
        logger.info("Seeded %d existing experiments into Optuna study", seeded)

    # --- Log search space summary ---
    total_combos = 1
    for values in search_space.values():
        total_combos *= len(values)

    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed_trials)

    mode_label = "TPE Optimization" if sampler_mode != "random" else "Random Search"
    space_lines = "".join(
        f"\n    {dim}: {len(values)} values" for dim, values in search_space.items()
    )
    summary = (
        f"\n{'=' * 70}\n"
        f"{mode_label}: {study_name}\n"
        f"  Sampler: {type(sampler).__name__}\n"
        f"  Metric: {optimize_metric} (maximize)\n"
        f"  Trials: {n_trials} (target)\n"
        f"  Storage: {storage_path}\n"
        f"  Search space:{space_lines}\n"
        f"  Total possible combinations: {total_combos:,}"
    )
    # Note conditional hybrid dimensions
    rag_cfg = config.get("rag", {})
    cond_dims = []
    if rag_cfg.get("rrf_k_values"):
        cond_dims.append(f"rrf_k: {len(rag_cfg['rrf_k_values'])} values")
    if rag_cfg.get("alpha_values"):
        cond_dims.append(f"alpha: {len(rag_cfg['alpha_values'])} values")
    if cond_dims:
        summary += f"\n  Conditional (hybrid-only): {', '.join(cond_dims)}"
    if completed_trials > 0:
        summary += (
            f"\n  Completed trials: {completed_trials} ({seeded} seeded from disk)"
            f"\n  New trials to run: {remaining}"
        )
    if fixed_dims and fixed_combos:
        n_combos = len(fixed_combos)
        trials_per = remaining // n_combos if remaining > 0 and n_combos > 0 else 0
        leftover = remaining % n_combos if remaining > 0 and n_combos > 0 else 0
        summary += (
            f"\n  Stratified dims: {', '.join(fixed_dims)}"
            f"\n  Fixed combos: {n_combos}"
            f" ({' x '.join(f'{d}({len(search_space[d])})' for d in fixed_dims)})"
            f"\n  Trials per combo: ~{trials_per}"
            f"{f' (+1 for {leftover} combos)' if leftover else ''}"
        )
    # --- Trial timeout (seconds) ---
    trial_timeout: int = sampling_cfg.get("trial_timeout", 1200)

    # --- Context feasibility constraints ---
    constraints = sampling_cfg.get("constraints", {})
    summary += f"\n  Trial timeout: {trial_timeout}s ({trial_timeout // 60}min)"
    if constraints:
        n_constrained = len(constraints.get("model_context_lengths", {}))
        summary += f"\n  Context constraints: {n_constrained} model(s) with context limits"
    summary += f"\n{'=' * 70}"
    logger.info(summary)

    if remaining == 0:
        logger.info("All trials already completed. Use a higher --sample value to run more.")
        _print_study_summary(study, optimize_metric, n_trials)
        return study

    # --- Define objective function ---
    def objective(trial: optuna.Trial) -> float:
        """Run one experiment and return the target metric."""
        spec = _trial_to_spec(
            trial,
            search_space,
            retriever_lookup,
            reranker_lookup,
            config,
        )

        trial_num = trial.number + 1
        model_short = spec.model.split("/")[-1] if "/" in spec.model else spec.model
        agent_info = ""
        if spec.agent_type:
            ap = spec.get_agent_params_dict()
            agent_info = f", agent={spec.agent_type}"
            if ap:
                agent_info += f"({', '.join(f'{k}={v}' for k, v in ap.items())})"
        logger.info(
            "\n[Trial %d/%d] %s\n"
            "  model=%s, retriever=%s, "
            "top_k=%s, prompt=%s, "
            "qt=%s, rr=%s, "
            "dataset=%s%s",
            trial_num,
            n_trials,
            spec.name,
            model_short,
            spec.retriever,
            spec.top_k,
            spec.prompt,
            spec.query_transform or "none",
            spec.reranker or "none",
            spec.dataset,
            agent_info,
        )

        # --- Feasibility check: prune if prompt would exceed context ---
        feasible, reason = _check_context_feasibility(
            model=spec.model,
            retriever_name=spec.retriever,
            top_k=spec.top_k,
            prompt=spec.prompt,
            retriever_lookup=retriever_lookup,
            constraints=constraints,
        )
        if not feasible:
            logger.info("  -> pruned: %s", reason)
            raise optuna.TrialPruned(reason)

        # --- Duplicate check: prune if experiment already exists on disk ---
        # When a config was already seeded (or ran earlier in this session),
        # run_spec returns "complete" instantly.  Recording it as a new
        # COMPLETE trial inflates completed_trials and wastes trial slots
        # on subsequent resumes.  Prune instead — TPE already has the
        # metric from the seeded trial.
        exp_out = output_dir / spec.name
        if exp_out.exists():
            existing_metric = _get_experiment_metric(
                spec.name,
                output_dir,
                optimize_metric,
            )
            if existing_metric is not None:
                logger.info(
                    "  -> already complete on disk (f1=%.4f), pruning duplicate", existing_metric
                )
                raise optuna.TrialPruned(f"experiment already complete: {spec.name}")

        # Run the experiment
        status = run_spec(
            spec=spec,
            limit=limit,
            metrics=metrics,
            out=output_dir,
            judge_model=judge_model,
            llm_judge_config=llm_judge_config,
            force=False,
            use_subprocess=True,
            timeout=trial_timeout,
        )

        # Extract metric from results
        if status in ("complete", "ran", "resumed"):
            value = _get_experiment_metric(spec.name, output_dir, optimize_metric)
            if value is not None:
                logger.info("  -> %s = %.4f", optimize_metric, value)
                return value

        # Failed/timeout experiments should be pruned so Optuna knows the
        # trial was invalid, rather than treating 0.0 as a legitimate score.
        logger.info("  -> experiment %s, pruning trial", status)
        raise optuna.TrialPruned(f"Experiment {status}")

    # --- Run optimization ---
    if fixed_dims and fixed_combos:
        # Stratified search: round-robin over fixed-dimension combos so
        # every (dataset, model) pair gets equal trial budget.  Pairs are
        # shuffled each epoch so TPE doesn't see a predictable order
        # (which would bias its surrogate model towards the first pairs).
        rng = random.Random(seed)
        schedule: list[dict[str, Any]] = []
        while len(schedule) < remaining:
            epoch = [dict(zip(fixed_dims, combo, strict=True)) for combo in fixed_combos]
            rng.shuffle(epoch)
            schedule.extend(epoch)
        schedule = schedule[:remaining]

        for i, fixed_params in enumerate(schedule):
            dims_str = ", ".join(f"{k}={v}" for k, v in fixed_params.items())
            logger.debug(
                "Stratified trial %d/%d — fixed: %s",
                i + 1,
                remaining,
                dims_str,
            )
            study.sampler = PartialFixedSampler(
                fixed_params=fixed_params,
                base_sampler=sampler,
            )
            study.optimize(objective, n_trials=1, show_progress_bar=False)
    else:
        # No fixed dimensions — standard optimization over all dims
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    # --- Print summary ---
    _print_study_summary(study, optimize_metric, n_trials)

    return study


def _print_study_summary(
    study: optuna.Study,
    optimize_metric: str,
    n_trials: int,
) -> None:
    """Print a summary of the Optuna study results."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    lines = [
        f"\n{'=' * 70}",
        f"Optuna Study Complete: {study.study_name}",
        f"  Total trials: {len(study.trials)}",
    ]

    if not completed:
        lines.append("  No completed trials.")
        lines.append(f"{'=' * 70}")
        logger.info("\n".join(lines))
        return

    lines.append(f"  Best {optimize_metric}: {study.best_value:.4f}")
    lines.append("  Best params:")
    for key, value in study.best_params.items():
        display = value
        if key == "model" and isinstance(value, str) and "/" in value:
            display = value.split("/")[-1]
        lines.append(f"    {key}: {display}")

    # Top 5 trials
    trials_sorted = sorted(completed, key=lambda t: t.value or 0.0, reverse=True)
    lines.append("\n  Top 5 trials:")
    for t in trials_sorted[:5]:
        model_short = t.params.get("model", "?").split("/")[-1]
        agent = t.params.get("agent_type", "fixed_rag")
        agent_str = f", agent={agent}" if agent != "fixed_rag" else ""
        lines.append(
            f"    #{t.number}: {optimize_metric}={t.value:.4f} "
            f"({model_short}, {t.params.get('retriever', '?')}, "
            f"top_k={t.params.get('top_k', '?')}, "
            f"prompt={t.params.get('prompt', '?')}, "
            f"{t.params.get('dataset', '?')}{agent_str})"
        )

    lines.append(f"{'=' * 70}")
    logger.info("\n".join(lines))
