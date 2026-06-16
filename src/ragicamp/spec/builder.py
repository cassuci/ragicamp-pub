"""Experiment specification builder.

Builds ExperimentSpec objects from YAML configuration files.
This is the main entry point for generating the experiment matrix.

Supports three modes:
- Grid search: Full Cartesian product of all dimensions
- Random search: Sample N random combinations from the grid
- Singleton: Explicit experiment definitions for hypothesis-driven research
"""

import random
from itertools import product as cartesian
from typing import Any

from ragicamp.core.logging import get_logger
from ragicamp.spec.experiment import ExperimentSpec
from ragicamp.spec.naming import name_direct, name_rag

logger = get_logger(__name__)


def build_specs(
    config: dict[str, Any],
    sampling_override: dict[str, Any] | None = None,
    exclude_names: set[str] | None = None,
) -> list[ExperimentSpec]:
    """Build experiment specs from YAML config.

    Generates experiment matrix with support for three modes:

    1. Grid Search (default): Full Cartesian product of all dimensions
       - Models x Datasets x Prompts x Quantizations (for direct)
       - Models x Retrievers x TopK x QueryTransforms x Rerankers x Prompts x Datasets (for RAG)

    2. Random Search: Sample N random combinations from the RAG grid
       - Configure via rag.sampling in YAML or --sample CLI flag
       - Supports 'random' mode (uniform) or 'stratified' mode (ensure coverage)
       - Only affects RAG experiments, not baselines or singletons

    3. Singleton: Explicit experiment definitions via 'experiments' list
       - Always included (not affected by sampling)
       - Use for hypothesis-driven research or agent-based strategies

    Baseline Behavior:
        Direct LLM experiments (no retrieval) are ALWAYS included as baselines.
        They are never subject to sampling, ensuring proper comparison.

    Args:
        config: Study configuration dict (loaded from YAML)
        sampling_override: Optional dict to override sampling config
            Example: {"mode": "random", "n_experiments": 50, "seed": 42}
        exclude_names: Optional set of experiment names to exclude from sampling
            Used to avoid resampling already-completed experiments when resuming

    Returns:
        List of ExperimentSpec objects for all experiments

    Example:
        >>> with open("conf/study/my_study.yaml") as f:
        ...     config = yaml.safe_load(f)
        >>> specs = build_specs(config)
        >>> print(f"Generated {len(specs)} experiments")

        # With random sampling (only affects RAG, baselines always included):
        >>> specs = build_specs(config, sampling_override={"mode": "random", "n_experiments": 50})
    """
    specs: list[ExperimentSpec] = []

    # Warn about YAML keys that are not used by the spec builder
    _KNOWN_KEYS = {
        "name",
        "datasets",
        "batch_size",
        "metrics",
        "direct",
        "rag",
        "experiments",
        "models",
        "description",
        "version",
    }
    unknown_keys = set(config.keys()) - _KNOWN_KEYS
    if unknown_keys:
        import warnings

        warnings.warn(
            f"Unrecognized top-level config keys: {unknown_keys}. "
            f"These have no effect. Known keys: {sorted(_KNOWN_KEYS)}",
            UserWarning,
            stacklevel=2,
        )

    # Global settings
    datasets = config.get("datasets", ["nq"])
    batch_size = config.get("batch_size", 8)
    metrics = config.get("metrics", ["f1", "exact_match"])

    # Build direct experiment specs (baselines - never sampled)
    # Direct LLM baselines are always included for proper comparison
    direct = config.get("direct", {})
    if direct.get("enabled"):
        direct_specs = _build_direct_specs(direct, datasets, batch_size, metrics)
        # Note: Direct baselines are NOT sampled - they're essential for comparison
        specs.extend(direct_specs)

    # Build RAG experiment specs (grid search, optionally sampled)
    rag = config.get("rag", {})
    if rag.get("enabled"):
        rag_specs = _build_rag_specs(rag, datasets, batch_size, metrics)

        # Apply sampling if configured
        sampling_config = sampling_override or rag.get("sampling")
        if sampling_config:
            rag_specs = _apply_sampling(
                rag_specs, sampling_config, "rag", exclude_names=exclude_names
            )

        specs.extend(rag_specs)

    # Build singleton experiments (hypothesis-driven) - never sampled
    experiments = config.get("experiments", [])
    if experiments:
        specs.extend(_build_singleton_specs(experiments, config, batch_size, metrics))

    return specs


def _apply_sampling(
    specs: list[ExperimentSpec],
    sampling_config: dict[str, Any],
    spec_type: str,
    exclude_names: set[str] | None = None,
) -> list[ExperimentSpec]:
    """Apply sampling to reduce experiment count.

    Args:
        specs: Full list of experiment specs from grid search
        sampling_config: Sampling configuration dict with:
            - mode: 'random' (uniform) or 'stratified' (ensure coverage)
            - n_experiments: Number of experiments to sample
            - seed: Random seed for reproducibility
            - stratify_by: List of dimensions to stratify (for stratified mode)
        spec_type: Type of specs ('direct' or 'rag') for logging
        exclude_names: Optional set of experiment names to exclude from sampling
            (already-completed experiments that shouldn't be resampled)

    Returns:
        Sampled list of experiment specs
    """
    mode = sampling_config.get("mode", "random")
    n_experiments = sampling_config.get("n_experiments", len(specs))
    seed = sampling_config.get("seed")

    # Filter out already-completed experiments before sampling
    if exclude_names:
        original_count = len(specs)
        specs = [s for s in specs if s.name not in exclude_names]
        excluded_count = original_count - len(specs)
        if excluded_count > 0:
            logger.info(
                "[%s] Excluding %d already-completed experiments from sampling pool",
                spec_type,
                excluded_count,
            )

    if n_experiments >= len(specs):
        return specs

    rng = random.Random(seed)

    if mode == "random":
        # Simple random sampling
        sampled = rng.sample(specs, n_experiments)
        logger.info("[%s] Random sampling: %d/%d experiments", spec_type, len(sampled), len(specs))
        return sampled

    elif mode == "stratified":
        # Stratified sampling: ensure at least one experiment per stratum
        stratify_by = sampling_config.get("stratify_by", ["model", "retriever"])
        sampled = _stratified_sample(specs, n_experiments, stratify_by, rng=rng)
        logger.info(
            "[%s] Stratified sampling by %s: %d/%d experiments",
            spec_type,
            stratify_by,
            len(sampled),
            len(specs),
        )
        return sampled

    else:
        logger.warning("Unknown sampling mode '%s', using all experiments", mode)
        return specs


def _stratified_sample(
    specs: list[ExperimentSpec],
    n_experiments: int,
    stratify_by: list[str],
    rng: random.Random | None = None,
) -> list[ExperimentSpec]:
    """Stratified sampling ensuring coverage across specified dimensions.

    Algorithm:
    1. Group specs by unique combinations of stratify_by dimensions
    2. Sample at least 1 from each group (if budget allows)
    3. Fill remaining budget with random samples from all specs

    Args:
        specs: Full list of experiment specs
        n_experiments: Target number of experiments
        stratify_by: Dimensions to stratify by (e.g., ['model', 'retriever'])
        rng: Seeded random generator for reproducibility

    Returns:
        Stratified sample of specs
    """
    from collections import defaultdict

    if rng is None:
        rng = random.Random()

    # Group specs by stratification key
    groups: dict[tuple, list[ExperimentSpec]] = defaultdict(list)
    for spec in specs:
        key = tuple(getattr(spec, dim, None) for dim in stratify_by)
        groups[key].append(spec)

    sampled: list[ExperimentSpec] = []
    sampled_set: set[str] = set()

    # Phase 1: Sample at least 1 from each group
    for group_specs in groups.values():
        if len(sampled) >= n_experiments:
            break
        choice = rng.choice(group_specs)
        if choice.name not in sampled_set:
            sampled.append(choice)
            sampled_set.add(choice.name)

    # Phase 2: Fill remaining budget with random samples
    remaining_budget = n_experiments - len(sampled)
    if remaining_budget > 0:
        remaining_specs = [s for s in specs if s.name not in sampled_set]
        if remaining_specs:
            additional = rng.sample(remaining_specs, min(remaining_budget, len(remaining_specs)))
            sampled.extend(additional)

    return sampled


def _build_direct_specs(
    direct_config: dict[str, Any],
    datasets: list[str],
    batch_size: int,
    metrics: list[str],
) -> list[ExperimentSpec]:
    """Build specs for direct (non-RAG) experiments."""
    specs: list[ExperimentSpec] = []

    models = direct_config.get("models", [])
    prompts = direct_config.get("prompts", ["default"])

    for model in models:
        for prompt in prompts:
            for dataset in datasets:
                name = name_direct(model, prompt, dataset)
                specs.append(
                    ExperimentSpec(
                        name=name,
                        exp_type="direct",
                        model=model,
                        dataset=dataset,
                        prompt=prompt,
                        batch_size=batch_size,
                        metrics=metrics,
                    )
                )

    return specs


def _build_rag_specs(
    rag_config: dict[str, Any],
    datasets: list[str],
    batch_size: int,
    metrics: list[str],
) -> list[ExperimentSpec]:
    """Build specs for RAG experiments.

    Supports two ways to specify retrievers for grid search:
    1. retriever_names: List of retriever names to use (references full configs in 'retrievers')
    2. retrievers: List of retriever configs (legacy, uses 'name' field from each config)

    The 'retrievers' list is always used for index building, but 'retriever_names' takes
    precedence for grid search dimension when specified.
    """
    specs: list[ExperimentSpec] = []

    models = rag_config.get("models", [])
    top_k_values = rag_config.get("top_k_values", [5])
    prompts = rag_config.get("prompts", ["default"])

    # Build retriever config lookup
    retrievers = rag_config.get("retrievers", [])
    retriever_configs: dict[str, dict[str, Any]] = {}
    for ret_config in retrievers:
        if isinstance(ret_config, dict):
            retriever_configs[ret_config["name"]] = ret_config
        elif isinstance(ret_config, str):
            # String-format retriever: use name directly (no extra config)
            retriever_configs[ret_config] = {"name": ret_config}

    # Get retriever names for grid search
    # Priority: retriever_names > extracting names from retrievers list
    retriever_names = rag_config.get("retriever_names")
    if retriever_names is None:
        retriever_names = list(retriever_configs.keys())

    # If no retrievers specified, return empty (indexes only, no experiments)
    if not retriever_names or not models:
        return specs

    # Fetch-K configuration: docs to retrieve before reranking
    # Can be explicit value or multiplier of top_k
    fetch_k_config = rag_config.get("fetch_k")
    fetch_k_multiplier = rag_config.get("fetch_k_multiplier", 4)  # Default: 4x top_k

    # Query transform options
    query_transforms = rag_config.get("query_transform", ["none"])
    if not query_transforms:
        query_transforms = ["none"]

    # RRF-K values for hybrid retrieval tuning
    rrf_k_values = rag_config.get("rrf_k_values")

    # Reranker configs - support both old and new format
    reranker_config = rag_config.get("reranker", {})
    if isinstance(reranker_config, dict):
        if "configs" in reranker_config:
            # New format: explicit list of reranker configs
            reranker_cfgs = reranker_config["configs"]
        elif reranker_config.get("enabled"):
            # Legacy format: simple {enabled: true, model: bge}
            reranker_cfgs = [{"enabled": True, "name": reranker_config.get("model", "bge")}]
        else:
            # No reranker configured
            reranker_cfgs = [{"enabled": False, "name": "none"}]
    else:
        reranker_cfgs = [{"enabled": False, "name": "none"}]

    if not reranker_cfgs:
        reranker_cfgs = [{"enabled": False, "name": "none"}]

    for model, ret_name, top_k, prompt, qt, rr_cfg, dataset in cartesian(
        models,
        retriever_names,
        top_k_values,
        prompts,
        query_transforms,
        reranker_cfgs,
        datasets,
    ):
        ret_cfg = retriever_configs.get(ret_name, {})
        embedding_index = ret_cfg.get("embedding_index")
        sparse_index = ret_cfg.get("sparse_index")

        rr_name = rr_cfg.get("name", "none") if rr_cfg.get("enabled") else "none"
        rr_model = rr_cfg.get("model") if rr_cfg.get("enabled") else None

        # Compute fetch_k: explicit > multiplier (if reranking) > None
        has_reranker = rr_name != "none" and rr_cfg.get("enabled")
        if fetch_k_config is not None:
            fetch_k = fetch_k_config
        elif has_reranker:
            fetch_k = top_k * fetch_k_multiplier
        else:
            fetch_k = None

        # RRF-K values — only relevant for hybrid retrievers
        is_hybrid = ret_cfg.get("type") == "hybrid"
        rrf_k_list = rrf_k_values if (is_hybrid and rrf_k_values) else [None]

        for rrf_k in rrf_k_list:
            name = name_rag(
                model,
                prompt,
                dataset,
                ret_name,
                top_k,
                qt,
                rr_name,
                rrf_k=rrf_k,
            )

            specs.append(
                ExperimentSpec(
                    name=name,
                    exp_type="rag",
                    model=model,
                    dataset=dataset,
                    prompt=prompt,
                    retriever=ret_name,
                    embedding_index=embedding_index,
                    sparse_index=sparse_index,
                    top_k=top_k,
                    fetch_k=fetch_k,
                    query_transform=qt if qt != "none" else None,
                    reranker=rr_name if rr_name != "none" else None,
                    reranker_model=rr_model,
                    rrf_k=rrf_k,
                    batch_size=batch_size,
                    metrics=metrics,
                )
            )

    return specs


def _build_singleton_specs(
    experiments: list[dict[str, Any]],
    config: dict[str, Any],
    batch_size: int,
    metrics: list[str],
) -> list[ExperimentSpec]:
    """Build specs from explicit singleton experiment definitions.

    This enables hypothesis-driven research where each experiment is
    explicitly defined rather than generated from a grid search.

    If an experiment doesn't specify a dataset, it will be expanded
    across all datasets in the config (like grid search).

    Args:
        experiments: List of experiment definition dicts
        config: Full study config (for defaults)
        batch_size: Default batch size
        metrics: Default metrics list

    Returns:
        List of ExperimentSpec objects
    """
    specs: list[ExperimentSpec] = []

    # Get defaults from config
    # Handle models as list (from YAML anchor) or dict (with "default" key)
    models_config = config.get("models")
    if isinstance(models_config, dict):
        default_model = models_config.get("default")
    elif isinstance(models_config, list) and models_config:
        default_model = models_config[0]  # Use first model as default
    else:
        default_model = None

    default_datasets = config.get("datasets", ["nq"])
    default_prompt = "concise"

    # Build retriever config lookup for embedding_index resolution
    rag_config = config.get("rag", {})
    retrievers = rag_config.get("retrievers", [])
    retriever_configs: dict[str, dict[str, Any]] = {}
    for ret_config in retrievers:
        if isinstance(ret_config, dict):
            retriever_configs[ret_config["name"]] = ret_config

    for exp in experiments:
        # Required field
        base_name = exp["name"]

        # Determine exp_type from agent_type or presence of retriever
        agent_type = exp.get("agent_type")
        has_retriever = exp.get("retriever") is not None

        if agent_type == "direct" or (not has_retriever and agent_type is None):
            exp_type = "direct"
        else:
            exp_type = "rag"

        # Get experiment-specific or default values
        model = exp.get("model", default_model)
        if model is None:
            raise ValueError(f"Experiment '{base_name}' requires a model")

        # If experiment specifies dataset, use only that one
        # Otherwise, expand across all default datasets
        exp_datasets = [exp["dataset"]] if "dataset" in exp else default_datasets

        prompt = exp.get("prompt", default_prompt)

        # RAG-specific fields
        retriever = exp.get("retriever")
        top_k = exp.get("top_k", 5)
        fetch_k = exp.get("fetch_k")
        query_transform = exp.get("query_transform")
        reranker = exp.get("reranker")
        reranker_model = exp.get("reranker_model")
        rrf_k = exp.get("rrf_k")

        # Get embedding_index and sparse_index from retriever config
        ret_cfg = retriever_configs.get(retriever, {}) if retriever else {}
        embedding_index = ret_cfg.get("embedding_index")
        sparse_index = ret_cfg.get("sparse_index")

        # If reranker is specified as short name, map to model
        if reranker and not reranker_model:
            reranker_map = {
                "bge": "bge",
                "ms-marco": "ms-marco",
            }
            reranker_model = reranker_map.get(reranker, reranker)

        # Singleton-specific fields
        hypothesis = exp.get("hypothesis")

        # Extract agent_params from agent-type-specific config blocks
        agent_params = {}
        if agent_type:
            # Check for agent-specific config block (e.g., iterative_rag: {...})
            agent_config_block = exp.get(agent_type, {})
            agent_params.update(agent_config_block)

        # Also check common agent param names at top level
        for param_name in [
            "max_iterations",
            "stop_on_sufficient",
            "retrieval_threshold",
            "verify_answer",
            "fallback_to_direct",
        ]:
            if param_name in exp:
                agent_params[param_name] = exp[param_name]

        # Convert agent_params dict to tuple for frozen dataclass
        agent_params_tuple = tuple(agent_params.items())

        # Override batch settings if specified
        exp_batch_size = exp.get("batch_size", batch_size)
        exp_metrics = exp.get("metrics", metrics)

        # Create one spec per dataset
        for dataset in exp_datasets:
            # Add dataset suffix to name if expanding across multiple datasets
            if len(exp_datasets) > 1:
                name = f"{base_name}_{dataset}"
            else:
                name = base_name

            specs.append(
                ExperimentSpec(
                    name=name,
                    exp_type=exp_type,
                    model=model,
                    dataset=dataset,
                    prompt=prompt,
                    retriever=retriever,
                    embedding_index=embedding_index,
                    sparse_index=sparse_index,
                    top_k=top_k,
                    fetch_k=fetch_k,
                    query_transform=query_transform,
                    reranker=reranker,
                    reranker_model=reranker_model,
                    rrf_k=rrf_k,
                    batch_size=exp_batch_size,
                    metrics=exp_metrics,
                    agent_type=agent_type,
                    hypothesis=hypothesis,
                    agent_params=agent_params_tuple,
                )
            )

    return specs
