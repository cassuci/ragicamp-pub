"""Study runner for CLI.

This module provides the main orchestration for running studies from YAML config files.
The actual implementation is delegated to specialized modules:
- config.validation: Config validation
- indexes.builder: Index building
- execution.runner: Experiment execution
- factory: Component creation
"""

from datetime import datetime
from pathlib import Path
from typing import Any

# Re-export for backward compatibility
from ragicamp.config.validation import (
    validate_config,
    validate_model_spec,
)
from ragicamp.core.constants import Defaults
from ragicamp.core.logging import add_file_handler, get_logger
from ragicamp.execution.runner import run_spec
from ragicamp.indexes.builders import build_embedding_index, build_hierarchical_index
from ragicamp.models import OpenAIModel
from ragicamp.spec.builder import build_specs
from ragicamp.utils.artifacts import get_artifact_manager

_study_logger = get_logger(__name__)


def _log_print(msg: str) -> None:
    """Print to terminal AND log to study.log (strips emoji for the log)."""
    print(msg)
    _study_logger.info(msg.strip())


def _index_exists(index_path: Path) -> bool:
    """Check if index exists, supporting both old and new formats.

    New format: config.json + index.faiss + documents.pkl
    Old format: retriever_config.json + index.faiss OR just index.faiss
    Sharded: config.json + shard_*/ directories
    """
    # New format
    if (index_path / "config.json").exists():
        return True

    # Old format with retriever_config.json
    if (index_path / "retriever_config.json").exists():
        return True

    # Very old format - just the FAISS index
    if (index_path / "index.faiss").exists():
        return True

    # Sharded index
    shard_dirs = list(index_path.glob("shard_*"))
    if shard_dirs and any((s / "index.faiss").exists() for s in shard_dirs):
        return True

    return False


def _sparse_index_exists(sparse_path: Path) -> bool:
    """Check if sparse index exists."""
    return (sparse_path / "config.json").exists()


def _build_sparse_index(
    sparse_name: str,
    embedding_index_path: Path,
    sparse_method: str,
) -> None:
    """Build sparse index from documents in the embedding index.

    Args:
        sparse_name: Name for the sparse index
        embedding_index_path: Path to the dense embedding index (to get documents)
        sparse_method: 'tfidf' or 'bm25'
    """
    from ragicamp.indexes.sparse import SparseIndex, SparseMethod
    from ragicamp.indexes.vector_index import VectorIndex

    _study_logger.info("Building sparse index: %s (method=%s)", sparse_name, sparse_method)

    # Load documents from the dense index
    _study_logger.info("Loading documents from: %s", embedding_index_path)
    vector_index = VectorIndex.load(embedding_index_path, use_mmap=True)
    documents = vector_index.documents

    _study_logger.info("Building %s index from %d documents...", sparse_method, len(documents))

    # Build sparse index
    sparse_index = SparseIndex(
        name=sparse_name,
        method=SparseMethod(sparse_method),
    )
    sparse_index.build(documents, show_progress=True)

    # Save
    saved_path = sparse_index.save()
    _study_logger.info("Sparse index saved to: %s", saved_path)


def _save_retriever_config(
    manager,
    retriever_name: str,
    config: dict,
    index_name: str,
    sparse_name: str | None = None,
) -> None:
    """Save retriever config to artifacts/retrievers/{name}/config.json.

    This creates a retriever config that references the actual index paths,
    enabling experiment.py to load the correct indexes at runtime.
    """
    retriever_path = manager.get_retriever_path(retriever_name)
    config_path = retriever_path / "config.json"

    retriever_type = config.get("type", "dense")

    retriever_cfg = {
        "name": retriever_name,
        "type": retriever_type,
        "embedding_index": index_name,
        "embedding_model": config.get("embedding_model", Defaults.EMBEDDING_MODEL),
        "embedding_backend": config.get("embedding_backend", Defaults.EMBEDDING_BACKEND),
        "chunk_size": config.get("chunk_size", Defaults.CHUNK_SIZE),
        "chunk_overlap": config.get("chunk_overlap", Defaults.CHUNK_OVERLAP),
    }

    if retriever_type == "hybrid" and sparse_name:
        retriever_cfg["sparse_index"] = sparse_name
        retriever_cfg["sparse_method"] = config.get("sparse_method", "bm25")
        retriever_cfg["alpha"] = config.get("alpha", Defaults.HYBRID_ALPHA)

    if retriever_type == "hierarchical":
        retriever_cfg["parent_chunk_size"] = config.get("parent_chunk_size", 2048)
        retriever_cfg["child_chunk_size"] = config.get("child_chunk_size", 448)

    manager.save_json(retriever_cfg, config_path)
    _study_logger.info("Saved retriever config: %s -> %s", retriever_name, config_path)


def ensure_indexes_exist(retriever_configs: list, corpus_config: dict) -> None:
    """Ensure all required indexes exist, building them if necessary.

    For dense retrievers: builds embedding index
    For hybrid retrievers: builds embedding index + sparse index
    For hierarchical retrievers: builds hierarchical index

    Args:
        retriever_configs: List of retriever configuration dicts
        corpus_config: Corpus configuration dict
    """
    manager = get_artifact_manager()

    for config in retriever_configs:
        retriever_type = config.get("type", "dense")
        name = config.get("name", "")

        # ===================================================================
        # Step 1: Check/Build dense embedding index
        # ===================================================================
        # All retriever types use the same index name resolution
        index_name = config.get("embedding_index", name)
        index_path = manager.get_embedding_index_path(index_name)

        if _index_exists(index_path):
            _study_logger.info("Dense index exists: %s (at %s)", name, index_path)
        else:
            # Build the dense index
            _study_logger.info("Building dense index: %s", name)

            if retriever_type == "hierarchical":
                build_hierarchical_index(
                    retriever_config=config,
                    corpus_config=corpus_config,
                    doc_batch_size=corpus_config.get("doc_batch_size", 5000),
                    embedding_batch_size=corpus_config.get("embedding_batch_size", 4096),
                    embedding_backend=corpus_config.get("embedding", {}).get("backend", "vllm"),
                    vllm_gpu_memory_fraction=corpus_config.get("embedding", {}).get(
                        "vllm_gpu_memory_fraction", Defaults.VLLM_GPU_MEMORY_FRACTION
                    ),
                )
            else:
                build_embedding_index(
                    index_name=index_name,
                    embedding_model=config.get("embedding_model", Defaults.EMBEDDING_MODEL),
                    chunk_size=config.get("chunk_size", Defaults.CHUNK_SIZE),
                    chunk_overlap=config.get("chunk_overlap", Defaults.CHUNK_OVERLAP),
                    corpus_config=corpus_config,
                    embedding_backend=config.get("embedding_backend", "vllm"),
                )

        # ===================================================================
        # Step 2: For hybrid retrievers, check/build sparse index
        # ===================================================================
        sparse_name = None
        if retriever_type == "hybrid":
            sparse_method = config.get("sparse_method", "tfidf")

            # Determine sparse index name
            # Priority: explicit sparse_index > auto-generated from embedding_index
            sparse_name = config.get("sparse_index")
            if not sparse_name:
                # Auto-generate: {embedding_index}_sparse_{method}
                sparse_name = f"{index_name}_sparse_{sparse_method}"

            sparse_path = manager.get_sparse_index_path(sparse_name)

            if _sparse_index_exists(sparse_path):
                _study_logger.info("Sparse index exists: %s (at %s)", sparse_name, sparse_path)
            else:
                # Build sparse index from the dense index documents
                _build_sparse_index(
                    sparse_name=sparse_name,
                    embedding_index_path=index_path,
                    sparse_method=sparse_method,
                )

        # ===================================================================
        # Step 3: Save retriever config to artifacts/retrievers/{name}/
        # ===================================================================
        _save_retriever_config(manager, name, config, index_name, sparse_name)


def get_prompt_builder(prompt_type: str, dataset: str):
    """Get a PromptBuilder configured for the given prompt type and dataset.

    Args:
        prompt_type: One of "default", "concise", "fewshot", etc.
        dataset: Dataset name (for loading appropriate fewshot examples)

    Returns:
        Configured PromptBuilder instance
    """
    from ragicamp.utils.prompts import PromptBuilder

    return PromptBuilder.from_config(prompt_type, dataset=dataset)


def create_generator_provider(spec: str):
    """Create a GeneratorProvider from spec."""
    from ragicamp.factory import ProviderFactory

    validate_model_spec(spec)
    return ProviderFactory.create_generator(spec)


def create_dataset(name: str, limit: int | None = None):
    """Create dataset using DatasetFactory."""
    from ragicamp.factory import DatasetFactory

    config = DatasetFactory.parse_spec(name, limit=limit)
    return DatasetFactory.create(config)


def create_judge_model(llm_judge_config: dict[str, Any] | None):
    """Create LLM judge model from config.

    Supports OpenAI and DeepInfra (or any OpenAI-compatible provider).
    Provider is auto-detected from available API keys:
    - DEEPINFRA_API_KEY → DeepInfra endpoint
    - OPENAI_API_KEY → OpenAI endpoint

    Config format in YAML:
        llm_judge:
          model: "openai:gpt-4o-mini"       # or "deepinfra:meta-llama/Llama-3.3-70B-Instruct"
          base_url: "https://..."            # optional explicit override
    """
    import os

    if not llm_judge_config:
        return None

    model_spec = llm_judge_config.get("model", "openai:gpt-4o-mini")
    explicit_base_url = llm_judge_config.get("base_url")

    # Parse provider prefix
    if ":" in model_spec:
        provider_prefix, model_name = model_spec.split(":", 1)
    else:
        provider_prefix, model_name = "openai", model_spec

    # Resolve API key and base_url
    if explicit_base_url:
        api_key = os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = explicit_base_url
    elif provider_prefix == "deepinfra":
        api_key = os.environ.get("DEEPINFRA_API_KEY")
        base_url = "https://api.deepinfra.com/v1/openai"
    else:
        # Auto-detect: prefer DeepInfra if key exists, else OpenAI
        deepinfra_key = os.environ.get("DEEPINFRA_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        if deepinfra_key:
            api_key = deepinfra_key
            base_url = "https://api.deepinfra.com/v1/openai"
        elif openai_key:
            api_key = openai_key
            base_url = None
        else:
            _study_logger.warning(
                "No API key found for LLM judge (DEEPINFRA_API_KEY or OPENAI_API_KEY)"
            )
            return None

    if not api_key:
        _study_logger.warning("No API key found for LLM judge provider: %s", provider_prefix)
        return None

    _study_logger.info("LLM judge: %s (base_url=%s)", model_name, base_url or "default")
    return OpenAIModel(model_name=model_name, api_key=api_key, base_url=base_url)


def _run_spec_list(
    specs: list,
    *,
    limit: int | None,
    metrics: list[str],
    out: Path,
    judge_model: Any = None,
    llm_judge_config: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Run a list of ExperimentSpecs, tracking results.

    This is the shared execution loop used by all study modes.

    Args:
        specs: List of ExperimentSpec objects to run.
        limit: Max questions per experiment.
        metrics: List of metric names to compute.
        out: Output directory for experiment results.
        judge_model: Optional LLM judge model instance.
        llm_judge_config: Optional LLM judge configuration.
        force: Force re-run of completed/failed experiments.

    Returns:
        Dict with counts: ``{"completed": N, "failed": N, "skipped": N}``.
    """
    results = {"completed": 0, "failed": 0, "skipped": 0}

    for i, spec in enumerate(specs):
        _log_print(f"\n[{i + 1}/{len(specs)}] {spec.name}")

        status = run_spec(
            spec=spec,
            limit=limit,
            metrics=metrics,
            out=out,
            judge_model=judge_model,
            llm_judge_config=llm_judge_config,
            force=force,
            use_subprocess=True,
        )

        if status in ("complete", "ran", "resumed"):
            results["completed"] += 1
        elif status == "failed":
            results["failed"] += 1
        else:
            results["skipped"] += 1

    return results


def _build_specs_for_mode(
    config: dict[str, Any],
    sampling_override: dict[str, Any] | None,
) -> tuple[list, dict[str, Any]]:
    """Build experiment specs based on the active sampling mode.

    When a sampling mode is set (random, tpe, etc.), only baseline + singleton
    specs are built here; RAG specs are generated trial-by-trial by Optuna.

    When no sampling mode is set, the full experiment grid is built.

    Args:
        config: Full study YAML config dict.
        sampling_override: Optional CLI sampling override.

    Returns:
        Tuple of (specs, effective_sampling_config).
    """
    rag_config = config.get("rag", {})
    effective_sampling = sampling_override or rag_config.get("sampling", {})
    if not isinstance(effective_sampling, dict):
        effective_sampling = {}

    sampling_mode = effective_sampling.get("mode")

    if sampling_mode:
        # Any sampling mode → Optuna handles RAG; only build baselines + singletons
        baseline_config = dict(config)
        baseline_config["rag"] = {**rag_config, "enabled": False}
        specs = build_specs(baseline_config)
    else:
        # No sampling → full grid
        specs = build_specs(config)

    return specs, effective_sampling


def run_study(
    config: dict[str, Any],
    dry_run: bool = False,
    skip_existing: bool = True,
    validate_only: bool = False,
    limit: int | None = None,
    force: bool = False,
    experiment_filter: str | None = None,
    sampling_override: dict[str, Any] | None = None,
) -> None:
    """Run a study from a configuration dictionary.

    Orchestrates the full study lifecycle:

    1. Validate config
    2. Ensure required indexes exist
    3. Build experiment specs (mode-aware)
    4. Run baseline/singleton experiments
    5. Run Optuna-driven RAG trials (if sampling mode is set)
    6. Print summary

    Sampling modes (``rag.sampling.mode`` in YAML or ``--sample-mode`` CLI):

    - **(none)**: Full Cartesian product of all dimensions.
    - **random**: Uniform random search via Optuna ``RandomSampler``.
    - **tpe**: Bayesian optimization via Optuna ``TPESampler`` that learns
      from previous trials to maximize the target metric.

    All sampling modes go through Optuna, which provides SQLite persistence,
    transparent resume, and warm-starting from existing experiments on disk.

    Args:
        config: Study configuration dict (already loaded from YAML).
        dry_run: If True, just print what would be done.
        skip_existing: If True, skip completed experiments.
        validate_only: If True, just validate config and exit.
        limit: Optional limit on examples per experiment.
        force: Force re-run of completed/failed experiments.
        experiment_filter: Optional filter pattern for experiment names.
        sampling_override: Optional sampling config to override YAML settings.
            Example: ``{"mode": "tpe", "n_experiments": 100, "optimize_metric": "f1"}``
    """
    # ===================================================================
    # 1. Validate
    # ===================================================================
    warnings = validate_config(config)
    for w in warnings:
        _log_print(f"⚠️  {w}")

    if validate_only:
        _log_print("✓ Configuration is valid")
        return

    study_name = config["name"]
    description = config.get("description", "")
    out = Path(config.get("output_dir", f"outputs/{study_name}"))
    out.mkdir(parents=True, exist_ok=True)

    # Enable file logging for the study
    log_path = add_file_handler(out / "study.log")
    _study_logger.info("Study log: %s", log_path)

    _log_print(f"\n{'=' * 70}")
    _log_print(f"Study: {study_name}")
    if description:
        _log_print(f"  {description}")
    _log_print(f"{'=' * 70}")

    if limit is None:
        limit = config.get("num_questions")

    metrics = config.get("metrics", ["f1", "exact_match"])
    llm_judge_config = config.get("llm_judge")
    judge_model = create_judge_model(llm_judge_config)

    # ===================================================================
    # 2. Ensure indexes exist
    # ===================================================================
    rag_config = config.get("rag", {})
    if rag_config.get("enabled") and not dry_run:
        corpus_config = rag_config.get("corpus", {})
        all_retrievers = rag_config.get("retrievers", [])
        active_retriever_names = set(rag_config.get("retriever_names", []))
        retriever_configs = [r for r in all_retrievers if r.get("name") in active_retriever_names]
        ensure_indexes_exist(retriever_configs, corpus_config)

    # ===================================================================
    # 3. Build experiment specs (mode-aware)
    # ===================================================================
    specs, effective_sampling = _build_specs_for_mode(config, sampling_override)
    sampling_mode = effective_sampling.get("mode")

    if experiment_filter:
        specs = [s for s in specs if experiment_filter in s.name]

    # ===================================================================
    # 4. Preview
    # ===================================================================
    label = "Baseline/singleton" if sampling_mode else "Total"
    _log_print(f"\n📋 {label} experiments: {len(specs)}")
    for s in specs:
        _log_print(f"   - {s.name}")

    if sampling_mode and rag_config.get("enabled"):
        n_trials = effective_sampling.get("n_experiments", 50)
        optimize_metric = effective_sampling.get("optimize_metric", "f1")
        mode_desc = "optimizing" if sampling_mode != "random" else "sampling"
        _log_print(f"🔬 + {n_trials} RAG trials ({sampling_mode} {mode_desc} {optimize_metric})")

    if dry_run:
        _log_print("\n[DRY RUN] Would run the above experiments")
        return

    # ===================================================================
    # 5. Run experiments
    # ===================================================================
    should_force = force or not skip_existing

    results = _run_spec_list(
        specs,
        limit=limit,
        metrics=metrics,
        out=out,
        judge_model=judge_model,
        llm_judge_config=llm_judge_config,
        force=should_force,
    )

    # ===================================================================
    # 6. Optuna-driven RAG trials
    # ===================================================================
    optuna_study = None
    if sampling_mode and rag_config.get("enabled"):
        from ragicamp.optimization.optuna_search import run_optuna_study as _optuna

        optuna_study = _optuna(
            config=config,
            n_trials=effective_sampling.get("n_experiments", 50),
            output_dir=out,
            sampler_mode=sampling_mode,
            optimize_metric=effective_sampling.get("optimize_metric", "f1"),
            limit=limit,
            judge_model=judge_model,
            llm_judge_config=llm_judge_config,
            seed=effective_sampling.get("seed"),
        )

    # ===================================================================
    # 7. Summary
    # ===================================================================
    _log_print(f"\n{'=' * 70}")
    _log_print(f"Study Complete: {study_name}")
    _log_print(
        f"  Baselines — Completed: {results['completed']}, "
        f"Failed: {results['failed']}, Skipped: {results['skipped']}"
    )
    if optuna_study is not None:
        n_complete = len(optuna_study.trials)
        n_pruned = len([t for t in optuna_study.trials if t.state.name == "PRUNED"])
        _log_print(f"  Optuna — Trials: {n_complete}, Pruned: {n_pruned}")
        if optuna_study.best_trial:
            _log_print(
                f"  Best trial: {optuna_study.best_value:.4f} "
                f"(trial #{optuna_study.best_trial.number})"
            )
    _log_print(f"{'=' * 70}")

    meta = {
        "name": study_name,
        "num_experiments": len(specs),
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    from ragicamp.utils.experiment_io import atomic_write_json

    atomic_write_json(meta, out / "study_meta.json")


def compare(out: Path) -> None:
    """Compare experiment results in a study output directory.

    Args:
        out: Path to study output directory
    """
    from ragicamp.analysis.comparison import compare_results
    from ragicamp.analysis.loader import ResultsLoader

    loader = ResultsLoader(out)
    results = loader.load_all()
    if results:
        grouped = compare_results(results)
        for group_name, metrics in grouped.items():
            _study_logger.info("%s: %s", group_name, metrics)
    else:
        _study_logger.warning("No experiment results found in %s", out)
