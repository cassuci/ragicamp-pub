"""CLI command implementations."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)

# DeepInfra OpenAI-compatible endpoint
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


def _resolve_judge_config(
    model_name: str,
    base_url: str | None = None,
) -> tuple[str, str | None, str]:
    """Resolve API key and base_url for the LLM judge.

    Checks DEEPINFRA_API_KEY first, then OPENAI_API_KEY.
    When DEEPINFRA_API_KEY is found and no explicit base_url is given,
    automatically sets the DeepInfra endpoint.

    Args:
        model_name: Model identifier
        base_url: Explicit base URL override (from --judge-base-url)

    Returns:
        Tuple of (api_key, base_url, provider_label)

    Raises:
        SystemExit: If no API key is found
    """
    deepinfra_key = os.environ.get("DEEPINFRA_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if base_url:
        # Explicit base_url — use whichever key is available
        api_key = deepinfra_key or openai_key
        if not api_key:
            print("Error: No API key found. Set DEEPINFRA_API_KEY or OPENAI_API_KEY.")
            raise SystemExit(1)
        return api_key, base_url, "custom"

    if deepinfra_key:
        return deepinfra_key, DEEPINFRA_BASE_URL, "DeepInfra"

    if openai_key:
        return openai_key, None, "OpenAI"

    print("Error: No API key found for LLM judge.")
    print("Set one of:")
    print("  export DEEPINFRA_API_KEY='your-deepinfra-key'")
    print("  export OPENAI_API_KEY='your-openai-key'")
    raise SystemExit(1)


def cmd_run(args: argparse.Namespace) -> int:
    """Run a study from config file."""
    from ragicamp.cli.study import run_study

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        return 1

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Build sampling override from CLI args
    sampling_override = None
    if args.sample:
        sampling_override = {
            "mode": args.sample_mode,
            "n_experiments": args.sample,
            "seed": args.sample_seed,
            "optimize_metric": getattr(args, "optimize_metric", "f1"),
        }
        print(f"🎲 Sampling mode: {args.sample_mode}, n={args.sample}")
        if args.sample_seed:
            print(f"   Seed: {args.sample_seed}")

    run_study(
        config,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        validate_only=args.validate,
        sampling_override=sampling_override,
        limit=getattr(args, "limit", None),
        force=getattr(args, "force", False),
    )
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """Build retrieval indexes."""
    from ragicamp.corpus import ChunkConfig, CorpusConfig, DocumentChunker, WikipediaCorpus
    from ragicamp.factory import ProviderFactory
    from ragicamp.indexes import IndexBuilder

    # Map short names to embedding models
    embedding_models = {
        "minilm": "all-MiniLM-L6-v2",
        "e5": "intfloat/e5-small-v2",
        "mpnet": "all-mpnet-base-v2",
        "mock": "mock-embedder",
    }
    embedding_model = embedding_models.get(args.embedding, args.embedding)

    # Map short names to corpus versions
    corpus_versions = {
        "simple": "20231101.simple",
        "en": "20231101.en",
    }
    corpus_version = corpus_versions.get(args.corpus, args.corpus)

    index_name = f"{args.corpus}_{args.embedding}_recursive_{args.chunk_size}"
    print(f"Building index: {index_name}")

    # Load corpus
    corpus_config = CorpusConfig(
        name=f"wikipedia_{args.corpus}",
        source="wikimedia/wikipedia",
        version=corpus_version,
        max_docs=args.max_docs,
    )
    corpus = WikipediaCorpus(corpus_config)
    docs = list(corpus.load())
    print(f"Loaded {len(docs)} documents")

    # Chunk config
    from ragicamp.core.constants import Defaults

    chunk_config = ChunkConfig(
        strategy="recursive",
        chunk_size=args.chunk_size,
        chunk_overlap=Defaults.CHUNK_OVERLAP,
    )
    chunker = DocumentChunker(chunk_config)

    # Create embedder provider
    embedder_backend = "mock" if args.embedding == "mock" else "sentence_transformers"
    embedder_provider = ProviderFactory.create_embedder(
        embedding_model,
        backend=embedder_backend,  # Default for small real models
    )

    # Build and save index using provider pattern
    from ragicamp.utils.artifacts import get_artifact_manager

    manager = get_artifact_manager()
    index_path = manager.get_embedding_index_path(index_name)
    builder = IndexBuilder(embedder_provider=embedder_provider, chunker=chunker)
    index = builder.build(docs, output_path=index_path)
    print(f"Created index with {len(index.documents)} chunks")
    print(f"Index saved: {index_path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare experiment results."""
    from ragicamp.analysis import (
        ResultsLoader,
        best_by,
        compare_results,
        format_comparison_table,
        pivot_results,
        summarize_results,
    )

    loader = ResultsLoader(args.output_dir)
    results = loader.load_all()

    if not results:
        print("No results found")
        return 1

    print(f"Loaded {len(results)} experiments\n")

    # Summary
    summary = summarize_results(results)
    print(f"Models: {', '.join(summary['models'])}")
    print(f"Datasets: {', '.join(summary['datasets'])}")
    print(f"Best F1: {summary['best_f1']['value']:.4f} ({summary['best_f1']['model']})")
    print()

    # Comparison by requested dimension
    stats = compare_results(results, group_by=args.group_by, metric=args.metric)
    print(format_comparison_table(stats, title=f"By {args.group_by}", metric=args.metric))

    # Pivot table if requested
    if args.pivot:
        pivot = pivot_results(results, rows=args.pivot[0], cols=args.pivot[1], metric=args.metric)
        print(f"\nPivot: {args.pivot[0]} x {args.pivot[1]}")
        for row, cols in sorted(pivot.items()):
            print(f"  {row[:20]}: {', '.join(f'{c}={v:.3f}' for c, v in sorted(cols.items()))}")

    # Top N
    print(f"\nTop {args.top} by {args.metric}:")
    for i, r in enumerate(best_by(results, metric=args.metric, n=args.top), 1):
        val = getattr(r, args.metric, 0)
        print(f"  {i}. {r.name[:50]} = {val:.4f}")

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Compute metrics on predictions file."""
    from ragicamp.evaluation import compute_metrics_from_file
    from ragicamp.factory import MetricFactory

    metric_names = list(args.metrics)

    # Build judge model if needed for LLM metrics
    judge_model = None
    if any(m in ("llm_judge", "llm_judge_qa") for m in metric_names):
        from ragicamp.models.openai import OpenAIModel

        judge_base_url = getattr(args, "judge_base_url", None)
        api_key, base_url, provider = _resolve_judge_config(args.judge_model, judge_base_url)
        print(
            f"Using judge model: {args.judge_model} via {provider}"
            f" (max_concurrent={args.max_concurrent})"
        )
        judge_model = OpenAIModel(
            args.judge_model, api_key=api_key, base_url=base_url, temperature=0.0
        )

    metrics = MetricFactory.create(metric_names, judge_model=judge_model)

    if not metrics:
        print(
            "No valid metrics specified. Available: f1, exact_match, llm_judge_qa, bertscore, bleurt"
        )
        return 1

    print(f"Computing metrics: {[m.name for m in metrics]}")
    print(f"Predictions file: {args.predictions}")

    results = compute_metrics_from_file(
        predictions_path=str(args.predictions),
        metrics=metrics,
        output_path=str(args.output) if args.output else None,
    )

    print("\nResults:")
    for key, value in results.get("aggregate", {}).items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Check health of experiments in a directory."""
    from ragicamp.state import check_health

    output_dir = args.output_dir
    if not output_dir.exists():
        print(f"Directory not found: {output_dir}")
        return 1

    # Find experiment directories
    exp_dirs = [
        d
        for d in output_dir.iterdir()
        if d.is_dir()
        and (
            (d / "state.json").exists()
            or (d / "predictions.json").exists()
            or (d / "results.json").exists()
        )
    ]

    if not exp_dirs:
        print(f"No experiments found in {output_dir}")
        return 1

    print(f"Checking {len(exp_dirs)} experiments in {output_dir}\n")

    # Status counts
    complete = 0
    incomplete = 0
    failed = 0

    for exp_dir in sorted(exp_dirs):
        health = check_health(exp_dir, args.metrics.split(",") if args.metrics else None)
        print(f"  {health.summary()} - {exp_dir.name}")

        if health.is_complete:
            complete += 1
        elif health.phase.value == "failed":
            failed += 1
        else:
            incomplete += 1

    print(f"\nSummary: {complete} complete, {incomplete} incomplete, {failed} failed")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Resume incomplete experiments."""
    from ragicamp.state import check_health

    output_dir = args.output_dir
    if not output_dir.exists():
        print(f"Directory not found: {output_dir}")
        return 1

    # Find experiment directories
    exp_dirs = [d for d in output_dir.iterdir() if d.is_dir()]

    if not exp_dirs:
        print(f"No experiments found in {output_dir}")
        return 1

    # Find incomplete experiments
    to_resume = []
    for exp_dir in sorted(exp_dirs):
        health = check_health(exp_dir)
        if health.can_resume and not health.is_complete:
            to_resume.append((exp_dir, health))

    if not to_resume:
        print("All experiments are complete or failed.")
        return 0

    print(f"Found {len(to_resume)} experiments to resume:\n")
    for exp_dir, health in to_resume:
        print(f"  {health.summary()} - {exp_dir.name}")

    if args.dry_run:
        print("\n[DRY RUN] - no changes made")
        return 0

    print("\nTo resume, run the original study config with --skip-existing=False")
    print("Or use `ragicamp metrics <dir>` to recompute just metrics")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    """Recompute metrics for an experiment."""
    from ragicamp.evaluation import compute_metrics_from_file
    from ragicamp.factory import MetricFactory

    exp_dir = args.exp_dir
    predictions_path = exp_dir / "predictions.json"

    if not predictions_path.exists():
        print(f"Predictions not found: {predictions_path}")
        return 1

    # Parse metrics
    metric_names = [m.strip() for m in args.metrics.split(",")]

    # Build judge model if needed
    judge_model = None
    if any(m in ("llm_judge", "llm_judge_qa") for m in metric_names):
        from ragicamp.models.openai import OpenAIModel

        judge_base_url = getattr(args, "judge_base_url", None)
        api_key, base_url, provider = _resolve_judge_config(args.judge_model, judge_base_url)
        print(f"Using judge model: {args.judge_model} via {provider}")
        judge_model = OpenAIModel(
            args.judge_model, api_key=api_key, base_url=base_url, temperature=0.0
        )

    metrics = MetricFactory.create(metric_names, judge_model=judge_model)

    print(f"Computing metrics: {metric_names}")
    print(f"Experiment: {exp_dir.name}")

    results = compute_metrics_from_file(
        predictions_path=str(predictions_path),
        metrics=metrics,
        output_path=str(predictions_path),  # Update in place
    )

    print("\nResults:")
    for key, value in results.get("aggregate", {}).items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # Update results.json if it exists
    results_path = exp_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            result_data = json.load(f)
        result_data["metrics"].update(results.get("aggregate", {}))
        from ragicamp.utils.experiment_io import atomic_write_json

        atomic_write_json(result_data, results_path)
        print(f"\n✓ Updated {results_path}")

    return 0


def _pickle_needs_migration(pkl_path: Path) -> bool:
    """Check if a pickle file uses the old module path."""
    if not pkl_path.exists():
        return False
    try:
        # Read first 4KB and check for old module path
        with open(pkl_path, "rb") as f:
            header = f.read(4096)
        return b"ragicamp.retrievers.base" in header
    except Exception:
        return False


def _find_indexes_to_migrate(
    args: argparse.Namespace,
) -> tuple[list[Path], bool]:
    """Find all index paths that should be considered for migration.

    Returns:
        Tuple of (index_paths, force_flag).
    """
    from ragicamp.utils.artifacts import get_artifact_manager

    manager = get_artifact_manager()
    indexes_dir = manager.indexes_dir
    force = getattr(args, "force", False)

    if args.index_name:
        return [manager.get_embedding_index_path(args.index_name)], force

    return [p for p in indexes_dir.iterdir() if p.is_dir()], force


def _migrate_single_index(
    index_path: Path,
    *,
    dry_run: bool,
    force: bool,
) -> str:
    """Migrate a single index directory.

    Returns:
        One of "migrated", "skipped", "failed".
    """
    import pickle

    from ragicamp.indexes.vector_index import VectorIndex

    name = index_path.name

    has_faiss = (index_path / "index.faiss").exists()
    has_shards = any((index_path / f"shard_{i}" / "index.faiss").exists() for i in range(10))

    if not has_faiss and not has_shards:
        print(f"  [SKIP] {name}: not a valid index")
        return "skipped"

    needs_config = not (index_path / "config.json").exists()
    docs_path = index_path / "documents.pkl"
    shard_dirs = sorted(index_path.glob("shard_*"))

    needs_pickle = False
    if docs_path.exists():
        needs_pickle = _pickle_needs_migration(docs_path)
    else:
        for shard_dir in shard_dirs:
            if _pickle_needs_migration(shard_dir / "documents.pkl"):
                needs_pickle = True
                break

    if not needs_config and not needs_pickle and not force:
        print(f"  [SKIP] {name}: already migrated")
        return "skipped"

    what_to_do = []
    if needs_config:
        what_to_do.append("config")
    if needs_pickle:
        what_to_do.append("pickle")

    try:
        if dry_run:
            print(f"  [DRY RUN] {name}: would migrate {', '.join(what_to_do)}")
            return "migrated"

        print(f"  Migrating {name} ({', '.join(what_to_do)})...")
        index = VectorIndex.load(index_path, use_mmap=False)

        if needs_config or force:
            config_path = index_path / "config.json"
            with open(config_path, "w") as f:
                json.dump(index.config.to_dict(), f, indent=2)

        if needs_pickle or force:
            if docs_path.exists():
                with open(docs_path, "wb") as f:
                    pickle.dump(index.documents, f)
                print(f"  [OK] {name}: migrated ({len(index.documents)} docs)")
            else:
                for shard_dir in shard_dirs:
                    shard_docs = shard_dir / "documents.pkl"
                    if shard_docs.exists():
                        with open(shard_docs, "rb") as f:
                            shard_documents = pickle.load(f)
                        shard_documents = VectorIndex._ensure_document_type(shard_documents)
                        with open(shard_docs, "wb") as f:
                            pickle.dump(shard_documents, f)
                print(f"  [OK] {name}: migrated {len(shard_dirs)} shards")
        else:
            print(f"  [OK] {name}: config updated")

        return "migrated"

    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return "failed"


def cmd_migrate_indexes(args: argparse.Namespace) -> int:
    """Migrate old index format to new format."""
    index_paths, force = _find_indexes_to_migrate(args)

    if not index_paths:
        print("No indexes found to migrate.")
        return 0

    migrated = skipped = failed = 0
    for index_path in index_paths:
        status = _migrate_single_index(index_path, dry_run=args.dry_run, force=force)
        if status == "migrated":
            migrated += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1

    print()
    print(f"Migration complete: {migrated} migrated, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


# =============================================================================
# Batch metric computation
# =============================================================================


def cmd_compute_metrics(args: argparse.Namespace) -> int:
    """Compute metrics across all experiments in a study output directory."""
    from ragicamp.execution.runner import run_metrics_only
    from ragicamp.state import check_health

    output_dir = args.output_dir
    if not output_dir.exists():
        print(f"Directory not found: {output_dir}")
        return 1

    metric_names = [m.strip() for m in args.metrics.split(",")]
    force = getattr(args, "force", False)

    # Find experiment directories (same pattern as cmd_health)
    exp_dirs = [
        d
        for d in output_dir.iterdir()
        if d.is_dir()
        and (
            (d / "state.json").exists()
            or (d / "predictions.json").exists()
            or (d / "results.json").exists()
        )
    ]

    if not exp_dirs:
        print(f"No experiments found in {output_dir}")
        return 1

    # Filter to experiments that need the metric
    to_compute = []
    skipped = 0
    for exp_dir in sorted(exp_dirs):
        health = check_health(exp_dir, metric_names)
        if health.predictions_complete == 0:
            skipped += 1
            continue
        if force or health.metrics_missing:
            to_compute.append(exp_dir)
        else:
            skipped += 1

    print(f"Found {len(exp_dirs)} experiments, {len(to_compute)} need metrics, {skipped} skipped")

    if not to_compute:
        print("Nothing to compute.")
        return 0

    if args.dry_run:
        print(f"\n[DRY RUN] Would compute {metric_names} for:")
        for exp_dir in to_compute:
            print(f"  {exp_dir.name}")
        return 0

    # Build judge model if needed
    judge_model = None
    if any(m in ("llm_judge", "llm_judge_qa") for m in metric_names):
        from ragicamp.models.openai import OpenAIModel

        judge_base_url = getattr(args, "judge_base_url", None)
        api_key, base_url, provider = _resolve_judge_config(args.judge_model, judge_base_url)
        print(f"Using judge model: {args.judge_model} via {provider}")
        judge_model = OpenAIModel(
            args.judge_model, api_key=api_key, base_url=base_url, temperature=0.0
        )

    # Compute metrics for each experiment
    computed = 0
    failed = 0
    for i, exp_dir in enumerate(to_compute, 1):
        print(f"\n[{i}/{len(to_compute)}] {exp_dir.name}")
        try:
            status = run_metrics_only(
                exp_name=exp_dir.name,
                output_path=exp_dir,
                metrics=metric_names,
                judge_model=judge_model,
                force=force,
            )
            if status == "failed":
                print("  FAILED")
                failed += 1
            else:
                print("  OK")
                computed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\nDone: {computed} computed, {failed} failed, {skipped} skipped")
    return 0 if failed == 0 else 1


# =============================================================================
# Cache management
# =============================================================================


def cmd_cache(args: argparse.Namespace) -> int:
    """Manage the embedding cache (stats / clear)."""
    from ragicamp.cache import EmbeddingStore

    store = EmbeddingStore.default()
    action = args.cache_action

    if action == "stats":
        info = store.stats()
        print(f"\nEmbedding cache: {info['db_path']}")
        print(f"  Total entries : {info['total_entries']:,}")
        print(f"  Total size    : {info['total_size_mb']:.2f} MB")
        if info["models"]:
            print(f"  Models cached : {len(info['models'])}")
            for model, minfo in sorted(info["models"].items()):
                print(f"    {model}: {minfo['entries']:,} entries ({minfo['size_mb']:.2f} MB)")
        else:
            print("  (empty)")
        print()
        return 0

    elif action == "clear":
        model = getattr(args, "model", None)
        if model:
            print(f"Clearing cache for model: {model}")
        else:
            print("Clearing entire embedding cache...")

        deleted = store.clear(model=model)
        print(f"Deleted {deleted:,} entries.")
        return 0

    else:
        print(f"Unknown cache action: {action}")
        return 1
