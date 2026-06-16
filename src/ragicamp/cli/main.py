#!/usr/bin/env python3
"""RAGiCamp CLI - Unified command-line interface.

Commands:
    run              Run a study from config
    index            Build retrieval indexes
    compare          Compare experiment results
    evaluate         Compute metrics on predictions
    compute-metrics  Compute metrics for all experiments in a study
"""

# ============================================================================
# CRITICAL: Configure environment BEFORE any library imports!
# ============================================================================
import os

# TensorFlow: Prevent grabbing all GPU memory on import
if "TF_FORCE_GPU_ALLOW_GROWTH" not in os.environ:
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# vLLM: Use 'spawn' for multiprocessing to avoid CUDA fork issues
# See: https://github.com/vllm-project/vllm/issues/6152
if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import sys
from pathlib import Path

from ragicamp.cli.commands import (
    cmd_cache,
    cmd_compare,
    cmd_compute_metrics,
    cmd_evaluate,
    cmd_health,
    cmd_index,
    cmd_metrics,
    cmd_migrate_indexes,
    cmd_resume,
    cmd_run,
)


def _add_run_parser(subparsers) -> None:
    """Add 'run' subcommand."""
    p = subparsers.add_parser("run", help="Run a study from config")
    p.add_argument("config", type=Path, help="Study config YAML")
    p.add_argument("--dry-run", action="store_true", help="Preview only")
    p.add_argument("--skip-existing", action="store_true", help="Skip completed")
    p.add_argument("--validate", action="store_true", help="Validate config only")
    p.add_argument(
        "--sample", "-s", type=int, metavar="N", help="Sample N experiments (enables search)"
    )
    p.add_argument(
        "--sample-mode", choices=["random", "tpe"], default="random", help="Sampling mode"
    )
    p.add_argument("--sample-seed", type=int, default=None, help="Random seed for sampling")
    p.add_argument(
        "--optimize-metric", type=str, default="f1", help="Metric to optimize (default: f1)"
    )
    p.add_argument("--limit", type=int, default=None, help="Max examples per experiment")
    p.add_argument("--force", action="store_true", help="Force re-run even if complete/failed")
    p.set_defaults(func=cmd_run)


def _add_index_parser(subparsers) -> None:
    """Add 'index' subcommand."""
    p = subparsers.add_parser("index", help="Build retrieval index")
    p.add_argument("--corpus", default="simple", help="Corpus: simple, en, or full version string")
    p.add_argument(
        "--embedding", default="minilm", help="Embedding: minilm, e5, mpnet, or full model name"
    )
    p.add_argument("--chunk-size", type=int, default=512, help="Chunk size in chars")
    p.add_argument("--max-docs", type=int, default=None, help="Max documents to index")
    p.set_defaults(func=cmd_index)


def _add_compare_parser(subparsers) -> None:
    """Add 'compare' subcommand."""
    p = subparsers.add_parser("compare", help="Compare results")
    p.add_argument("output_dir", type=Path, help="Output directory")
    p.add_argument("--top", type=int, default=10, help="Show top N results")
    p.add_argument("--metric", "-m", default="f1", help="Metric to compare")
    p.add_argument(
        "--group-by",
        "-g",
        default="model",
        choices=["model", "dataset", "prompt", "retriever", "quantization", "type"],
        help="Dimension to group by",
    )
    p.add_argument("--pivot", nargs=2, metavar=("ROWS", "COLS"), help="Create pivot table")
    p.set_defaults(func=cmd_compare)


def _add_evaluate_parser(subparsers) -> None:
    """Add 'evaluate' subcommand."""
    p = subparsers.add_parser("evaluate", help="Compute metrics")
    p.add_argument("predictions", type=Path, help="Predictions JSON")
    p.add_argument("--metrics", nargs="+", default=["f1", "exact_match"], help="Metrics to compute")
    p.add_argument("--output", type=Path, help="Output file")
    p.add_argument("--judge-model", default="gpt-4o-mini", help="Model for LLM judge")
    p.add_argument(
        "--judge-base-url",
        default=None,
        help="Base URL for OpenAI-compatible API (auto-detected from DEEPINFRA_API_KEY)",
    )
    p.add_argument("--judgment-type", choices=["binary", "ternary"], default="binary")
    p.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent LLM judge calls")
    p.set_defaults(func=cmd_evaluate)


def _add_health_parser(subparsers) -> None:
    """Add 'health' subcommand."""
    p = subparsers.add_parser("health", help="Check experiment health")
    p.add_argument("output_dir", type=Path, help="Output directory")
    p.add_argument("--metrics", default=None, help="Comma-separated metrics to check")
    p.set_defaults(func=cmd_health)


def _add_resume_parser(subparsers) -> None:
    """Add 'resume' subcommand."""
    p = subparsers.add_parser("resume", help="Resume incomplete experiments")
    p.add_argument("output_dir", type=Path, help="Output directory")
    p.add_argument("--dry-run", action="store_true", help="Preview only")
    p.set_defaults(func=cmd_resume)


def _add_metrics_parser(subparsers) -> None:
    """Add 'metrics' subcommand."""
    p = subparsers.add_parser("metrics", help="Recompute metrics for an experiment")
    p.add_argument("exp_dir", type=Path, help="Experiment directory")
    p.add_argument("--metrics", "-m", required=True, help="Comma-separated metrics")
    p.add_argument("--judge-model", default="gpt-4o-mini", help="Model for LLM judge")
    p.add_argument(
        "--judge-base-url",
        default=None,
        help="Base URL for OpenAI-compatible API (auto-detected from DEEPINFRA_API_KEY)",
    )
    p.set_defaults(func=cmd_metrics)


def _add_compute_metrics_parser(subparsers) -> None:
    """Add 'compute-metrics' subcommand."""
    p = subparsers.add_parser(
        "compute-metrics", help="Compute metrics for all experiments in a study"
    )
    p.add_argument("output_dir", type=Path, help="Study output directory")
    p.add_argument("--metrics", "-m", required=True, help="Comma-separated metrics to compute")
    p.add_argument("--dry-run", action="store_true", help="Preview only, list experiments")
    p.add_argument("--judge-model", default="gpt-4o-mini", help="Model for LLM judge")
    p.add_argument(
        "--judge-base-url",
        default=None,
        help="Base URL for OpenAI-compatible API (auto-detected from DEEPINFRA_API_KEY)",
    )
    p.add_argument("--force", action="store_true", help="Recompute even if metric already exists")
    p.set_defaults(func=cmd_compute_metrics)


def _add_cache_parser(subparsers) -> None:
    """Add 'cache' subcommand."""
    p = subparsers.add_parser("cache", help="Manage the embedding cache")
    sub = p.add_subparsers(dest="cache_action", help="Cache actions")
    sub.required = True
    sub.add_parser("stats", help="Show cache statistics")
    clear = sub.add_parser("clear", help="Clear cached embeddings")
    clear.add_argument("--model", default=None, help="Only clear for this model")
    p.set_defaults(func=cmd_cache)


def _add_migrate_parser(subparsers) -> None:
    """Add 'migrate-indexes' subcommand."""
    p = subparsers.add_parser("migrate-indexes", help="Migrate old index format to new format")
    p.add_argument("--index-name", "-n", default=None, help="Specific index to migrate")
    p.add_argument("--dry-run", action="store_true", help="Preview only")
    p.add_argument("--force", "-f", action="store_true", help="Re-migrate even if already migrated")
    p.set_defaults(func=cmd_migrate_indexes)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="ragicamp",
        description="RAGiCamp - RAG Experimentation Framework",
    )
    from ragicamp import __version__

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _add_run_parser(subparsers)
    _add_index_parser(subparsers)
    _add_compare_parser(subparsers)
    _add_evaluate_parser(subparsers)
    _add_health_parser(subparsers)
    _add_resume_parser(subparsers)
    _add_metrics_parser(subparsers)
    _add_compute_metrics_parser(subparsers)
    _add_cache_parser(subparsers)
    _add_migrate_parser(subparsers)

    return parser


def main(argv: list | None = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
