#!/usr/bin/env python3
"""Run a single experiment in isolation (subprocess wrapper).

This script is called by the study runner to isolate experiments from CUDA crashes.
It reuses all the existing abstractions from cli/study.py.

Usage:
    python run_single_experiment.py --spec-json '{"name": "...", ...}' --output-dir outputs/study
"""

# ============================================================================
# CRITICAL: Configure environment BEFORE any imports!
# ============================================================================
import os
import shutil

# TensorFlow: Prevent grabbing all GPU memory on import
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info logs

# vLLM: Use 'spawn' for multiprocessing to avoid CUDA fork issues
# See: https://github.com/vllm-project/vllm/issues/6152
if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# vLLM: Use TORCH_SDPA attention backend if nvcc is not available
# FlashInfer requires nvcc for JIT compilation
if not shutil.which("nvcc") and "VLLM_ATTENTION_BACKEND" not in os.environ:
    os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"

import argparse
import json
import sys
from pathlib import Path


def _log_gpu_mem(label: str) -> None:
    """Log GPU memory usage for debugging."""
    try:
        import torch

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            if alloc > 0.1:  # Only log if significant
                print(f"  [GPU] {label}: {alloc:.2f} GiB allocated")
    except Exception:
        pass


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

_log_gpu_mem("before imports")

# Reuse existing abstractions - NO DUPLICATION
from ragicamp.cli.study import create_judge_model  # noqa: E402
from ragicamp.execution.runner import run_spec  # noqa: E402
from ragicamp.spec.experiment import ExperimentSpec  # noqa: E402

_log_gpu_mem("after imports")


def main():
    parser = argparse.ArgumentParser(description="Run single experiment in subprocess")
    parser.add_argument("--spec-json", required=True, help="Experiment spec as JSON string")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--metrics", default="f1,exact_match", help="Comma-separated metrics")
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    parser.add_argument("--llm-judge-config", help="LLM judge config as JSON string")

    args = parser.parse_args()

    spec_dict = json.loads(args.spec_json)
    metrics = args.metrics.split(",")

    # Parse llm_judge config if provided
    llm_judge_config = None
    judge_model = None
    if args.llm_judge_config:
        llm_judge_config = json.loads(args.llm_judge_config)
        # Create the judge model in this subprocess
        judge_model = create_judge_model(llm_judge_config)

    # Convert dict back to ExperimentSpec dataclass
    # Use metrics from spec_dict (passed from parent) or fall back to CLI arg
    spec_metrics = spec_dict.get("metrics", metrics)

    # Convert agent_params dict to tuple for frozen dataclass
    agent_params = spec_dict.get("agent_params", {})
    agent_params_tuple = tuple(agent_params.items()) if agent_params else ()

    spec = ExperimentSpec(
        name=spec_dict["name"],
        exp_type=spec_dict["exp_type"],
        model=spec_dict["model"],
        dataset=spec_dict["dataset"],
        prompt=spec_dict["prompt"],
        retriever=spec_dict.get("retriever"),
        embedding_index=spec_dict.get("embedding_index"),
        sparse_index=spec_dict.get("sparse_index"),
        top_k=spec_dict.get("top_k", 5),
        fetch_k=spec_dict.get("fetch_k"),
        query_transform=spec_dict.get("query_transform"),
        reranker=spec_dict.get("reranker"),
        reranker_model=spec_dict.get("reranker_model"),
        rrf_k=spec_dict.get("rrf_k"),
        alpha=spec_dict.get("alpha"),
        batch_size=spec_dict.get("batch_size", 8),
        metrics=spec_metrics,
        agent_type=spec_dict.get("agent_type"),
        agent_params=agent_params_tuple,
    )

    # Reuse the existing run_spec function with subprocess disabled
    # (we're already in a subprocess, no need to nest)
    status = run_spec(
        spec=spec,
        limit=args.limit,
        metrics=metrics,
        out=args.output_dir,
        judge_model=judge_model,
        llm_judge_config=llm_judge_config,
        force=True,  # Always retry in subprocess
        use_subprocess=False,  # We ARE the subprocess
    )

    print(f"\n__RESULT__:{json.dumps({'status': status})}")

    if status in ("failed", "crashed"):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
