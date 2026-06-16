#!/usr/bin/env python3
"""Repair metadata.json for experiments with missing/inconsistent fields.

Addresses three data quality issues:

1. **Empty model field** (37 experiments): The INIT phase only saved a sparse
   metadata.json; the full metadata is only written by the runner on successful
   completion.  Experiments that failed or stalled lack model, type, prompt,
   retriever, top_k, etc.  This script infers the model spec from the directory
   name and backfills from spec fields recoverable from the experiment name.

2. **Dataset name inconsistency** (``natural_questions`` vs ``nq``): The INIT
   phase used ``context.dataset.name`` (the dataset class name, e.g.
   ``"natural_questions"``) instead of ``spec.dataset`` (the canonical short
   name from the YAML, e.g. ``"nq"``).

3. **Name mismatch** (``_none`` suffix): Some experiments have a trailing
   ``_none`` in ``metadata.json`` ``name`` field that doesn't match the
   directory name.

Usage:
    # Dry-run (default):
    python scripts/repair_metadata.py --study outputs/smart_retrieval_slm

    # Execute:
    python scripts/repair_metadata.py --study outputs/smart_retrieval_slm --execute
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKIP_DIRS = {"_archived_fake_reranked", "analysis", "__pycache__", ".ipynb_checkpoints"}

# Reverse mapping: normalised substring (no hyphens, lowercase) -> full model spec.
# Built from the known models in the study.
MODEL_REVERSE_MAP = {
    "qwen2.51.5binstruct": "vllm:Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.53binstruct": "vllm:Qwen/Qwen2.5-3B-Instruct",
    "qwen2.57binstruct": "vllm:Qwen/Qwen2.5-7B-Instruct",
    "gemma22bit": "vllm:google/gemma-2-2b-it",
    "gemma29bit": "vllm:google/gemma-2-9b-it",
    "llama3.23binstruct": "vllm:meta-llama/Llama-3.2-3B-Instruct",
    "phi3mini4kinstruct": "vllm:microsoft/Phi-3-mini-4k-instruct",
    "mistral7binstructv0.3": "vllm:mistralai/Mistral-7B-Instruct-v0.3",
}

DATASET_ALIASES = {
    "natural_questions": "nq",
}

# Known prompts (most specific first to avoid partial matches)
KNOWN_PROMPTS = [
    "concise_strict",
    "concise_json",
    "extractive_quoted",
    "cot_final",
    "fewshot_3",
    "fewshot_1",
    "fewshot",
    "concise",
    "structured",
    "cot",
    "extractive",
    "cited",
]

KNOWN_DATASETS = ["hotpotqa", "triviaqa", "nq"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _infer_model_spec(dir_name: str) -> str | None:
    """Infer full model spec from experiment directory name."""
    name_lower = dir_name.lower()
    # Try most specific patterns first (longer keys first)
    for key in sorted(MODEL_REVERSE_MAP.keys(), key=len, reverse=True):
        if key.lower() in name_lower:
            return MODEL_REVERSE_MAP[key]
    return None


def _infer_exp_type(dir_name: str) -> str:
    """Infer experiment type from directory name."""
    if dir_name.startswith("direct_"):
        return "direct"
    return "rag"


def _infer_dataset(dir_name: str) -> str | None:
    """Infer dataset from directory name (appears at end)."""
    for ds in KNOWN_DATASETS:
        if dir_name.endswith(f"_{ds}"):
            return ds
    return None


def _infer_prompt(dir_name: str, dataset: str | None) -> str | None:
    """Infer prompt style from directory name.

    Prompt appears just before the dataset suffix.
    """
    # Remove dataset suffix for easier matching
    search_name = dir_name
    if dataset:
        search_name = dir_name[: -(len(dataset) + 1)]  # strip _dataset

    for prompt in KNOWN_PROMPTS:
        if search_name.endswith(f"_{prompt}"):
            return prompt
    return None


def _infer_top_k(dir_name: str) -> int | None:
    """Extract top_k from _k{N}_ segment."""
    m = re.search(r"_k(\d+)_", dir_name)
    return int(m.group(1)) if m else None


def _infer_retriever(dir_name: str) -> str | None:
    """Extract retriever name from RAG experiment directory name.

    Format: rag_{model}_{retriever}_k{N}_...
    The retriever is between the model spec and _k{N}.
    """
    k_match = re.search(r"_k(\d+)_", dir_name)
    if not k_match:
        return None

    # Find where the model spec ends by looking for known model patterns
    # The retriever starts right after the model and ends at _k{N}
    name_lower = dir_name.lower()
    model_end = 0
    for key in sorted(MODEL_REVERSE_MAP.keys(), key=len, reverse=True):
        idx = name_lower.find(key.lower())
        if idx >= 0:
            model_end = idx + len(key)
            # Skip the underscore after model
            if model_end < len(dir_name) and dir_name[model_end] == "_":
                model_end += 1
            break

    if model_end == 0:
        return None

    retriever = dir_name[model_end : k_match.start()]
    # Clean up: strip trailing underscore
    return retriever.rstrip("_") if retriever else None


# ---------------------------------------------------------------------------
# Repair logic
# ---------------------------------------------------------------------------


def plan_repairs(study_path: Path) -> list[dict]:
    """Scan study and plan metadata repairs.

    Returns a list of repair actions with keys:
        - dir_name: experiment directory name
        - repairs: list of (field, old_value, new_value, reason) tuples
    """
    actions = []

    for exp_dir in sorted(study_path.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in SKIP_DIRS:
            continue

        metadata_path = exp_dir / "metadata.json"
        metadata = _load_json(metadata_path)

        if metadata is None:
            # No metadata.json at all — try to create one from directory name
            repairs = _plan_missing_metadata(exp_dir.name)
            if repairs:
                actions.append(
                    {
                        "dir_name": exp_dir.name,
                        "dir_path": exp_dir,
                        "create_new": True,
                        "repairs": repairs,
                    }
                )
            continue

        repairs = []

        # --- Fix 1: Empty model ---
        current_model = metadata.get("model", "")
        if not current_model:
            inferred = _infer_model_spec(exp_dir.name)
            if inferred:
                repairs.append(("model", current_model, inferred, "inferred from directory name"))

        # --- Fix 1b: Missing type ---
        if not metadata.get("type"):
            exp_type = _infer_exp_type(exp_dir.name)
            repairs.append(
                ("type", metadata.get("type", ""), exp_type, "inferred from directory name")
            )

        # --- Fix 1c: Missing prompt ---
        if not metadata.get("prompt"):
            dataset = metadata.get("dataset") or _infer_dataset(exp_dir.name)
            prompt = _infer_prompt(exp_dir.name, dataset)
            if prompt:
                repairs.append(("prompt", "", prompt, "inferred from directory name"))

        # --- Fix 1d: Missing retriever/top_k for RAG ---
        exp_type = metadata.get("type") or _infer_exp_type(exp_dir.name)
        if exp_type == "rag":
            if not metadata.get("retriever"):
                retriever = _infer_retriever(exp_dir.name)
                if retriever:
                    repairs.append(("retriever", "", retriever, "inferred from directory name"))
            if metadata.get("top_k") is None:
                top_k = _infer_top_k(exp_dir.name)
                if top_k is not None:
                    repairs.append(("top_k", None, top_k, "inferred from directory name"))

        # --- Fix 2: Dataset alias ---
        current_dataset = metadata.get("dataset", "")
        canonical = DATASET_ALIASES.get(current_dataset)
        if canonical:
            repairs.append(("dataset", current_dataset, canonical, "normalize alias"))

        # --- Fix 3: Name mismatch ---
        current_name = metadata.get("name", "")
        if current_name and current_name != exp_dir.name:
            repairs.append(("name", current_name, exp_dir.name, "match directory name"))

        if repairs:
            actions.append(
                {
                    "dir_name": exp_dir.name,
                    "dir_path": exp_dir,
                    "create_new": False,
                    "repairs": repairs,
                }
            )

    return actions


def _plan_missing_metadata(dir_name: str) -> list[tuple]:
    """Plan repairs for an experiment with no metadata.json at all."""
    repairs = []
    model = _infer_model_spec(dir_name)
    exp_type = _infer_exp_type(dir_name)
    dataset = _infer_dataset(dir_name)
    prompt = _infer_prompt(dir_name, dataset)

    repairs.append(("name", None, dir_name, "create from directory name"))
    if model:
        repairs.append(("model", None, model, "inferred from directory name"))
    if exp_type:
        repairs.append(("type", None, exp_type, "inferred from directory name"))
    if dataset:
        repairs.append(("dataset", None, dataset, "inferred from directory name"))
    if prompt:
        repairs.append(("prompt", None, prompt, "inferred from directory name"))

    if exp_type == "rag":
        top_k = _infer_top_k(dir_name)
        retriever = _infer_retriever(dir_name)
        if top_k is not None:
            repairs.append(("top_k", None, top_k, "inferred from directory name"))
        if retriever:
            repairs.append(("retriever", None, retriever, "inferred from directory name"))

    return repairs


def execute_repairs(actions: list[dict]) -> None:
    """Apply planned repairs to metadata.json files."""
    for action in actions:
        metadata_path = action["dir_path"] / "metadata.json"

        if action["create_new"]:
            metadata = {}
        else:
            metadata = _load_json(metadata_path) or {}

        for field, _old, new, _reason in action["repairs"]:
            metadata[field] = new

        _save_json(metadata_path, metadata)

    # Also update results.json name/dataset fields to stay consistent
    for action in actions:
        results_path = action["dir_path"] / "results.json"
        results = _load_json(results_path)
        if results is None:
            continue

        changed = False
        for field, _old, new, _reason in action["repairs"]:
            if field == "name" and results.get("name") != new:
                results["name"] = new
                changed = True
            elif field == "dataset" and results.get("dataset") != new:
                results["dataset"] = new
                changed = True

        if changed:
            _save_json(results_path, results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Repair metadata.json for experiments with missing/inconsistent fields.",
    )
    parser.add_argument(
        "--study",
        type=Path,
        required=True,
        help="Path to the study output directory (e.g. outputs/smart_retrieval_slm)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the repairs (default is dry-run)",
    )
    args = parser.parse_args()

    study_path = args.study
    if not study_path.is_dir():
        print(f"Error: {study_path} does not exist or is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {study_path} for metadata issues...\n")
    actions = plan_repairs(study_path)

    if not actions:
        print("No issues found. All metadata looks good.")
        return

    # Print summary
    total_repairs = sum(len(a["repairs"]) for a in actions)
    print(f"Found {len(actions)} experiments needing {total_repairs} repairs:\n")

    # Group by repair type for high-level summary
    repair_types: dict[str, int] = {}
    for action in actions:
        for field, _old, _new, reason in action["repairs"]:
            key = f"{field} ({reason})"
            repair_types[key] = repair_types.get(key, 0) + 1

    print("Repair summary:")
    for key, count in sorted(repair_types.items(), key=lambda x: -x[1]):
        print(f"  {count:3d}x  {key}")

    # Print details
    print("\nDetails:\n")
    for action in actions:
        prefix = "[CREATE]" if action["create_new"] else "[UPDATE]"
        print(f"  {prefix} {action['dir_name']}")
        for field, old, new, reason in action["repairs"]:
            old_str = repr(old) if old is not None else "<missing>"
            print(f"         {field}: {old_str} -> {new!r}  ({reason})")

    if args.execute:
        print(f"\nExecuting {total_repairs} repairs...")
        execute_repairs(actions)
        print("Done.")
    else:
        print("\n[DRY RUN] No changes made. Use --execute to apply.")


if __name__ == "__main__":
    main()
