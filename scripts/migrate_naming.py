#!/usr/bin/env python3
"""Migrate experiment directories from old long names to new hash-based names.

Reads metadata.json from each experiment, computes the new canonical name using
the updated naming functions, and renames the directory.

Modes:
    - **dry-run** (default): Print what would change.  No modifications.
    - ``--execute``: Apply renames, update metadata.json and results.json.
    - ``--reset-optuna``: Also delete optuna_study.db so TPE re-seeds from
      renamed directories on the next run.

Usage::

    # Dry run:
    python scripts/migrate_naming.py --study outputs/smart_retrieval_slm

    # Execute:
    python scripts/migrate_naming.py --study outputs/smart_retrieval_slm --execute

    # Execute + reset Optuna:
    python scripts/migrate_naming.py --study outputs/smart_retrieval_slm --execute --reset-optuna
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragicamp.spec.naming import name_direct, name_rag  # noqa: E402

SKIP_DIRS = {
    "_archived_fake_reranked",
    "_tainted",
    "_collisions",
    "_incomplete",
    "_quarantined",
    "analysis",
    "__pycache__",
    ".ipynb_checkpoints",
}


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _compute_new_name(meta: dict) -> str | None:
    """Compute new hash-based name from experiment metadata."""
    exp_type = meta.get("exp_type") or meta.get("type")
    if not exp_type:
        return None

    model = meta.get("model")
    prompt = meta.get("prompt")
    dataset = meta.get("dataset")

    if not all([model, prompt, dataset]):
        return None

    if exp_type == "direct":
        return name_direct(model, prompt, dataset)

    # RAG experiment
    retriever = meta.get("retriever")
    if not retriever:
        return None

    top_k = meta.get("top_k", 5)
    qt = meta.get("query_transform") or "none"
    reranker = meta.get("reranker") or "none"
    rrf_k = meta.get("rrf_k")
    alpha = meta.get("alpha")
    agent_type = meta.get("agent_type")
    agent_params = meta.get("agent_params")

    # Convert agent_params dict to tuple of tuples for hashing consistency
    if isinstance(agent_params, dict) and agent_params:
        ap = tuple(sorted(agent_params.items()))
    else:
        ap = None

    return name_rag(
        model,
        prompt,
        dataset,
        retriever,
        top_k,
        query_transform=qt,
        reranker=reranker,
        rrf_k=rrf_k,
        alpha=alpha,
        agent_type=agent_type,
        agent_params=ap,
    )


def plan_migration(study_path: Path) -> list[tuple[Path, str, str]]:
    """Scan study directory and plan renames.

    Returns list of (exp_dir, old_name, new_name) for experiments that
    need renaming.
    """
    renames: list[tuple[Path, str, str]] = []

    for exp_dir in sorted(study_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name in SKIP_DIRS or exp_dir.name.startswith((".", "_")):
            continue

        meta = _load_json(exp_dir / "metadata.json")
        if meta is None:
            continue

        new_name = _compute_new_name(meta)
        if new_name is None:
            continue

        if new_name != exp_dir.name:
            renames.append((exp_dir, exp_dir.name, new_name))

    return renames


def execute_migration(
    renames: list[tuple[Path, str, str]],
    study_path: Path,
    reset_optuna: bool = False,
) -> None:
    """Execute the planned renames."""
    renamed = 0
    collisions = 0

    for exp_dir, old_name, new_name in renames:
        target = study_path / new_name

        if target.exists():
            print(f"  COLLISION: {old_name} -> {new_name} (target exists, skipping)")
            collisions += 1
            continue

        # Update JSON files
        for fname in ("metadata.json", "results.json"):
            data = _load_json(exp_dir / fname)
            if data is None:
                continue
            data["name"] = new_name
            if fname == "results.json" and isinstance(data.get("metadata"), dict):
                data["metadata"]["name"] = new_name
            _save_json(exp_dir / fname, data)

        # Rename directory
        exp_dir.rename(target)
        print(f"  {old_name} -> {new_name}")
        renamed += 1

    if reset_optuna:
        for fname in ("optuna_study.db", "study_summary.json"):
            fpath = study_path / fname
            if fpath.exists():
                fpath.unlink()
                print(f"  removed: {fname}")

    print(f"\n  Done: {renamed} renamed, {collisions} collisions (skipped).")
    if reset_optuna:
        print("  Optuna DB removed -- TPE will re-seed from renamed experiments.")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate experiment names to new hash-based scheme.",
    )
    parser.add_argument(
        "--study",
        required=True,
        type=Path,
        help="Path to study output directory",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply renames (default: dry-run)",
    )
    parser.add_argument(
        "--reset-optuna",
        action="store_true",
        help="Delete optuna_study.db after migration",
    )
    args = parser.parse_args()

    study_path = args.study
    if not study_path.is_dir():
        print(f"Error: {study_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    renames = plan_migration(study_path)

    if not renames:
        print("No experiments need renaming.")
        return

    print(f"\nMigration plan: {len(renames)} experiments to rename\n")
    for _, old_name, new_name in renames[:50]:
        print(f"  {old_name}")
        print(f"    -> {new_name}")
    if len(renames) > 50:
        print(f"  ... and {len(renames) - 50} more")

    if args.execute:
        confirm = input(f"\nExecute {len(renames)} renames? [y/N] ")
        if confirm.lower() == "y":
            execute_migration(renames, study_path, reset_optuna=args.reset_optuna)
        else:
            print("Aborted.")
    else:
        print("\nDry run. Use --execute to apply changes.")


if __name__ == "__main__":
    main()
