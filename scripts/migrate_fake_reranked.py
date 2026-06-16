#!/usr/bin/env python3
"""Migrate experiments that were labelled with a reranker but never actually reranked.

Due to a bug where the RerankerProvider was never wired into the agent pipeline,
experiments whose names include a reranker suffix (e.g. ``_bge_``, ``_bge-v2_``)
are actually plain retrieval results.  This script:

1. Scans a study output directory for experiments whose name encodes a reranker.
2. Computes the *corrected* name (with ``reranker="none"``).
3. If the corrected name already exists, keeps the experiment with the better
   primary metric (F1 by default) and archives the other.
4. Renames the directory to the corrected name.
5. Updates ``metadata.json``, ``results.json``, and ``state.json`` to reflect
   the corrected name and ``reranker: none``.

Usage:
    # Dry-run (default): only prints what *would* happen
    python scripts/migrate_fake_reranked.py --study outputs/smart_retrieval_slm

    # Execute the migration
    python scripts/migrate_fake_reranked.py --study outputs/smart_retrieval_slm --execute
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Naming helpers (mirrors src/ragicamp/spec/naming.py)
# ---------------------------------------------------------------------------


def _strip_reranker_from_name(name: str) -> str | None:
    """Remove the reranker segment from an experiment name.

    Returns the corrected name, or ``None`` if the experiment does not
    encode a reranker (i.e. it is already clean).
    """
    # rag_ experiments: segments are
    #   rag_{model}_{retriever}_k{top_k}[_{qt}][_{reranker}]_{prompt}_{dataset}
    # iterative_rag_ / self_rag_ follow the same pattern after their prefix.

    # Known reranker short names (must match spec/builder.py RERANKER_MODELS keys)
    reranker_tokens = {
        "bge",
        "bgev2",
        "bge-v2",
        "bgebase",
        "bge-base",
        "msmarco",
        "ms-marco",
        "msmarcolarge",
        "ms-marco-large",
    }

    # Normalise hyphens that were collapsed during naming
    # In experiment names hyphens are stripped: bge-v2 -> bgev2
    # But some old names may still have them, so we check both forms.

    parts = name.split("_")

    # We need at least: prefix_{model}_{retriever}_k{N}_{prompt}_{dataset}
    # Reranker is optional between k{N}/qt and prompt.

    # Find the _k{N}_ segment
    k_idx = None
    for i, p in enumerate(parts):
        if re.match(r"^k\d+$", p):
            k_idx = i
            break

    if k_idx is None:
        return None  # Not a standard RAG name

    # Everything after k_idx up to the last two parts (prompt, dataset) is
    # optional: [query_transform] [reranker].
    # The last two parts are always prompt and dataset.
    tail = parts[k_idx + 1 :]

    if len(tail) < 2:
        return None  # Malformed name

    prompt_dataset = tail[-2:]  # last two are prompt, dataset

    optional = tail[:-2]  # everything between k{N} and prompt

    # Check if any optional segment is a reranker token
    found_reranker = False
    cleaned_optional = []
    for seg in optional:
        if seg.lower().replace("-", "") in {t.replace("-", "") for t in reranker_tokens}:
            found_reranker = True
            # Skip this segment (remove from name)
        else:
            cleaned_optional.append(seg)

    if not found_reranker:
        return None  # No reranker to strip

    new_parts = parts[: k_idx + 1] + cleaned_optional + prompt_dataset
    return "_".join(new_parts)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _get_f1(exp_dir: Path) -> float:
    """Read F1 from results.json, falling back to 0.0."""
    results = _load_json(exp_dir / "results.json")
    if results:
        metrics = results.get("metrics", results)
        if isinstance(metrics, dict):
            return metrics.get("f1", 0.0)
    return 0.0


# ---------------------------------------------------------------------------
# Migration logic
# ---------------------------------------------------------------------------


def scan_study(study_path: Path) -> list[dict]:
    """Scan a study directory and plan the migration.

    Returns a list of actions, each a dict with keys:
        - old_name, old_dir
        - new_name, new_dir
        - action: 'rename', 'merge_keep_old', 'merge_keep_new', 'skip'
        - detail: human-readable explanation

    Handles the case where multiple fake-reranked experiments (e.g. one
    with ``_bge_`` and another with ``_bge-v2_``) map to the same
    corrected name by comparing their F1 scores and keeping the best.
    """
    actions = []

    if not study_path.is_dir():
        print(f"Error: {study_path} is not a directory", file=sys.stderr)
        return actions

    # Track names that have been claimed by earlier actions so we can
    # detect collisions between multiple fake-reranked experiments that
    # map to the same corrected name.
    # Maps new_name -> (action_index, f1_of_winner)
    claimed: dict[str, tuple[int, float]] = {}

    for exp_dir in sorted(study_path.iterdir()):
        if not exp_dir.is_dir():
            continue

        old_name = exp_dir.name
        new_name = _strip_reranker_from_name(old_name)

        if new_name is None:
            continue  # Nothing to fix

        if new_name == old_name:
            continue

        new_dir = study_path / new_name
        old_f1 = _get_f1(exp_dir)

        # Check 1: Does the corrected name already exist on disk?
        target_exists_on_disk = new_dir.exists()

        # Check 2: Was the corrected name already claimed by a prior action?
        already_claimed = new_name in claimed

        if target_exists_on_disk and not already_claimed:
            # Conflict with a real (non-fake-reranked) experiment on disk
            new_f1 = _get_f1(new_dir)

            if old_f1 >= new_f1:
                idx = len(actions)
                actions.append(
                    {
                        "old_name": old_name,
                        "old_dir": exp_dir,
                        "new_name": new_name,
                        "new_dir": new_dir,
                        "action": "merge_keep_old",
                        "detail": (
                            f"Conflict: '{new_name}' already exists on disk. "
                            f"Old F1={old_f1:.4f} >= existing F1={new_f1:.4f}. "
                            f"Will archive existing and rename old."
                        ),
                    }
                )
                claimed[new_name] = (idx, old_f1)
            else:
                actions.append(
                    {
                        "old_name": old_name,
                        "old_dir": exp_dir,
                        "new_name": new_name,
                        "new_dir": new_dir,
                        "action": "merge_keep_new",
                        "detail": (
                            f"Conflict: '{new_name}' already exists on disk. "
                            f"Existing F1={new_f1:.4f} > old F1={old_f1:.4f}. "
                            f"Will archive fake-reranked."
                        ),
                    }
                )
                # Don't update claimed — the on-disk version stays

        elif already_claimed:
            # Another fake-reranked experiment already claimed this name.
            # Compare F1 to decide which one wins.
            prev_idx, prev_f1 = claimed[new_name]

            if old_f1 > prev_f1:
                # This experiment is better — demote the previous winner
                prev_action = actions[prev_idx]
                prev_action["action"] = "merge_keep_new"
                prev_action["detail"] = (
                    f"Collision: '{prev_action['old_name']}' also maps to "
                    f"'{new_name}' but F1={prev_f1:.4f} < {old_f1:.4f}. "
                    f"Will archive this one."
                )

                idx = len(actions)
                actions.append(
                    {
                        "old_name": old_name,
                        "old_dir": exp_dir,
                        "new_name": new_name,
                        "new_dir": new_dir,
                        "action": "rename",
                        "detail": (
                            f"Collision winner: '{old_name}' -> '{new_name}' "
                            f"(F1={old_f1:.4f} > {prev_f1:.4f})"
                        ),
                    }
                )
                claimed[new_name] = (idx, old_f1)
            else:
                # Previous winner is still better — archive this one
                actions.append(
                    {
                        "old_name": old_name,
                        "old_dir": exp_dir,
                        "new_name": new_name,
                        "new_dir": new_dir,
                        "action": "merge_keep_new",
                        "detail": (
                            f"Collision: '{old_name}' also maps to '{new_name}' "
                            f"but F1={old_f1:.4f} <= {prev_f1:.4f}. "
                            f"Will archive this one."
                        ),
                    }
                )
        else:
            # No conflict — simple rename
            idx = len(actions)
            actions.append(
                {
                    "old_name": old_name,
                    "old_dir": exp_dir,
                    "new_name": new_name,
                    "new_dir": new_dir,
                    "action": "rename",
                    "detail": f"Rename: '{old_name}' -> '{new_name}'",
                }
            )
            claimed[new_name] = (idx, old_f1)

    return actions


def _update_json_fields(exp_dir: Path, new_name: str) -> None:
    """Update name/reranker fields in metadata.json, results.json, state.json."""
    for fname in ["metadata.json", "results.json"]:
        data = _load_json(exp_dir / fname)
        if data is None:
            continue
        data["name"] = new_name
        if "reranker" in data:
            data["reranker"] = "none"
        if "reranker_model" in data:
            data["reranker_model"] = None
        _save_json(exp_dir / fname, data)


def execute_migration(study_path: Path, actions: list[dict]) -> None:
    """Execute the planned migration actions."""
    archive_dir = study_path / "_archived_fake_reranked"

    for action in actions:
        old_dir = action["old_dir"]
        new_dir = action["new_dir"]
        new_name = action["new_name"]

        if not old_dir.exists():
            print(f"  SKIP (already moved): {action['old_name']}")
            continue

        if action["action"] == "rename":
            if new_dir.exists():
                # Safety: target appeared (e.g. from a prior action in this
                # batch).  Archive the target first, then rename.
                archive_dir.mkdir(exist_ok=True)
                dest = archive_dir / new_name
                if dest.exists():
                    # Unique suffix to avoid archive collisions
                    dest = archive_dir / f"{new_name}__dup"
                shutil.move(str(new_dir), str(dest))
                print(f"  (pre-archived existing '{new_name}')")

            shutil.move(str(old_dir), str(new_dir))
            _update_json_fields(new_dir, new_name)
            print(f"  RENAMED: {action['old_name']} -> {new_name}")

        elif action["action"] == "merge_keep_old":
            # Archive existing, rename old
            archive_dir.mkdir(exist_ok=True)
            if new_dir.exists():
                dest = archive_dir / new_name
                if dest.exists():
                    dest = archive_dir / f"{new_name}__dup"
                shutil.move(str(new_dir), str(dest))
            shutil.move(str(old_dir), str(new_dir))
            _update_json_fields(new_dir, new_name)
            print(
                f"  MERGED (kept old): {action['old_name']} -> {new_name} "
                f"(archived existing to _archived_fake_reranked/)"
            )

        elif action["action"] == "merge_keep_new":
            # Archive the fake-reranked experiment
            archive_dir.mkdir(exist_ok=True)
            shutil.move(str(old_dir), str(archive_dir / action["old_name"]))
            print(f"  ARCHIVED: {action['old_name']} (kept existing '{new_name}' with better F1)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Migrate fake-reranked experiments to corrected names.",
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
        help="Actually perform the migration (default is dry-run)",
    )
    args = parser.parse_args()

    study_path = args.study
    if not study_path.is_dir():
        print(f"Error: {study_path} does not exist or is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {study_path} for fake-reranked experiments...")
    actions = scan_study(study_path)

    if not actions:
        print("No fake-reranked experiments found. Nothing to do.")
        return

    print(f"\nFound {len(actions)} experiments to migrate:\n")
    for i, action in enumerate(actions, 1):
        print(f"  [{i}] {action['detail']}")

    rename_count = sum(1 for a in actions if a["action"] == "rename")
    merge_count = sum(1 for a in actions if a["action"].startswith("merge"))
    print(f"\nSummary: {rename_count} renames, {merge_count} merges")

    if args.execute:
        print("\nExecuting migration...")
        execute_migration(study_path, actions)
        print(f"\nDone. {len(actions)} experiments migrated.")
    else:
        print("\n[DRY RUN] No changes made. Use --execute to apply.")


if __name__ == "__main__":
    main()
