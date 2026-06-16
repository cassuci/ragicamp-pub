#!/usr/bin/env python3
"""Migrate experiments that were labelled with a query transform but never actually transformed.

Due to a bug where the query transform (HyDE, MultiQuery) was never wired into the
agent pipeline (the factory just emitted a warning), experiments whose names include
a query_transform suffix (e.g. ``_hyde_``, ``_multiquery_``) are actually plain
retrieval results identical to ``query_transform=none``.  This script:

1. Scans a study output directory for experiments whose name encodes a query_transform.
2. Computes the *corrected* name (without the query_transform segment).
3. If the corrected name already exists, keeps the experiment with the better
   primary metric (F1 by default) and archives the other.
4. Renames the directory to the corrected name.
5. Updates ``metadata.json`` and ``results.json`` to reflect ``query_transform: none``.

Usage:
    # Dry-run (default): only prints what *would* happen
    python scripts/migrate_fake_query_transform.py --study outputs/smart_retrieval_slm

    # Execute the migration
    python scripts/migrate_fake_query_transform.py --study outputs/smart_retrieval_slm --execute
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Known query-transform tokens (must match naming.py append logic)
# ---------------------------------------------------------------------------
# naming.py appends qt as-is: parts.append(query_transform)
# Model names are normalised (hyphens stripped), but qt tokens are short
# lowercase strings that don't contain hyphens.
QUERY_TRANSFORM_TOKENS = {"hyde", "multiquery"}


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def _strip_query_transform_from_name(name: str) -> tuple[str | None, str | None]:
    """Remove the query_transform segment from an experiment name.

    Returns (corrected_name, qt_token) or (None, None) if the experiment
    does not encode a query_transform.

    Experiment name format (from naming.py):
        rag_{model}_{retriever}_k{top_k}[_{qt}][_{reranker}]_{prompt}_{dataset}

    The qt segment appears right after k{N} and before any reranker or prompt.
    """
    parts = name.split("_")

    # Find the _k{N}_ segment
    k_idx = None
    for i, p in enumerate(parts):
        if re.match(r"^k\d+$", p):
            k_idx = i
            break

    if k_idx is None:
        return None, None  # Not a standard RAG name

    # Everything after k_idx up to the last two parts (prompt, dataset) is
    # optional: [query_transform] [reranker].
    tail = parts[k_idx + 1 :]

    if len(tail) < 2:
        return None, None  # Malformed name

    prompt_dataset = tail[-2:]  # last two are prompt, dataset
    optional = tail[:-2]  # everything between k{N} and prompt

    # Check if any optional segment is a query_transform token
    found_qt = None
    cleaned_optional = []
    for seg in optional:
        if seg.lower() in QUERY_TRANSFORM_TOKENS:
            found_qt = seg
            # Skip this segment (remove from name)
        else:
            cleaned_optional.append(seg)

    if found_qt is None:
        return None, None  # No query_transform to strip

    new_parts = parts[: k_idx + 1] + cleaned_optional + prompt_dataset
    return "_".join(new_parts), found_qt


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
        - qt_token: the query_transform token that was stripped
        - action: 'rename', 'merge_keep_old', 'merge_keep_new'
        - detail: human-readable explanation
    """
    actions = []

    if not study_path.is_dir():
        print(f"Error: {study_path} is not a directory", file=sys.stderr)
        return actions

    # Track names that have been claimed by earlier actions so we can
    # detect collisions between multiple fake-qt experiments that
    # map to the same corrected name (e.g. _hyde_ and _multiquery_ both -> none).
    # Maps new_name -> (action_index, f1_of_winner)
    claimed: dict[str, tuple[int, float]] = {}

    for exp_dir in sorted(study_path.iterdir()):
        if not exp_dir.is_dir():
            continue

        old_name = exp_dir.name
        new_name, qt_token = _strip_query_transform_from_name(old_name)

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
            # Conflict with a real (non-fake-qt) experiment on disk
            new_f1 = _get_f1(new_dir)

            if old_f1 >= new_f1:
                idx = len(actions)
                actions.append(
                    {
                        "old_name": old_name,
                        "old_dir": exp_dir,
                        "new_name": new_name,
                        "new_dir": new_dir,
                        "qt_token": qt_token,
                        "action": "merge_keep_old",
                        "detail": (
                            f"Conflict: '{new_name}' already exists on disk. "
                            f"Old ({qt_token}) F1={old_f1:.4f} >= existing F1={new_f1:.4f}. "
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
                        "qt_token": qt_token,
                        "action": "merge_keep_new",
                        "detail": (
                            f"Conflict: '{new_name}' already exists on disk. "
                            f"Existing F1={new_f1:.4f} > old ({qt_token}) F1={old_f1:.4f}. "
                            f"Will archive fake-qt experiment."
                        ),
                    }
                )

        elif already_claimed:
            # Another fake-qt experiment already claimed this name
            # (e.g. _hyde_ and _multiquery_ both map to same corrected name).
            prev_idx, prev_f1 = claimed[new_name]

            if old_f1 > prev_f1:
                # This experiment is better -- demote the previous winner
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
                        "qt_token": qt_token,
                        "action": "rename",
                        "detail": (
                            f"Collision winner: '{old_name}' -> '{new_name}' "
                            f"(F1={old_f1:.4f} > {prev_f1:.4f})"
                        ),
                    }
                )
                claimed[new_name] = (idx, old_f1)
            else:
                # Previous winner is still better -- archive this one
                actions.append(
                    {
                        "old_name": old_name,
                        "old_dir": exp_dir,
                        "new_name": new_name,
                        "new_dir": new_dir,
                        "qt_token": qt_token,
                        "action": "merge_keep_new",
                        "detail": (
                            f"Collision: '{old_name}' ({qt_token}) also maps to '{new_name}' "
                            f"but F1={old_f1:.4f} <= {prev_f1:.4f}. "
                            f"Will archive this one."
                        ),
                    }
                )
        else:
            # No conflict -- simple rename
            idx = len(actions)
            actions.append(
                {
                    "old_name": old_name,
                    "old_dir": exp_dir,
                    "new_name": new_name,
                    "new_dir": new_dir,
                    "qt_token": qt_token,
                    "action": "rename",
                    "detail": f"Rename: '{old_name}' -> '{new_name}' (strip {qt_token})",
                }
            )
            claimed[new_name] = (idx, old_f1)

    return actions


def _update_json_fields(exp_dir: Path, new_name: str) -> None:
    """Update name/query_transform fields in metadata.json and results.json."""
    for fname in ["metadata.json", "results.json"]:
        data = _load_json(exp_dir / fname)
        if data is None:
            continue
        data["name"] = new_name
        if "query_transform" in data:
            data["query_transform"] = None
        _save_json(exp_dir / fname, data)


def execute_migration(study_path: Path, actions: list[dict]) -> None:
    """Execute the planned migration actions."""
    archive_dir = study_path / "_archived_fake_query_transform"

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
                f"(archived existing to _archived_fake_query_transform/)"
            )

        elif action["action"] == "merge_keep_new":
            # Archive the fake-qt experiment
            archive_dir.mkdir(exist_ok=True)
            shutil.move(str(old_dir), str(archive_dir / action["old_name"]))
            print(f"  ARCHIVED: {action['old_name']} (kept existing '{new_name}' with better F1)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Migrate fake query-transform experiments to corrected names.",
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
        print(
            f"Error: {study_path} does not exist or is not a directory",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Scanning {study_path} for fake query-transform experiments...")
    actions = scan_study(study_path)

    if not actions:
        print("No fake query-transform experiments found. Nothing to do.")
        return

    print(f"\nFound {len(actions)} experiments to migrate:\n")
    for i, action in enumerate(actions, 1):
        print(f"  [{i}] {action['detail']}")

    rename_count = sum(1 for a in actions if a["action"] == "rename")
    merge_count = sum(1 for a in actions if a["action"].startswith("merge"))
    archive_count = sum(1 for a in actions if a["action"] == "merge_keep_new")
    print(f"\nSummary: {rename_count} renames, {merge_count} merges ({archive_count} archived)")

    if args.execute:
        print("\nExecuting migration...")
        execute_migration(study_path, actions)
        print(f"\nDone. {len(actions)} experiments processed.")
    else:
        print("\n[DRY RUN] No changes made. Use --execute to apply.")


if __name__ == "__main__":
    main()
