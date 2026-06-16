#!/usr/bin/env python3
"""Diagnose and fix experiment results that were incorrectly named.

Due to bugs where ``query_transform`` and ``reranker`` were configured but
never wired into the agent pipeline, many experiment names encode parameters
that had no effect.  For example::

    rag_..._k5_hyde_bge_concise_nq    # hyde + bge never wired
    rag_..._k5_concise_nq             # ← identical results

This script:

1. **Strips fake tokens** from experiment names (``_hyde_``, ``_multiquery_``,
   ``_bge_``, ``_bgev2_``, etc.) and renames the directory.
2. **Verifies duplicates** by comparing F1 scores between experiments that
   map to the same corrected name (within a tolerance).
3. **Resolves collisions**: when multiple experiments map to the same
   corrected name, keeps the one with the best F1 and archives the rest.
4. **Updates JSON files** (metadata.json, results.json) to reflect the
   corrected name and set ``query_transform`` / ``reranker`` to ``none``.
5. Optionally **resets Optuna** (``--reset-optuna``) to let TPE restart.

Modes:

- **diagnose** (default): Print what would happen.  No changes.
- ``--execute``: Apply renames, archive collisions, update JSON files.

Usage::

    # Dry run (default):
    python scripts/clean_study.py --study outputs/smart_retrieval_slm

    # Execute renames:
    python scripts/clean_study.py --study outputs/smart_retrieval_slm --execute

    # Also reset Optuna DB:
    python scripts/clean_study.py --study outputs/smart_retrieval_slm --execute --reset-optuna

    # Also clean incomplete/failed experiments:
    python scripts/clean_study.py --study outputs/smart_retrieval_slm --execute --include-incomplete
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================================
# Constants
# ============================================================================

SKIP_DIRS = {
    "_archived_fake_reranked",
    "_tainted",
    "_collisions",
    "analysis",
    "__pycache__",
    ".ipynb_checkpoints",
}

KNOWN_RERANKER_TOKENS = {
    "bge",
    "bgev2",
    "bgev2m3",
    "bgebase",
    "msmarco",
    "msmarcolarge",
    # Hyphenated forms (collapsed in names)
    "bge-v2",
    "bge-base",
    "ms-marco",
    "ms-marco-large",
}

KNOWN_QUERY_TRANSFORMS = {"hyde", "multiquery"}

DATASET_ALIASES = {
    "natural_questions": "nq",
}

# F1 tolerance for verifying that "duplicates" really produced the same results
F1_TOLERANCE = 0.015


# ============================================================================
# Name manipulation
# ============================================================================


def _strip_fake_tokens(name: str) -> tuple[str | None, list[str]]:
    """Strip fake reranker and query_transform tokens from an experiment name.

    Returns:
        (corrected_name, list_of_stripped_tokens)
        corrected_name is None if nothing to strip.
    """
    # Skip hash-based names (new naming scheme: prefix_model_dataset_hash8)
    if re.match(r"^(rag|direct|iterative|self)_\w+_\w+_[0-9a-f]{8}$", name):
        return None, []

    parts = name.split("_")

    # Find _k{N}_ segment
    k_idx = None
    for i, p in enumerate(parts):
        if re.match(r"^k\d+$", p):
            k_idx = i
            break

    if k_idx is None:
        return None, []

    tail = parts[k_idx + 1 :]
    if len(tail) < 2:
        return None, []

    prompt_dataset = tail[-2:]
    middle = tail[:-2]

    stripped = []
    cleaned = []
    # Normalise for comparison: collapse hyphens
    rr_normalised = {t.replace("-", "") for t in KNOWN_RERANKER_TOKENS}

    for token in middle:
        token_norm = token.lower().replace("-", "")
        if token_norm in rr_normalised:
            stripped.append(token)
        elif token.lower() in KNOWN_QUERY_TRANSFORMS:
            stripped.append(token)
        else:
            cleaned.append(token)

    if not stripped:
        return None, []

    new_parts = parts[: k_idx + 1] + cleaned + prompt_dataset
    return "_".join(new_parts), stripped


# ============================================================================
# Data model
# ============================================================================


@dataclass
class Action:
    """A planned rename/archive/delete action."""

    old_name: str
    old_dir: Path
    new_name: str
    action: str  # "rename", "archive_collision", "archive_incomplete", "keep"
    detail: str
    f1: float = 0.0
    stripped_tokens: list[str] = field(default_factory=list)
    verified_duplicate: bool = False  # True if F1 matches within tolerance


# ============================================================================
# Helpers
# ============================================================================


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _get_f1(exp_dir: Path) -> float:
    """Read F1 from results.json or predictions.json aggregate_metrics."""
    results = _load_json(exp_dir / "results.json")
    if results:
        metrics = results.get("metrics", {})
        if isinstance(metrics, dict) and metrics.get("f1"):
            return metrics["f1"]

    # Fallback: try predictions.json aggregate
    preds = _load_json(exp_dir / "predictions.json")
    if preds:
        agg = preds.get("aggregate_metrics", {})
        if isinstance(agg, dict) and agg.get("f1"):
            return agg["f1"]

    return 0.0


def _get_phase(exp_dir: Path) -> str | None:
    state = _load_json(exp_dir / "state.json")
    if state:
        return state.get("phase")
    return None


def _update_json_fields(exp_dir: Path, new_name: str, stripped: list[str]) -> None:
    """Update name, reranker, query_transform in metadata/results JSON."""
    rr_normalised = {t.replace("-", "") for t in KNOWN_RERANKER_TOKENS}
    has_rr = any(t.lower().replace("-", "") in rr_normalised for t in stripped)
    has_qt = any(t.lower() in KNOWN_QUERY_TRANSFORMS for t in stripped)

    for fname in ["metadata.json", "results.json"]:
        data = _load_json(exp_dir / fname)
        if data is None:
            continue

        data["name"] = new_name

        if has_rr:
            if "reranker" in data:
                data["reranker"] = None
            if "reranker_model" in data:
                data["reranker_model"] = None

        if has_qt:
            if "query_transform" in data:
                data["query_transform"] = None

        # Also fix nested metadata in results.json
        if fname == "results.json" and "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"]["name"] = new_name
            if has_rr:
                data["metadata"]["reranker"] = None
                data["metadata"]["reranker_model"] = None
            if has_qt:
                data["metadata"]["query_transform"] = None

        _save_json(exp_dir / fname, data)


# ============================================================================
# Planning
# ============================================================================


def plan_actions(
    study_path: Path,
    include_incomplete: bool = False,
) -> list[Action]:
    """Scan the study directory and plan all actions.

    Returns a list of Action objects describing what to do.
    """
    actions: list[Action] = []

    # Phase 1: Collect all experiments and their corrected names
    # Maps corrected_name -> list of (old_name, old_dir, f1, stripped_tokens)
    corrections: dict[str, list[dict]] = defaultdict(list)
    incomplete: list[Path] = []

    for exp_dir in sorted(study_path.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in SKIP_DIRS:
            continue
        if exp_dir.name.startswith("."):
            continue

        phase = _get_phase(exp_dir)

        # Track incomplete/failed experiments
        if phase and phase not in ("complete",):
            incomplete.append(exp_dir)

        corrected, stripped = _strip_fake_tokens(exp_dir.name)

        if corrected is None:
            # Name is already clean — nothing to rename
            continue

        f1 = _get_f1(exp_dir)

        corrections[corrected].append(
            {
                "old_name": exp_dir.name,
                "old_dir": exp_dir,
                "f1": f1,
                "stripped": stripped,
                "phase": phase,
            }
        )

    # Phase 2: Resolve collisions and plan renames
    for corrected_name, candidates in sorted(corrections.items()):
        target_dir = study_path / corrected_name
        target_exists_on_disk = target_dir.exists() and target_dir.name not in [
            c["old_name"] for c in candidates
        ]

        # Collect all competitors: candidates + existing on-disk experiment
        all_competitors = list(candidates)
        if target_exists_on_disk:
            all_competitors.append(
                {
                    "old_name": corrected_name,
                    "old_dir": target_dir,
                    "f1": _get_f1(target_dir),
                    "stripped": [],
                    "phase": _get_phase(target_dir),
                    "is_existing": True,
                }
            )

        # Verify that these are true duplicates (F1 within tolerance)
        f1_values = [c["f1"] for c in all_competitors if c["f1"] > 0]
        if len(f1_values) >= 2:
            f1_range = max(f1_values) - min(f1_values)
            verified = f1_range <= F1_TOLERANCE
        elif len(f1_values) == 1:
            verified = True
        else:
            verified = False  # All zeros (failed experiments)

        # Pick the winner: highest F1, prefer complete
        complete_candidates = [c for c in all_competitors if c.get("phase") == "complete"]
        pool = complete_candidates if complete_candidates else all_competitors
        winner = max(pool, key=lambda c: c["f1"])

        for candidate in candidates:
            if candidate["old_name"] == winner["old_name"]:
                if candidate["old_name"] == corrected_name:
                    # Already has the correct name — nothing to do
                    continue
                actions.append(
                    Action(
                        old_name=candidate["old_name"],
                        old_dir=candidate["old_dir"],
                        new_name=corrected_name,
                        action="rename",
                        detail=(f"Rename: strip {candidate['stripped']} → '{corrected_name}'"),
                        f1=candidate["f1"],
                        stripped_tokens=candidate["stripped"],
                        verified_duplicate=verified,
                    )
                )
            else:
                # Loser in collision
                actions.append(
                    Action(
                        old_name=candidate["old_name"],
                        old_dir=candidate["old_dir"],
                        new_name=corrected_name,
                        action="archive_collision",
                        detail=(
                            f"Collision loser: f1={candidate['f1']:.4f} vs winner "
                            f"f1={winner['f1']:.4f} ('{winner['old_name']}')"
                        ),
                        f1=candidate["f1"],
                        stripped_tokens=candidate["stripped"],
                        verified_duplicate=verified,
                    )
                )

    # Phase 3: Handle incomplete experiments (optional)
    if include_incomplete:
        already_planned = {a.old_name for a in actions}
        for exp_dir in incomplete:
            if exp_dir.name not in already_planned:
                phase = _get_phase(exp_dir) or "unknown"
                actions.append(
                    Action(
                        old_name=exp_dir.name,
                        old_dir=exp_dir,
                        new_name=exp_dir.name,
                        action="archive_incomplete",
                        detail=f"Incomplete experiment: phase={phase}",
                        f1=_get_f1(exp_dir),
                    )
                )

    return actions


# ============================================================================
# Reporting
# ============================================================================

_RED = "\033[91m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def print_report(actions: list[Action], study_path: Path) -> None:
    """Print a human-readable diagnostic report."""
    renames = [a for a in actions if a.action == "rename"]
    collisions = [a for a in actions if a.action == "archive_collision"]
    incomplete_acts = [a for a in actions if a.action == "archive_incomplete"]

    # Count total experiments
    total = sum(
        1
        for d in study_path.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith(".")
    )

    print(f"\n{_BOLD}{'=' * 70}")
    print("  STUDY CLEANUP REPORT")
    print(f"{'=' * 70}{_RESET}\n")

    print(f"  Total experiments on disk: {total}")
    print(f"  {_GREEN}Rename (strip fake tokens): {len(renames)}{_RESET}")
    print(f"  {_YELLOW}Archive (collision losers):  {len(collisions)}{_RESET}")
    if incomplete_acts:
        print(f"  {_YELLOW}Archive (incomplete/failed): {len(incomplete_acts)}{_RESET}")

    # Verify duplicates
    verified = [a for a in renames + collisions if a.verified_duplicate]
    unverified = [a for a in renames + collisions if not a.verified_duplicate]
    if verified:
        print(
            f"\n  {_GREEN}Verified duplicates (F1 within {F1_TOLERANCE}): {len(verified)}{_RESET}"
        )
    if unverified:
        print(f"  {_YELLOW}Unverified (F1 differs or all zero):     {len(unverified)}{_RESET}")

    # --- Renames ---
    if renames:
        print(f"\n{_BOLD}--- RENAMES (strip fake tokens, keep results) ---{_RESET}")
        for a in renames[:50]:  # Cap at 50 for readability
            tokens = ", ".join(a.stripped_tokens)
            check = f"{_GREEN}✓{_RESET}" if a.verified_duplicate else f"{_YELLOW}?{_RESET}"
            print(f"  {check} {_DIM}{a.old_name}{_RESET}")
            print(f"    → {a.new_name}  {_DIM}(strip: {tokens}, f1={a.f1:.4f}){_RESET}")
        if len(renames) > 50:
            print(f"  ... and {len(renames) - 50} more")

    # --- Collisions ---
    if collisions:
        print(f"\n{_BOLD}{_YELLOW}--- COLLISION LOSERS (archive to _collisions/) ---{_RESET}")
        for a in collisions[:30]:
            check = f"{_GREEN}✓{_RESET}" if a.verified_duplicate else f"{_YELLOW}?{_RESET}"
            print(f"  {check} {a.old_name}")
            print(f"    {_DIM}{a.detail}{_RESET}")
        if len(collisions) > 30:
            print(f"  ... and {len(collisions) - 30} more")

    # --- Incomplete ---
    if incomplete_acts:
        print(f"\n{_BOLD}{_YELLOW}--- INCOMPLETE / FAILED (archive to _incomplete/) ---{_RESET}")
        for a in incomplete_acts[:20]:
            print(f"  {_YELLOW}⚠{_RESET} {a.old_name}  {_DIM}({a.detail}){_RESET}")
        if len(incomplete_acts) > 20:
            print(f"  ... and {len(incomplete_acts) - 20} more")

    # Summary
    print(f"\n{_BOLD}--- SUMMARY ---{_RESET}")
    print(f"  Renames:            {_GREEN}{len(renames)}{_RESET}")
    print(f"  Archived (collision): {_YELLOW}{len(collisions)}{_RESET}")
    if incomplete_acts:
        print(f"  Archived (incomplete): {_YELLOW}{len(incomplete_acts)}{_RESET}")
    untouched = total - len(renames) - len(collisions) - len(incomplete_acts)
    print(f"  Untouched:          {untouched}")
    print()


# ============================================================================
# Execution
# ============================================================================


def execute_actions(
    actions: list[Action],
    study_path: Path,
    reset_optuna: bool = False,
) -> None:
    """Execute the planned actions."""
    archive_collision_dir = study_path / "_collisions"
    archive_incomplete_dir = study_path / "_incomplete"

    renamed = 0
    archived = 0

    for action in actions:
        if action.action == "rename":
            target = study_path / action.new_name

            # If the target already exists, the existing experiment at that path
            # is the "winner" that was already there. Archive the current one
            # as a collision if target exists AND is a different directory.
            if target.exists() and target != action.old_dir:
                # Target already exists — archive the loser instead
                archive_collision_dir.mkdir(exist_ok=True)
                dest = archive_collision_dir / action.old_name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(action.old_dir), str(dest))
                print(f"  collision → _collisions/{action.old_name}")
                archived += 1
                continue

            # Update JSON fields first (while dir still has old name)
            _update_json_fields(action.old_dir, action.new_name, action.stripped_tokens)

            # Rename directory
            action.old_dir.rename(target)
            print(f"  renamed: {action.old_name} → {action.new_name}")
            renamed += 1

        elif action.action == "archive_collision":
            archive_collision_dir.mkdir(exist_ok=True)
            dest = archive_collision_dir / action.old_name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(action.old_dir), str(dest))
            print(f"  archived (collision): {action.old_name}")
            archived += 1

        elif action.action == "archive_incomplete":
            archive_incomplete_dir.mkdir(exist_ok=True)
            dest = archive_incomplete_dir / action.old_name
            if dest.exists():
                shutil.rmtree(dest)
            if action.old_dir.exists():
                shutil.move(str(action.old_dir), str(dest))
                print(f"  archived (incomplete): {action.old_name}")
                archived += 1

    # Reset Optuna + study summaries
    if reset_optuna:
        for fname in ["optuna_study.db", "study_summary.json", "comparison.json"]:
            fpath = study_path / fname
            if fpath.exists():
                fpath.unlink()
                print(f"  removed: {fname}")

    # Final summary
    remaining = sum(
        1
        for d in study_path.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith((".", "_"))
    )
    print(f"\n  Done: {renamed} renamed, {archived} archived.")
    print(f"  Clean experiments remaining: {remaining}")
    if reset_optuna:
        print("  Optuna DB removed — TPE will re-seed from clean experiments on next run.")
    print()


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose and fix incorrectly-named experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Apply renames and archive collisions (default: dry-run)",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Also archive incomplete/failed experiments",
    )
    parser.add_argument(
        "--reset-optuna",
        action="store_true",
        help="Remove optuna_study.db and study summaries",
    )
    args = parser.parse_args()

    study_path = args.study
    if not study_path.is_dir():
        print(f"Error: {study_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    actions = plan_actions(study_path, include_incomplete=args.include_incomplete)
    print_report(actions, study_path)

    if not actions:
        print("Nothing to do.")
        return

    if args.execute:
        renames = sum(1 for a in actions if a.action == "rename")
        collisions = sum(1 for a in actions if a.action == "archive_collision")
        inc = sum(1 for a in actions if a.action == "archive_incomplete")

        prompt_parts = []
        if renames:
            prompt_parts.append(f"{renames} renames")
        if collisions:
            prompt_parts.append(f"{collisions} collision archives")
        if inc:
            prompt_parts.append(f"{inc} incomplete archives")

        confirm = input(f"Execute {', '.join(prompt_parts)}? [y/N] ")
        if confirm.lower() == "y":
            execute_actions(actions, study_path, reset_optuna=args.reset_optuna)
        else:
            print("Aborted.")
    else:
        print(f"{_DIM}Dry run. Use --execute to apply changes.{_RESET}\n")


if __name__ == "__main__":
    main()
