#!/usr/bin/env python3
"""Migration script for 2026-02-09 audit fixes.

Scans existing experiment data and fixes known issues:

1. Duplicate predictions (B4): Dedup by idx, keeping last occurrence
2. Failed-but-complete experiments (B3): Flag experiments where state says
   COMPLETE but predictions have errors
3. Stale checkpoints (A1): Remove agent_checkpoint.json files
4. BERTScore key migration (B6): Add "bertscore" key from "bertscore_f1"

Usage:
    # Dry run (report only, no changes)
    python scripts/migrate_audit_fixes.py outputs/smart_retrieval_slm

    # Apply fixes
    python scripts/migrate_audit_fixes.py outputs/smart_retrieval_slm --apply

    # Verbose output
    python scripts/migrate_audit_fixes.py outputs/smart_retrieval_slm --apply -v
"""

import argparse
import json
import sys
from pathlib import Path


def scan_experiment(exp_dir: Path, verbose: bool = False) -> dict:
    """Scan a single experiment for issues.

    Returns dict with issue counts and details.
    """
    issues = {
        "duplicate_predictions": 0,
        "failed_but_complete": False,
        "stale_checkpoint": False,
        "bertscore_key_missing": False,
        "details": [],
    }

    # Check for stale checkpoint
    checkpoint = exp_dir / "agent_checkpoint.json"
    if checkpoint.exists():
        issues["stale_checkpoint"] = True
        issues["details"].append("Has stale agent_checkpoint.json")

    # Check predictions for duplicates and errors
    predictions_path = exp_dir / "predictions.json"
    if predictions_path.exists():
        try:
            with open(predictions_path) as f:
                data = json.load(f)
            preds = data.get("predictions", [])

            # Check for duplicate idx
            seen_idx = set()
            dupes = 0
            for p in preds:
                idx = p.get("idx")
                if idx in seen_idx:
                    dupes += 1
                seen_idx.add(idx)
            issues["duplicate_predictions"] = dupes
            if dupes > 0:
                issues["details"].append(f"{dupes} duplicate predictions")

            # Check for error predictions
            errors = sum(
                1
                for p in preds
                if "[ERROR" in str(p.get("prediction", ""))
                or "[ABORTED" in str(p.get("prediction", ""))
                or p.get("error")
            )
            if errors > 0:
                issues["details"].append(f"{errors} error/aborted predictions")

            # Check BERTScore key
            agg = data.get("aggregate_metrics", {})
            if "bertscore_f1" in agg and "bertscore" not in agg:
                issues["bertscore_key_missing"] = True
                issues["details"].append("Missing 'bertscore' key (has bertscore_f1)")

        except (json.JSONDecodeError, OSError) as e:
            issues["details"].append(f"Cannot read predictions.json: {e}")

    # Check for failed-but-complete state
    state_path = exp_dir / "state.json"
    if state_path.exists() and predictions_path.exists():
        try:
            with open(state_path) as f:
                state = json.load(f)
            phase = state.get("phase", "")

            if phase == "complete":
                with open(predictions_path) as f:
                    data = json.load(f)
                preds = data.get("predictions", [])
                errors = sum(
                    1
                    for p in preds
                    if "[ERROR" in str(p.get("prediction", ""))
                    or "[ABORTED" in str(p.get("prediction", ""))
                )
                if errors > 0:
                    issues["failed_but_complete"] = True
                    issues["details"].append(
                        f"State says COMPLETE but {errors} predictions have errors"
                    )
        except (json.JSONDecodeError, OSError):
            pass

    return issues


def fix_experiment(exp_dir: Path, verbose: bool = False) -> list[str]:
    """Apply fixes to a single experiment. Returns list of actions taken."""
    actions = []

    # Fix 1: Remove stale checkpoint
    checkpoint = exp_dir / "agent_checkpoint.json"
    if checkpoint.exists():
        checkpoint.unlink()
        actions.append("Removed stale agent_checkpoint.json")

    # Fix 2: Dedup predictions
    predictions_path = exp_dir / "predictions.json"
    if predictions_path.exists():
        try:
            with open(predictions_path) as f:
                data = json.load(f)
            preds = data.get("predictions", [])

            # Dedup: keep last occurrence per idx
            seen = {}
            for p in preds:
                seen[p.get("idx")] = p  # Later overwrites earlier
            deduped = sorted(seen.values(), key=lambda p: p.get("idx", 0))

            if len(deduped) < len(preds):
                removed = len(preds) - len(deduped)
                data["predictions"] = deduped
                # Atomic write
                tmp = predictions_path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(data, f, indent=2)
                tmp.replace(predictions_path)
                actions.append(f"Removed {removed} duplicate predictions")

            # Fix 3: Add BERTScore key
            agg = data.get("aggregate_metrics", {})
            if "bertscore_f1" in agg and "bertscore" not in agg:
                agg["bertscore"] = agg["bertscore_f1"]
                data["aggregate_metrics"] = agg
                tmp = predictions_path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(data, f, indent=2)
                tmp.replace(predictions_path)
                actions.append("Added 'bertscore' key from bertscore_f1")

            # Also fix results.json if it has the same issue
            results_path = exp_dir / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    rdata = json.load(f)
                rmetrics = rdata.get("metrics", {})
                if "bertscore_f1" in rmetrics and "bertscore" not in rmetrics:
                    rmetrics["bertscore"] = rmetrics["bertscore_f1"]
                    rdata["metrics"] = rmetrics
                    tmp = results_path.with_suffix(".tmp")
                    with open(tmp, "w") as f:
                        json.dump(rdata, f, indent=2)
                    tmp.replace(results_path)
                    actions.append("Added 'bertscore' key to results.json")

        except (json.JSONDecodeError, OSError) as e:
            actions.append(f"ERROR reading predictions: {e}")

    # Fix 4: Update state if failed-but-complete
    state_path = exp_dir / "state.json"
    if state_path.exists() and predictions_path.exists():
        try:
            with open(state_path) as f:
                state = json.load(f)

            if state.get("phase") == "complete":
                with open(predictions_path) as f:
                    pdata = json.load(f)
                preds = pdata.get("predictions", [])
                errors = sum(
                    1
                    for p in preds
                    if "[ERROR" in str(p.get("prediction", ""))
                    or "[ABORTED" in str(p.get("prediction", ""))
                )
                if errors > 0:
                    state["phase"] = "failed"
                    state["error"] = f"{errors} predictions have errors"
                    tmp = state_path.with_suffix(".tmp")
                    with open(tmp, "w") as f:
                        json.dump(state, f, indent=2)
                    tmp.replace(state_path)
                    actions.append(f"Set state to FAILED ({errors} error predictions)")

        except (json.JSONDecodeError, OSError):
            pass

    return actions


def main():
    parser = argparse.ArgumentParser(
        description="Migrate experiment data for 2026-02-09 audit fixes"
    )
    parser.add_argument("study_dir", type=Path, help="Path to study output directory")
    parser.add_argument("--apply", action="store_true", help="Apply fixes (default: dry run)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not args.study_dir.is_dir():
        print(f"Error: {args.study_dir} is not a directory")
        return 1

    # Find all experiment directories
    exp_dirs = sorted(
        d for d in args.study_dir.iterdir() if d.is_dir() and (d / "state.json").exists()
    )

    if not exp_dirs:
        print(f"No experiments found in {args.study_dir}")
        return 1

    print(f"Scanning {len(exp_dirs)} experiments in {args.study_dir}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}\n")

    total_issues = {
        "duplicate_predictions": 0,
        "failed_but_complete": 0,
        "stale_checkpoints": 0,
        "bertscore_key_missing": 0,
    }
    total_actions = 0

    for exp_dir in exp_dirs:
        issues = scan_experiment(exp_dir, args.verbose)

        has_issues = bool(issues["details"])
        if has_issues:
            print(f"  {exp_dir.name}:")
            for detail in issues["details"]:
                print(f"    - {detail}")

            total_issues["duplicate_predictions"] += issues["duplicate_predictions"]
            total_issues["failed_but_complete"] += int(issues["failed_but_complete"])
            total_issues["stale_checkpoints"] += int(issues["stale_checkpoint"])
            total_issues["bertscore_key_missing"] += int(issues["bertscore_key_missing"])

            if args.apply:
                actions = fix_experiment(exp_dir, args.verbose)
                for action in actions:
                    print(f"    [FIXED] {action}")
                total_actions += len(actions)

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Experiments scanned: {len(exp_dirs)}")
    print(f"  Duplicate predictions: {total_issues['duplicate_predictions']}")
    print(f"  Failed-but-complete: {total_issues['failed_but_complete']}")
    print(f"  Stale checkpoints: {total_issues['stale_checkpoints']}")
    print(f"  BERTScore key missing: {total_issues['bertscore_key_missing']}")
    if args.apply:
        print(f"  Actions taken: {total_actions}")
    else:
        print("\n  Run with --apply to fix issues")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
