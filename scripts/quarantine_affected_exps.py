#!/usr/bin/env python3
"""Quarantine experiments affected by known bugs.

Scans experiment directories, reads metadata.json to identify experiments
affected by bugs found in the 2026-02-09 post-audit investigation:

1. HYBRID_RRF: Experiments using hybrid retrievers had zero-score sparse
   documents inflating RRF rankings, contaminating top-k results.

2. ITERATIVE_HYDE: Experiments using iterative_rag with max_iterations > 1
   AND query_transform != none. HyDE/multiquery was only applied on
   iteration 0, making cross-agent comparisons unfair.
   (max_iterations=1 is equivalent to fixed_rag, so those are fine.)

3. ITERATIVE_RERANK: Same issue for reranking — only applied on iteration 0
   when max_iterations > 1.

Actions:
  --report   Show affected experiments (default)
  --move     Move affected dirs to {study_dir}/_quarantined/
  --rename   Prefix affected dirs with _q_ (in-place)

Usage:
    python scripts/quarantine_affected_exps.py outputs/smart_retrieval_slm
    python scripts/quarantine_affected_exps.py outputs/smart_retrieval_slm --move
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REASONS = {
    "HYBRID_RRF": "hybrid retriever with RRF zero-score padding bug",
    "ITERATIVE_HYDE": "iterative_rag (iters>1) + query_transform — only applied on iter 0",
    "ITERATIVE_RERANK": "iterative_rag (iters>1) + reranker — only applied on iter 0",
}


def classify_experiment(exp_dir: Path) -> list[str]:
    """Check an experiment directory for known issues.

    Returns list of issue tags (empty if clean).
    """
    issues = []

    # Try metadata.json first (has spec fields), fall back to state.json
    metadata_path = exp_dir / "metadata.json"
    if not metadata_path.exists():
        return issues

    try:
        with open(metadata_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return issues

    retriever = meta.get("retriever") or ""
    agent_type = meta.get("agent_type") or "fixed_rag"
    query_transform = meta.get("query_transform")
    reranker = meta.get("reranker")

    # Get agent_params — may be in metadata directly or nested
    agent_params = meta.get("agent_params") or {}
    max_iterations = agent_params.get("max_iterations", 1)

    # Issue 1: hybrid retriever → RRF zero-score bug
    if "hybrid" in retriever.lower():
        issues.append("HYBRID_RRF")

    # Issue 2: iterative_rag + query_transform + max_iterations > 1
    if (
        agent_type == "iterative_rag"
        and max_iterations > 1
        and query_transform
        and query_transform not in ("none", None)
    ):
        issues.append("ITERATIVE_HYDE")

    # Issue 3: iterative_rag + reranker + max_iterations > 1
    if (
        agent_type == "iterative_rag"
        and max_iterations > 1
        and reranker
        and reranker not in ("none", None)
    ):
        issues.append("ITERATIVE_RERANK")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Quarantine experiments affected by known bugs")
    parser.add_argument("study_dir", type=Path, help="Path to study output directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--move",
        action="store_true",
        help="Move affected dirs to _quarantined/ subfolder",
    )
    group.add_argument(
        "--rename",
        action="store_true",
        help="Prefix affected dirs with _q_ in-place",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.study_dir.is_dir():
        print(f"Error: {args.study_dir} is not a directory")
        return 1

    # Find experiment directories (have metadata.json or state.json)
    exp_dirs = sorted(
        d
        for d in args.study_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "metadata.json").exists()
    )

    if not exp_dirs:
        print(f"No experiments found in {args.study_dir}")
        return 1

    mode = "MOVE" if args.move else "RENAME" if args.rename else "REPORT"
    print(f"Scanning {len(exp_dirs)} experiments in {args.study_dir}")
    print(f"Mode: {mode}\n")

    # Classify all experiments
    affected: dict[str, list[str]] = {}  # dir_name → list of issue tags
    clean_count = 0

    for exp_dir in exp_dirs:
        issues = classify_experiment(exp_dir)
        if issues:
            affected[exp_dir.name] = issues
        else:
            clean_count += 1

    # Report
    by_issue: dict[str, list[str]] = {tag: [] for tag in REASONS}
    for name, issues in affected.items():
        for issue in issues:
            by_issue[issue].append(name)

    print("Issue Summary:")
    for tag, desc in REASONS.items():
        names = by_issue[tag]
        print(f"  {tag}: {len(names)} experiments")
        print(f"    ({desc})")
        if args.verbose and names:
            for n in names[:5]:
                print(f"      - {n}")
            if len(names) > 5:
                print(f"      ... and {len(names) - 5} more")

    print(f"\n  Clean: {clean_count}")
    print(f"  Affected: {len(affected)}")
    print()

    if not affected:
        print("No experiments need quarantining.")
        return 0

    # Apply action
    if args.move:
        quarantine_dir = args.study_dir / "_quarantined"
        quarantine_dir.mkdir(exist_ok=True)

        moved = 0
        for name, issues in sorted(affected.items()):
            src = args.study_dir / name
            dst = quarantine_dir / name
            if dst.exists():
                print(f"  SKIP {name} (already in _quarantined/)")
                continue
            shutil.move(str(src), str(dst))

            # Write a reason file
            reason_path = dst / "_quarantine_reason.json"
            with open(reason_path, "w") as f:
                json.dump(
                    {"issues": issues, "reasons": [REASONS[i] for i in issues]},
                    f,
                    indent=2,
                )
            moved += 1
            if args.verbose:
                print(f"  MOVED {name} → _quarantined/ ({', '.join(issues)})")

        print(f"\nMoved {moved} experiments to {quarantine_dir}")

    elif args.rename:
        renamed = 0
        for name, issues in sorted(affected.items()):
            src = args.study_dir / name
            dst = args.study_dir / f"_q_{name}"
            if dst.exists():
                print(f"  SKIP {name} (already renamed)")
                continue
            src.rename(dst)
            renamed += 1
            if args.verbose:
                print(f"  RENAMED {name} → _q_{name} ({', '.join(issues)})")

        print(f"\nRenamed {renamed} experiments with _q_ prefix")

    else:
        # Report mode — list all affected
        print("Affected experiments:")
        for name, issues in sorted(affected.items()):
            print(f"  {name}")
            for issue in issues:
                print(f"    [{issue}] {REASONS[issue]}")

        print("\nRun with --move or --rename to quarantine these experiments.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
