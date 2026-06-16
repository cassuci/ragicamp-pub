#!/usr/bin/env python
"""Post-hoc retrieval analysis using saved predictions.

Analyzes whether the answer appeared in retrieved context
and correlates with generation success.

Usage:
    python scripts/analyze_retrieval.py outputs/smart_retrieval_slm
    python scripts/analyze_retrieval.py outputs/smart_retrieval_slm --experiment rag_*_nq
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievalStats:
    """Stats for a single experiment."""

    name: str
    total: int = 0
    answer_in_context: int = 0  # Answer text found in prompt context
    correct_with_context: int = 0  # Got it right AND answer was in context
    correct_without_context: int = 0  # Got it right but answer NOT in context (lucky?)
    wrong_with_context: int = 0  # Had the answer but still wrong (generation fail)
    wrong_without_context: int = 0  # Didn't have answer, got it wrong (retrieval fail)

    @property
    def retrieval_recall(self) -> float:
        """% of questions where answer appeared in context."""
        return self.answer_in_context / self.total if self.total else 0

    @property
    def generation_given_context(self) -> float:
        """% correct when answer was in context (generation quality)."""
        total_with = self.correct_with_context + self.wrong_with_context
        return self.correct_with_context / total_with if total_with else 0

    @property
    def bottleneck(self) -> str:
        """Identify the main bottleneck."""
        if self.retrieval_recall < 0.5:
            return "RETRIEVAL"
        elif self.generation_given_context < 0.5:
            return "GENERATION"
        else:
            return "BALANCED"


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def answer_in_prompt(answer: str, prompt: str) -> bool:
    """Check if any answer variant appears in the prompt context."""
    if not prompt or not answer:
        return False

    prompt_normalized = normalize_text(prompt)

    # Handle multiple possible answers (list)
    if isinstance(answer, list):
        answers = answer
    else:
        answers = [answer]

    for ans in answers:
        ans_normalized = normalize_text(str(ans))
        if len(ans_normalized) < 2:
            continue
        if ans_normalized in prompt_normalized:
            return True

    return False


def is_correct(prediction: str, expected: list[str], threshold: float = 0.5) -> bool:
    """Check if prediction matches any expected answer (fuzzy)."""
    if not prediction or not expected:
        return False

    pred_normalized = normalize_text(prediction)

    for exp in expected:
        exp_normalized = normalize_text(str(exp))
        if not exp_normalized:
            continue
        # Check substring match in either direction
        if exp_normalized in pred_normalized or pred_normalized in exp_normalized:
            return True
        # Check word overlap
        pred_words = set(pred_normalized.split())
        exp_words = set(exp_normalized.split())
        if exp_words and pred_words:
            overlap = len(pred_words & exp_words) / len(exp_words)
            if overlap >= threshold:
                return True

    return False


def analyze_experiment(exp_dir: Path) -> RetrievalStats | None:
    """Analyze a single experiment directory."""
    pred_file = exp_dir / "predictions.json"
    if not pred_file.exists():
        return None

    with open(pred_file) as f:
        data = json.load(f)

    preds = data.get("predictions", [])
    if not preds:
        return None

    # Skip direct experiments (no retrieval)
    metadata_file = exp_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        if metadata.get("type") == "direct":
            return None

    stats = RetrievalStats(name=exp_dir.name)

    for p in preds:
        prompt = p.get("prompt", "")
        prediction = p.get("prediction", "")
        expected = p.get("expected", [])

        if not expected:
            continue

        stats.total += 1
        has_answer = answer_in_prompt(expected, prompt)
        got_correct = is_correct(prediction, expected)

        if has_answer:
            stats.answer_in_context += 1
            if got_correct:
                stats.correct_with_context += 1
            else:
                stats.wrong_with_context += 1
        else:
            if got_correct:
                stats.correct_without_context += 1
            else:
                stats.wrong_without_context += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze retrieval quality post-hoc")
    parser.add_argument("study_dir", help="Path to study output directory")
    parser.add_argument("--experiment", "-e", help="Filter experiments by pattern (glob)")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    args = parser.parse_args()

    study_dir = Path(args.study_dir)
    if not study_dir.exists():
        print(f"Directory not found: {study_dir}")
        return

    # Find experiment directories
    exp_dirs = [d for d in study_dir.iterdir() if d.is_dir() and (d / "predictions.json").exists()]

    if args.experiment:
        import fnmatch

        exp_dirs = [d for d in exp_dirs if fnmatch.fnmatch(d.name, args.experiment)]

    if not exp_dirs:
        print("No experiments found")
        return

    # Analyze each
    all_stats: list[RetrievalStats] = []
    for exp_dir in sorted(exp_dirs):
        stats = analyze_experiment(exp_dir)
        if stats and stats.total > 0:
            all_stats.append(stats)

    if not all_stats:
        print("No RAG experiments with predictions found")
        return

    # Output
    if args.csv:
        print(
            "experiment,total,recall,gen_given_ctx,bottleneck,correct_w_ctx,wrong_w_ctx,correct_wo_ctx,wrong_wo_ctx"
        )
        for s in all_stats:
            print(
                f"{s.name},{s.total},{s.retrieval_recall:.3f},{s.generation_given_context:.3f},{s.bottleneck},"
                f"{s.correct_with_context},{s.wrong_with_context},{s.correct_without_context},{s.wrong_without_context}"
            )
    else:
        print(f"\n{'Experiment':<60} {'Total':>6} {'Recall':>8} {'Gen|Ctx':>8} {'Bottleneck':>12}")
        print("=" * 100)

        for s in all_stats:
            print(
                f"{s.name[:60]:<60} {s.total:>6} {s.retrieval_recall:>7.1%} {s.generation_given_context:>7.1%} {s.bottleneck:>12}"
            )

        # Summary
        print("\n" + "=" * 100)
        print("\nLegend:")
        print("  Recall    = % of questions where answer appeared in retrieved context")
        print("  Gen|Ctx   = % correct when answer WAS in context (generation quality)")
        print("  Bottleneck: RETRIEVAL if Recall<50%, GENERATION if Gen|Ctx<50%")

        # Aggregate
        total = sum(s.total for s in all_stats)
        avg_recall = sum(s.answer_in_context for s in all_stats) / total if total else 0
        total_with_ctx = sum(s.correct_with_context + s.wrong_with_context for s in all_stats)
        avg_gen = (
            sum(s.correct_with_context for s in all_stats) / total_with_ctx if total_with_ctx else 0
        )

        print(f"\nAggregate ({len(all_stats)} experiments, {total} questions):")
        print(f"  Average Retrieval Recall: {avg_recall:.1%}")
        print(f"  Average Generation|Context: {avg_gen:.1%}")


if __name__ == "__main__":
    main()
