"""Answer-in-context retrieval proxy metric.

Checks whether the gold answer text appears in the retrieved context.
Pure text matching — no model required. Useful as a cheap proxy for
retrieval quality: if the answer isn't in the retrieved passages, the
generator has no chance of producing it from context alone.

Adapted from ``scripts/analyze_retrieval.py:answer_in_prompt()``.
"""

import re
from typing import Any

from ragicamp.metrics.base import Metric


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class AnswerInContextMetric(Metric):
    """Binary metric: 1.0 if any gold answer appears in retrieved context, else 0.0.

    Multi-reference: returns 1.0 if *any* reference variant is found.
    Answers shorter than ``min_answer_len`` (after normalization) are skipped
    to avoid trivially short matches (e.g. "a", "no").
    """

    def __init__(self, min_answer_len: int = 2, **kwargs: Any):
        super().__init__(name="answer_in_context", **kwargs)
        self.min_answer_len = min_answer_len

    def compute(
        self,
        predictions: list[str],
        references: list[str | list[str]],
        contexts: list[list[str]] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute answer-in-context for a batch.

        Args:
            predictions: Model predictions (unused — metric compares reference vs context).
            references: Gold answers. Each entry may be a str or list[str] for multi-ref.
            contexts: Retrieved doc lists, one per prediction. If None, all scores are 0.0.

        Returns:
            Dict with ``"answer_in_context"`` aggregate score.
        """
        scores: list[float] = []
        for i, ref in enumerate(references):
            ctx = contexts[i] if contexts and i < len(contexts) else None
            scores.append(self._score_single(ref, ctx))

        self._last_per_item = scores
        avg = sum(scores) / len(scores) if scores else 0.0
        return {"answer_in_context": avg}

    def _score_single(self, reference: str | list[str], context: list[str] | None) -> float:
        if not context:
            return 0.0

        context_normalized = _normalize(" ".join(context))
        refs = reference if isinstance(reference, list) else [reference]

        for ref in refs:
            ref_normalized = _normalize(str(ref))
            if len(ref_normalized) < self.min_answer_len:
                continue
            if ref_normalized in context_normalized:
                return 1.0

        return 0.0
