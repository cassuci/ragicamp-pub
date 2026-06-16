"""Context recall metric for RAG evaluation.

Sentence-level recall: checks what fraction of the reference answer's
sentences are supported by the retrieved context.  No model required —
uses normalized substring matching with a word-overlap fallback.

Useful for diagnosing retrieval quality alongside ``AnswerInContextMetric``
(binary) by providing a *graded* signal.
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


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on [.!?]+, keeping non-trivial fragments."""
    parts = re.split(r"[.!?]+", text)
    return [s.strip() for s in parts if s.strip()]


def _word_overlap(a: str, b: str) -> float:
    """Fraction of words in *a* that also appear in *b*."""
    words_a = set(a.split())
    if not words_a:
        return 0.0
    words_b = set(b.split())
    return len(words_a & words_b) / len(words_a)


class ContextRecallMetric(Metric):
    """Fraction of reference-answer sentences found in the retrieved context.

    For each sentence in the reference:
      1. Check if its normalized form is a substring of the normalized context.
      2. If not, fall back to word-overlap >= ``overlap_threshold``.

    Multi-reference: takes the *best* recall across all valid answer variants.
    Sentences with fewer than ``min_sentence_words`` words are skipped.
    """

    def __init__(
        self,
        min_sentence_words: int = 3,
        overlap_threshold: float = 0.8,
        **kwargs: Any,
    ):
        super().__init__(name="context_recall", **kwargs)
        self.min_sentence_words = min_sentence_words
        self.overlap_threshold = overlap_threshold

    def compute(
        self,
        predictions: list[str],
        references: list[str | list[str]],
        contexts: list[list[str]] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute context recall for a batch.

        Args:
            predictions: Model predictions (unused).
            references: Gold answers. Each may be str or list[str].
            contexts: Retrieved doc lists per prediction. If None, all 0.0.

        Returns:
            Dict with ``"context_recall"`` aggregate score.
        """
        scores: list[float] = []
        for i, ref in enumerate(references):
            ctx = contexts[i] if contexts and i < len(contexts) else None
            scores.append(self._score_single(ref, ctx))

        self._last_per_item = scores
        avg = sum(scores) / len(scores) if scores else 0.0
        return {"context_recall": avg}

    # ------------------------------------------------------------------

    def _score_single(self, reference: str | list[str], context: list[str] | None) -> float:
        if not context:
            return 0.0

        context_normalized = _normalize(" ".join(context))
        refs = reference if isinstance(reference, list) else [reference]

        best_recall = 0.0
        for ref in refs:
            ref_str = str(ref)
            if not ref_str.strip():
                continue
            recall = self._recall_for_ref(ref_str, context_normalized)
            if recall > best_recall:
                best_recall = recall

        return best_recall

    def _recall_for_ref(self, reference: str, context_normalized: str) -> float:
        sentences = _split_sentences(reference)
        # Filter trivial fragments
        sentences = [s for s in sentences if len(s.split()) >= self.min_sentence_words]
        if not sentences:
            # Fall back to whole-reference substring check
            ref_norm = _normalize(reference)
            if not ref_norm or len(ref_norm.split()) < self.min_sentence_words:
                return 0.0
            return 1.0 if ref_norm in context_normalized else 0.0

        found = 0
        for sent in sentences:
            sent_norm = _normalize(sent)
            if not sent_norm:
                continue
            if sent_norm in context_normalized:
                found += 1
            elif _word_overlap(sent_norm, context_normalized) >= self.overlap_threshold:
                found += 1

        return found / len(sentences)
