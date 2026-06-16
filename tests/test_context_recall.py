"""Tests for ContextRecallMetric.

Covers:
- All / no / half sentences found
- Empty context, single-word reference
- Word-overlap fallback
- Multi-reference (best recall)
- Sentence splitting edge cases
- Batch computation, compute_with_details
"""

from ragicamp.metrics.context_recall import (
    ContextRecallMetric,
    _normalize,
    _split_sentences,
    _word_overlap,
)


class TestHelpers:
    def test_normalize(self):
        assert _normalize("Hello, World!") == "hello world"

    def test_split_sentences(self):
        assert _split_sentences("First. Second! Third?") == [
            "First",
            "Second",
            "Third",
        ]

    def test_split_sentences_empty(self):
        assert _split_sentences("") == []

    def test_word_overlap_full(self):
        assert _word_overlap("hello world", "hello world foo") == 1.0

    def test_word_overlap_partial(self):
        assert _word_overlap("hello world bar", "hello world foo") == 2 / 3

    def test_word_overlap_empty(self):
        assert _word_overlap("", "hello") == 0.0


class TestContextRecallBasic:
    def test_all_sentences_found(self):
        metric = ContextRecallMetric()
        result = metric.compute(
            predictions=["unused"],
            references=["The war ended in 1945. It was devastating."],
            contexts=[["The war ended in 1945. It was devastating and costly."]],
        )
        assert result["context_recall"] == 1.0

    def test_no_sentences_found(self):
        metric = ContextRecallMetric()
        result = metric.compute(
            predictions=["unused"],
            references=["Paris is the capital of France. It has the Eiffel Tower."],
            contexts=[["Berlin is in Germany."]],
        )
        assert result["context_recall"] == 0.0

    def test_half_sentences_found(self):
        metric = ContextRecallMetric()
        result = metric.compute(
            predictions=["unused"],
            references=["The war ended in 1945. Paris is beautiful."],
            contexts=[["The war ended in 1945."]],
        )
        assert result["context_recall"] == 0.5

    def test_empty_context_returns_zero(self):
        metric = ContextRecallMetric()
        result = metric.compute(
            predictions=["unused"],
            references=["Some reference."],
            contexts=[[]],
        )
        assert result["context_recall"] == 0.0

    def test_none_context_returns_zero(self):
        metric = ContextRecallMetric()
        result = metric.compute(
            predictions=["unused"],
            references=["Some reference."],
            contexts=None,
        )
        assert result["context_recall"] == 0.0

    def test_short_ref_falls_through(self):
        """Single-word reference shorter than min_sentence_words uses whole-ref fallback."""
        metric = ContextRecallMetric(min_sentence_words=3)
        # "Hi" has fewer than 3 words, so it uses whole-ref fallback
        # which also checks min_sentence_words — so it returns 0.0
        result = metric.compute(
            predictions=["unused"],
            references=["Hi"],
            contexts=[["Hi there!"]],
        )
        assert result["context_recall"] == 0.0

    def test_word_overlap_fallback(self):
        """When substring doesn't match, word overlap >= threshold counts."""
        metric = ContextRecallMetric(overlap_threshold=0.7)
        # "the big brown fox jumped" has 5 content words
        # context has "the", "big", "brown", "fox" = 4/5 = 0.8 >= 0.7
        result = metric.compute(
            predictions=["unused"],
            references=["The big brown fox jumped."],
            contexts=[["The big brown fox was here."]],
        )
        assert result["context_recall"] > 0.0


class TestContextRecallMultiRef:
    def test_best_recall_across_refs(self):
        """Multi-reference: takes best recall across answer variants."""
        metric = ContextRecallMetric()
        result = metric.compute(
            predictions=["unused"],
            references=[["Some random text nobody has.", "The war ended in 1945."]],
            contexts=[["The war ended in 1945."]],
        )
        assert result["context_recall"] == 1.0

    def test_multi_ref_none_found(self):
        metric = ContextRecallMetric()
        result = metric.compute(
            predictions=["unused"],
            references=[["Alpha beta gamma.", "Delta epsilon zeta."]],
            contexts=[["Totally unrelated context here."]],
        )
        assert result["context_recall"] == 0.0


class TestContextRecallBatch:
    def test_batch_computation(self):
        metric = ContextRecallMetric()
        result = metric.compute(
            predictions=["a", "b"],
            references=[
                "The war ended in 1945.",
                "Paris is the capital of France.",
            ],
            contexts=[
                ["The war ended in 1945."],
                ["Berlin is in Germany."],
            ],
        )
        # First: 1.0, Second: 0.0 → avg 0.5
        assert result["context_recall"] == 0.5

    def test_compute_with_details(self):
        metric = ContextRecallMetric()
        detail = metric.compute_with_details(
            predictions=["a", "b"],
            references=[
                "The war ended in 1945.",
                "Paris is the capital of France.",
            ],
            contexts=[
                ["The war ended in 1945."],
                ["Berlin is in Germany."],
            ],
        )
        assert detail.name == "context_recall"
        assert detail.aggregate == 0.5
        assert len(detail.per_item) == 2
        assert detail.per_item[0] == 1.0
        assert detail.per_item[1] == 0.0

    def test_per_item_scores(self):
        metric = ContextRecallMetric()
        metric.compute(
            predictions=["a", "b"],
            references=["The war ended in 1945.", "Paris is beautiful."],
            contexts=[
                ["The war ended in 1945."],
                ["Paris is beautiful."],
            ],
        )
        scores = metric.get_per_item_scores()
        assert scores == [1.0, 1.0]
