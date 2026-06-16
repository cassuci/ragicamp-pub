"""Tests for AnswerInContextMetric.

Covers:
- Perfect match, no match, empty/None context
- Case-insensitive, punctuation-normalized matching
- Short answer skip (min_answer_len)
- Multi-reference (any found / none found)
- Batch computation
- compute_with_details and per_item scores
"""

from ragicamp.metrics.answer_in_context import AnswerInContextMetric, _normalize


class TestNormalize:
    def test_lowercases(self):
        assert _normalize("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert _normalize("hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize("hello   world") == "hello world"

    def test_strips_edges(self):
        assert _normalize("  hello  ") == "hello"


class TestAnswerInContextBasic:
    def test_perfect_match(self):
        metric = AnswerInContextMetric()
        result = metric.compute(
            predictions=["1945"],
            references=["1945"],
            contexts=[["World War II ended in 1945."]],
        )
        assert result["answer_in_context"] == 1.0

    def test_no_match(self):
        metric = AnswerInContextMetric()
        result = metric.compute(
            predictions=["1945"],
            references=["1945"],
            contexts=[["The Eiffel Tower is in Paris."]],
        )
        assert result["answer_in_context"] == 0.0

    def test_none_context_returns_zero(self):
        metric = AnswerInContextMetric()
        result = metric.compute(
            predictions=["answer"],
            references=["answer"],
            contexts=None,
        )
        assert result["answer_in_context"] == 0.0

    def test_empty_context_list_returns_zero(self):
        metric = AnswerInContextMetric()
        result = metric.compute(
            predictions=["answer"],
            references=["answer"],
            contexts=[[]],
        )
        assert result["answer_in_context"] == 0.0

    def test_case_insensitive(self):
        metric = AnswerInContextMetric()
        result = metric.compute(
            predictions=["Paris"],
            references=["paris"],
            contexts=[["The capital of France is PARIS."]],
        )
        assert result["answer_in_context"] == 1.0

    def test_punctuation_normalized(self):
        """Punctuation is stripped so '(Paris)' matches 'Paris'."""
        metric = AnswerInContextMetric()
        result = metric.compute(
            predictions=["(Paris)"],
            references=["(Paris)"],
            contexts=[["The capital is Paris."]],
        )
        assert result["answer_in_context"] == 1.0

    def test_short_answer_skipped(self):
        """Answers shorter than min_answer_len are skipped (trivially short)."""
        metric = AnswerInContextMetric(min_answer_len=2)
        result = metric.compute(
            predictions=["a"],
            references=["a"],
            contexts=[["a is a vowel."]],
        )
        assert result["answer_in_context"] == 0.0

    def test_short_answer_custom_threshold(self):
        """With min_answer_len=1, even single-char answers match."""
        metric = AnswerInContextMetric(min_answer_len=1)
        result = metric.compute(
            predictions=["x"],
            references=["x"],
            contexts=[["x marks the spot."]],
        )
        assert result["answer_in_context"] == 1.0


class TestAnswerInContextMultiRef:
    def test_multi_ref_any_found(self):
        metric = AnswerInContextMetric()
        result = metric.compute(
            predictions=["answer"],
            references=[["George Washington", "Washington"]],
            contexts=[["Washington was the first president."]],
        )
        assert result["answer_in_context"] == 1.0

    def test_multi_ref_none_found(self):
        metric = AnswerInContextMetric()
        result = metric.compute(
            predictions=["answer"],
            references=[["George Washington", "Washington"]],
            contexts=[["Napoleon was emperor of France."]],
        )
        assert result["answer_in_context"] == 0.0


class TestAnswerInContextBatch:
    def test_batch_mixed(self):
        metric = AnswerInContextMetric()
        result = metric.compute(
            predictions=["a", "b"],
            references=["1945", "Paris"],
            contexts=[
                ["The war ended in 1945."],
                ["Berlin is in Germany."],
            ],
        )
        assert result["answer_in_context"] == 0.5

    def test_compute_with_details(self):
        metric = AnswerInContextMetric()
        detail = metric.compute_with_details(
            predictions=["a", "b"],
            references=["1945", "Paris"],
            contexts=[
                ["The war ended in 1945."],
                ["Berlin is in Germany."],
            ],
        )
        assert detail.name == "answer_in_context"
        assert detail.aggregate == 0.5
        assert detail.per_item == [1.0, 0.0]

    def test_per_item_scores(self):
        metric = AnswerInContextMetric()
        metric.compute(
            predictions=["a", "b", "c"],
            references=["cat", "dog", "fish"],
            contexts=[
                ["I have a cat."],
                ["I have a cat."],
                ["I have a fish."],
            ],
        )
        assert metric.get_per_item_scores() == [1.0, 0.0, 1.0]

    def test_multiple_context_docs(self):
        """Answer found across multiple retrieved docs."""
        metric = AnswerInContextMetric()
        result = metric.compute(
            predictions=["answer"],
            references=["Einstein"],
            contexts=[["Albert was a physicist.", "Einstein developed relativity."]],
        )
        assert result["answer_in_context"] == 1.0
