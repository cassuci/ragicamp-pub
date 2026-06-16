"""Tests for formatting utilities."""

import pytest

from ragicamp.core.types import Document
from ragicamp.utils.formatting import (
    PERCENTAGE_METRICS,
    ContextFormatter,
    format_metrics_summary,
)


class TestPercentageMetrics:
    """Test PERCENTAGE_METRICS constant."""

    def test_percentage_metrics_is_frozenset(self):
        """Test that PERCENTAGE_METRICS is a frozenset."""
        assert isinstance(PERCENTAGE_METRICS, frozenset)

    def test_percentage_metrics_contents(self):
        """Test that PERCENTAGE_METRICS contains expected metrics."""
        expected_metrics = {"f1", "exact_match", "bertscore_f1", "bleurt", "llm_judge_qa"}
        assert PERCENTAGE_METRICS == expected_metrics

    def test_percentage_metrics_immutable(self):
        """Test that PERCENTAGE_METRICS cannot be modified."""
        with pytest.raises(AttributeError):
            PERCENTAGE_METRICS.add("new_metric")


class TestFormatMetricsSummary:
    """Test format_metrics_summary function."""

    def test_percentage_metrics_formatting(self):
        """Test that percentage metrics are formatted as percentages."""
        metrics = {"f1": 0.853, "exact_match": 0.75}
        result = format_metrics_summary(metrics)

        assert "f1=85.3%" in result
        assert "exact_match=75.0%" in result

    def test_non_percentage_metrics_formatting(self):
        """Test that non-percentage metrics are formatted as floats."""
        metrics = {"loss": 0.123456, "latency": 0.789}
        result = format_metrics_summary(metrics)

        assert "loss=0.123" in result
        assert "latency=0.789" in result

    def test_mixed_metrics(self):
        """Test formatting with both percentage and non-percentage metrics."""
        metrics = {
            "f1": 0.85,
            "exact_match": 0.90,
            "loss": 0.123,
            "latency": 0.456,
        }
        result = format_metrics_summary(metrics)

        assert "f1=85.0%" in result
        assert "exact_match=90.0%" in result
        assert "loss=0.123" in result
        assert "latency=0.456" in result

    def test_empty_dict(self):
        """Test formatting with empty metrics dict."""
        metrics = {}
        result = format_metrics_summary(metrics)

        assert result == "no metrics"

    def test_zero_values(self):
        """Test formatting with zero values."""
        metrics = {"f1": 0.0, "loss": 0.0}
        result = format_metrics_summary(metrics)

        assert "f1=0.0%" in result
        assert "loss=0.000" in result

    def test_one_value_percentage(self):
        """Test formatting with value of 1.0 for percentage metric."""
        metrics = {"exact_match": 1.0}
        result = format_metrics_summary(metrics)

        assert result == "exact_match=100.0%"

    def test_all_percentage_metrics(self):
        """Test that all metrics in PERCENTAGE_METRICS are formatted as percentages."""
        metrics = {
            "f1": 0.5,
            "exact_match": 0.6,
            "bertscore_f1": 0.7,
            "bleurt": 0.8,
            "llm_judge_qa": 0.9,
        }
        result = format_metrics_summary(metrics)

        assert "f1=50.0%" in result
        assert "exact_match=60.0%" in result
        assert "bertscore_f1=70.0%" in result
        assert "bleurt=80.0%" in result
        assert "llm_judge_qa=90.0%" in result

    def test_non_numeric_values_skipped(self):
        """Test that non-numeric values (str, etc.) are skipped, but int is kept."""
        metrics = {
            "f1": 0.85,
            "status": "ok",
            "count": 10,
            "loss": 0.123,
        }
        result = format_metrics_summary(metrics)

        assert "f1=85.0%" in result
        assert "loss=0.123" in result
        assert "status" not in result
        # int values are formatted as floats
        assert "count=10.000" in result

    def test_none_values_skipped(self):
        """Test that None values are skipped."""
        metrics = {
            "f1": 0.85,
            "loss": None,
            "exact_match": 0.90,
        }
        result = format_metrics_summary(metrics)

        assert "f1=85.0%" in result
        assert "exact_match=90.0%" in result
        assert "loss" not in result

    def test_int_values(self):
        """Test that integer values are handled correctly (int is numeric)."""
        metrics = {
            "f1": 1,  # int value
            "loss": 0,  # int zero
            "count": 5,  # int non-zero
        }
        result = format_metrics_summary(metrics)

        assert "f1=100.0%" in result
        assert "loss=0.000" in result
        assert "count=5.000" in result  # int is formatted as float

    def test_float_precision(self):
        """Test that float precision is handled correctly."""
        metrics = {
            "f1": 0.8537,  # Should round to 85.4%
            "loss": 0.123456789,  # Should round to 0.123
        }
        result = format_metrics_summary(metrics)

        assert "f1=85.4%" in result
        assert "loss=0.123" in result

    def test_negative_values(self):
        """Test formatting with negative values."""
        metrics = {
            "f1": -0.5,
            "loss": -0.123,
        }
        result = format_metrics_summary(metrics)

        assert "f1=-50.0%" in result
        assert "loss=-0.123" in result

    def test_very_small_values(self):
        """Test formatting with very small values."""
        metrics = {
            "f1": 0.0001,
            "loss": 0.000001,
        }
        result = format_metrics_summary(metrics)

        assert "f1=0.0%" in result
        assert "loss=0.000" in result

    def test_very_large_values(self):
        """Test formatting with very large values."""
        metrics = {
            "f1": 1.5,  # > 1.0 for percentage
            "loss": 100.0,
        }
        result = format_metrics_summary(metrics)

        assert "f1=150.0%" in result
        assert "loss=100.000" in result


class TestContextFormatter:
    """Test ContextFormatter class."""

    def test_format_documents_basic(self):
        """Test basic document formatting."""
        docs = [
            Document(id="1", text="First document", metadata={}, score=None),
            Document(id="2", text="Second document", metadata={}, score=None),
        ]
        result = ContextFormatter.format_documents(docs)

        assert "[1] First document" in result
        assert "[2] Second document" in result

    def test_format_documents_empty(self):
        """Test formatting with empty document list."""
        docs = []
        result = ContextFormatter.format_documents(docs)

        assert result == "No relevant context found."

    def test_format_documents_custom_template(self):
        """Test formatting with custom template."""
        docs = [
            Document(id="1", text="Test", metadata={}, score=0.9),
        ]
        result = ContextFormatter.format_documents(
            docs, template="Doc {idx} (score={score:.2f}): {text}"
        )

        assert result == "Doc 1 (score=0.90): Test"

    def test_format_documents_custom_separator(self):
        """Test formatting with custom separator."""
        docs = [
            Document(id="1", text="First", metadata={}, score=None),
            Document(id="2", text="Second", metadata={}, score=None),
        ]
        result = ContextFormatter.format_documents(docs, separator=" | ")

        assert result == "[1] First | [2] Second"

    def test_format_documents_custom_empty_message(self):
        """Test formatting with custom empty message."""
        docs = []
        result = ContextFormatter.format_documents(docs, empty_message="No docs available")

        assert result == "No docs available"

    def test_format_documents_max_length(self):
        """Test formatting with max_length truncation."""
        docs = [
            Document(
                id="1",
                text="This is a very long document that should be truncated",
                metadata={},
                score=None,
            ),
        ]
        result = ContextFormatter.format_documents(docs, max_length=20)

        assert result == "[1] This is a very long ..."
        # Text part is truncated with " ..."
        text_part = result.split("] ")[1]
        assert len(text_part) <= 30  # reasonably bounded

    def test_format_documents_max_length_no_truncation(self):
        """Test formatting when text is shorter than max_length."""
        docs = [
            Document(id="1", text="Short", metadata={}, score=None),
        ]
        result = ContextFormatter.format_documents(docs, max_length=20)

        assert result == "[1] Short"

    def test_format_documents_include_metadata(self):
        """Test formatting with metadata included."""
        docs = [
            Document(
                id="1",
                text="Test",
                metadata={"title": "Test Title", "source": "test.txt"},
                score=None,
            ),
        ]
        result = ContextFormatter.format_documents(
            docs, template="{idx}: {text} (title={title}, source={source})", include_metadata=True
        )

        assert "title=Test Title" in result
        assert "source=test.txt" in result

    def test_format_documents_with_scores(self):
        """Test formatting documents with scores."""
        docs = [
            Document(id="1", text="First", metadata={}, score=0.9),
            Document(id="2", text="Second", metadata={}, score=0.85),
        ]
        result = ContextFormatter.format_documents(docs)

        # Default template includes score via {score} placeholder
        assert "[1] First" in result
        assert "[2] Second" in result

    def test_format_documents_score_none(self):
        """Test formatting when score is None."""
        docs = [
            Document(id="1", text="Test", metadata={}, score=None),
        ]
        result = ContextFormatter.format_documents(docs, template="Score: {score}")

        assert result == "Score: 0.0"

    def test_format_with_scores(self):
        """Test format_with_scores method."""
        docs = [
            Document(id="1", text="Example", metadata={}, score=0.856),
        ]
        result = ContextFormatter.format_with_scores(docs)

        assert "[1] (score: 0.856) Example" in result

    def test_format_with_scores_no_show(self):
        """Test format_with_scores with show_score=False."""
        docs = [
            Document(id="1", text="Example", metadata={}, score=0.856),
        ]
        result = ContextFormatter.format_with_scores(docs, show_score=False)

        assert result == "[1] Example"

    def test_format_with_scores_custom_format(self):
        """Test format_with_scores with custom score format."""
        docs = [
            Document(id="1", text="Example", metadata={}, score=0.856),
        ]
        result = ContextFormatter.format_with_scores(docs, score_format="{score:.1f}")

        assert "[1] (score: 0.9) Example" in result

    def test_format_numbered(self):
        """Test format_numbered method."""
        docs = [
            Document(id="1", text="First document", metadata={}, score=None),
            Document(id="2", text="Second document", metadata={}, score=None),
        ]
        result = ContextFormatter.format_numbered(docs)

        assert "--- Passage 1 ---" in result
        assert "First document" in result
        assert "--- Passage 2 ---" in result
        assert "Second document" in result

    def test_format_numbered_empty(self):
        """Test format_numbered with empty list."""
        docs = []
        result = ContextFormatter.format_numbered(docs)

        assert result == "No relevant passages found."

    def test_format_numbered_from_docs(self):
        """Test format_numbered_from_docs method."""

        class DocLike:
            def __init__(self, text):
                self.text = text

        docs = [DocLike("First"), DocLike("Second")]
        result = ContextFormatter.format_numbered_from_docs(docs)

        assert "--- Passage 1 ---" in result
        assert "First" in result
        assert "--- Passage 2 ---" in result
        assert "Second" in result

    def test_format_numbered_from_docs_empty(self):
        """Test format_numbered_from_docs with empty list."""
        docs = []
        result = ContextFormatter.format_numbered_from_docs(docs)

        assert result == "No relevant passages found."

    def test_format_numbered_from_docs_no_text_attr(self):
        """Test format_numbered_from_docs with object without text attribute."""

        class DocLike:
            def __init__(self, content):
                self.content = content

        docs = [DocLike("Test")]
        result = ContextFormatter.format_numbered_from_docs(docs)

        # Should convert to string
        assert "--- Passage 1 ---" in result

    def test_format_with_titles(self):
        """Test format_with_titles method."""
        docs = [
            Document(
                id="1",
                text="Content here",
                metadata={"title": "Document Title"},
                score=None,
            ),
        ]
        result = ContextFormatter.format_with_titles(docs)

        assert "[1] Title: Document Title" in result
        assert "Content: Content here" in result

    def test_format_with_titles_default_title(self):
        """Test format_with_titles with missing title in metadata."""
        docs = [
            Document(id="1", text="Content", metadata={}, score=None),
        ]
        result = ContextFormatter.format_with_titles(docs)

        assert "[1] Title: Document 1" in result
        assert "Content: Content" in result

    def test_format_with_titles_custom_key(self):
        """Test format_with_titles with custom title key."""
        docs = [
            Document(
                id="1",
                text="Content",
                metadata={"heading": "Custom Heading"},
                score=None,
            ),
        ]
        result = ContextFormatter.format_with_titles(docs, title_key="heading")

        assert "Title: Custom Heading" in result

    def test_format_with_titles_empty(self):
        """Test format_with_titles with empty list."""
        docs = []
        result = ContextFormatter.format_with_titles(docs)

        assert result == "No relevant context found."

    def test_format_with_titles_multiple_docs(self):
        """Test format_with_titles with multiple documents."""
        docs = [
            Document(
                id="1",
                text="First content",
                metadata={"title": "First Title"},
                score=None,
            ),
            Document(
                id="2",
                text="Second content",
                metadata={"title": "Second Title"},
                score=None,
            ),
        ]
        result = ContextFormatter.format_with_titles(docs)

        assert "[1] Title: First Title" in result
        assert "First content" in result
        assert "[2] Title: Second Title" in result
        assert "Second content" in result
        assert "\n\n" in result  # Should have separator


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
