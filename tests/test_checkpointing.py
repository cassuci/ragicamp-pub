"""Tests for LLM judge async computation.

Note: Checkpointing functionality was removed when LLMJudgeQAMetric
was refactored to use AsyncAPIMetric base class with parallel async
API calls. The new implementation handles errors gracefully per-item
without checkpointing.
"""

import pytest

from ragicamp.metrics.llm_judge_qa import LLMJudgeQAMetric


class MockAsyncJudgeModel:
    """Mock async LLM judge model for testing."""

    def __init__(self, responses: list[str] | None = None):
        self.model_name = "mock_judge"
        self.call_count = 0
        self.responses = responses or []

    async def agenerate_single(self, prompt: str, **kwargs) -> str:
        """Generate mock judgment asynchronously."""
        self.call_count += 1
        if self.responses and self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        return "JUDGMENT: CORRECT\nThe answer is right."


class TestLLMJudgeQAMetric:
    """Test LLM judge async computation."""

    def test_binary_judgment_correct(self):
        """Test binary judgment returns correct score."""
        judge_model = MockAsyncJudgeModel(responses=["JUDGMENT: CORRECT\nThe answer is accurate."])
        metric = LLMJudgeQAMetric(judge_model=judge_model, judgment_type="binary")

        results = metric.compute(
            predictions=["Paris"],
            references=[["Paris"]],
            questions=["What is the capital of France?"],
        )

        assert results["llm_judge_qa"] == 1.0

    def test_binary_judgment_incorrect(self):
        """Test binary judgment returns incorrect score."""
        judge_model = MockAsyncJudgeModel(responses=["JUDGMENT: INCORRECT\nThe answer is wrong."])
        metric = LLMJudgeQAMetric(judge_model=judge_model, judgment_type="binary")

        results = metric.compute(
            predictions=["London"],
            references=[["Paris"]],
            questions=["What is the capital of France?"],
        )

        assert results["llm_judge_qa"] == 0.0

    def test_ternary_judgment_partial(self):
        """Test ternary judgment returns partial score."""
        judge_model = MockAsyncJudgeModel(
            responses=["JUDGMENT: PARTIALLY_CORRECT\nClose but missing details."]
        )
        metric = LLMJudgeQAMetric(judge_model=judge_model, judgment_type="ternary")

        results = metric.compute(
            predictions=["France"],
            references=[["Paris, France"]],
            questions=["Where is the Eiffel Tower?"],
        )

        assert results["llm_judge_qa"] == 0.5

    def test_multiple_predictions(self):
        """Test batch of predictions."""
        judge_model = MockAsyncJudgeModel(
            responses=[
                "JUDGMENT: CORRECT\nRight.",
                "JUDGMENT: INCORRECT\nWrong.",
                "JUDGMENT: CORRECT\nRight.",
            ]
        )
        metric = LLMJudgeQAMetric(judge_model=judge_model, judgment_type="binary")

        results = metric.compute(
            predictions=["A", "B", "C"],
            references=[["A"], ["X"], ["C"]],
            questions=["Q1", "Q2", "Q3"],
        )

        # Average: (1.0 + 0.0 + 1.0) / 3 = 0.666...
        assert abs(results["llm_judge_qa"] - 0.666) < 0.01

    def test_error_handling_graceful(self):
        """Test that errors are handled gracefully per-item."""

        class FailingMockModel:
            model_name = "failing_mock"

            async def agenerate_single(self, prompt: str, **kwargs) -> str:
                raise Exception("API Error")

        metric = LLMJudgeQAMetric(
            judge_model=FailingMockModel(),
            judgment_type="binary",
            show_progress=False,
        )

        # Should not raise, but return error score
        results = metric.compute(
            predictions=["Answer"],
            references=[["Ref"]],
            questions=["Question"],
        )

        # Error should result in 0.0 score
        assert results["llm_judge_qa"] == 0.0

    def test_categories_tracked(self):
        """Test that category proportions are tracked."""
        judge_model = MockAsyncJudgeModel(
            responses=[
                "JUDGMENT: CORRECT",
                "JUDGMENT: CORRECT",
                "JUDGMENT: INCORRECT",
            ]
        )
        metric = LLMJudgeQAMetric(judge_model=judge_model, judgment_type="binary")

        results = metric.compute(
            predictions=["A", "B", "C"],
            references=[["A"], ["B"], ["C"]],
            questions=["Q1", "Q2", "Q3"],
        )

        # 2 correct, 1 incorrect
        assert abs(results["llm_judge_qa_correct"] - 0.666) < 0.01
        assert abs(results["llm_judge_qa_incorrect"] - 0.333) < 0.01


class TestJudgmentParsing:
    """Test judgment text parsing."""

    def test_parse_correct(self):
        """Test parsing CORRECT judgment."""
        judge_model = MockAsyncJudgeModel()
        metric = LLMJudgeQAMetric(judge_model=judge_model, judgment_type="binary")

        category, score = metric._extract_judgment("JUDGMENT: CORRECT\nGood answer.")
        assert category == "correct"
        assert score == 1.0

    def test_parse_incorrect(self):
        """Test parsing INCORRECT judgment."""
        judge_model = MockAsyncJudgeModel()
        metric = LLMJudgeQAMetric(judge_model=judge_model, judgment_type="binary")

        category, score = metric._extract_judgment("JUDGMENT: INCORRECT\nBad answer.")
        assert category == "incorrect"
        assert score == 0.0

    def test_parse_partial_ternary(self):
        """Test parsing PARTIALLY_CORRECT judgment in ternary mode."""
        judge_model = MockAsyncJudgeModel()
        metric = LLMJudgeQAMetric(judge_model=judge_model, judgment_type="ternary")

        category, score = metric._extract_judgment("JUDGMENT: PARTIALLY_CORRECT")
        assert category == "partially_correct"
        assert score == 0.5

    def test_parse_partial_in_binary_mode(self):
        """Test that partial is mapped to incorrect in binary mode."""
        judge_model = MockAsyncJudgeModel()
        metric = LLMJudgeQAMetric(judge_model=judge_model, judgment_type="binary")

        category, score = metric._extract_judgment("JUDGMENT: PARTIALLY_CORRECT")
        assert category == "incorrect"
        assert score == 0.0

    def test_parse_fallback_conservative(self):
        """Test that unparseable text defaults to incorrect."""
        judge_model = MockAsyncJudgeModel()
        metric = LLMJudgeQAMetric(judge_model=judge_model, judgment_type="binary")

        category, score = metric._extract_judgment("Some random text without judgment")
        assert category == "incorrect"
        assert score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
