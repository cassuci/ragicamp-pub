"""Enhanced LLM-as-a-judge for QA evaluation with categorical judgments.

This module provides async-capable LLM-as-a-judge metrics for evaluating
QA systems. Uses parallel async API calls for efficient batch evaluation.

Example:
    >>> from ragicamp.models import OpenAIModel
    >>> from ragicamp.metrics import LLMJudgeQAMetric
    >>>
    >>> judge = OpenAIModel("gpt-4o-mini")
    >>> metric = LLMJudgeQAMetric(judge_model=judge, judgment_type="binary")
    >>>
    >>> # Sync usage (internally uses async)
    >>> scores = metric.compute(predictions, references, questions)
    >>>
    >>> # Async usage (for integration with async pipelines)
    >>> scores = await metric.acompute(predictions, references, questions)
"""

import re
from typing import Any

from ragicamp.core.logging import get_logger
from ragicamp.metrics.async_base import AsyncAPIMetric
from ragicamp.models.base import LanguageModel

logger = get_logger(__name__)


class LLMJudgeQAMetric(AsyncAPIMetric):
    """LLM-as-a-judge for QA with categorical evaluation (correct/partial/incorrect).

    Uses GPT-4 or similar to evaluate answer quality with clear categorical judgments.
    Particularly useful when standard metrics don't capture semantic correctness.

    Inherits from AsyncAPIMetric for efficient parallel async API calls.
    """

    def __init__(
        self,
        judge_model: LanguageModel,
        judgment_type: str = "binary",  # "binary" or "ternary"
        max_concurrent: int = 20,  # Number of concurrent API calls
        show_progress: bool = True,
        **kwargs: Any,
    ):
        """Initialize LLM judge for QA evaluation.

        Args:
            judge_model: The LLM to use as judge (e.g., GPT-4, must have agenerate_single)
            judgment_type: Type of judgment
                - "binary": correct (1.0) or incorrect (0.0)
                - "ternary": correct (1.0), partial (0.5), or incorrect (0.0)
            max_concurrent: Maximum concurrent API calls (default: 20)
            show_progress: Show progress bar during computation
            **kwargs: Additional configuration
        """
        super().__init__(
            name="llm_judge_qa",
            max_concurrent=max_concurrent,
            show_progress=show_progress,
            **kwargs,
        )
        self.judge_model = judge_model
        self.judgment_type = judgment_type
        self._traces: list[dict[str, Any]] = []

        # Define categories
        if judgment_type == "binary":
            self.categories = ["correct", "incorrect"]
        else:  # ternary
            self.categories = ["correct", "partially_correct", "incorrect"]

    async def acompute(
        self,
        predictions: list[str],
        references: list[str],
        questions: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Override to clear traces before each computation run."""
        self._traces.clear()
        return await super().acompute(predictions, references, questions, **kwargs)

    def get_traces(self) -> list[dict[str, Any]]:
        """Return collected traces from the last computation."""
        return list(self._traces)

    async def acompute_single(
        self,
        prediction: str,
        reference: str | list[str],
        question: str | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute judgment for a single prediction against one or more references.

        Args:
            prediction: Predicted answer
            reference: Reference answer(s) — str or list of valid answers
            question: Question for context (highly recommended)
            **kwargs: Additional parameters

        Returns:
            Dict with score and category info
        """
        prompt = self._create_judgment_prompt(
            prediction=prediction,
            reference=reference,
        )

        judgment = await self._call_judge(prompt)

        # Retry once on empty response (reasoning models sometimes exhaust
        # their thinking budget and produce no visible output)
        if not judgment.strip():
            logger.info("Empty judge response, retrying once")
            judgment = await self._call_judge(prompt)

        # Extract categorical judgment
        category, score = self._extract_judgment(judgment)

        # Store trace for debugging/export
        self._traces.append(
            {
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "prompt": prompt,
                "response": judgment,
                "category": category,
                "score": score,
            }
        )

        return {
            "llm_judge_qa": score,
            f"llm_judge_qa_{category}": 1.0,  # For category counting
        }

    async def _call_judge(self, prompt: str) -> str:
        """Call the judge model, handling sync/async transparently."""
        sys_msg = self._get_system_message()
        if hasattr(self.judge_model, "agenerate_single"):
            return await self.judge_model.agenerate_single(
                prompt,
                temperature=0.0,
                max_tokens=16384,
                system_message=sys_msg,
            )
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.judge_model.generate(prompt, temperature=0.0)
        )

    def _aggregate_results(self, results: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate individual results with category statistics.

        Args:
            results: List of per-item result dictionaries

        Returns:
            Aggregated metrics including category proportions
        """
        if not results:
            return {"llm_judge_qa": 0.0}

        total = len(results)

        # Sum scores
        avg_score = sum(r.get("llm_judge_qa", 0.0) for r in results) / total

        # Count categories
        correct_count = sum(1 for r in results if r.get("llm_judge_qa_correct", 0) > 0)
        incorrect_count = sum(1 for r in results if r.get("llm_judge_qa_incorrect", 0) > 0)

        aggregated = {
            "llm_judge_qa": avg_score,
            "llm_judge_qa_correct": correct_count / total,
            "llm_judge_qa_incorrect": incorrect_count / total,
        }

        # Add partial category for ternary
        if self.judgment_type == "ternary":
            partial_count = sum(
                1 for r in results if r.get("llm_judge_qa_partially_correct", 0) > 0
            )
            aggregated["llm_judge_qa_partial"] = partial_count / total

        return aggregated

    def _get_system_message(self) -> str:
        """System message that locks the model into judge role."""
        if self.judgment_type == "binary":
            return (
                "You are a QA judge. Check if <pred> contains an answer that "
                "matches any answer in <valid>. Ignore extra text, focus on "
                "whether the correct answer appears anywhere in the prediction. "
                "Reply CORRECT or INCORRECT as your first word."
            )
        return (
            "You are a QA judge. Check if <pred> contains an answer that "
            "matches any answer in <valid>. Ignore extra text, focus on "
            "whether the correct answer appears anywhere in the prediction. "
            "Reply CORRECT, PARTIALLY_CORRECT, or INCORRECT as your first word."
        )

    @staticmethod
    def _sanitize(text: str, max_len: int = 500) -> str:
        """Strip injection delimiters and truncate."""
        # Only strip clear injection delimiters; keep content like \n\n
        # and "Note:" since the model may self-correct after them
        text = re.sub(r"-{3,}.*", "", text)  # --- END ---, --- etc.
        return text.strip()[:max_len]

    def _create_judgment_prompt(
        self, prediction: str, reference: str | list[str], **_kwargs: Any
    ) -> str:
        """Create the user message with sanitized data in XML tags."""
        if isinstance(reference, list):
            ref_str = " | ".join(reference)
        else:
            ref_str = reference

        pred_clean = self._sanitize(prediction)
        return f"<valid>{ref_str}</valid>\n<pred>{pred_clean}</pred>"

    def _extract_judgment(self, judgment_text: str) -> tuple:
        """Extract categorical judgment and convert to score.

        Checks the first line first (model is instructed to put judgment there),
        then falls back to scanning the full text.

        Returns:
            Tuple of (category, score) where score is 0.0, 0.5, or 1.0
        """
        # Check first line first (most reliable with system message)
        first_line = judgment_text.strip().split("\n")[0].lower().strip() if judgment_text else ""

        for text in (first_line, judgment_text.lower()):
            if not text:
                continue
            # Check in order of specificity
            if "partially_correct" in text or "partially correct" in text or "partial" in text:
                if self.judgment_type == "binary":
                    return ("incorrect", 0.0)
                return ("partially_correct", 0.5)
            if re.search(r"\bincorrect\b", text):
                return ("incorrect", 0.0)
            if re.search(r"\bcorrect\b", text):
                return ("correct", 1.0)

        # Default to incorrect if can't parse (conservative)
        logger.warning("Could not parse judgment (defaulting to incorrect): %.120s", judgment_text)
        return ("incorrect", 0.0)


# Convenience functions for creating pre-configured judges


def create_binary_judge(
    judge_model: LanguageModel,
    max_concurrent: int = 20,
) -> LLMJudgeQAMetric:
    """Create a binary LLM judge (correct/incorrect).

    Args:
        judge_model: GPT-4 or similar model
        max_concurrent: Maximum concurrent API calls

    Returns:
        Configured LLMJudgeQAMetric
    """
    return LLMJudgeQAMetric(
        judge_model=judge_model,
        judgment_type="binary",
        max_concurrent=max_concurrent,
    )


def create_ternary_judge(
    judge_model: LanguageModel,
    max_concurrent: int = 20,
) -> LLMJudgeQAMetric:
    """Create a ternary LLM judge (correct/partial/incorrect).

    Args:
        judge_model: GPT-4 or similar model
        max_concurrent: Maximum concurrent API calls

    Returns:
        Configured LLMJudgeQAMetric
    """
    return LLMJudgeQAMetric(
        judge_model=judge_model,
        judgment_type="ternary",
        max_concurrent=max_concurrent,
    )
