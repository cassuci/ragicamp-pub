"""Faithfulness/groundedness metrics for RAG evaluation.

Measures whether answers are actually supported by the retrieved context.
This is critical for RAG systems to prevent hallucination and ensure answers
are grounded in the provided documents.
"""

from typing import Any

from ragicamp.core.logging import get_logger
from ragicamp.metrics.base import Metric

logger = get_logger(__name__)


class FaithfulnessMetric(Metric):
    """Evaluates if answers are grounded in retrieved context.

    Uses Natural Language Inference (NLI) to determine if the answer
    is entailed by the retrieved documents.

    Methods:
    - "nli": NLI model (default)
    - "token_overlap": Simple baseline using token overlap
    - "llm": Use LLM for judgment (requires judge_model)
    """

    def __init__(
        self,
        method: str = "nli",
        nli_model: str = "microsoft/deberta-base-mnli",
        threshold: float = 0.5,
        batch_size: int = 8,
        judge_model: Any | None = None,
        **kwargs: Any,
    ):
        super().__init__(name="faithfulness", **kwargs)
        self.method = method
        self.nli_model_name = nli_model
        self.threshold = threshold
        self.batch_size = batch_size
        self.judge_model = judge_model
        self._nli_pipeline = None

    def _get_nli_pipeline(self):
        """Lazy load NLI pipeline."""
        if self._nli_pipeline is None:
            try:
                from transformers import pipeline

                self._nli_pipeline = pipeline(
                    "text-classification",
                    model=self.nli_model_name,
                    device=0 if self._has_cuda() else -1,
                    batch_size=self.batch_size,
                )
            except ImportError as e:
                raise ImportError(
                    "transformers library required for NLI-based faithfulness. "
                    "Install with: pip install transformers"
                ) from e
        return self._nli_pipeline

    def _has_cuda(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        contexts: list[list[str]] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute faithfulness scores for a batch.

        Args:
            predictions: List of predicted answers
            references: List of reference answers (unused for faithfulness)
            contexts: List of context doc lists, one per prediction.
                      If None, all scores are 0.0.

        Returns:
            Dict with "faithfulness" aggregate score
        """
        try:
            scores = []
            for i, prediction in enumerate(predictions):
                ctx = contexts[i] if contexts and i < len(contexts) else None
                scores.append(self._score_single(prediction, ctx))

            self._last_per_item = scores
            avg = sum(scores) / len(scores) if scores else 0.0
            return {"faithfulness": avg}
        finally:
            self._unload_pipeline()

    def _score_single(self, prediction: str, context: list[str] | None) -> float:
        """Score a single prediction against its context."""
        if not context:
            return 0.0
        if not prediction or not prediction.strip():
            return 0.0

        if self.method == "nli":
            return self._compute_nli_faithfulness(prediction, context)
        elif self.method == "token_overlap":
            return self._compute_token_overlap(prediction, context)
        elif self.method == "llm":
            return self._compute_llm_faithfulness(prediction, context)
        else:
            raise ValueError(f"Unknown faithfulness method: {self.method}")

    def _unload_pipeline(self) -> None:
        """Unload the NLI pipeline to free GPU memory."""
        if self._nli_pipeline is not None:
            del self._nli_pipeline
            self._nli_pipeline = None
            try:
                from ragicamp.utils.resource_manager import ResourceManager

                ResourceManager.clear_gpu_memory()
            except Exception:
                pass
            logger.info("Faithfulness NLI pipeline unloaded")

    def _compute_nli_faithfulness(self, prediction: str, context: list[str]) -> float:
        """Compute faithfulness using NLI entailment."""
        nli = self._get_nli_pipeline()

        max_entailment = 0.0
        for passage in context:
            if not passage or not passage.strip():
                continue

            # A4 fix: pass premise/hypothesis as dict pair, not literal [SEP]
            result = nli({"text": passage, "text_pair": prediction}, top_k=None)

            for label_result in result:
                if "entailment" in label_result["label"].lower():
                    max_entailment = max(max_entailment, label_result["score"])
                    break

            if max_entailment > 0.9:
                break

        return float(max_entailment)

    def _compute_token_overlap(self, prediction: str, context: list[str]) -> float:
        """Simple token overlap baseline."""
        pred_tokens = set(prediction.lower().split())
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "and",
            "or",
            "but",
        }
        pred_tokens = pred_tokens - stopwords
        if not pred_tokens:
            return 0.0

        context_tokens = set(" ".join(context).lower().split())
        overlap = len(pred_tokens & context_tokens)
        return float(overlap / len(pred_tokens))

    def _compute_llm_faithfulness(self, prediction: str, context: list[str]) -> float:
        """Use LLM to judge faithfulness."""
        if self.judge_model is None:
            raise ValueError("judge_model required for LLM-based faithfulness")

        combined_context = "\n\n".join(context[:3])
        prompt = (
            "Given the following context and answer, determine if the answer "
            "is fully supported by the context.\n\n"
            f"Context:\n{combined_context}\n\n"
            f"Answer:\n{prediction}\n\n"
            "Is the answer fully supported by the context? Respond with only:\n"
            '- "YES" if every claim in the answer is supported by the context\n'
            '- "NO" if the answer contains unsupported claims\n\n'
            "Response:"
        )

        try:
            response = self.judge_model.generate(prompt, max_tokens=10, temperature=0.0)
            response = response.strip().upper()
            if "YES" in response:
                return 1.0
            elif "NO" in response:
                return 0.0
            return 0.5
        except Exception as e:
            logger.warning("LLM faithfulness check failed: %s", e)
            return 0.5
