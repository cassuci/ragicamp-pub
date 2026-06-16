"""Hallucination detection metrics for RAG evaluation.

Detects when the model generates content not supported by retrieved documents.
Complementary to faithfulness - focuses on identifying problematic outputs.
"""

from typing import Any

from ragicamp.core.logging import get_logger
from ragicamp.metrics.base import Metric

logger = get_logger(__name__)


class HallucinationMetric(Metric):
    """Detects hallucinations in generated answers.

    A hallucination occurs when the model makes claims that are:
    1. Not supported by the retrieved context
    2. Contradicted by the context
    3. Invented from parametric knowledge

    Methods:
    - "nli": NLI-based detection (checks for contradiction)
    - "simple": Token-based heuristic (fast baseline)
    """

    def __init__(
        self,
        method: str = "nli",
        nli_model: str = "microsoft/deberta-base-mnli",
        threshold: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(name="hallucination", **kwargs)
        self.method = method
        self.nli_model_name = nli_model
        self.threshold = threshold
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
                )
            except ImportError as e:
                raise ImportError(
                    "transformers library required for NLI-based hallucination detection. "
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
        """Compute hallucination scores for a batch.

        Args:
            predictions: List of predicted answers
            references: List of reference answers (unused)
            contexts: List of context doc lists, one per prediction.
                      If None, all scores are 1.0 (assume hallucinated).

        Returns:
            Dict with "hallucination" aggregate score (0=grounded, 1=hallucinated)
        """
        try:
            scores = []
            for i, prediction in enumerate(predictions):
                ctx = contexts[i] if contexts and i < len(contexts) else None
                scores.append(self._score_single(prediction, ctx))

            self._last_per_item = scores
            avg = sum(scores) / len(scores) if scores else 0.0
            return {"hallucination": avg}
        finally:
            self._unload_pipeline()

    def _score_single(self, prediction: str, context: list[str] | None) -> float:
        """Score a single prediction."""
        if not context:
            return 1.0
        if not prediction or not prediction.strip():
            return 0.0

        if self.method == "nli":
            return self._detect_nli_hallucination(prediction, context)
        elif self.method == "simple":
            return self._detect_simple_hallucination(prediction, context)
        elif self.method == "claim_analysis":
            return self._detect_nli_hallucination(prediction, context)
        else:
            raise ValueError(f"Unknown hallucination method: {self.method}")

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

    def _detect_nli_hallucination(self, prediction: str, context: list[str]) -> float:
        """Detect hallucination using NLI."""
        nli = self._get_nli_pipeline()

        max_entailment = 0.0
        max_contradiction = 0.0

        for passage in context:
            if not passage or not passage.strip():
                continue

            # A4 fix: pass premise/hypothesis as dict pair, not literal [SEP]
            result = nli({"text": passage, "text_pair": prediction}, top_k=None)

            for label_result in result:
                label = label_result["label"].lower()
                score = label_result["score"]
                if "entailment" in label:
                    max_entailment = max(max_entailment, score)
                elif "contradiction" in label:
                    max_contradiction = max(max_contradiction, score)

        if max_contradiction > 0.7:
            return float(max_contradiction)

        return float(1.0 - max_entailment)

    def _detect_simple_hallucination(self, prediction: str, context: list[str]) -> float:
        """Simple token-based hallucination detection."""
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
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "they",
            "their",
        }
        content_tokens = pred_tokens - stopwords
        if not content_tokens:
            return 0.0

        context_tokens = set(" ".join(context).lower().split())
        unsupported = content_tokens - context_tokens
        return float(len(unsupported) / len(content_tokens))
