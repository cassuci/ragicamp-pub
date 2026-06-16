"""BERTScore metric implementation with lazy loading and proper GPU cleanup."""

import gc
import os
from typing import Any

# Fix matplotlib backend BEFORE any imports that might use it
# This prevents errors when running in non-interactive environments (scripts vs notebooks)
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

from ragicamp.core.logging import get_logger
from ragicamp.metrics.base import Metric

logger = get_logger(__name__)


class BERTScoreMetric(Metric):
    """BERTScore for semantic similarity evaluation.

    Uses lazy loading - model is only loaded when compute() is called,
    and unloaded immediately after to free GPU memory.
    """

    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli", **kwargs: Any):
        """Initialize BERTScore metric.

        Args:
            model_type: Model to use for computing BERTScore
            **kwargs: Additional configuration
        """
        super().__init__(name="bertscore", **kwargs)
        self.model_type = model_type
        self._scorer: Any | None = None  # Lazy loaded

    def _load_scorer(self) -> None:
        """Load the BERTScore model (lazy loading)."""
        if self._scorer is not None:
            return

        try:
            import torch
            from bert_score import BERTScorer

            # Use GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading BERTScore model: %s (device=%s)", self.model_type, device)
            self._scorer = BERTScorer(model_type=self.model_type, lang="en", device=device)
        except ImportError as e:
            raise ImportError(
                "bert-score is required for BERTScoreMetric. Install with: pip install bert-score"
            ) from e

    def _unload_scorer(self) -> None:
        """Unload the BERTScore model to free GPU memory."""
        if self._scorer is None:
            return

        import torch

        # Delete the scorer and its model
        if hasattr(self._scorer, "model"):
            del self._scorer.model
        del self._scorer
        self._scorer = None

        # Force GPU memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("BERTScore model unloaded")

    def compute(
        self, predictions: list[str], references: list[str], **kwargs: Any
    ) -> dict[str, float]:
        """Compute BERTScore with automatic batching and OOM retry.

        Loads model, computes scores in batches, then unloads to free GPU memory.
        If OOM occurs, automatically retries with smaller batch size.

        Args:
            predictions: List of predicted answers
            references: List of reference answers (one per prediction)

        Returns:
            Dict with precision, recall, and F1 scores
        """
        import torch

        try:
            # Load model (lazy)
            self._load_scorer()

            # Start with batch_size=128, halve on OOM
            batch_size = 128
            min_batch_size = 1
            all_P: list = []
            all_R: list = []
            all_F1: list = []
            processed = 0  # Track how many items completed successfully

            while processed < len(predictions):
                try:
                    for i in range(processed, len(predictions), batch_size):
                        batch_preds = predictions[i : i + batch_size]
                        batch_refs = references[i : i + batch_size]

                        # Clear cache before each batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        P, R, F1 = self._scorer.score(batch_preds, batch_refs)
                        all_P.append(P)
                        all_R.append(R)
                        all_F1.append(F1)
                        processed += len(batch_preds)

                    break  # All done

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" in str(e).lower() or isinstance(
                        e, torch.cuda.OutOfMemoryError
                    ):
                        # Clear memory and retry failed batch with smaller size
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                        old_batch_size = batch_size
                        batch_size = batch_size // 2
                        if batch_size >= min_batch_size:
                            logger.warning(
                                "OOM with batch_size=%d, retrying from item %d with %d",
                                old_batch_size,
                                processed,
                                batch_size,
                            )
                        else:
                            raise RuntimeError(
                                "BERTScore OOM even with batch_size=1. "
                                "Text too long or GPU memory insufficient."
                            ) from e
                    else:
                        raise

            # Concatenate results
            P = torch.cat(all_P)
            R = torch.cat(all_R)
            F1 = torch.cat(all_F1)

            # Store per-item scores for detailed analysis
            self._last_scores = F1.tolist()
            self._last_per_item = self._last_scores  # B5 fix: base class compat

            return {
                "bertscore": F1.mean().item(),  # B6 fix: primary key matches metric.name
                "bertscore_precision": P.mean().item(),
                "bertscore_recall": R.mean().item(),
                "bertscore_f1": F1.mean().item(),
            }

        finally:
            # ALWAYS unload after computation to free GPU
            self._unload_scorer()

    def get_per_item_scores(self) -> list[float]:
        """Get per-item F1 scores from last compute() call."""
        return getattr(self, "_last_scores", [])
