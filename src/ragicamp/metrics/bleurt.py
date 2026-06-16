"""BLEURT metric implementation using PyTorch (via bleurt-pytorch)."""

import gc
from typing import Any

from ragicamp.core.logging import get_logger
from ragicamp.metrics.base import Metric

logger = get_logger(__name__)


class BLEURTMetric(Metric):
    """BLEURT using PyTorch implementation (bleurt-pytorch).

    Uses the lucadiliello/bleurt-pytorch package which provides a PyTorch
    reimplementation of BLEURT, compatible with modern CUDA versions.
    """

    def __init__(self, model_name: str = "lucadiliello/BLEURT-20", **kwargs: Any):
        """Initialize BLEURT metric.

        Args:
            model_name: HuggingFace model name for BLEURT
                Options: lucadiliello/BLEURT-20, lucadiliello/BLEURT-20-D12,
                         lucadiliello/BLEURT-20-D6, lucadiliello/BLEURT-20-D3
                Default: lucadiliello/BLEURT-20 (best quality)
            **kwargs: Additional configuration
        """
        super().__init__(name="bleurt", **kwargs)
        self.model_name = model_name
        self._scorer: Any | None = None
        self._tokenizer: Any | None = None

    def _load_scorer(self) -> None:
        """Load the BLEURT model (lazy loading)."""
        if self._scorer is not None:
            return

        try:
            import torch
            from bleurt_pytorch import (
                BleurtForSequenceClassification,
                BleurtTokenizer,
            )
        except ImportError as e:
            raise ImportError(
                "bleurt-pytorch is required for BLEURTMetric. "
                "Install with: uv pip install bleurt-pytorch"
            ) from e

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading BLEURT model: %s (device=%s)", self.model_name, device)

        self._tokenizer = BleurtTokenizer.from_pretrained(self.model_name)
        self._scorer = BleurtForSequenceClassification.from_pretrained(self.model_name)
        self._scorer = self._scorer.to(device)
        self._scorer.eval()
        self._device = device

        logger.info("BLEURT (PyTorch) loaded")

    def _unload_scorer(self) -> None:
        """Unload the BLEURT model to free GPU memory."""
        if self._scorer is None:
            return

        import torch

        del self._scorer
        del self._tokenizer
        self._scorer = None
        self._tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("BLEURT model unloaded")

    def compute(
        self, predictions: list[str], references: list[str], **kwargs: Any
    ) -> dict[str, float]:
        """Compute BLEURT scores with automatic batching and OOM retry.

        Args:
            predictions: List of predicted answers
            references: List of reference answers (one per prediction)

        Returns:
            Dict with average BLEURT score
        """
        import torch

        try:
            self._load_scorer()

            # Start with batch_size=128, halve on OOM
            batch_size = 128
            min_batch_size = 1

            while batch_size >= min_batch_size:
                try:
                    all_scores = []

                    with torch.no_grad():
                        for i in range(0, len(predictions), batch_size):
                            batch_preds = predictions[i : i + batch_size]
                            batch_refs = references[i : i + batch_size]

                            # Tokenize
                            inputs = self._tokenizer(
                                batch_refs,
                                batch_preds,
                                padding=True,
                                truncation=True,
                                max_length=512,
                                return_tensors="pt",
                            )
                            inputs = {k: v.to(self._device) for k, v in inputs.items()}

                            # Get scores
                            outputs = self._scorer(**inputs)
                            scores = outputs.logits.flatten().cpu().tolist()
                            all_scores.extend(scores)

                            # Clear cache between batches
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    # Store per-item scores
                    self._last_scores = all_scores
                    self._last_per_item = all_scores  # B5 fix: base class compat

                    return {
                        "bleurt": float(sum(all_scores) / len(all_scores)) if all_scores else 0.0
                    }

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" in str(e).lower() or isinstance(
                        e, torch.cuda.OutOfMemoryError
                    ):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                        old_batch_size = batch_size
                        batch_size = batch_size // 2
                        if batch_size >= min_batch_size:
                            logger.warning(
                                "OOM with batch_size=%d, retrying with %d",
                                old_batch_size,
                                batch_size,
                            )
                        else:
                            raise RuntimeError(
                                "BLEURT OOM even with batch_size=1. "
                                "Text too long or GPU memory insufficient."
                            ) from e
                    else:
                        raise

            raise RuntimeError("BLEURT computation failed")

        finally:
            self._unload_scorer()

    def get_per_item_scores(self) -> list[float]:
        """Get per-item scores from last compute() call."""
        return getattr(self, "_last_scores", [])
