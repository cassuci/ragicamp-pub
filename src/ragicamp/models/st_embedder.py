"""SentenceTransformer-based embedder.

Wraps SentenceTransformer with a consistent interface matching the Embedder protocol.
Handles Flash Attention 2 and torch.compile optimizations.
"""

import os
from typing import TYPE_CHECKING

# Disable tokenizers parallelism to avoid fork warnings with multiprocessing
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from ragicamp.core.logging import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


class SentenceTransformerEmbedder:
    """SentenceTransformer-based embedding model.

    Provides the same interface as VLLMEmbedder for consistency.
    Supports Flash Attention 2 and torch.compile optimizations.
    """

    def __init__(
        self,
        model_name: str,
        use_flash_attn: bool = True,
        use_compile: bool = True,
        trust_remote_code: bool = True,
    ):
        """Initialize SentenceTransformer embedder.

        Args:
            model_name: HuggingFace model name
            use_flash_attn: Use Flash Attention 2 if available
            use_compile: Apply torch.compile for speedup (PyTorch 2.0+)
            trust_remote_code: Trust remote code in model
        """
        self.model_name = model_name
        self.use_flash_attn = use_flash_attn
        self.use_compile = use_compile
        self.trust_remote_code = trust_remote_code

        self._model: SentenceTransformer | None = None
        self._embedding_dim: int | None = None

    @property
    def model(self) -> "SentenceTransformer":
        """Lazy load the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceTransformer model: %s", self.model_name)

            # Check for Flash Attention 2
            model_kwargs = {}
            if self.use_flash_attn:
                try:
                    import flash_attn  # noqa: F401

                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Flash Attention 2 enabled")
                except ImportError:
                    logger.info("flash-attn not installed, using default attention")

            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                model_kwargs=model_kwargs if model_kwargs else None,
            )

            # Apply torch.compile for additional speedup
            if self.use_compile:
                try:
                    import torch

                    if hasattr(torch, "compile") and torch.cuda.is_available():
                        self._model = torch.compile(self._model, mode="reduce-overhead")
                        logger.info("Applied torch.compile() to embedding model")
                except Exception as e:
                    logger.warning("torch.compile() not applied: %s", e)

            logger.info("SentenceTransformer model loaded: %s", self.model_name)

        return self._model

    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is None:
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info("Embedding dimension: %d", self._embedding_dim)
        return self._embedding_dim

    def encode(
        self,
        sentences: list[str] | str,
        batch_size: int = 256,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Encode sentences to embeddings.

        Args:
            sentences: Single string or list of strings to encode
            batch_size: Batch size for processing
            show_progress_bar: Show progress bar
            normalize_embeddings: L2 normalize embeddings
            **kwargs: Additional arguments passed to SentenceTransformer.encode()

        Returns:
            numpy array of embeddings, shape (n_sentences, embedding_dim)
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            **kwargs,
        )

        return embeddings.astype(np.float32)

    def unload(self) -> None:
        """Unload the model from GPU memory."""
        if self._model is not None:
            import gc

            del self._model
            self._model = None
            self._embedding_dim = None
            gc.collect()

            # Clear CUDA cache
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("SentenceTransformer model unloaded: %s", self.model_name)
