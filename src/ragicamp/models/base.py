"""Base class for language models.

LEGACY INTERFACE: For new code, prefer using providers:
- GeneratorProvider: For text generation with GPU lifecycle management
- EmbedderProvider: For embeddings with GPU lifecycle management
- RerankerProvider: For cross-encoder reranking

This LanguageModel base class is still used for:
- OpenAIModel: External API calls (no GPU lifecycle needed)
- LLM Judge metric: Uses OpenAI for evaluation

See ragicamp.models.providers for the new architecture.
"""

from abc import ABC, abstractmethod
from typing import Any


class LanguageModel(ABC):
    """Base class for all language models.

    LEGACY: For local GPU models, prefer GeneratorProvider which provides:
    - Lazy loading (model not loaded until needed)
    - Context manager for automatic cleanup
    - Sequential resource management (one model at a time)

    This class is still appropriate for:
    - OpenAI/Anthropic API calls (no local GPU)
    - LLM-as-judge evaluation
    """

    def __init__(self, model_name: str, **kwargs: Any):
        """Initialize the language model.

        Args:
            model_name: Identifier for the model
            **kwargs: Model-specific configuration
        """
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str | list[str],
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> str | list[str]:
        """Generate text from a prompt.

        Args:
            prompt: Single prompt string or list of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            **kwargs: Additional generation parameters

        Returns:
            Generated text (single string or list matching input)
        """
        pass

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts.

        Override in subclasses that support embeddings (e.g. OpenAIModel).
        Generator-only models can leave the default which raises.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support embeddings")

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        # Default simple approximation - override in subclasses
        return len(text.split())

    def batch_generate(self, prompts: list[str], batch_size: int = 8, **kwargs: Any) -> list[str]:
        """Generate text for multiple prompts in batches.

        Args:
            prompts: List of prompts
            batch_size: Batch size for generation
            **kwargs: Generation parameters

        Returns:
            List of generated texts
        """
        # Default implementation - can be overridden for efficiency
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            results.extend(self.generate(batch, **kwargs))
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
