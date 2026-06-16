"""Language model interfaces.

New architecture:
- Providers: lazy loading with context managers
- Legacy: direct model classes (still supported)
"""

from ragicamp.models.base import LanguageModel
from ragicamp.models.embedder import Embedder, create_embedder
from ragicamp.models.huggingface import HuggingFaceModel
from ragicamp.models.openai import OpenAIModel
from ragicamp.models.providers import (
    EmbedderConfig,
    EmbedderProvider,
    GeneratorConfig,
    GeneratorProvider,
    ModelProvider,
    RerankerConfig,
    RerankerProvider,
)

# vLLM is optional - only import if available
try:
    from ragicamp.models.vllm import VLLMModel

    _VLLM_AVAILABLE = True
except ImportError:
    VLLMModel = None  # type: ignore
    _VLLM_AVAILABLE = False

__all__ = [
    # Providers (new architecture)
    "ModelProvider",
    "EmbedderProvider",
    "EmbedderConfig",
    "GeneratorProvider",
    "GeneratorConfig",
    "RerankerProvider",
    "RerankerConfig",
    # Legacy interfaces
    "LanguageModel",
    "HuggingFaceModel",
    "OpenAIModel",
    "VLLMModel",
    "Embedder",
    "create_embedder",
]
