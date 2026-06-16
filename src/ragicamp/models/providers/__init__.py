"""Model providers with lazy loading and lifecycle management.

Design principles:
- Models are not loaded until explicitly requested
- Context managers ensure proper cleanup
- Each model can use full GPU when it has exclusive access
- True batch operations for throughput

Split into sub-modules for maintainability:
- base: ModelProvider ABC
- embedder: EmbedderProvider, ManagedEmbedder, wrappers
- generator: GeneratorProvider, Generator, wrappers
- reranker: RerankerProvider, RerankerWrapper
- gpu_profile: GPUProfile auto-detection
"""

import os
import shutil

# Set vLLM attention backend fallback if nvcc is not available
# This must happen before vLLM is imported
if not shutil.which("nvcc") and "VLLM_ATTENTION_BACKEND" not in os.environ:
    os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"

from .base import ModelProvider
from .embedder import (
    EmbedderConfig,
    EmbedderProvider,
    ManagedEmbedder,
    SentenceTransformerWrapper,
    VLLMEmbedderWrapper,
)
from .generator import (
    Generator,
    GeneratorConfig,
    GeneratorProvider,
    HFGeneratorWrapper,
    VLLMGeneratorWrapper,
)
from .gpu_profile import GPUProfile
from .reranker import RerankerConfig, RerankerProvider, RerankerWrapper

__all__ = [
    # Base
    "ModelProvider",
    # Embedder
    "EmbedderConfig",
    "EmbedderProvider",
    "ManagedEmbedder",
    "VLLMEmbedderWrapper",
    "SentenceTransformerWrapper",
    # Generator
    "GeneratorConfig",
    "GeneratorProvider",
    "Generator",
    "VLLMGeneratorWrapper",
    "HFGeneratorWrapper",
    # Reranker
    "RerankerConfig",
    "RerankerProvider",
    "RerankerWrapper",
    # GPU
    "GPUProfile",
]
