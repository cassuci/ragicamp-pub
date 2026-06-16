"""Resource management for GPU/RAM lifecycle.

This module provides clean abstractions for managing memory during experiments.
Use context managers to ensure proper cleanup.

Example:
    with GPUMemoryManager() as gpu:
        model = load_model()
        results = model.generate(...)
    # Model automatically cleaned up, GPU memory freed

GPU Memory Partitioning:
    When using vLLM + FAISS GPU together, memory is partitioned:
    - vLLM: 60% (configurable via Defaults.VLLM_GPU_MEMORY_FRACTION)
    - FAISS: 35% (configurable via Defaults.FAISS_GPU_MEMORY_FRACTION)
    - Overhead: 5% for PyTorch, embeddings, etc.
"""

import gc
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)


class ResourceManager:
    """Centralized resource management for experiments.

    Handles GPU memory, model lifecycle, and cleanup.
    """

    @staticmethod
    def get_gpu_memory_info() -> dict:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"available": False}

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved

        return {
            "available": True,
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(free, 2),
        }

    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache."""
        gc.collect()  # Release Python refs to GPU tensors first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def clear_faiss_gpu_resources():
        """Release FAISS GPU resources to free memory.

        Call this before loading a large LLM if FAISS was using GPU.
        """
        try:
            from ragicamp.indexes.vector_index import release_faiss_gpu_resources

            release_faiss_gpu_resources()
        except ImportError:
            pass

    @staticmethod
    def print_memory_status(phase: str = ""):
        """Print current memory status."""
        info = ResourceManager.get_gpu_memory_info()
        if info["available"]:
            prefix = f"[{phase}] " if phase else ""
            logger.info(
                "%sGPU: %.1f/%.1f GiB (free: %.1f GiB)",
                prefix,
                info["allocated_gb"],
                info["total_gb"],
                info["free_gb"],
            )

    @staticmethod
    def print_memory_partitioning():
        """Print recommended GPU memory partitioning."""
        from ragicamp.core.constants import Defaults

        info = ResourceManager.get_gpu_memory_info()
        if info["available"]:
            total = info["total_gb"]
            vllm_gb = total * Defaults.VLLM_GPU_MEMORY_FRACTION
            faiss_gb = total * Defaults.FAISS_GPU_MEMORY_FRACTION
            logger.info("GPU Memory Partitioning (total: %.1f GiB):", total)
            logger.info(
                "   vLLM:  %.1f GiB (%.0f%%)", vllm_gb, Defaults.VLLM_GPU_MEMORY_FRACTION * 100
            )
            logger.info(
                "   FAISS: %.1f GiB (%.0f%%)", faiss_gb, Defaults.FAISS_GPU_MEMORY_FRACTION * 100
            )


@contextmanager
def managed_model(model_factory: Callable[[], Any], name: str = "model"):
    """Context manager for model lifecycle.

    Ensures model is properly cleaned up after use.

    Args:
        model_factory: Callable that creates the model (lazy loading)
        name: Name for logging

    Yields:
        The created model

    Example:
        def create_model():
            return HuggingFaceModel("google/gemma-2-2b-it", load_in_4bit=True)

        with managed_model(create_model, "Gemma") as model:
            output = model.generate("Hello")
        # Model automatically cleaned up
    """
    logger.info("Loading %s...", name)
    ResourceManager.print_memory_status("before")

    model = model_factory()

    ResourceManager.print_memory_status("after load")

    try:
        yield model
    finally:
        # Cleanup
        logger.info("Unloading %s...", name)

        # Try to move to CPU first (for non-quantized models)
        if hasattr(model, "model") and model.model is not None:
            try:
                if hasattr(model, "_use_quantization") and not model._use_quantization:
                    model.model = model.model.to("cpu")
            except Exception:
                pass

            # Delete the model
            del model.model
            model.model = None

        del model
        ResourceManager.clear_gpu_memory()
        ResourceManager.print_memory_status("after cleanup")


@contextmanager
def gpu_memory_scope(name: str = "operation"):
    """Context manager that ensures GPU memory is freed after scope.

    Useful for wrapping phases that load temporary models.

    Args:
        name: Name of the operation for logging

    Example:
        with gpu_memory_scope("BERTScore computation"):
            scorer = BERTScorer(...)
            scores = scorer.score(...)
        # GPU memory freed
    """
    ResourceManager.print_memory_status(f"{name} start")

    try:
        yield
    finally:
        ResourceManager.clear_gpu_memory()
        ResourceManager.print_memory_status(f"{name} end")
