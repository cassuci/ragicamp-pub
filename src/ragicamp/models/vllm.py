"""vLLM model implementation with PagedAttention and efficient memory management.

vLLM provides:
- PagedAttention: Eliminates KV cache fragmentation, enabling longer contexts
- Continuous batching: Better throughput for multiple requests
- Efficient memory management: Use more of your GPU for actual computation

This is the recommended backend for running local models with long context
without needing to quantize model weights.

GPU Memory Partitioning:
    vLLM defaults to 90% GPU memory utilization since FAISS runs on CPU.
    Adjust via gpu_memory_utilization param if needed.
"""

import gc
from typing import Any

from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.models.base import LanguageModel

logger = get_logger(__name__)

# Lazy import to avoid requiring vllm when not using this model
_vllm_available = None


def _check_vllm_available() -> bool:
    """Check if vLLM is available."""
    global _vllm_available
    if _vllm_available is None:
        try:
            import vllm  # noqa: F401

            _vllm_available = True
        except ImportError:
            _vllm_available = False
    return _vllm_available


class VLLMModel(LanguageModel):
    """Language model implementation using vLLM for efficient inference.

    vLLM automatically provides:
    - PagedAttention for efficient KV cache management
    - Continuous batching for throughput
    - Memory-efficient attention

    Supports optional quantization but defaults to full precision (BF16).
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        quantization: str | None = None,
        gpu_memory_utilization: float | None = None,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        enforce_eager: bool = False,
        kv_cache_dtype: str | None = None,
        enable_prefix_caching: bool = True,
        **kwargs: Any,
    ):
        """Initialize vLLM model.

        Args:
            model_name: HuggingFace model identifier or local path
            dtype: Model weight dtype ('bfloat16', 'float16', 'float32', 'auto')
                   Default is 'bfloat16' for full precision without quantization.
            quantization: Optional quantization method ('awq', 'gptq', 'squeezellm', None)
                         Default is None (no quantization - full precision).
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0).
                                   Default is Defaults.VLLM_GPU_MEMORY_FRACTION (0.90).
            max_model_len: Maximum context length. If None, uses model's default.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            trust_remote_code: Whether to trust remote code in model repos.
            enforce_eager: Disable CUDA graph optimization (useful for debugging).
            kv_cache_dtype: KV cache dtype ('auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3').
                           Use 'fp8' for ~2x KV cache memory reduction.
            enable_prefix_caching: Enable automatic prefix caching for efficiency.
            **kwargs: Additional vLLM engine arguments.
        """
        # Default to full GPU - agents manage loading sequence
        if gpu_memory_utilization is None:
            gpu_memory_utilization = Defaults.VLLM_GPU_MEMORY_FRACTION_FULL
        super().__init__(model_name, **kwargs)

        if not _check_vllm_available():
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm\n"
                "Note: vLLM requires CUDA and a compatible GPU."
            )

        from vllm import LLM

        self.dtype = dtype
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self._is_loaded = False

        # Build engine arguments
        engine_kwargs = {
            "model": model_name,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "enforce_eager": enforce_eager,
            "enable_prefix_caching": enable_prefix_caching,
        }

        # Add optional parameters
        if quantization:
            engine_kwargs["quantization"] = quantization
            logger.info("Using quantization: %s", quantization)

        if max_model_len:
            engine_kwargs["max_model_len"] = max_model_len

        if tensor_parallel_size > 1:
            engine_kwargs["tensor_parallel_size"] = tensor_parallel_size

        if kv_cache_dtype:
            engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
            logger.info("Using KV cache dtype: %s", kv_cache_dtype)

        # Merge any additional kwargs
        engine_kwargs.update(kwargs)

        logger.info(
            "Loading vLLM model: %s (dtype=%s, quantization=%s, gpu_util=%.0f%%)",
            model_name,
            dtype,
            quantization or "none",
            gpu_memory_utilization * 100,
        )

        self.llm = LLM(**engine_kwargs)
        self._is_loaded = True

        logger.info("vLLM model loaded successfully with PagedAttention enabled")

    def unload(self) -> None:
        """Unload model to free GPU memory.

        Call this when done with the model to release GPU resources.
        """
        if not self._is_loaded:
            return

        try:
            import torch

            # Delete the LLM engine
            if hasattr(self, "llm") and self.llm is not None:
                del self.llm
                self.llm = None

            # Force GPU memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._is_loaded = False
            logger.info("vLLM model %s unloaded from GPU", self.model_name)

        except Exception as e:
            logger.warning("Error during vLLM model unload: %s", e)

    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._is_loaded

    def generate(
        self,
        prompt: str | list[str],
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> str | list[str]:
        """Generate text using vLLM.

        Args:
            prompt: Single prompt string or list of prompts
            max_tokens: Maximum tokens to generate (default: 256)
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Nucleus sampling parameter
            stop: Optional stop sequences
            **kwargs: Additional sampling parameters

        Returns:
            Generated text (single string or list matching input)
        """
        from vllm import SamplingParams

        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        # Build sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens or 256,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )

        # Generate (disable tqdm to avoid noisy per-request progress bars)
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        # Extract generated text
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)

        return results if is_batch else results[0]

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for texts.

        Note: vLLM is optimized for generation, not embeddings.
        For embeddings, consider using sentence-transformers directly.

        Raises:
            NotImplementedError: vLLM doesn't support embeddings natively
        """
        raise NotImplementedError(
            "vLLM is optimized for generation, not embeddings. "
            "Use sentence-transformers or a dedicated embedding model."
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        tokenizer = self.llm.get_tokenizer()
        return len(tokenizer.encode(text))

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "is_loaded": self._is_loaded,
            "backend": "vllm",
            "features": [
                "paged_attention",
                "continuous_batching",
                "prefix_caching",
            ],
        }
