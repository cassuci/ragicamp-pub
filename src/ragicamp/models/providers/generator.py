"""Generator provider with lazy loading and lifecycle management."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.utils.resource_manager import ResourceManager

from .base import ModelProvider
from .gpu_profile import GPUProfile

logger = get_logger(__name__)


@dataclass
class GeneratorConfig:
    """Configuration for generator."""

    model_name: str
    backend: str = "vllm"  # "vllm" or "hf"
    dtype: str = "auto"
    trust_remote_code: bool = True
    max_model_len: int | None = None


class GeneratorProvider(ModelProvider):
    """Provides generator with lazy loading and proper cleanup.

    Usage:
        provider = GeneratorProvider(GeneratorConfig("meta-llama/Llama-3.2-3B-Instruct"))

        with provider.load() as generator:
            answers = generator.batch_generate(prompts)
        # Generator unloaded, GPU memory freed
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self._generator = None
        self._refcount: int = 0

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @contextmanager
    def load(self, gpu_fraction: float | None = None) -> Iterator["Generator"]:
        """Load generator, yield it, then unload.

        Supports ref-counting: nested ``with provider.load()`` calls reuse
        the already-loaded model and only unload when the outermost context
        exits.  This is fully backward-compatible — single ``load()`` calls
        behave identically to before.
        """
        self._refcount += 1
        try:
            if self._refcount == 1:
                # First caller — actually load the model
                from time import perf_counter as _pc

                if gpu_fraction is None:
                    gpu_fraction = Defaults.VLLM_GPU_MEMORY_FRACTION_FULL

                logger.info(
                    "Loading generator: %s (gpu=%.0f%%)", self.model_name, gpu_fraction * 100
                )
                _t0 = _pc()

                if self.config.backend == "vllm":
                    generator = self._load_vllm(gpu_fraction)
                else:
                    generator = self._load_hf()

                _load_s = _pc() - _t0
                logger.info("Generator loaded in %.1fs: %s", _load_s, self.model_name)
                self._generator = generator
            else:
                logger.debug(
                    "Generator already loaded (refcount=%d): %s", self._refcount, self.model_name
                )

            yield self._generator

        finally:
            self._refcount -= 1
            if self._refcount == 0:
                self._unload()

    def _load_vllm(self, gpu_fraction: float) -> "Generator":
        """Load vLLM generator with GPU-optimized settings."""
        from vllm import LLM

        profile = GPUProfile.detect()

        llm_kwargs = {
            "model": self.config.model_name,
            "dtype": self.config.dtype,
            "gpu_memory_utilization": gpu_fraction,
            "trust_remote_code": self.config.trust_remote_code,
            "enable_prefix_caching": True,  # Cache common prompt prefixes
        }

        if self.config.max_model_len:
            llm_kwargs["max_model_len"] = self.config.max_model_len

        # Apply GPU-tier optimizations
        if profile.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = profile.max_num_seqs
            llm_kwargs["max_num_batched_tokens"] = profile.max_num_batched_tokens
            logger.info(
                "%s detected (%.0fGB): seqs=%d, tokens=%d",
                profile.tier,
                profile.gpu_mem_gb,
                profile.max_num_seqs,
                profile.max_num_batched_tokens,
            )

        llm = LLM(**llm_kwargs)
        return VLLMGeneratorWrapper(llm, self.config.model_name)

    def _load_hf(self) -> "Generator":
        """Load HuggingFace generator."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.dtype == "float16" else "auto",
            trust_remote_code=self.config.trust_remote_code,
            low_cpu_mem_usage=True,
        )

        if torch.cuda.is_available():
            model = model.to("cuda")

        model.eval()

        return HFGeneratorWrapper(model, tokenizer, self.config.model_name)

    def _unload(self):
        """Unload generator and free GPU memory."""
        if self._generator is not None:
            if hasattr(self._generator, "unload"):
                self._generator.unload()
            self._generator = None

        ResourceManager.clear_gpu_memory()
        logger.info("Generator unloaded: %s", self.model_name)


class Generator(ABC):
    """Protocol for generators with batch operations."""

    @abstractmethod
    def batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses for multiple prompts."""
        ...

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response for single prompt."""
        return self.batch_generate([prompt], **kwargs)[0]

    def unload(self) -> None:  # noqa: B027
        """Unload model (optional override)."""


class VLLMGeneratorWrapper(Generator):
    """Wrapper around vLLM LLM implementing Generator protocol."""

    def __init__(self, llm, model_name: str):
        self._llm = llm
        self._model_name = model_name

    def batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", Defaults.MAX_TOKENS),
            temperature=kwargs.get("temperature", 0.0),
        )

        outputs = self._llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def unload(self):
        # Destroy NCCL process group before deleting the engine to avoid
        # "destroy_process_group() was not called" warning at exit.
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except (RuntimeError, ImportError):
            pass
        del self._llm


class HFGeneratorWrapper(Generator):
    """Wrapper around HuggingFace model implementing Generator protocol."""

    def __init__(self, model, tokenizer, model_name: str):
        self._model = model
        self._tokenizer = tokenizer
        self._model_name = model_name

    def batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
        import torch

        max_tokens = kwargs.get("max_tokens", Defaults.MAX_TOKENS)
        temperature = kwargs.get("temperature", 0.0)

        # Tokenize with padding
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=Defaults.MAX_INPUT_LENGTH,
        )

        # Move to model device
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only new tokens
        input_length = inputs["input_ids"].shape[1]
        generated = outputs[:, input_length:]

        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)

    def unload(self):
        import gc

        import torch

        del self._model
        del self._tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
