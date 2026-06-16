"""GPU profile detection for optimized model loading.

Replaces hardcoded magic numbers with a structured dataclass
that auto-detects GPU capabilities.
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class GPUProfile:
    """GPU capability profile for optimizing vLLM settings.

    Use ``GPUProfile.detect()`` to auto-detect the current GPU tier
    and get recommended vLLM parameters.
    """

    tier: str  # "high", "mid", "low", "cpu"
    gpu_mem_gb: float
    max_num_seqs: int | None
    max_num_batched_tokens: int | None

    # Tier thresholds (GB)
    _HIGH_THRESHOLD: ClassVar[float] = 160  # B200, A100-80GB×2, etc.
    _MID_THRESHOLD: ClassVar[float] = 80  # A100-80GB, H100, etc.

    def embedder_batch_params(self) -> tuple[int, int]:
        """Get recommended (max_num_seqs, max_num_batched_tokens) for vLLM embedder.

        Embedders need much larger batch params than generators since embeddings
        are smaller and don't generate tokens.
        """
        if self.gpu_mem_gb >= 160:
            return 16384, 262144  # 256k tokens
        elif self.gpu_mem_gb >= 80:
            return 8192, 131072  # 128k tokens
        elif self.gpu_mem_gb >= 40:
            return 4096, 65536  # 64k tokens
        elif self.gpu_mem_gb >= 16:
            return 1024, 16384  # 16k tokens
        else:
            return 256, 4096  # 4k tokens (CPU/small GPU)

    @classmethod
    def detect(cls) -> "GPUProfile":
        """Detect GPU and return an appropriate profile.

        Returns:
            GPUProfile with recommended settings for the detected GPU.
        """
        gpu_mem_gb = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except ImportError:
            pass

        if gpu_mem_gb >= cls._HIGH_THRESHOLD:
            return cls(
                tier="high",
                gpu_mem_gb=gpu_mem_gb,
                max_num_seqs=512,
                max_num_batched_tokens=32768,
            )
        elif gpu_mem_gb >= cls._MID_THRESHOLD:
            return cls(
                tier="mid",
                gpu_mem_gb=gpu_mem_gb,
                max_num_seqs=256,
                max_num_batched_tokens=16384,
            )
        elif gpu_mem_gb > 0:
            return cls(
                tier="low",
                gpu_mem_gb=gpu_mem_gb,
                max_num_seqs=None,  # Use vLLM defaults
                max_num_batched_tokens=None,
            )
        else:
            return cls(
                tier="cpu",
                gpu_mem_gb=0,
                max_num_seqs=None,
                max_num_batched_tokens=None,
            )
