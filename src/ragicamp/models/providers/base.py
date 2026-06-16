"""Base model provider protocol.

Defines the abstract interface all providers must implement:
lazy loading with context-manager lifecycle.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class ModelProvider(ABC):
    """Base class for lazy model loading with lifecycle management."""

    @abstractmethod
    @contextmanager
    def load(self, gpu_fraction: float | None = None) -> Iterator[Any]:
        """Load model and yield it, then unload on exit.

        Usage:
            with provider.load() as model:
                result = model.encode(texts)
            # Model automatically unloaded
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name/identifier."""
        ...
