"""Core infrastructure for RAGiCamp.

This module provides foundational components:
- Types: Core data types (Document, SearchResult)
- Exceptions: Custom exception hierarchy
- Logging: Structured logging configuration
- Constants: Enums and constants
"""

from ragicamp.core.constants import (
    AgentType,
    MetricType,
)
from ragicamp.core.exceptions import (
    ConfigError,
    ConfigurationError,
    EvaluationError,
    ModelError,
    RAGiCampError,
)
from ragicamp.core.logging import configure_logging, get_logger
from ragicamp.core.types import Document, SearchResult

__all__ = [
    # Core types
    "Document",
    "SearchResult",
    # Exceptions
    "RAGiCampError",
    "ConfigError",
    "ConfigurationError",  # Alias for ConfigError
    "ModelError",
    "EvaluationError",
    # Logging
    "get_logger",
    "configure_logging",
    # Constants
    "AgentType",
    "MetricType",
]
