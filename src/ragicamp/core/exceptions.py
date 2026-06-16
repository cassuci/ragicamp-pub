"""Custom exceptions for RAGiCamp.

All exceptions inherit from RAGiCampError, making it easy to:
- Catch all RAGiCamp-specific errors
- Distinguish from third-party errors
- Add context to error messages

Usage:
    try:
        model.generate(prompt)
    except RAGiCampError as e:
        logger.error("RAGiCamp error: %s", e)
"""

from typing import Any


class RAGiCampError(Exception):
    """Base exception for all RAGiCamp errors.

    Attributes:
        message: Human-readable error message
        details: Optional dictionary with additional context
        cause: Optional original exception that caused this error
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        self.message = message
        self.details = details or {}
        self.cause = cause
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with details."""
        msg = self.message
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            msg = f"{msg} ({detail_str})"
        if self.cause:
            msg = f"{msg} [caused by: {type(self.cause).__name__}: {self.cause}]"
        return msg


class ConfigError(RAGiCampError):
    """Configuration error.

    Raised when:
    - Missing required config field
    - Invalid config value
    - Config file not found
    - Validation failed
    """

    pass


class ModelError(RAGiCampError):
    """Model-related error.

    Raised when:
    - Model loading failed
    - Generation failed
    - Token limit exceeded
    - API error
    """

    pass


class EvaluationError(RAGiCampError):
    """Evaluation-related error.

    Raised when:
    - Metric computation failed
    - Agent crashed during evaluation
    - Resource exhausted
    - Checkpoint error
    """

    pass


class RecoverableError(RAGiCampError):
    """Recoverable error that can be retried.

    Raised when:
    - CUDA out of memory (can reduce batch size)
    - Temporary GPU errors (can retry)
    - Resource allocation failures (can retry with different config)

    These errors should be caught and handled with retry logic,
    not propagated to crash the entire experiment.

    Example:
        >>> try:
        ...     model.generate(prompts, batch_size=32)
        ... except torch.cuda.OutOfMemoryError as e:
        ...     raise RecoverableError("CUDA OOM", details={"batch_size": 32}, cause=e)
    """

    pass


# Aliases for backward compatibility
ConfigurationError = ConfigError
