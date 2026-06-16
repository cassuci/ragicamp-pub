"""Structured logging for RAGiCamp.

Provides consistent, configurable logging across all modules.

Usage:
    from ragicamp.core import get_logger

    logger = get_logger(__name__)
    logger.info("Starting evaluation")
    logger.debug("Config: %s", config)
    logger.warning("GPU memory low: %d MB", available_mb)
    logger.error("Failed to load model", exc_info=True)
"""

import logging
import os
import sys
from pathlib import Path

# Default format for console output
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%H:%M:%S"

# Detailed format for file output
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Environment variable for log level
LOG_LEVEL_ENV = "RAGICAMP_LOG_LEVEL"

# Track if logging has been configured
_configured = False


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels for terminal output."""

    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, datefmt: str, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = self.COLORS.get(record.levelno, "")
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Get a configured logger for a module.

    Args:
        name: Logger name, typically __name__
        level: Optional override for log level

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing %d examples", 100)
    """
    # Auto-configure on first call
    if not _configured:
        configure_logging()

    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    return logger


def configure_logging(
    level: int | str | None = None,
    format_string: str | None = None,
    log_file: str | Path | None = None,
    use_colors: bool = True,
    quiet: bool = False,
) -> None:
    """Configure logging for RAGiCamp.

    Should be called once at application startup. If not called explicitly,
    get_logger() will call it with defaults.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL or int)
        format_string: Custom format string
        log_file: Optional file path to also log to
        use_colors: Whether to use colored output (default: True)
        quiet: If True, only show WARNING and above

    Environment Variables:
        RAGICAMP_LOG_LEVEL: Override log level (e.g., "DEBUG", "INFO")

    Example:
        >>> from ragicamp.core import configure_logging
        >>> configure_logging(level="DEBUG", log_file="experiment.log")
    """
    global _configured

    # Determine log level
    if level is None:
        env_level = os.environ.get(LOG_LEVEL_ENV)
        if env_level:
            level = env_level
        elif quiet:
            level = logging.WARNING
        else:
            level = logging.INFO

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger for ragicamp
    root_logger = logging.getLogger("ragicamp")
    root_logger.setLevel(level)
    root_logger.propagate = False  # Don't propagate to Python root logger

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        format_string or DEFAULT_FORMAT,
        DEFAULT_DATE_FORMAT,
        use_colors=use_colors,
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_formatter = logging.Formatter(FILE_FORMAT, FILE_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy_logger in [
        "transformers",
        "tokenizers",
        "datasets",
        "httpx",
        "urllib3",
        "filelock",
        "vllm",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _configured = True


def log_exception(
    logger: logging.Logger,
    message: str,
    exception: Exception,
    level: int = logging.ERROR,
) -> None:
    """Log an exception with full context.

    Args:
        logger: Logger instance
        message: Context message
        exception: The exception to log
        level: Log level (default: ERROR)

    Example:
        >>> try:
        ...     do_something()
        ... except Exception as e:
        ...     log_exception(logger, "Operation failed", e)
    """
    logger.log(
        level,
        "%s: %s: %s",
        message,
        type(exception).__name__,
        str(exception),
        exc_info=True,
    )


class LogContext:
    """Context manager for structured logging with context.

    Example:
        >>> with LogContext(logger, "Evaluating", dataset="NQ", model="gemma"):
        ...     # Logs: "Evaluating started (dataset=NQ, model=gemma)"
        ...     do_evaluation()
        ...     # Logs: "Evaluating completed in 5.2s"
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        level: int = logging.INFO,
        **context,
    ):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.context = context
        self.start_time = None

    def __enter__(self):
        import time

        self.start_time = time.time()

        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        if context_str:
            self.logger.log(self.level, "%s started (%s)", self.operation, context_str)
        else:
            self.logger.log(self.level, "%s started", self.operation)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        elapsed = time.time() - self.start_time

        if exc_type is not None:
            self.logger.error(
                "%s failed after %.2fs: %s",
                self.operation,
                elapsed,
                exc_val,
            )
        else:
            self.logger.log(
                self.level,
                "%s completed in %.2fs",
                self.operation,
                elapsed,
            )

        return False  # Don't suppress exceptions


def add_file_handler(
    log_file: str | Path,
    level: int = logging.DEBUG,
) -> Path:
    """Add a file handler to the ragicamp root logger.

    Use this to capture all logs to a file without disrupting existing
    console logging. The file receives all messages at the specified level
    (default: DEBUG), giving a complete record for post-mortem debugging.

    Args:
        log_file: Path for the log file (parent dirs created automatically)
        level: Minimum level for the file handler (default: DEBUG)

    Returns:
        Resolved path to the log file

    Example:
        >>> from ragicamp.core.logging import add_file_handler
        >>> log_path = add_file_handler("outputs/my_study/study.log")
        >>> # All ragicamp.* logs now also go to that file
    """
    if not _configured:
        configure_logging()

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("ragicamp")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, FILE_DATE_FORMAT))
    root_logger.addHandler(file_handler)

    root_logger.info("File logging enabled: %s", log_path)
    return log_path


def create_experiment_logger(
    experiment_name: str,
    output_dir: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger for a specific experiment.

    Creates both console and file handlers, with the file going to
    the experiment's output directory.

    Args:
        experiment_name: Name of the experiment
        output_dir: Directory for log file (default: outputs/)
        level: Log level

    Returns:
        Configured logger

    Example:
        >>> logger = create_experiment_logger("gemma_baseline", Path("outputs/run1"))
        >>> # Logs to console AND outputs/run1/gemma_baseline.log
    """
    output_dir = output_dir or Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"{experiment_name}.log"

    logger = logging.getLogger(f"ragicamp.experiment.{experiment_name}")
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT))
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, FILE_DATE_FORMAT))
    logger.addHandler(file_handler)

    logger.info("Experiment logger initialized: %s", log_file)

    return logger
