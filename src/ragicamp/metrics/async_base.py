"""Base class for async API-based metrics.

This module provides a base class for metrics that call external APIs
(like OpenAI for LLM-as-a-judge), with support for:
- Async parallel execution
- Rate limiting via semaphore
- Progress tracking
- Backward-compatible sync interface

Example:
    >>> class MyLLMMetric(AsyncAPIMetric):
    ...     async def acompute_single(self, prediction, reference, question=None, **kwargs):
    ...         # Call OpenAI API
    ...         response = await self.client.chat.completions.create(...)
    ...         return self._parse_score(response)
    >>>
    >>> metric = MyLLMMetric(max_concurrent=10)
    >>> scores = metric.compute(predictions, references)  # Sync interface
    >>> # Or use async directly:
    >>> scores = await metric.acompute(predictions, references)
"""

import asyncio
import random
import time
from abc import abstractmethod
from typing import Any

from tqdm.asyncio import tqdm as atqdm

from ragicamp.core.logging import get_logger
from ragicamp.metrics.base import Metric

logger = get_logger(__name__)


class AsyncAPIMetric(Metric):
    """Base class for metrics that call external APIs asynchronously.

    Subclasses should implement `acompute_single()` which handles a single
    prediction-reference pair (1-to-1). The base class handles parallelization,
    rate limiting, and progress tracking.

    Multi-reference aggregation is handled externally by compute_metrics_batched().

    Attributes:
        max_concurrent: Maximum number of concurrent API calls
        show_progress: Whether to show progress bar
    """

    # Default RPM limit for API calls (0 = unlimited)
    DEFAULT_RPM_LIMIT: int = 1200
    # Retry config for transient errors (429, 5xx)
    MAX_RETRIES: int = 5
    RETRY_BASE_DELAY: float = 2.0

    def __init__(
        self,
        name: str,
        max_concurrent: int = 5,
        rpm_limit: int | None = None,
        show_progress: bool = True,
        **kwargs: Any,
    ):
        """Initialize async API metric.

        Args:
            name: Metric identifier
            max_concurrent: Maximum concurrent API calls (default: 5)
            rpm_limit: Maximum requests per minute (default: 1200, 0 = unlimited)
            show_progress: Show progress bar during computation
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.max_concurrent = max_concurrent
        self.rpm_limit = rpm_limit if rpm_limit is not None else self.DEFAULT_RPM_LIMIT
        self.show_progress = show_progress

    @abstractmethod
    async def acompute_single(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute metric for a single prediction-reference pair (async, 1-to-1).

        Subclasses must implement this method.

        Args:
            prediction: Predicted answer
            reference: Single reference answer
            question: Optional question for context
            **kwargs: Additional parameters

        Returns:
            Dict of metric scores for this single item
        """
        pass

    async def acompute(
        self,
        predictions: list[str],
        references: list[str],
        questions: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute metric for all predictions using async parallel execution.

        Args:
            predictions: List of predicted answers
            references: List of reference answers (one per prediction)
            questions: Optional list of questions for context
            **kwargs: Additional parameters

        Returns:
            Dict of aggregated metric scores
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Token-bucket rate limiter for RPM enforcement
        rpm = self.rpm_limit
        if rpm > 0:
            min_interval = 60.0 / rpm  # seconds between requests
            last_request_time: list[float] = [0.0]  # mutable container for closure
            rpm_lock = asyncio.Lock()
        else:
            min_interval = 0.0
            last_request_time = [0.0]
            rpm_lock = None

        async def rate_limited_compute(
            idx: int,
            pred: str,
            ref: str,
            q: str | None,
        ) -> dict[str, float]:
            """Compute with concurrency, RPM rate limiting, and retry on 429."""
            async with semaphore:
                for attempt in range(self.MAX_RETRIES + 1):
                    # Enforce RPM limit by spacing out request starts
                    if rpm_lock is not None:
                        async with rpm_lock:
                            now = time.monotonic()
                            wait = last_request_time[0] + min_interval - now
                            if wait > 0:
                                await asyncio.sleep(wait)
                            last_request_time[0] = time.monotonic()
                    try:
                        return await self.acompute_single(pred, ref, q, **kwargs)
                    except Exception as e:
                        is_retryable = "429" in str(e) or "5" == str(e)[:1]
                        if is_retryable and attempt < self.MAX_RETRIES:
                            delay = self.RETRY_BASE_DELAY * (2**attempt) + random.random()
                            logger.info(
                                "Item %d: 429/5xx, retry %d/%d in %.1fs",
                                idx,
                                attempt + 1,
                                self.MAX_RETRIES,
                                delay,
                            )
                            await asyncio.sleep(delay)
                            continue
                        # Non-retryable or exhausted retries
                        logger.warning("Error computing metric for item %d: %s", idx, e)
                        return {self.name: 0.0, f"{self.name}_error": 1.0}
            # Should never reach here, but satisfy type checker
            return {self.name: 0.0, f"{self.name}_error": 1.0}

        # Build tasks
        tasks = []
        for i, (pred, ref) in enumerate(zip(predictions, references, strict=True)):
            q = questions[i] if questions and i < len(questions) else None
            tasks.append(rate_limited_compute(i, pred, ref, q))

        # Execute with progress bar
        if self.show_progress:
            results = await atqdm.gather(
                *tasks,
                desc=f"Computing {self.name}",
                total=len(tasks),
            )
        else:
            results = await asyncio.gather(*tasks)

        # Store per-item results for detailed analysis
        self._last_results = results
        self._last_per_item = [r.get(self.name, 0.0) for r in results]

        # Aggregate results
        return self._aggregate_results(results)

    def get_per_item_scores(self) -> list[float]:
        """Get per-item main scores from last compute() call."""
        results = getattr(self, "_last_results", [])
        return [r.get(self.name, 0.0) for r in results]

    def get_per_item_results(self) -> list[dict[str, float]]:
        """Get full per-item result dicts from last compute() call."""
        return getattr(self, "_last_results", [])

    def _aggregate_results(self, results: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate individual results into final metrics.

        Args:
            results: List of per-item result dictionaries

        Returns:
            Aggregated metrics dictionary
        """
        if not results:
            return {self.name: 0.0}

        # Collect all metric keys
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())

        # Compute mean for each metric
        aggregated = {}
        for key in all_keys:
            values = [r.get(key, 0.0) for r in results if key in r]
            if values:
                aggregated[key] = sum(values) / len(values)

        return aggregated

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        questions: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute metric synchronously (wrapper around async).

        This provides backward compatibility with the sync Metric interface.

        Args:
            predictions: List of predicted answers
            references: List of reference answers (one per prediction)
            questions: Optional list of questions
            **kwargs: Additional parameters

        Returns:
            Dict of metric scores
        """
        # Check if we're already in an event loop
        try:
            asyncio.get_running_loop()
            # We're in an async context, can't use asyncio.run()
            # Create a new thread to run the async code
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, self.acompute(predictions, references, questions, **kwargs)
                )
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(self.acompute(predictions, references, questions, **kwargs))

    def compute_single(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute metric for a single item (sync wrapper).

        Args:
            prediction: Predicted answer
            reference: Single reference answer
            question: Optional question for context
            **kwargs: Additional parameters

        Returns:
            Dict of metric scores
        """
        return self.compute(
            [prediction],
            [reference],
            [question] if question else None,
            **kwargs,
        )
