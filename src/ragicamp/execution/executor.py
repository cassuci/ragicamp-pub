"""Resilient executor for agent calls.

This module handles the execution-layer concerns:
- Batch processing for throughput
- GPU memory management between batches
- Error handling and progress tracking

The executor is agnostic to what it's executing - it just needs something
with a `batch_answer` or `answer` method.

Note: With vLLM as the backend, batch size just controls how many prompts
are queued at once. vLLM handles its own GPU memory management internally.
"""

from collections.abc import Callable
from typing import Any, Protocol

from tqdm import tqdm

from ragicamp.core.logging import get_logger
from ragicamp.utils.resource_manager import ResourceManager

logger = get_logger(__name__)


def _build_result_item(
    resp: Any,
    orig_idx: int,
    query: str,
    expected: list[str],
) -> dict[str, Any]:
    """Build a result dict from a successful agent response."""
    result_item: dict[str, Any] = {
        "idx": orig_idx,
        "query": query,
        "prediction": resp.answer,
        "expected": expected,
        "prompt": getattr(resp, "prompt", None),
        "error": None,
    }
    metadata = getattr(resp, "metadata_dict", None) or getattr(resp, "metadata", {}) or {}
    if metadata:
        result_item["metadata"] = metadata
    context = getattr(resp, "context", None)
    if context and hasattr(context, "intermediate_steps") and context.intermediate_steps:
        result_item["intermediate_steps"] = context.intermediate_steps
    return result_item


def _build_error_item(
    orig_idx: int,
    query: str,
    expected: list[str],
    error: str,
) -> dict[str, Any]:
    """Build a result dict for a failed query."""
    return {
        "idx": orig_idx,
        "query": query,
        "prediction": f"[ERROR: {error[:50]}]",
        "expected": expected,
        "prompt": None,
        "error": error,
    }


class Answerable(Protocol):
    """Protocol for anything that can answer questions."""

    def answer(self, query: str, **kwargs: Any) -> Any:
        """Answer a single query."""
        ...


class BatchAnswerable(Answerable, Protocol):
    """Protocol for anything that can answer questions in batches."""

    def batch_answer(self, queries: list[str], **kwargs: Any) -> list[Any]:
        """Answer multiple queries in a batch."""
        ...


class ResilientExecutor:
    """Executes agent calls with batch processing.

    With vLLM as the backend, batch size just controls how many prompts
    are queued at once. vLLM handles its own GPU memory management.

    Example:
        executor = ResilientExecutor(agent, batch_size=32)
        results = executor.execute(queries, expected_answers)

        for result in results:
            print(f"Q: {result['query']}")
            print(f"A: {result['prediction']}")
            if result.get('error'):
                print(f"Error: {result['error']}")
    """

    def __init__(
        self,
        agent: Answerable,
        batch_size: int = 32,
    ):
        """Initialize the executor.

        Args:
            agent: The agent to execute (must have answer() or batch_answer())
            batch_size: Number of prompts to process per batch
        """
        self.agent = agent
        self.batch_size = batch_size
        self._supports_batch = hasattr(agent, "batch_answer")

    def execute(
        self,
        queries: list[tuple[int, str, list[str]]],
        progress: bool = True,
        checkpoint_every: int = 0,
        checkpoint_callback: Callable[[list[dict]], None] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute queries with batch processing.

        Args:
            queries: List of (index, question, expected_answers) tuples
            progress: Show progress bar
            checkpoint_every: Save checkpoint every N items (0 = disabled)
            checkpoint_callback: Called with results after each checkpoint
            **kwargs: Additional arguments passed to agent.answer()

        Returns:
            List of result dicts with keys: idx, query, prediction, expected, prompt, error
        """
        if not queries:
            return []

        pending = list(queries)

        if self._supports_batch and self.batch_size > 1:
            return self._execute_batched(
                pending, progress, checkpoint_every, checkpoint_callback, **kwargs
            )
        else:
            return self._execute_sequential(
                pending, progress, checkpoint_every, checkpoint_callback, **kwargs
            )

    def _execute_batched(
        self,
        queries: list[tuple[int, str, list[str]]],
        progress: bool,
        checkpoint_every: int,
        checkpoint_callback: Callable[[list[dict]], None] | None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute with batch processing."""
        results: list[dict[str, Any]] = []
        idx = 0
        items_since_checkpoint = 0
        batches_since_gc = 0
        pbar = tqdm(total=len(queries), desc="Generating", disable=not progress)
        consecutive_failures = 0
        max_consecutive_failures = 5

        while idx < len(queries):
            batch = queries[idx : idx + self.batch_size]
            batch_queries = [q for _, q, _ in batch]

            try:
                responses = self.agent.batch_answer(batch_queries, **kwargs)

                for (orig_idx, query, expected), resp in zip(batch, responses, strict=True):
                    results.append(_build_result_item(resp, orig_idx, query, expected))

                pbar.update(len(batch))
                idx += self.batch_size
                consecutive_failures = 0

            except Exception as e:
                error_str = str(e)
                consecutive_failures += 1

                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        "Hit %d consecutive failures - model may be broken. Aborting execution.",
                        consecutive_failures,
                    )
                    for i in range(idx, len(queries)):
                        orig_idx, query, expected = queries[i]
                        results.append(
                            {
                                "idx": orig_idx,
                                "query": query,
                                "prediction": f"[ABORTED: {consecutive_failures} consecutive failures]",
                                "expected": expected,
                                "prompt": None,
                                "error": f"Aborted after {consecutive_failures} failures: {error_str}",
                            }
                        )
                    pbar.update(len(queries) - idx)
                    break

                logger.warning(
                    "Batch failed (attempt %d/%d): %s",
                    consecutive_failures,
                    max_consecutive_failures,
                    error_str[:100],
                )
                for orig_idx, query, expected in batch:
                    results.append(_build_error_item(orig_idx, query, expected, error_str))
                pbar.update(len(batch))
                idx += len(batch)

            items_since_checkpoint += len(batch)
            if checkpoint_every and items_since_checkpoint >= checkpoint_every:
                if checkpoint_callback:
                    checkpoint_callback(results)
                items_since_checkpoint = 0

            # Only clear GPU memory every 10 batches to avoid overhead
            batches_since_gc += 1
            if batches_since_gc >= 10:
                ResourceManager.clear_gpu_memory()
                batches_since_gc = 0

        pbar.close()
        return results

    def _execute_sequential(
        self,
        queries: list[tuple[int, str, list[str]]],
        progress: bool,
        checkpoint_every: int,
        checkpoint_callback: Callable[[list[dict]], None] | None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute queries one at a time."""
        results: list[dict[str, Any]] = []
        items_since_checkpoint = 0
        items_since_gc = 0

        for orig_idx, query, expected in tqdm(queries, desc="Generating", disable=not progress):
            try:
                resp = self.agent.answer(query, **kwargs)
                results.append(_build_result_item(resp, orig_idx, query, expected))
            except Exception as e:
                results.append(_build_error_item(orig_idx, query, expected, str(e)))

            items_since_checkpoint += 1
            if checkpoint_every and items_since_checkpoint >= checkpoint_every:
                if checkpoint_callback:
                    checkpoint_callback(results)
                items_since_checkpoint = 0

            # Only clear GPU memory every 10 items to avoid overhead
            items_since_gc += 1
            if items_since_gc >= 10:
                ResourceManager.clear_gpu_memory()
                items_since_gc = 0

        return results
