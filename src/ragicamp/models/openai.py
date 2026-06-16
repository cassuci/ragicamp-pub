"""OpenAI model implementation with async support.

This module provides both sync and async interfaces to the OpenAI API.
Use async methods (agenerate, agenerate_single) for efficient parallel
API calls, especially in LLM-as-a-judge metrics.

Example:
    >>> model = OpenAIModel("gpt-4o-mini")
    >>>
    >>> # Sync usage (backward compatible)
    >>> response = model.generate("Hello, world!")
    >>>
    >>> # Async usage (for parallel calls)
    >>> responses = await model.agenerate(["prompt1", "prompt2", "prompt3"])
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import openai
import tiktoken
from openai import AsyncOpenAI

from ragicamp.core.logging import get_logger
from ragicamp.models.base import LanguageModel

logger = get_logger(__name__)


class OpenAIModel(LanguageModel):
    """Language model implementation using OpenAI-compatible APIs.

    Supports both synchronous and asynchronous generation.
    Use async methods for efficient parallel API calls.

    Works with any OpenAI-compatible provider (OpenAI, DeepInfra, etc.)
    by setting the base_url parameter.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        max_workers: int = 10,  # Parallel API calls
        max_concurrent_async: int = 20,  # Max concurrent async calls
        **kwargs: Any,
    ):
        """Initialize OpenAI-compatible model.

        Args:
            model_name: Model identifier (e.g. 'gpt-4o-mini', 'meta-llama/Llama-3.3-70B-Instruct')
            api_key: API key (or uses OPENAI_API_KEY env var)
            base_url: API base URL for OpenAI-compatible providers.
                      E.g. 'https://api.deepinfra.com/v1/openai' for DeepInfra.
                      If None, uses the default OpenAI endpoint.
            max_workers: Maximum parallel thread-based API calls (default: 10)
            max_concurrent_async: Maximum concurrent async API calls (default: 20)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        self.max_workers = max_workers
        self.max_concurrent_async = max_concurrent_async
        self._api_key = api_key
        self._base_url = base_url

        # Create sync client with optional base_url
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self._sync_client = openai.OpenAI(**client_kwargs)

        # Initialize tokenizer for counting
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        # Lazy-initialized async client
        self._async_client: AsyncOpenAI | None = None

    @property
    def async_client(self) -> AsyncOpenAI:
        """Get or create the async OpenAI client."""
        if self._async_client is None:
            client_kwargs: dict[str, Any] = {}
            if self._api_key:
                client_kwargs["api_key"] = self._api_key
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._async_client = AsyncOpenAI(**client_kwargs)
        return self._async_client

    def _supports_sampling_params(self) -> bool:
        """Check if model supports temperature/top_p parameters.

        Newer models (o1, o3, gpt-5) don't support these sampling parameters.
        """
        model_lower = self.model_name.lower()
        # Models that don't support temperature/top_p
        unsupported_prefixes = ("o1", "o3", "gpt-5")
        return not any(model_lower.startswith(prefix) for prefix in unsupported_prefixes)

    def _single_generate(
        self,
        prompt: str,
        max_tokens: int | None,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        **kwargs: Any,
    ) -> str:
        """Generate text for a single prompt."""
        # Build API params, excluding unsupported values
        api_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        # Only include sampling params for models that support them
        if self._supports_sampling_params():
            api_params["temperature"] = temperature
            api_params["top_p"] = top_p
        if max_tokens is not None:
            # Reasoning models (o1, o3) require max_completion_tokens;
            # third-party providers (DeepInfra, etc.) only understand max_tokens.
            if self._supports_sampling_params():
                api_params["max_tokens"] = max_tokens
            else:
                api_params["max_completion_tokens"] = max_tokens
        if stop is not None:
            api_params["stop"] = stop

        response = self._sync_client.chat.completions.create(**api_params)
        content = response.choices[0].message.content
        if not content:
            logger.warning(
                "Empty response from %s (finish_reason=%s, prompt=%.80s...)",
                self.model_name,
                response.choices[0].finish_reason,
                prompt[:80],
            )
        return content or ""

    def generate(
        self,
        prompt: str | list[str],
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        parallel: bool = True,
        **kwargs: Any,
    ) -> str | list[str]:
        """Generate text using OpenAI API.

        Args:
            prompt: Single prompt or list of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            parallel: Use parallel API calls for batches (default: True)
            **kwargs: Additional API parameters

        Returns:
            Generated text or list of texts
        """
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        # Single prompt - no parallelization needed
        if len(prompts) == 1:
            result = self._single_generate(
                prompts[0], max_tokens, temperature, top_p, stop, **kwargs
            )
            return [result] if is_batch else result

        # Multiple prompts - use parallel execution
        if parallel and len(prompts) > 1:
            results = [None] * len(prompts)

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks with their indices
                future_to_idx = {
                    executor.submit(
                        self._single_generate, p, max_tokens, temperature, top_p, stop, **kwargs
                    ): i
                    for i, p in enumerate(prompts)
                }

                # Collect results maintaining order
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error("OpenAI API error for prompt %d: %s", idx, e)
                        results[idx] = f"[ERROR: {type(e).__name__}: {str(e)[:100]}]"

            return results
        else:
            # Sequential fallback
            results = []
            for p in prompts:
                results.append(
                    self._single_generate(p, max_tokens, temperature, top_p, stop, **kwargs)
                )
            return results

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using OpenAI embeddings API."""
        response = self._sync_client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.encoding.encode(text))

    # =========================================================================
    # Async Methods
    # =========================================================================

    async def agenerate_single(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        system_message: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text for a single prompt asynchronously.

        Args:
            prompt: The prompt to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            system_message: Optional system message
            **kwargs: Additional API parameters

        Returns:
            Generated text
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        # Build API params, excluding unsupported values
        api_params = {
            "model": self.model_name,
            "messages": messages,
            **kwargs,
        }
        # Only include sampling params for models that support them
        if self._supports_sampling_params():
            api_params["temperature"] = temperature
            api_params["top_p"] = top_p
        if max_tokens is not None:
            if self._supports_sampling_params():
                api_params["max_tokens"] = max_tokens
            else:
                api_params["max_completion_tokens"] = max_tokens
        if stop is not None:
            api_params["stop"] = stop

        response = await self.async_client.chat.completions.create(**api_params)
        content = response.choices[0].message.content
        if not content:
            logger.warning(
                "Empty response from %s (finish_reason=%s, prompt=%.80s...)",
                self.model_name,
                response.choices[0].finish_reason,
                prompt[:80],
            )
        return content or ""

    async def agenerate(
        self,
        prompts: list[str],
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        system_message: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate text for multiple prompts asynchronously with rate limiting.

        Uses asyncio.Semaphore to limit concurrent API calls.

        Args:
            prompts: List of prompts to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            system_message: Optional system message for all prompts
            **kwargs: Additional API parameters

        Returns:
            List of generated texts (in same order as prompts)
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_async)

        async def rate_limited_generate(idx: int, prompt: str) -> tuple[int, str]:
            """Generate with rate limiting, returning index for ordering."""
            async with semaphore:
                try:
                    result = await self.agenerate_single(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        system_message=system_message,
                        **kwargs,
                    )
                    return (idx, result)
                except Exception as e:
                    logger.error("OpenAI async API error for prompt %d: %s", idx, e)
                    return (idx, f"[ERROR: {type(e).__name__}: {str(e)[:100]}]")

        # Create tasks for all prompts
        tasks = [rate_limited_generate(i, p) for i, p in enumerate(prompts)]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Sort by index to maintain order
        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]

    async def aget_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using OpenAI embeddings API asynchronously.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        response = await self.async_client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]
