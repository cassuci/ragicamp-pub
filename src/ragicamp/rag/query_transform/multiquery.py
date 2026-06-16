"""Multi-query transformer for query expansion.

Multi-query generates multiple variations of the original query
to capture different phrasings and aspects of the user's intent.
Results from all queries are merged and deduplicated.

Uses GeneratorProvider for clean GPU lifecycle management.
"""

import re
from typing import TYPE_CHECKING

from ragicamp.rag.query_transform.base import QueryTransformer

if TYPE_CHECKING:
    from ragicamp.models.providers import GeneratorProvider


class MultiQueryTransformer(QueryTransformer):
    """Rewrite query into multiple variations for better coverage.

    This transformer asks an LLM to generate alternative phrasings
    of the original question. Searching with multiple queries helps:
    - Capture different vocabulary (synonyms, technical terms)
    - Address different aspects of the question
    - Improve recall by diversifying the search

    Uses GeneratorProvider pattern for clean GPU lifecycle.
    """

    DEFAULT_PROMPT = """You are an AI assistant helping to improve search queries.
Given the following question, generate {num_queries} alternative versions of the same question.
Each version should capture a different angle or use different words, but ask about the same thing.

Original question: {query}

Generate {num_queries} alternative questions, one per line. Only output the questions, no numbering or explanations.

Alternative questions:"""

    def __init__(
        self,
        generator_provider: "GeneratorProvider",
        num_queries: int = 3,
        prompt_template: str | None = None,
        include_original: bool = True,
        max_tokens: int = 200,
    ):
        """Initialize multi-query transformer.

        Args:
            generator_provider: Provider for the LLM (lazy loading)
            num_queries: Number of alternative queries to generate
            prompt_template: Custom prompt template with {query} and {num_queries} placeholders
            include_original: Whether to include the original query
            max_tokens: Maximum tokens for generation
        """
        self.generator_provider = generator_provider
        self.num_queries = num_queries
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.include_original = include_original
        self.max_tokens = max_tokens

    def transform(self, query: str) -> list[str]:
        """Generate multiple query variations.

        Args:
            query: The original user query

        Returns:
            List of query variations including original (if enabled)
        """
        queries = []

        # Include original query first
        if self.include_original:
            queries.append(query)

        # Generate variations using provider
        prompt = self.prompt_template.format(query=query, num_queries=self.num_queries)

        with self.generator_provider.load() as generator:
            responses = generator.batch_generate(
                [prompt],
                max_tokens=self.max_tokens,
                temperature=0.7,
            )
            response = responses[0] if responses else ""

        # Parse the response into individual queries
        variations = self._parse_variations(response)

        # Add unique variations
        for variation in variations:
            variation = variation.strip()
            # Skip empty or duplicate queries
            if variation and variation.lower() != query.lower() and variation not in queries:
                queries.append(variation)

        return queries

    def _parse_variations(self, response: str) -> list[str]:
        """Parse LLM response into individual query variations.

        Args:
            response: Raw LLM response

        Returns:
            List of parsed query strings
        """
        # Split by newlines
        lines = response.strip().split("\n")

        variations = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove common prefixes like "1.", "- ", "* ", etc.
            line = re.sub(r"^[\d]+[.\)]\s*", "", line)
            line = re.sub(r"^[-*•]\s*", "", line)
            line = line.strip()

            # Skip if it looks like a label or instruction
            if line.lower().startswith(("alternative", "question", "version")):
                continue

            if line:
                variations.append(line)

        return variations[: self.num_queries]  # Limit to requested number

    def batch_transform(self, queries: list[str]) -> list[list[str]]:
        """Generate query variations for multiple queries in one batch call.

        This is much faster than calling transform() sequentially because
        it uses a single batched LLM call for all queries.

        Args:
            queries: List of original user queries

        Returns:
            List of query lists, one per input query
        """
        if not queries:
            return []

        # Build prompts for all queries
        prompts = [
            self.prompt_template.format(query=q, num_queries=self.num_queries) for q in queries
        ]

        # Batch generate variations (single batched LLM call!)
        with self.generator_provider.load() as generator:
            responses = generator.batch_generate(
                prompts,
                max_tokens=self.max_tokens,
                temperature=0.7,
            )

        # Parse each response
        results = []
        for query, response in zip(queries, responses, strict=True):
            query_list = []
            if self.include_original:
                query_list.append(query)

            variations = self._parse_variations(response)
            for variation in variations:
                variation = variation.strip()
                if variation and variation.lower() != query.lower() and variation not in query_list:
                    query_list.append(variation)

            results.append(query_list)

        return results

    def __repr__(self) -> str:
        return (
            f"MultiQueryTransformer(num_queries={self.num_queries}, "
            f"include_original={self.include_original})"
        )
