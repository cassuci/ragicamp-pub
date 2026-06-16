"""HyDE (Hypothetical Document Embeddings) query transformer.

HyDE generates a hypothetical answer to the query using an LLM,
then uses that hypothetical answer as the search query instead
of the original question. This often improves retrieval because
the hypothetical answer is more similar to actual documents.

Reference: https://arxiv.org/abs/2212.10496

Uses GeneratorProvider for clean GPU lifecycle management.
"""

from typing import TYPE_CHECKING

from ragicamp.rag.query_transform.base import QueryTransformer

if TYPE_CHECKING:
    from ragicamp.models.providers import GeneratorProvider


class HyDETransformer(QueryTransformer):
    """Generate hypothetical answers and search with those.

    HyDE (Hypothetical Document Embeddings) asks an LLM to generate
    a hypothetical answer to the user's question, then uses that
    answer for retrieval instead of the original query.

    This works because:
    - The hypothetical answer is written in "document style"
    - It contains domain vocabulary that matches real documents
    - Embedding similarity works better for document-document matching

    Uses GeneratorProvider pattern for clean GPU lifecycle.
    """

    DEFAULT_PROMPT = """Please write a short paragraph that directly answers the following question.
Write as if you are explaining the answer in a textbook or encyclopedia.
Be factual and informative.

Question: {query}

Answer:"""

    def __init__(
        self,
        generator_provider: "GeneratorProvider",
        prompt_template: str | None = None,
        include_original: bool = True,
        num_hypothetical: int = 1,
        max_tokens: int = 150,
    ):
        """Initialize HyDE transformer.

        Args:
            generator_provider: Provider for the LLM (lazy loading)
            prompt_template: Custom prompt template with {query} placeholder
            include_original: Whether to also search with the original query
            num_hypothetical: Number of hypothetical answers to generate
            max_tokens: Maximum tokens for hypothetical answer generation
        """
        self.generator_provider = generator_provider
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.include_original = include_original
        self.num_hypothetical = num_hypothetical
        self.max_tokens = max_tokens
        self.last_hypothetical: str | None = None

    def transform(self, query: str) -> list[str]:
        """Generate hypothetical answer(s) and return as search queries.

        Args:
            query: The original user query

        Returns:
            List containing hypothetical answer(s), optionally with original query
        """
        queries = []

        # Optionally include original query
        if self.include_original:
            queries.append(query)

        # Generate hypothetical answer(s) using provider
        prompt = self.prompt_template.format(query=query)

        with self.generator_provider.load() as generator:
            for _ in range(self.num_hypothetical):
                hypotheticals = generator.batch_generate(
                    [prompt],
                    max_tokens=self.max_tokens,
                    temperature=0.7,
                )
                hypothetical = hypotheticals[0].strip() if hypotheticals else ""
                if hypothetical:
                    queries.append(hypothetical)
                    self.last_hypothetical = hypothetical

        return queries

    def batch_transform(self, queries: list[str]) -> list[list[str]]:
        """Generate hypothetical answers for multiple queries in one batch call.

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
        prompts = [self.prompt_template.format(query=q) for q in queries]

        # Batch generate hypothetical answers (single batched LLM call!)
        with self.generator_provider.load() as generator:
            hypotheticals = generator.batch_generate(
                prompts,
                max_tokens=self.max_tokens,
                temperature=0.7,
            )

        # Build result: for each query, return [original, hypothetical]
        results = []
        for query, hyp in zip(queries, hypotheticals, strict=True):
            query_list = []
            if self.include_original:
                query_list.append(query)
            if hyp and hyp.strip():
                query_list.append(hyp.strip())
            results.append(query_list)

        return results

    def __repr__(self) -> str:
        return (
            f"HyDETransformer(include_original={self.include_original}, "
            f"num_hypothetical={self.num_hypothetical})"
        )
