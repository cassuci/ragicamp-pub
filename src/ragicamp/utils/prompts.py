"""Prompt building utilities for RAG systems.

This module provides a centralized way to build prompts for different
experiment types (direct, RAG) with support for few-shot examples.

Prompt Types:
- concise: Short, direct answers (baseline)
- structured: Clear delimiters, explicit instructions
- extractive: Strict extraction from context only
- cot: Chain-of-thought reasoning
- cited: Answers with passage citations

Usage:
    builder = PromptBuilder.from_config("structured", dataset="hotpotqa")
    prompt = builder.build_rag(query="What is AI?", context="...")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Module-level cache for fewshot examples loaded from YAML.
# Loaded once per process, never invalidated (static file).
_fewshot_cache: dict[str, Any] | None = None


@dataclass
class FewShotExample:
    """A single few-shot example."""

    question: str
    answer: str


@dataclass
class PromptConfig:
    """Configuration for prompt building."""

    style: str = ""
    stop_instruction: str = ""
    knowledge_instruction: str = ""
    examples: list[FewShotExample] = field(default_factory=list)
    # New fields for advanced prompt types
    system_instruction: str = ""
    use_delimiters: bool = False
    include_reasoning: bool = False
    require_citation: bool = False
    strict_extraction: bool = False


class PromptBuilder:
    """Builds clean, well-structured prompts for QA experiments.

    Design principles:
    1. Clear task instruction at the top
    2. Examples in a labeled section (if fewshot)
    3. Clear transition to the actual question
    4. Consistent formatting throughout
    5. Last instruction = most important (LLMs follow it better)

    Prompt types:
    - concise: Minimal, just answer
    - structured: Clear delimiters, explicit format
    - extractive: Only from context, no model knowledge
    - cot: Chain-of-thought for complex questions
    - cited: Answers with [1], [2] citations
    """

    def __init__(self, config: PromptConfig):
        self.config = config

    def build_direct(self, query: str) -> str:
        """Build prompt for direct (no retrieval) QA.

        Structure:
            [Task context - what you're doing]

            [Examples section - if fewshot]

            Question: {query}

            [Format instruction - at END for better following]
            Answer:
        """
        parts = []

        # Task context - brief description (no formatting instructions here)
        if self.config.system_instruction:
            parts.append(self.config.system_instruction)
        else:
            parts.append("Answer the question using your knowledge.")

        # Examples section (if any)
        if self.config.examples:
            parts.append(self._format_examples())

        # The actual question
        parts.append(f"Question: {query}")

        # Format instructions at END - LLMs follow last instruction better
        format_instruction = []
        if self.config.style:
            format_instruction.append(self.config.style)
        if self.config.stop_instruction:
            format_instruction.append(self.config.stop_instruction)
        format_instruction.append("If you don't know, answer 'Unknown'.")

        parts.append(" ".join(format_instruction) + "\nAnswer:")

        return "\n\n".join(parts)

    def build_rag(self, query: str, context: str) -> str:
        """Build prompt for RAG (with retrieved context).

        Dispatches to specialized builders based on config.
        """
        if self.config.use_delimiters:
            return self._build_rag_structured(query, context)
        if self.config.include_reasoning:
            return self._build_rag_cot(query, context)
        if self.config.require_citation:
            return self._build_rag_cited(query, context)
        if self.config.strict_extraction:
            return self._build_rag_extractive(query, context)
        return self._build_rag_default(query, context)

    def _build_rag_default(self, query: str, context: str) -> str:
        """Default RAG prompt - simple and effective."""
        parts = []

        # Task context
        task = "Answer the question using the retrieved passages below."
        if not self.config.strict_extraction:
            task += " If the passages don't contain the answer, you may use your own knowledge."
        if self.config.knowledge_instruction:
            task += f" {self.config.knowledge_instruction}"
        parts.append(task)

        # Examples section (if any)
        if self.config.examples:
            parts.append(self._format_examples())

        # Retrieved passages
        parts.append(f"Retrieved Passages:\n{context}")

        # The actual question
        parts.append(f"Question: {query}")

        # Format instructions at END
        format_instruction = []
        if self.config.style:
            format_instruction.append(self.config.style)
        if self.config.stop_instruction:
            format_instruction.append(self.config.stop_instruction)
        format_instruction.append("If you don't know, answer 'Unknown'.")

        parts.append(" ".join(format_instruction) + "\nAnswer:")

        return "\n\n".join(parts)

    def _build_rag_structured(self, query: str, context: str) -> str:
        """Structured prompt with clear delimiters (works well with instruction-tuned models)."""
        prompt = f"""### Instruction
You are a helpful assistant that answers questions based on the provided context.
Read the context carefully and give a precise, concise answer.

### Context
{context}

### Question
{query}

### Requirements
- Answer based on the context above
- Be concise - give only the answer, no explanations
- If the answer is not in the context, say "Unknown"

### Answer
"""
        return prompt

    def _build_rag_extractive(self, query: str, context: str) -> str:
        """Strict extractive prompt - answer MUST come from context only."""
        prompt = f"""You are an extractive QA system. Your answer MUST be extracted directly from the given passages.
Do NOT use any knowledge outside of these passages.

---PASSAGES---
{context}
---END PASSAGES---

Question: {query}

Instructions:
1. Find the exact answer in the passages above
2. Copy the answer exactly as it appears
3. If the answer is NOT in the passages, respond with "Unknown"
4. Do NOT make up information or use external knowledge

Answer:"""
        return prompt

    def _build_rag_cot(self, query: str, context: str) -> str:
        """Chain-of-thought prompt for complex reasoning (good for HotpotQA)."""
        prompt = f"""Answer the question by reasoning step-by-step through the provided passages.

Passages:
{context}

Question: {query}

Let's think through this step by step:
1. First, identify relevant information in the passages
2. Then, connect the facts to answer the question
3. Finally, give a concise final answer

Reasoning:"""
        return prompt

    def _build_rag_cited(self, query: str, context: str) -> str:
        """Citation-aware prompt - answers reference passage numbers."""
        prompt = f"""Answer the question using the numbered passages below. Include citation numbers in your answer.

{context}

Question: {query}

Instructions:
- Use [1], [2], etc. to cite which passage(s) support your answer
- Be concise but include the citation
- If no passage contains the answer, say "Unknown"

Answer:"""
        return prompt

    def _format_examples(self) -> str:
        """Format examples in a clear, labeled section."""
        lines = ["Examples:"]
        for ex in self.config.examples:
            lines.append(f"Q: {ex.question}")
            lines.append(f"A: {ex.answer}")
            lines.append("")  # blank line between examples
        return "\n".join(lines).rstrip()

    @staticmethod
    def _load_fewshot_file() -> dict[str, Any]:
        """Load fewshot examples from YAML file (cached at module level)."""
        global _fewshot_cache
        if _fewshot_cache is not None:
            return _fewshot_cache

        from ragicamp.utils.paths import get_project_root

        paths = [
            get_project_root() / "conf" / "prompts" / "fewshot_examples.yaml",
            Path("conf/prompts/fewshot_examples.yaml"),
        ]

        for path in paths:
            if path.exists():
                with open(path) as f:
                    _fewshot_cache = yaml.safe_load(f) or {}
                    return _fewshot_cache

        _fewshot_cache = {}
        return _fewshot_cache

    @classmethod
    def from_config(cls, prompt_type: str = "default", dataset: str = "nq") -> "PromptBuilder":
        """Create a PromptBuilder from a config type.

        Args:
            prompt_type: Prompt style to use:
                - "default": Basic RAG prompt
                - "concise": Minimal, just answer
                - "structured": Clear delimiters (### sections)
                - "extractive": Strict context-only answers
                - "cot": Chain-of-thought reasoning
                - "cited": Answers with [1], [2] citations
                - "fewshot", "fewshot_3", "fewshot_1": With examples
            dataset: Dataset name for loading appropriate fewshot examples

        Returns:
            Configured PromptBuilder
        """
        # =================================================================
        # Basic prompts
        # =================================================================
        if prompt_type == "default":
            return cls(
                PromptConfig(
                    style="Give only the answer, no explanations.",
                    knowledge_instruction="Read the passages carefully to find the answer.",
                )
            )

        if prompt_type == "concise":
            return cls(
                PromptConfig(
                    style="Reply with just the answer.",
                    knowledge_instruction="The answer can be found in the passages.",
                )
            )

        # concise_strict: More aggressive anti-hallucination version
        if prompt_type == "concise_strict":
            return cls(
                PromptConfig(
                    style="Give ONLY the answer - a single word, name, date, or short phrase.",
                    stop_instruction="STOP immediately after answering. Do NOT add explanations or follow-up questions.",
                    knowledge_instruction="Find the answer in the passages.",
                )
            )

        # concise_json: Request JSON format for easier parsing
        if prompt_type == "concise_json":
            return cls(
                PromptConfig(
                    style='Reply with JSON: {"answer": "your answer here"}',
                    knowledge_instruction="The answer can be found in the passages.",
                )
            )

        # =================================================================
        # Advanced prompts
        # =================================================================
        # Note: structured kept for backwards compatibility but not recommended
        if prompt_type == "structured":
            return cls(
                PromptConfig(
                    use_delimiters=True,
                    style="Be concise.",
                )
            )

        if prompt_type == "extractive":
            return cls(
                PromptConfig(
                    strict_extraction=True,
                    style="Extract the exact answer from the passages.",
                    knowledge_instruction="Only use information from the passages.",
                )
            )

        # extractive_quoted: Forces model to quote from passages
        if prompt_type == "extractive_quoted":
            return cls(
                PromptConfig(
                    strict_extraction=True,
                    style="Copy the exact answer as it appears in the passages. Use quotation marks.",
                    knowledge_instruction="The answer MUST appear verbatim in the passages. Quote it exactly.",
                )
            )

        if prompt_type == "cot":
            return cls(
                PromptConfig(
                    include_reasoning=True,
                    style="Show your reasoning, then give a final answer.",
                )
            )

        # cot_final: Chain-of-thought with clear final answer marker
        if prompt_type == "cot_final":
            return cls(
                PromptConfig(
                    include_reasoning=True,
                    style="Think step by step, then end with 'FINAL ANSWER: <your answer>'",
                )
            )

        if prompt_type == "cited":
            return cls(
                PromptConfig(
                    require_citation=True,
                    style="Include passage citations [1], [2], etc.",
                )
            )

        # =================================================================
        # Fewshot prompts
        # =================================================================
        if prompt_type.startswith("fewshot"):
            n_examples = {"fewshot": 5, "fewshot_3": 3, "fewshot_1": 1}.get(prompt_type, 5)

            fewshot_data = cls._load_fewshot_file()
            dataset_config = fewshot_data.get(dataset, {})

            raw_examples = dataset_config.get("examples", [])[:n_examples]
            examples = [
                FewShotExample(question=ex["question"], answer=ex["answer"]) for ex in raw_examples
            ]

            return cls(
                PromptConfig(
                    style=dataset_config.get("style", "Give a short, direct answer."),
                    stop_instruction=dataset_config.get("stop_instruction", ""),
                    knowledge_instruction=dataset_config.get("knowledge_instruction", ""),
                    examples=examples,
                )
            )

        return cls(PromptConfig())

    # Legacy compatibility
    @staticmethod
    def create_default() -> "PromptBuilder":
        return PromptBuilder.from_config("default")

    @staticmethod
    def create_concise() -> "PromptBuilder":
        return PromptBuilder.from_config("concise")
