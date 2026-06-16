"""TechQA dataset loader.

TechQA is IBM's technical support question answering dataset.
It contains questions about IBM products with answers extracted
from technical support documents (Technotes).

Dataset: https://huggingface.co/datasets/ibm/techqa
Paper: https://arxiv.org/abs/1911.02984
"""

from pathlib import Path
from typing import Any

from datasets import load_dataset

from ragicamp.datasets.base import QADataset, QAExample


class TechQADataset(QADataset):
    """Loader for IBM TechQA dataset.

    TechQA is a domain-specific QA dataset focused on technical
    support for IBM products. Questions are real user queries
    and answers come from technical documentation.

    This is a good dataset for testing RAG on technical/specialized domains.
    """

    def __init__(
        self,
        split: str = "train",
        cache_dir: Path | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ):
        """Initialize TechQA dataset.

        Args:
            split: Dataset split (train/dev/test)
            cache_dir: Optional directory to cache processed datasets
            use_cache: Whether to try loading from cache first
            **kwargs: Additional configuration
        """
        super().__init__(name="techqa", split=split, cache_dir=cache_dir, **kwargs)

        # Try loading from cache first
        if use_cache and self.load_from_cache():
            return

        # Otherwise load from HuggingFace
        self.load()

    def load(self) -> None:
        """Load TechQA from HuggingFace datasets."""
        # Map split names (TechQA uses 'dev' instead of 'validation')
        split_map = {
            "train": "train",
            "validation": "dev",
            "dev": "dev",
            "test": "test",
        }
        hf_split = split_map.get(self.split, self.split)

        # Load from HuggingFace
        try:
            dataset = load_dataset("ibm/techqa", split=hf_split)
        except Exception as e:
            # TechQA might require authentication or have access restrictions
            raise RuntimeError(
                f"Failed to load TechQA dataset. Error: {e}\n"
                "Note: TechQA may require accepting terms on HuggingFace. "
                "Visit https://huggingface.co/datasets/ibm/techqa"
            ) from e

        # Convert to our format
        for item in dataset:
            # TechQA structure:
            # - question: The user's question
            # - answer: The answer text (can be empty for unanswerable)
            # - context: The technote context
            # - document: Full document text
            # - is_answerable: Boolean flag

            answer = item.get("answer", "")
            if not answer or not answer.strip():
                # Skip unanswerable questions
                continue

            # Use technote context if available, otherwise full document
            context = item.get("context") or item.get("document", "")

            example = QAExample(
                id=str(item.get("id", len(self.examples))),
                question=item["question"],
                answers=[answer],
                context=context,
                metadata={
                    "source": "techqa",
                    "domain": "technical_support",
                    "document_id": item.get("document_id", ""),
                    "is_answerable": item.get("is_answerable", True),
                },
            )
            self.examples.append(example)

    def get_cache_path(self) -> Path:
        """Get cache path for TechQA."""
        return self.cache_dir / f"{self.name}_{self.split}.json"
