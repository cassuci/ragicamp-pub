"""Natural Questions dataset loader."""

from pathlib import Path
from typing import Any

from datasets import load_dataset

from ragicamp.datasets.base import QADataset, QAExample


class NaturalQuestionsDataset(QADataset):
    """Loader for Google's Natural Questions dataset.

    Natural Questions contains real Google search queries with
    answers from Wikipedia articles.
    """

    def __init__(
        self,
        split: str = "train",
        cache_dir: Path | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ):
        """Initialize NQ dataset.

        Args:
            split: Dataset split (train/validation)
            cache_dir: Optional directory to cache processed datasets
            use_cache: Whether to try loading from cache first
            **kwargs: Additional configuration
        """
        super().__init__(name="natural_questions", split=split, cache_dir=cache_dir, **kwargs)

        # Try loading from cache first
        if use_cache and self.load_from_cache():
            return

        # Otherwise load from HuggingFace
        self.load()

    def load(self) -> None:
        """Load Natural Questions from HuggingFace datasets."""
        # Load from HuggingFace
        dataset = load_dataset("nq_open", split=self.split)

        # Convert to our format
        for i, item in enumerate(dataset):
            example = QAExample(
                id=f"nq_{self.split}_{i}",
                question=item["question"],
                answers=item["answer"],  # NQ has multiple acceptable answers
                context=None,  # Open-domain version doesn't include context
                metadata={"source": "natural_questions"},
            )
            self.examples.append(example)
