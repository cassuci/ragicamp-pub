"""TriviaQA dataset loader."""

from pathlib import Path
from typing import Any

from datasets import load_dataset

from ragicamp.datasets.base import QADataset, QAExample


class TriviaQADataset(QADataset):
    """Loader for TriviaQA dataset.

    TriviaQA contains trivia questions with evidence from Wikipedia
    and web search results.
    """

    def __init__(
        self,
        split: str = "train",
        subset: str = "unfiltered",
        cache_dir: Path | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ):
        """Initialize TriviaQA dataset.

        Args:
            split: Dataset split (train/validation)
            subset: Dataset subset (unfiltered/filtered)
            cache_dir: Optional directory to cache processed datasets
            use_cache: Whether to try loading from cache first
            **kwargs: Additional configuration
        """
        super().__init__(name="triviaqa", split=split, cache_dir=cache_dir, **kwargs)
        self.subset = subset

        # Try loading from cache first
        if use_cache and self.load_from_cache():
            return

        # Otherwise load from HuggingFace
        self.load()

    def get_cache_path(self) -> Path:
        """Get cache path that includes subset parameter."""
        return self.cache_dir / f"{self.name}_{self.subset}_{self.split}.json"

    def load(self) -> None:
        """Load TriviaQA from HuggingFace datasets."""
        # Map split names
        hf_split = self.split if self.split in ["train", "validation", "test"] else "validation"

        # Load from HuggingFace
        dataset = load_dataset("trivia_qa", self.subset, split=hf_split)

        # Convert to our format
        for item in dataset:
            # Get all acceptable answer variants
            answers = list(
                set(
                    item.get("answer", {}).get("aliases", [])
                    + [item.get("answer", {}).get("value", "")]
                )
            )
            answers = [a for a in answers if a]  # Remove empty strings

            example = QAExample(
                id=item.get("question_id", f"triviaqa_{hf_split}_{len(self.examples)}"),
                question=item["question"],
                answers=answers,
                context=None,  # Can add evidence documents if needed
                metadata={
                    "source": "triviaqa",
                    "question_source": item.get("question_source", ""),
                },
            )
            self.examples.append(example)
