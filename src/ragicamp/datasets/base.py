"""Base classes for datasets."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QAExample:
    """Represents a question-answering example.

    Attributes:
        id: Unique example identifier
        question: The question text
        answers: List of acceptable answers
        context: Optional context/passage (for reading comprehension)
        metadata: Additional example metadata
    """

    id: str
    question: str
    answers: list[str]
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class QADataset(ABC):
    """Base class for QA datasets.

    Provides a unified interface for loading and accessing different
    QA datasets (NQ, HotpotQA, TriviaQA, etc.).
    """

    def __init__(
        self, name: str, split: str = "train", cache_dir: Path | None = None, **kwargs: Any
    ):
        """Initialize the dataset.

        Args:
            name: Dataset identifier
            split: Dataset split (train/validation/test)
            cache_dir: Optional directory to cache processed datasets
            **kwargs: Dataset-specific configuration
        """
        self.name = name
        self.split = split
        self.config = kwargs
        self.cache_dir = cache_dir or Path("data/datasets")
        self.examples: list[QAExample] = []

    @abstractmethod
    def load(self) -> None:
        """Load the dataset from source."""
        pass

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> QAExample:
        """Get example by index."""
        return self.examples[idx]

    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)

    def get_subset(self, n: int, seed: int | None = None) -> list[QAExample]:
        """Get a random subset of examples.

        Args:
            n: Number of examples to sample
            seed: Random seed for reproducibility

        Returns:
            List of sampled examples
        """
        import random

        rng = random.Random(seed)
        n = min(n, len(self.examples))
        return rng.sample(self.examples, n)

    def filter_with_answers(self) -> None:
        """Filter dataset to only include examples with explicit answers.

        Removes examples where answers list is empty or contains only empty strings.
        Updates self.examples in-place.
        """
        original_count = len(self.examples)
        self.examples = [
            ex
            for ex in self.examples
            if ex.answers and any(answer.strip() for answer in ex.answers)
        ]
        filtered_count = original_count - len(self.examples)
        if filtered_count > 0:
            logger.info("Filtered out %d examples without explicit answers", filtered_count)
            logger.info("Remaining: %d examples", len(self.examples))

    def get_examples_with_answers(self, n: int | None = None) -> list[QAExample]:
        """Get examples that have explicit answers.

        Args:
            n: Optional maximum number of examples to return

        Returns:
            List of examples with non-empty answers
        """
        filtered = [
            ex
            for ex in self.examples
            if ex.answers and any(answer.strip() for answer in ex.answers)
        ]

        if n is not None:
            filtered = filtered[:n]

        return filtered

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', split='{self.split}', size={len(self)})"
        )

    def get_cache_path(self) -> Path:
        """Get the path where this dataset should be cached.

        Child classes can override this to include dataset-specific parameters
        in the cache path (e.g., subset, distractor settings).

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{self.name}_{self.split}.json"

    def save_to_cache(self, info: dict[str, Any] | None = None) -> Path:
        """Save dataset to cache file.

        Args:
            info: Optional metadata to save with the dataset

        Returns:
            Path where dataset was saved
        """
        cache_path = self.get_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert examples to dicts
        examples_data = [
            {
                "id": ex.id,
                "question": ex.question,
                "answers": ex.answers,
                "context": ex.context,
                "metadata": ex.metadata,
            }
            for ex in self.examples
        ]

        # Prepare data to save
        data = {
            "info": info or {"dataset": self.name, "split": self.split, "size": len(self.examples)},
            "examples": examples_data,
        }

        # Save to JSON
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)

        return cache_path

    def load_from_cache(self) -> bool:
        """Load dataset from cache if available.

        Returns:
            True if loaded from cache, False otherwise
        """
        cache_path = self.get_cache_path()

        if not cache_path.exists():
            return False

        try:
            with open(cache_path) as f:
                data = json.load(f)

            # Load examples
            self.examples = [
                QAExample(
                    id=ex["id"],
                    question=ex["question"],
                    answers=ex["answers"],
                    context=ex.get("context"),
                    metadata=ex.get("metadata", {}),
                )
                for ex in data["examples"]
            ]

            return True
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load cache from %s: %s", cache_path, e)
            return False

    @classmethod
    def download_and_cache(
        cls,
        split: str = "validation",
        cache_dir: Path | None = None,
        max_examples: int | None = None,
        filter_no_answer: bool = True,
        force_download: bool = False,
        **kwargs: Any,
    ) -> "QADataset":
        """Download dataset and save to cache.

        This is a convenience method that:
        1. Creates a dataset instance
        2. Loads from HuggingFace (or cache if available)
        3. Applies filtering
        4. Saves to cache for future use

        Args:
            split: Dataset split to download
            cache_dir: Directory to cache the dataset
            max_examples: Optional limit on number of examples
            filter_no_answer: Whether to filter examples without answers
            force_download: Force re-download even if cache exists
            **kwargs: Additional dataset-specific arguments

        Returns:
            Loaded dataset instance
        """
        # Create instance (will auto-load)
        dataset = cls(split=split, cache_dir=cache_dir, **kwargs)

        # Check cache first unless force_download
        if not force_download and dataset.load_from_cache():
            logger.info("Loaded from cache: %s", dataset.get_cache_path())
            logger.info("  %d examples", len(dataset))

            # Apply max_examples if specified
            if max_examples and len(dataset) > max_examples:
                dataset.examples = dataset.examples[:max_examples]

            return dataset

        # Load from source (already done in __init__)
        logger.info("Loaded %d examples from HuggingFace", len(dataset))

        # Filter if requested
        if filter_no_answer:
            original_size = len(dataset)
            dataset.filter_with_answers()
            logger.info("Filtered: %d -> %d examples", original_size, len(dataset))

        # Limit if requested
        if max_examples and len(dataset) > max_examples:
            dataset.examples = dataset.examples[:max_examples]
            logger.info("Limited to %d examples", len(dataset))

        # Save to cache
        info = {
            "dataset": dataset.name,
            "split": split,
            "original_size": len(dataset),
            "filtered_size": len(dataset),
            "filter_no_answer": filter_no_answer,
        }
        cache_path = dataset.save_to_cache(info)
        logger.info("Saved to cache: %s", cache_path)

        return dataset
