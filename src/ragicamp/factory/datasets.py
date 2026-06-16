"""Dataset factory for creating QA datasets from configuration."""

from pathlib import Path
from typing import Any

from ragicamp.core.logging import get_logger
from ragicamp.datasets import (
    HotpotQADataset,
    NaturalQuestionsDataset,
    PubMedQADataset,
    QADataset,
    TechQADataset,
    TriviaQADataset,
)

logger = get_logger(__name__)


class DatasetFactory:
    """Factory for creating QA datasets from configuration."""

    @staticmethod
    def parse_spec(
        name: str,
        split: str = "validation",
        limit: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Parse a dataset name into a config dict.

        Args:
            name: Dataset name ('nq', 'triviaqa', 'hotpotqa' or full names)
            split: Dataset split ('train', 'validation', 'test')
            limit: Optional limit on number of examples
            **kwargs: Additional dataset parameters

        Returns:
            Config dict suitable for create()

        Example:
            >>> config = DatasetFactory.parse_spec("nq", limit=100)
            >>> dataset = DatasetFactory.create(config)
        """
        # Map short names to full names
        name_map = {
            "nq": "natural_questions",
            "triviaqa": "triviaqa",
            "hotpotqa": "hotpotqa",
            "techqa": "techqa",
            "pubmedqa": "pubmedqa",
            "natural_questions": "natural_questions",
        }
        full_name = name_map.get(name, name)

        config = {
            "name": full_name,
            "split": split,
        }
        if limit:
            config["num_examples"] = limit
        config.update(kwargs)
        return config

    @staticmethod
    def create(config: dict[str, Any]) -> QADataset:
        """Create a dataset from configuration.

        Args:
            config: Dataset configuration dict with 'name' and dataset-specific params

        Returns:
            Instantiated QADataset

        Example:
            >>> config = {"name": "natural_questions", "split": "validation"}
            >>> dataset = DatasetFactory.create(config)
        """
        dataset_name = config["name"]
        split = config.get("split", "validation")
        num_examples = config.get("num_examples")
        filter_no_answer = config.get("filter_no_answer", True)
        cache_dir = config.get("cache_dir", Path("data/datasets"))

        # Create appropriate dataset class
        if dataset_name == "natural_questions":
            dataset_class = NaturalQuestionsDataset
        elif dataset_name == "triviaqa":
            dataset_class = TriviaQADataset
        elif dataset_name == "hotpotqa":
            dataset_class = HotpotQADataset
        elif dataset_name == "techqa":
            dataset_class = TechQADataset
        elif dataset_name == "pubmedqa":
            dataset_class = PubMedQADataset
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: natural_questions, triviaqa, hotpotqa, techqa, pubmedqa"
            )

        # Load dataset
        dataset = dataset_class(split=split, cache_dir=Path(cache_dir))

        # Filter if needed
        if filter_no_answer:
            original_size = len(dataset)
            dataset.filter_with_answers()
            if len(dataset) < original_size:
                logger.debug("Filtered: %d → %d examples", original_size, len(dataset))

        # Limit if requested
        if num_examples and len(dataset) > num_examples:
            dataset.examples = dataset.examples[:num_examples]
            logger.debug("Dataset size: %d", len(dataset))

        return dataset
