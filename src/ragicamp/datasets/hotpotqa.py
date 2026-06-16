"""HotpotQA dataset loader."""

from pathlib import Path
from typing import Any

from datasets import load_dataset

from ragicamp.datasets.base import QADataset, QAExample


class HotpotQADataset(QADataset):
    """Loader for HotpotQA dataset.

    HotpotQA requires reasoning over multiple Wikipedia passages
    to answer questions (multi-hop reasoning).
    """

    def __init__(
        self,
        split: str = "train",
        distractor: bool = True,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ):
        """Initialize HotpotQA dataset.

        Args:
            split: Dataset split (train/validation)
            distractor: Whether to use distractor setting (with irrelevant contexts)
            cache_dir: Optional directory to cache processed datasets
            use_cache: Whether to try loading from cache first
            **kwargs: Additional configuration
        """
        super().__init__(name="hotpotqa", split=split, cache_dir=cache_dir, **kwargs)
        self.distractor = distractor

        # Try loading from cache first
        if use_cache and self.load_from_cache():
            return

        # Otherwise load from HuggingFace
        self.load()

    def get_cache_path(self) -> Path:
        """Get cache path that includes distractor parameter."""
        distractor_str = "distractor" if self.distractor else "fullwiki"
        return self.cache_dir / f"{self.name}_{distractor_str}_{self.split}.json"

    def load(self) -> None:
        """Load HotpotQA from HuggingFace datasets."""
        # Map split names
        hf_split = "train" if self.split == "train" else "validation"
        subset = "distractor" if self.distractor else "fullwiki"

        # Load from HuggingFace
        dataset = load_dataset("hotpot_qa", subset, split=hf_split)

        # Convert to our format
        for item in dataset:
            # Combine context passages
            context_parts = []
            for title, sentences in zip(
                item["context"]["title"], item["context"]["sentences"], strict=True
            ):
                context_parts.append(f"{title}: {' '.join(sentences)}")

            example = QAExample(
                id=item["id"],
                question=item["question"],
                answers=[item["answer"]],
                context="\n\n".join(context_parts),
                metadata={
                    "source": "hotpotqa",
                    "level": item.get("level", ""),
                    "type": item.get("type", ""),
                },
            )
            self.examples.append(example)
