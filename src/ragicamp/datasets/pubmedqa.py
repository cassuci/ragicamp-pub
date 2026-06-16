"""PubMedQA dataset loader.

PubMedQA is a biomedical question answering dataset built from
PubMed abstracts. Questions are derived from article titles,
and answers are extracted from the abstracts.

Dataset: https://huggingface.co/datasets/qiaojin/PubMedQA
Paper: https://arxiv.org/abs/1909.06146
"""

from pathlib import Path
from typing import Any

from datasets import load_dataset

from ragicamp.datasets.base import QADataset, QAExample


class PubMedQADataset(QADataset):
    """Loader for PubMedQA dataset.

    PubMedQA is a biomedical QA dataset where:
    - Questions are derived from research article titles
    - Context is the abstract of the paper
    - Answers are yes/no/maybe with explanations (long_answer)

    This dataset tests RAG performance on scientific/medical domains.

    Subsets:
    - pqa_labeled: Expert-annotated (small, high quality)
    - pqa_artificial: Automatically generated (large, lower quality)
    - pqa_unlabeled: Unlabeled data
    """

    def __init__(
        self,
        split: str = "train",
        subset: str = "pqa_labeled",
        cache_dir: Path | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ):
        """Initialize PubMedQA dataset.

        Args:
            split: Dataset split (train/validation/test)
            subset: Dataset subset ('pqa_labeled', 'pqa_artificial', 'pqa_unlabeled')
            cache_dir: Optional directory to cache processed datasets
            use_cache: Whether to try loading from cache first
            **kwargs: Additional configuration
        """
        super().__init__(name="pubmedqa", split=split, cache_dir=cache_dir, **kwargs)
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
        """Load PubMedQA from HuggingFace datasets."""
        # PubMedQA has specific split names
        # pqa_labeled only has 'train' split with 1000 examples
        hf_split = self.split
        if self.subset == "pqa_labeled":
            # pqa_labeled is small, only has train split
            # We'll split it ourselves if needed
            hf_split = "train"

        # Load from HuggingFace
        try:
            dataset = load_dataset("qiaojin/PubMedQA", self.subset, split=hf_split)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load PubMedQA dataset. Error: {e}\n"
                "Visit https://huggingface.co/datasets/qiaojin/PubMedQA"
            ) from e

        # For pqa_labeled, manually split if needed
        if self.subset == "pqa_labeled" and self.split in ("validation", "test"):
            # Use last 20% as validation/test
            total = len(dataset)
            if self.split == "validation":
                # Validation: 10% (indices 800-899 for 1000 examples)
                start_idx = int(total * 0.8)
                end_idx = int(total * 0.9)
                indices = range(start_idx, end_idx)
            else:  # test
                # Test: 10% (indices 900-999 for 1000 examples)
                start_idx = int(total * 0.9)
                indices = range(start_idx, total)

            dataset = dataset.select(indices)
        elif self.subset == "pqa_labeled" and self.split == "train":
            # Train: first 80%
            total = len(dataset)
            end_idx = int(total * 0.8)
            dataset = dataset.select(range(end_idx))

        # Convert to our format
        for item in dataset:
            # PubMedQA structure:
            # - pubid: PubMed ID
            # - question: The question (derived from title)
            # - context: Dict with 'contexts' (list of abstract sentences),
            #           'labels' (list), 'meshes' (list)
            # - long_answer: Detailed answer explanation
            # - final_decision: yes/no/maybe

            # Build context from abstract sentences
            context_data = item.get("context", {})
            if isinstance(context_data, dict):
                context_sentences = context_data.get("contexts", [])
                context = " ".join(context_sentences) if context_sentences else ""
            else:
                context = str(context_data)

            # Use long_answer as the answer (more informative than yes/no)
            long_answer = item.get("long_answer", "")
            final_decision = item.get("final_decision", "")

            # Skip if no answer
            if not long_answer and not final_decision:
                continue

            # Use long_answer if available, otherwise use final_decision
            answer = long_answer if long_answer else final_decision

            example = QAExample(
                id=str(item.get("pubid", len(self.examples))),
                question=item["question"],
                answers=[answer],
                context=context,
                metadata={
                    "source": "pubmedqa",
                    "domain": "biomedical",
                    "subset": self.subset,
                    "final_decision": final_decision,
                    "pubid": item.get("pubid", ""),
                },
            )
            self.examples.append(example)
