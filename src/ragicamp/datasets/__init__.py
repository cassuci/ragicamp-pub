"""Dataset loaders for QA datasets."""

from ragicamp.datasets.base import QADataset, QAExample
from ragicamp.datasets.hotpotqa import HotpotQADataset
from ragicamp.datasets.nq import NaturalQuestionsDataset
from ragicamp.datasets.pubmedqa import PubMedQADataset
from ragicamp.datasets.techqa import TechQADataset
from ragicamp.datasets.triviaqa import TriviaQADataset

__all__ = [
    "QADataset",
    "QAExample",
    "NaturalQuestionsDataset",
    "TriviaQADataset",
    "HotpotQADataset",
    "TechQADataset",
    "PubMedQADataset",
]
