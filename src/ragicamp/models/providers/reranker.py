"""Reranker provider with lazy loading and lifecycle management."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ragicamp.core.constants import RERANKER_MODELS
from ragicamp.core.logging import get_logger

if TYPE_CHECKING:
    from ragicamp.core.types import Document
from ragicamp.utils.resource_manager import ResourceManager

from .base import ModelProvider

logger = get_logger(__name__)


@dataclass
class RerankerConfig:
    """Configuration for reranker."""

    model_name: str = "bge"  # bge, bge-base, ms-marco, or HF path
    batch_size: int = 32


class RerankerProvider(ModelProvider):
    """Provides reranker with lazy loading and proper cleanup.

    Usage:
        provider = RerankerProvider(RerankerConfig("bge"))

        with provider.load() as reranker:
            reranked = reranker.rerank(query, documents, top_k=5)
        # Reranker unloaded, GPU memory freed
    """

    MODELS = RERANKER_MODELS

    def __init__(self, config: RerankerConfig):
        self.config = config
        self._reranker = None
        self._refcount: int = 0

    @property
    def model_name(self) -> str:
        return self.MODELS.get(self.config.model_name, self.config.model_name)

    @contextmanager
    def load(self, gpu_fraction: float | None = None) -> Iterator[RerankerWrapper]:
        """Load reranker, yield it, then unload.

        Supports ref-counting: nested ``with provider.load()`` calls reuse
        the already-loaded model and only unload when the outermost context
        exits.
        """
        self._refcount += 1
        try:
            if self._refcount == 1:
                # First caller — actually load the model
                import torch
                from sentence_transformers import CrossEncoder

                logger.info("Loading reranker: %s", self.model_name)

                device = "cuda" if torch.cuda.is_available() else "cpu"

                model = CrossEncoder(
                    self.model_name,
                    device=device,
                    trust_remote_code=True,
                )

                self._reranker = RerankerWrapper(model, self.config.batch_size)
            else:
                logger.debug(
                    "Reranker already loaded (refcount=%d): %s",
                    self._refcount,
                    self.model_name,
                )

            yield self._reranker

        finally:
            self._refcount -= 1
            if self._refcount == 0:
                self._unload()

    def _unload(self):
        """Unload reranker and free GPU memory."""
        if self._reranker is not None:
            self._reranker.unload()
            self._reranker = None

        ResourceManager.clear_gpu_memory()
        logger.info("Reranker unloaded: %s", self.model_name)


class RerankerWrapper:
    """Wrapper around CrossEncoder implementing reranker interface."""

    def __init__(self, model, batch_size: int = 32):
        self._model = model
        self._batch_size = batch_size

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        """Rerank documents based on query relevance.

        Args:
            query: Search query
            documents: List of Document objects
            top_k: Number to return

        Returns:
            Top-k documents sorted by reranker score
        """
        if not documents:
            return []

        # Create pairs
        pairs = [(query, doc.text) for doc in documents]

        # Score
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        # Copy documents to avoid mutating caller's objects
        docs_copy = [copy.copy(doc) for doc in documents]
        for doc, score in zip(docs_copy, scores, strict=True):
            doc.score = float(score)

        sorted_docs = sorted(docs_copy, key=lambda d: d.score, reverse=True)
        return sorted_docs[:top_k]

    def batch_rerank(
        self,
        queries: list[str],
        documents_list: list[list[Document]],
        top_k: int,
    ) -> list[list[Document]]:
        """Batch rerank for multiple queries.

        Args:
            queries: List of search queries
            documents_list: List of document lists
            top_k: Number to return per query

        Returns:
            List of top-k document lists
        """
        if not queries:
            return []

        # Build all pairs
        all_pairs = []
        pair_indices = []

        for q_idx, (query, docs) in enumerate(zip(queries, documents_list, strict=True)):
            for d_idx, doc in enumerate(docs):
                all_pairs.append((query, doc.text))
                pair_indices.append((q_idx, d_idx))

        if not all_pairs:
            return [[] for _ in queries]

        # Score all at once
        scores = self._model.predict(
            all_pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        # Copy documents to avoid mutating caller's objects, then assign scores
        copied_lists = [[copy.copy(d) for d in docs] for docs in documents_list]
        for (q_idx, d_idx), score in zip(pair_indices, scores, strict=True):
            copied_lists[q_idx][d_idx].score = float(score)

        # Sort and return
        results = []
        for docs in copied_lists:
            sorted_docs = sorted(docs, key=lambda d: d.score, reverse=True)
            results.append(sorted_docs[:top_k])

        return results

    def unload(self):
        """Unload model."""
        import gc

        import torch

        del self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
