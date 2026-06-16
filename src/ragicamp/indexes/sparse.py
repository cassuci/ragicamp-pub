"""Sparse index for keyword-based retrieval (TF-IDF or BM25).

Sparse indexes complement dense embeddings by handling:
- Exact keyword matches
- Technical terms and jargon
- Rare words that embeddings may not capture

Supports two methods:
- TF-IDF: Fast, simple, good for most use cases
- BM25: More sophisticated probabilistic ranking
"""

import pickle
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ragicamp.core.logging import get_logger
from ragicamp.core.types import Document
from ragicamp.utils.artifacts import get_artifact_manager

logger = get_logger(__name__)


class SparseMethod(str, Enum):
    """Sparse retrieval method."""

    TFIDF = "tfidf"
    BM25 = "bm25"


class SparseIndex:
    """Shared sparse index supporting TF-IDF or BM25.

    Sparse indexes complement dense embeddings by handling:
    - Exact keyword matches
    - Technical terms and jargon
    - Rare words that embeddings may not capture

    Can be:
    - Built once and saved to disk
    - Shared across multiple HybridSearchers
    - Loaded without rebuilding
    """

    def __init__(
        self,
        name: str,
        method: SparseMethod | str = SparseMethod.TFIDF,
        max_features: int = 50000,
    ):
        """Initialize sparse index.

        Args:
            name: Index identifier
            method: Sparse method ('tfidf' or 'bm25')
            max_features: Max vocabulary size for TF-IDF
        """
        self.name = name
        self.method = SparseMethod(method) if isinstance(method, str) else method
        self.max_features = max_features
        self.documents: list[Document] = []

        # TF-IDF components
        self._vectorizer: Any = None
        self._doc_vectors: Any = None

        # BM25 component (bm25s library)
        self._bm25: Any = None

    def build(self, documents: list[Document], show_progress: bool = True) -> None:
        """Build sparse index from documents.

        Args:
            documents: List of documents to index
            show_progress: Whether to show progress bar
        """
        self.documents = documents
        n_docs = len(documents)

        logger.info(
            "Building sparse index (%s) for %d documents...",
            self.method.value,
            n_docs,
        )

        # Extract texts
        if show_progress and n_docs > 10000:
            texts = [doc.text for doc in tqdm(documents, desc="Extracting texts")]
        else:
            texts = [doc.text for doc in documents]

        if self.method == SparseMethod.TFIDF:
            self._build_tfidf(texts)
        else:
            self._build_bm25(texts, show_progress)

        logger.info("Sparse index built: %s, %d documents", self.method.value, n_docs)

    def _build_tfidf(self, texts: list[str]) -> None:
        """Build TF-IDF index."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        logger.info("Fitting TF-IDF vectorizer (max_features=%d)...", self.max_features)
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
        )
        self._doc_vectors = self._vectorizer.fit_transform(texts)
        logger.info("TF-IDF: %d features", self._doc_vectors.shape[1])

    def _build_bm25(self, texts: list[str], show_progress: bool = True) -> None:
        """Build BM25 index using bm25s (vectorized sparse matrices)."""
        import bm25s

        logger.info("Tokenizing %d documents for BM25...", len(texts))
        corpus_tokens = bm25s.tokenize(
            texts,
            stopwords=None,
            stemmer=None,
            show_progress=show_progress,
        )
        logger.info("Tokenization complete, vocab size: %d", len(corpus_tokens.vocab))

        logger.info("Building BM25 sparse index...")
        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens, show_progress=show_progress)
        logger.info("BM25 index built (scipy sparse matrices)")

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Search sparse index.

        Args:
            query: Query string
            top_k: Number of results

        Returns:
            List of (doc_idx, score) tuples
        """
        if self.method == SparseMethod.TFIDF:
            return self._search_tfidf(query, top_k)
        else:
            return self._search_bm25(query, top_k)

    def _search_tfidf(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Search using TF-IDF."""
        from sklearn.metrics.pairwise import cosine_similarity

        if self._vectorizer is None or self._doc_vectors is None:
            return []

        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._doc_vectors)[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [
            (int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0.0
        ]

    def _search_bm25(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Search using BM25 (delegates to batch)."""
        if self._bm25 is None:
            return []
        return self._batch_search_bm25([query], top_k)[0]

    def _batch_search_bm25(self, queries: list[str], top_k: int) -> list[list[tuple[int, float]]]:
        """Vectorized batch BM25 search using bm25s.retrieve()."""
        import bm25s

        if self._bm25 is None:
            return [[] for _ in queries]

        query_tokens = bm25s.tokenize(
            queries,
            stopwords=None,
            stemmer=None,
            show_progress=False,
        )
        results_array, scores_array = self._bm25.retrieve(query_tokens, k=top_k)

        all_results: list[list[tuple[int, float]]] = []
        for i in range(len(queries)):
            query_results: list[tuple[int, float]] = []
            for j in range(top_k):
                doc_idx = int(results_array[i, j])
                score = float(scores_array[i, j])
                if doc_idx < 0 or score <= 0.0:
                    continue
                query_results.append((doc_idx, score))
            all_results.append(query_results)

        return all_results

    def batch_search(self, queries: list[str], top_k: int = 10) -> list[list[tuple[int, float]]]:
        """Batch search for multiple queries.

        Args:
            queries: List of query strings
            top_k: Number of results per query

        Returns:
            List of results lists
        """
        if self.method == SparseMethod.TFIDF:
            return self._batch_search_tfidf(queries, top_k)
        else:
            return self._batch_search_bm25(queries, top_k)

    def _batch_search_tfidf(self, queries: list[str], top_k: int) -> list[list[tuple[int, float]]]:
        """Batch TF-IDF search."""
        from sklearn.metrics.pairwise import cosine_similarity

        if self._vectorizer is None or self._doc_vectors is None:
            return [[] for _ in queries]

        query_vecs = self._vectorizer.transform(queries)
        all_similarities = cosine_similarity(query_vecs, self._doc_vectors)

        results = []
        for similarities in all_similarities:
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results.append(
                [
                    (int(idx), float(similarities[idx]))
                    for idx in top_indices
                    if similarities[idx] > 0.0
                ]
            )

        return results

    def get_document(self, idx: int) -> Document | None:
        """Get document by index."""
        if 0 <= idx < len(self.documents):
            return self.documents[idx]
        return None

    def save(self, path: Path | None = None) -> Path:
        """Save sparse index to disk.

        Args:
            path: Optional custom path

        Returns:
            Path where saved
        """
        manager = get_artifact_manager()

        if path is None:
            path = manager.get_sparse_index_path(self.name)
        else:
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            "name": self.name,
            "method": self.method.value,
            "max_features": self.max_features,
            "num_documents": len(self.documents),
        }
        manager.save_json(config, path / "config.json")

        # Save documents (reference by ID, actual docs come from embedding index)
        doc_ids = [doc.id for doc in self.documents]
        with open(path / "doc_ids.pkl", "wb") as f:
            pickle.dump(doc_ids, f)

        # Save method-specific components
        if self.method == SparseMethod.TFIDF:
            with open(path / "vectorizer.pkl", "wb") as f:
                pickle.dump(self._vectorizer, f)
            with open(path / "doc_vectors.pkl", "wb") as f:
                pickle.dump(self._doc_vectors, f)
        else:
            # bm25s native save (scipy sparse matrices + vocab_dict)
            self._bm25.save(str(path / "bm25s_data"))

        logger.info("Saved sparse index to: %s", path)
        return path

    @classmethod
    def load(
        cls,
        name: str,
        path: Path | None = None,
        documents: list[Document] | None = None,
    ) -> "SparseIndex":
        """Load sparse index from disk.

        Args:
            name: Index name
            path: Optional custom path
            documents: Optional documents (from embedding index)

        Returns:
            Loaded SparseIndex
        """
        manager = get_artifact_manager()

        if path is None:
            path = manager.get_sparse_index_path(name)
        else:
            path = Path(path)

        # Load config
        config = manager.load_json(path / "config.json")

        index = cls(
            name=config["name"],
            method=config["method"],
            max_features=config.get("max_features", 50000),
        )

        # Load documents if provided, otherwise just load IDs
        if documents is not None:
            index.documents = documents
        else:
            with open(path / "doc_ids.pkl", "rb") as f:
                doc_ids = pickle.load(f)
            # Create placeholder documents (will be linked later)
            index.documents = [Document(id=doc_id, text="") for doc_id in doc_ids]

        # SECURITY: pickle.load executes arbitrary code. These files are only
        # loaded from locally-built artifacts. Do NOT load from untrusted sources.
        if index.method == SparseMethod.TFIDF:
            with open(path / "vectorizer.pkl", "rb") as f:
                index._vectorizer = pickle.load(f)
            with open(path / "doc_vectors.pkl", "rb") as f:
                index._doc_vectors = pickle.load(f)
        else:
            import bm25s

            bm25_dir = path / "bm25s_data"

            # Detect old rank_bm25 pickle format
            if (path / "bm25.pkl").exists() and not bm25_dir.exists():
                raise ValueError(
                    f"BM25 index at {path} uses the old rank_bm25 pickle format. "
                    "Delete the sparse index directory and rebuild with: "
                    "make index-full (or make index-simple)"
                )

            index._bm25 = bm25s.BM25.load(str(bm25_dir), mmap=True)

        logger.info(
            "Loaded sparse index: %s (%s, %d docs)",
            name,
            index.method.value,
            len(index.documents),
        )
        return index

    def __len__(self) -> int:
        return len(self.documents)


def get_sparse_index_name(embedding_index_name: str, method: str = "tfidf") -> str:
    """Get canonical name for a sparse index.

    Sparse indexes are tied to embedding indexes (same documents).

    Args:
        embedding_index_name: Name of the embedding index
        method: Sparse method ('tfidf' or 'bm25')

    Returns:
        Sparse index name
    """
    return f"{embedding_index_name}_sparse_{method}"
