"""Hierarchical index for parent-child chunk retrieval."""

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.core.types import Document
from ragicamp.rag.chunking.hierarchical import HierarchicalChunker
from ragicamp.utils.artifacts import get_artifact_manager

if TYPE_CHECKING:
    from ragicamp.models.embedder import Embedder

logger = get_logger(__name__)


class HierarchicalIndex:
    """Index for hierarchical (parent-child) chunk retrieval.

    Stores:
    - Parent chunks (large, for context)
    - Child chunks (small, for precise matching)
    - Child-to-parent mapping
    - FAISS index of child embeddings

    The search finds similar children, then maps back to parents.
    This gives precise matching with rich context.
    """

    def __init__(
        self,
        name: str,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        parent_chunk_size: int = 1024,
        child_chunk_size: int = 256,
        parent_overlap: int = 100,
        child_overlap: int = 50,
        embedding_backend: str = "vllm",
        vllm_gpu_memory_fraction: float = Defaults.VLLM_GPU_MEMORY_FRACTION,
    ):
        """Initialize hierarchical index.

        Args:
            name: Index identifier
            embedding_model: Model for embeddings
            parent_chunk_size: Size of parent chunks
            child_chunk_size: Size of child chunks
            parent_overlap: Overlap between parent chunks
            child_overlap: Overlap between child chunks
            embedding_backend: 'vllm' or 'sentence_transformers'
            vllm_gpu_memory_fraction: GPU memory fraction for vLLM embeddings
        """
        self.name = name
        self.embedding_model_name = embedding_model
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap

        # Embedding backend configuration
        self.embedding_backend = embedding_backend
        self.vllm_gpu_memory_fraction = vllm_gpu_memory_fraction

        # Lazy load encoder
        self._encoder: Embedder | None = None
        self._embedding_dim: int | None = None

        # Chunker
        self._chunker: HierarchicalChunker | None = None

        # Storage
        self.parent_docs: list[Document] = []
        self.child_docs: list[Document] = []
        self.child_to_parent: dict[str, str] = {}
        self.parent_id_to_idx: dict[str, int] = {}

        # FAISS index for child chunks
        self.index: faiss.Index | None = None

    @property
    def encoder(self) -> "Embedder":
        """Lazy load encoder with backend-specific optimizations.

        Backends:
        - vllm: Uses vLLM's continuous batching for high throughput
        - sentence_transformers: Default, with Flash Attention, FP16, torch.compile
        """
        if self._encoder is None:
            if self.embedding_backend == "vllm":
                self._load_vllm_encoder()
            else:
                self._load_sentence_transformers_encoder()

        return self._encoder

    def _load_vllm_encoder(self):
        """Load vLLM embedding model."""
        from ragicamp.models.vllm_embedder import VLLMEmbedder

        # For inference (query encoding), use lower GPU fraction to leave room for generator
        gpu_fraction = Defaults.VLLM_EMBEDDER_GPU_MEMORY_FRACTION_SHARED

        logger.info(
            "Loading vLLM embedding model: %s (gpu_fraction=%.0f%% for inference)",
            self.embedding_model_name,
            gpu_fraction * 100,
        )
        self._encoder = VLLMEmbedder(
            model_name=self.embedding_model_name,
            gpu_memory_fraction=gpu_fraction,
            enforce_eager=False,  # Use CUDA graphs for speed
        )
        self._embedding_dim = self._encoder.get_sentence_embedding_dimension()
        logger.info("vLLM embedder loaded (dim=%d)", self._embedding_dim)

    def _load_sentence_transformers_encoder(self):
        """Load sentence-transformers model with optimizations."""
        import torch
        from sentence_transformers import SentenceTransformer

        model_kwargs = {}

        # Optimization 1: Flash Attention 2
        if Defaults.EMBEDDING_USE_FLASH_ATTENTION:
            try:
                import flash_attn  # noqa: F401

                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 enabled for embeddings")
            except ImportError:
                logger.debug("flash-attn not installed, using default attention")

        # Optimization 2: FP16/BF16 precision
        if Defaults.EMBEDDING_USE_FP16 and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                model_kwargs["torch_dtype"] = torch.bfloat16
                logger.info("Using BF16 precision for embeddings")
            else:
                model_kwargs["torch_dtype"] = torch.float16
                logger.info("Using FP16 precision for embeddings")

        self._encoder = SentenceTransformer(
            self.embedding_model_name,
            trust_remote_code=True,
            model_kwargs=model_kwargs if model_kwargs else None,
        )
        self._embedding_dim = self._encoder.get_sentence_embedding_dimension()

        # Optimization 3: torch.compile() for kernel fusion
        if Defaults.EMBEDDING_USE_TORCH_COMPILE:
            try:
                if hasattr(torch, "compile") and torch.cuda.is_available():
                    self._encoder = torch.compile(self._encoder, mode="reduce-overhead")
                    logger.info("Applied torch.compile() to embedding model")
            except Exception as e:
                logger.debug("torch.compile() not applied: %s", e)

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is None:
            _ = self.encoder
        return self._embedding_dim

    @property
    def chunker(self) -> HierarchicalChunker:
        """Lazy load chunker."""
        if self._chunker is None:
            self._chunker = HierarchicalChunker(
                parent_chunk_size=self.parent_chunk_size,
                parent_chunk_overlap=self.parent_overlap,
                child_chunk_size=self.child_chunk_size,
                child_chunk_overlap=self.child_overlap,
            )
        return self._chunker

    def build(self, documents: list[Document]) -> None:
        """Build hierarchical index from documents.

        Args:
            documents: List of documents to index
        """
        logger.info(
            "Building hierarchical index for %d documents (parent=%d, child=%d)",
            len(documents),
            self.parent_chunk_size,
            self.child_chunk_size,
        )

        # Create hierarchical chunks
        self.parent_docs, self.child_docs, self.child_to_parent = self.chunker.chunk_documents(
            iter(documents)
        )

        # Build parent ID lookup
        self.parent_id_to_idx = {doc.id: idx for idx, doc in enumerate(self.parent_docs)}

        logger.info(
            "Created %d parent chunks and %d child chunks",
            len(self.parent_docs),
            len(self.child_docs),
        )

        if not self.child_docs:
            logger.warning("No child chunks created!")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            return

        # Embed child chunks
        child_texts = [doc.text for doc in self.child_docs]
        embeddings = self.encoder.encode(child_texts, show_progress_bar=True)

        # Normalize for cosine similarity (guard against zero-norm)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embeddings = embeddings / norms

        # Create and populate FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype("float32"))

        logger.info("Hierarchical index built with %d child vectors", self.index.ntotal)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[tuple[int, float, int]]:
        """Search children and return parent indices.

        Args:
            query_embedding: Query vector (normalized)
            top_k: Number of parent results

        Returns:
            List of (parent_idx, score, best_child_idx) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Search more children than needed (multiple may map to same parent)
        num_children = top_k * 5

        query = query_embedding.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query, num_children)

        # Map children to parents and track best scores
        parent_scores: dict[str, float] = {}
        parent_best_child: dict[str, int] = {}

        for idx, score in zip(indices[0], scores[0], strict=True):
            if idx < 0 or idx >= len(self.child_docs):
                continue

            child_doc = self.child_docs[idx]
            parent_id = self.child_to_parent.get(child_doc.id)

            if parent_id is None:
                continue

            # Keep best score for each parent
            if parent_id not in parent_scores or score > parent_scores[parent_id]:
                parent_scores[parent_id] = float(score)
                parent_best_child[parent_id] = int(idx)

        # Sort by score and return top-k
        sorted_parents = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for parent_id, score in sorted_parents[:top_k]:
            parent_idx = self.parent_id_to_idx.get(parent_id)
            if parent_idx is not None:
                results.append((parent_idx, score, parent_best_child[parent_id]))

        return results

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query string.

        Args:
            query: Query text

        Returns:
            Normalized query embedding
        """
        embedding = self.encoder.encode([query]).astype("float32")
        faiss.normalize_L2(embedding)
        return embedding[0]

    def batch_encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode multiple queries at once (much faster than sequential).

        Args:
            queries: List of query texts

        Returns:
            Normalized query embeddings, shape (n_queries, embedding_dim)
        """
        embeddings = self.encoder.encode(queries, show_progress_bar=False).astype("float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    def batch_search(
        self, query_embeddings: np.ndarray, top_k: int = 10
    ) -> list[list[tuple[int, float, int]]]:
        """Search children for multiple queries and return parent indices.

        Args:
            query_embeddings: Query vectors, shape (n_queries, embedding_dim)
            top_k: Number of parent results per query

        Returns:
            List of [(parent_idx, score, best_child_idx), ...] for each query
        """
        if self.index is None or self.index.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]

        # Search more children than needed (multiple may map to same parent)
        num_children = top_k * 5

        scores_batch, indices_batch = self.index.search(
            query_embeddings.astype("float32"), num_children
        )

        all_results = []
        for q_idx in range(len(query_embeddings)):
            # Map children to parents and track best scores
            parent_scores: dict[str, float] = {}
            parent_best_child: dict[str, int] = {}

            for idx, score in zip(indices_batch[q_idx], scores_batch[q_idx], strict=True):
                if idx < 0 or idx >= len(self.child_docs):
                    continue

                child_doc = self.child_docs[idx]
                parent_id = self.child_to_parent.get(child_doc.id)

                if parent_id is None:
                    continue

                # Keep best score for each parent
                if parent_id not in parent_scores or score > parent_scores[parent_id]:
                    parent_scores[parent_id] = float(score)
                    parent_best_child[parent_id] = int(idx)

            # Sort by score and return top-k
            sorted_parents = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)

            results = []
            for parent_id, score in sorted_parents[:top_k]:
                parent_idx = self.parent_id_to_idx.get(parent_id)
                if parent_idx is not None:
                    results.append((parent_idx, score, parent_best_child[parent_id]))

            all_results.append(results)

        return all_results

    def get_parent(self, idx: int) -> Document | None:
        """Get parent document by index."""
        if 0 <= idx < len(self.parent_docs):
            return self.parent_docs[idx]
        return None

    def get_child(self, idx: int) -> Document | None:
        """Get child document by index."""
        if 0 <= idx < len(self.child_docs):
            return self.child_docs[idx]
        return None

    def get_document(self, idx: int) -> Document | None:
        """Get parent document (alias for get_parent)."""
        return self.get_parent(idx)

    def __len__(self) -> int:
        """Number of parent documents."""
        return len(self.parent_docs)

    def save(self, path: Path | None = None) -> Path:
        """Save index to disk.

        Args:
            path: Optional custom path

        Returns:
            Path where saved
        """
        manager = get_artifact_manager()

        if path is None:
            path = manager.get_embedding_index_path(self.name)
        else:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "child_index.faiss"))

        # Save documents and mappings
        with open(path / "parent_docs.pkl", "wb") as f:
            pickle.dump(self.parent_docs, f)

        with open(path / "child_docs.pkl", "wb") as f:
            pickle.dump(self.child_docs, f)

        with open(path / "child_to_parent.pkl", "wb") as f:
            pickle.dump(self.child_to_parent, f)

        # Save config
        config = {
            "name": self.name,
            "type": "hierarchical",
            "embedding_model": self.embedding_model_name,
            "embedding_backend": self.embedding_backend,
            "parent_chunk_size": self.parent_chunk_size,
            "child_chunk_size": self.child_chunk_size,
            "parent_overlap": self.parent_overlap,
            "child_overlap": self.child_overlap,
            "embedding_dim": self.embedding_dim,
            "num_parents": len(self.parent_docs),
            "num_children": len(self.child_docs),
        }
        manager.save_json(config, path / "config.json")

        logger.info("Saved hierarchical index to: %s", path)
        return path

    @classmethod
    def load(cls, name: str, path: Path | None = None) -> "HierarchicalIndex":
        """Load index from disk.

        Args:
            name: Index name
            path: Optional custom path

        Returns:
            Loaded HierarchicalIndex
        """
        manager = get_artifact_manager()

        if path is None:
            path = manager.get_embedding_index_path(name)
        else:
            path = Path(path)

        # Load config
        config = manager.load_json(path / "config.json")

        # Create index
        index = cls(
            name=config["name"],
            embedding_model=config["embedding_model"],
            parent_chunk_size=config["parent_chunk_size"],
            child_chunk_size=config["child_chunk_size"],
            parent_overlap=config.get("parent_overlap", 100),
            child_overlap=config.get("child_overlap", 50),
            embedding_backend=config.get("embedding_backend", "vllm"),
        )
        index._embedding_dim = config.get("embedding_dim")

        # Load FAISS index
        index.index = faiss.read_index(str(path / "child_index.faiss"))

        # Load documents and mappings
        # Handle both single-dump and batched pickle files
        def load_batched_list(filepath):
            """Load a pickle file that may contain multiple batched dumps."""
            items = []
            with open(filepath, "rb") as f:
                while True:
                    try:
                        batch = pickle.load(f)
                        if isinstance(batch, list):
                            items.extend(batch)
                        else:
                            items.append(batch)
                    except EOFError:
                        break
            return items

        def load_batched_dict(filepath):
            """Load a pickle file with batched dicts, merging them."""
            result = {}
            with open(filepath, "rb") as f:
                while True:
                    try:
                        batch = pickle.load(f)
                        if isinstance(batch, dict):
                            result.update(batch)
                        else:
                            # Single dump of full dict
                            return batch
                    except EOFError:
                        break
            return result

        index.parent_docs = load_batched_list(path / "parent_docs.pkl")
        index.child_docs = load_batched_list(path / "child_docs.pkl")
        index.child_to_parent = load_batched_dict(path / "child_to_parent.pkl")

        # Rebuild lookups
        index.parent_id_to_idx = {doc.id: i for i, doc in enumerate(index.parent_docs)}

        logger.info(
            "Loaded hierarchical index: %s (%d parents, %d children)",
            name,
            len(index.parent_docs),
            len(index.child_docs),
        )
        return index
