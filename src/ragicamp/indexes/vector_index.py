"""VectorIndex - Pure data index (FAISS + documents).

Design principles:
- Index is JUST DATA - no model ownership
- Embeddings provided externally by EmbedderProvider
- Supports sharded indexes for large corpora
- Memory-mapped loading for RAM efficiency
- GPU FAISS for fast search (optional)
"""

import json
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.core.types import Document, SearchResult

logger = get_logger(__name__)


# =============================================================================
# FAISS GPU Resources (thread-safe singleton)
# =============================================================================

_faiss_gpu_lock = threading.Lock()
_faiss_gpu_resources = None


def get_faiss_gpu_resources():
    """Get shared FAISS GPU resources (thread-safe).

    Returns the singleton ``StandardGpuResources`` instance, creating it on
    first call.  Access is protected by a lock so concurrent threads won't
    race to initialise the resource.
    """
    global _faiss_gpu_resources

    if not hasattr(faiss, "StandardGpuResources"):
        return None

    # Fast path – already initialised
    if _faiss_gpu_resources is not None:
        return _faiss_gpu_resources

    with _faiss_gpu_lock:
        # Double-check after acquiring the lock
        if _faiss_gpu_resources is not None:
            return _faiss_gpu_resources
        try:
            res = faiss.StandardGpuResources()
            res.setTempMemory(Defaults.FAISS_GPU_TEMP_MEMORY_MB * 1024 * 1024)
            _faiss_gpu_resources = res
            logger.info("Initialized FAISS GPU resources")
        except Exception as e:
            logger.warning("Failed to init FAISS GPU: %s", e)
            return None

    return _faiss_gpu_resources


def release_faiss_gpu_resources() -> None:
    """Release FAISS GPU resources to free memory.

    Called by ``ResourceManager.clear_faiss_gpu_resources()`` when
    transitioning from retrieval to generation phase.
    """
    global _faiss_gpu_resources

    with _faiss_gpu_lock:
        if _faiss_gpu_resources is not None:
            _faiss_gpu_resources = None
            logger.info("Released FAISS GPU resources")


# =============================================================================
# VectorIndex
# =============================================================================


@dataclass
class IndexConfig:
    """Configuration for vector index."""

    embedding_model: str
    embedding_dim: int
    index_type: str = "flat"  # flat, ivf, hnsw
    n_documents: int = 0
    chunk_size: int | None = None
    chunk_overlap: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "n_documents": self.n_documents,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexConfig":
        return cls(
            embedding_model=data["embedding_model"],
            embedding_dim=data["embedding_dim"],
            index_type=data.get("index_type", "flat"),
            n_documents=data.get("n_documents", data.get("num_documents", 0)),
            chunk_size=data.get("chunk_size"),
            chunk_overlap=data.get("chunk_overlap"),
        )


class VectorIndex:
    """FAISS-based vector index - pure data, no models.

    The index stores:
    - FAISS index with embeddings
    - Document list (id, text, metadata)
    - Configuration (model name, dimensions, etc.)

    Embeddings are provided externally - the index does NOT own an embedder.
    This allows clean GPU lifecycle management by agents.

    Usage:
        # Build with external embedder
        with embedder_provider.load() as embedder:
            embeddings = embedder.batch_encode(texts)
            index = VectorIndex.build(documents, embeddings, config)
            index.save("my_index")

        # Search with external embedder
        index = VectorIndex.load("my_index")
        with embedder_provider.load() as embedder:
            query_emb = embedder.batch_encode(queries)
            results = index.batch_search(query_emb, top_k=5)
    """

    def __init__(
        self,
        faiss_index: faiss.Index,
        documents: list[Document],
        config: IndexConfig,
        is_sharded: bool = False,
        shard_indexes: list[faiss.Index] | None = None,
        shard_offsets: list[int] | None = None,
    ):
        self.faiss_index = faiss_index
        self.documents = documents
        self.config = config

        # Sharding support
        self._is_sharded = is_sharded
        self._shard_indexes = shard_indexes or []
        self._shard_offsets = shard_offsets or [0]

        # GPU state
        self._is_gpu = False

    @classmethod
    def build(
        cls,
        documents: list[Document],
        embeddings: np.ndarray,
        config: IndexConfig,
        normalize: bool = True,
    ) -> "VectorIndex":
        """Build index from documents and pre-computed embeddings.

        Args:
            documents: List of documents (same order as embeddings)
            embeddings: Pre-computed embeddings, shape (n_docs, dim)
            config: Index configuration
            normalize: Whether to L2-normalize embeddings

        Returns:
            VectorIndex ready for search
        """
        n_docs, dim = embeddings.shape

        if len(documents) != n_docs:
            raise ValueError(f"Documents ({len(documents)}) != embeddings ({n_docs})")

        if dim != config.embedding_dim:
            raise ValueError(f"Embedding dim ({dim}) != config ({config.embedding_dim})")

        # Normalize for cosine similarity
        embeddings = embeddings.astype(np.float32)
        if normalize:
            faiss.normalize_L2(embeddings)

        # Create FAISS index
        if config.index_type == "hnsw":
            faiss_index = faiss.IndexHNSWFlat(
                dim, Defaults.FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT
            )
            faiss_index.hnsw.efConstruction = Defaults.FAISS_HNSW_EF_CONSTRUCTION
            faiss_index.hnsw.efSearch = Defaults.FAISS_HNSW_EF_SEARCH
        elif config.index_type == "ivf":
            nlist = min(Defaults.FAISS_IVF_NLIST, max(64, int(np.sqrt(n_docs) * 2)))
            quantizer = faiss.IndexFlatIP(dim)
            faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            faiss_index.train(embeddings)
            faiss_index.nprobe = min(Defaults.FAISS_IVF_NPROBE, nlist // 4)
        else:  # flat
            faiss_index = faiss.IndexFlatIP(dim)

        faiss_index.add(embeddings)
        config.n_documents = n_docs

        logger.info("Built %s index: %d vectors, dim=%d", config.index_type, n_docs, dim)
        return cls(faiss_index, documents, config)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query_embedding: Query vector (will be normalized)
            top_k: Number of results

        Returns:
            List of SearchResult
        """
        results = self.batch_search(query_embedding.reshape(1, -1), top_k)
        return results[0] if results else []

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> list[list[SearchResult]]:
        """Search for multiple queries at once.

        Args:
            query_embeddings: Query vectors, shape (n_queries, dim)
            top_k: Number of results per query

        Returns:
            List of SearchResult lists, one per query
        """
        queries = query_embeddings.astype(np.float32)
        faiss.normalize_L2(queries)

        if self._is_sharded:
            return self._batch_search_sharded(queries, top_k)

        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return [[] for _ in range(len(queries))]

        scores, indices = self.faiss_index.search(queries, top_k)

        all_results = []
        for q_idx in range(len(queries)):
            results = []
            for rank, (idx, score) in enumerate(zip(indices[q_idx], scores[q_idx], strict=True)):
                if 0 <= idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append(
                        SearchResult(
                            document=Document(
                                id=doc.id,
                                text=doc.text,
                                metadata=doc.metadata.copy(),
                                score=float(score),
                            ),
                            score=float(score),
                            rank=rank,
                        )
                    )
            all_results.append(results)

        return all_results

    def _batch_search_sharded(
        self,
        queries: np.ndarray,
        top_k: int,
    ) -> list[list[SearchResult]]:
        """Search across shards and merge results."""
        n_queries = len(queries)
        all_shard_results: list[list[tuple[int, float]]] = [[] for _ in range(n_queries)]

        for shard_idx, shard_index in enumerate(self._shard_indexes):
            if shard_index.ntotal == 0:
                continue

            scores, indices = shard_index.search(queries, top_k)
            doc_offset = self._shard_offsets[shard_idx]

            for q_idx in range(n_queries):
                for idx, score in zip(indices[q_idx], scores[q_idx], strict=True):
                    if idx >= 0:
                        global_idx = doc_offset + int(idx)
                        if 0 <= global_idx < len(self.documents):
                            all_shard_results[q_idx].append((global_idx, float(score)))

        # Sort and build results
        final_results = []
        for q_results in all_shard_results:
            q_results.sort(key=lambda x: x[1], reverse=True)
            results = []
            for rank, (idx, score) in enumerate(q_results[:top_k]):
                doc = self.documents[idx]
                results.append(
                    SearchResult(
                        document=Document(
                            id=doc.id,
                            text=doc.text,
                            metadata=doc.metadata.copy(),
                            score=score,
                        ),
                        score=score,
                        rank=rank,
                    )
                )
            final_results.append(results)

        return final_results

    def to_gpu(self) -> "VectorIndex":
        """Move index to GPU for faster search."""
        if self._is_gpu:
            return self

        if self.config.index_type == "hnsw":
            logger.warning("HNSW doesn't support GPU, staying on CPU")
            return self

        gpu_res = get_faiss_gpu_resources()
        if gpu_res is None:
            logger.warning("FAISS GPU not available")
            return self

        try:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True

            if self._is_sharded:
                self._shard_indexes = [
                    faiss.index_cpu_to_gpu(gpu_res, 0, idx, co) for idx in self._shard_indexes
                ]
            else:
                self.faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, self.faiss_index, co)

            self._is_gpu = True
            logger.info("Moved index to GPU")
        except Exception as e:
            logger.warning("Failed to move to GPU: %s", e)

        return self

    def save(self, path: Path | str) -> None:
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Convert GPU to CPU for saving
        index_to_save = self.faiss_index
        if self._is_gpu and not self._is_sharded:
            index_to_save = faiss.index_gpu_to_cpu(self.faiss_index)

        # Save FAISS index
        faiss.write_index(index_to_save, str(path / "index.faiss"))

        # Save documents
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info("Saved index to %s (%d docs)", path, len(self.documents))

    @classmethod
    def load(
        cls,
        path: Path | str,
        use_mmap: bool = True,
        use_gpu: bool = False,
    ) -> "VectorIndex":
        """Load index from disk.

        Args:
            path: Index directory
            use_mmap: Memory-map the index (reduces RAM for large indexes)
            use_gpu: Move index to GPU after loading

        Supports both old and new index formats:
        - New: config.json with full IndexConfig
        - Old: retriever_config.json or minimal auto-detection
        """
        path = Path(path)

        # Load config (try multiple formats)
        config = cls._load_config(path)

        # Check for sharded index
        shard_dirs = sorted(
            [d for d in path.iterdir() if d.is_dir() and d.name.startswith("shard_")]
        )

        if shard_dirs:
            index = cls._load_sharded(path, shard_dirs, config, use_mmap)
        else:
            index = cls._load_single(path, config, use_mmap)

        if use_gpu:
            index.to_gpu()

        logger.info(
            "Loaded index: %d docs, sharded=%s, gpu=%s",
            len(index.documents),
            index._is_sharded,
            index._is_gpu,
        )
        return index

    @classmethod
    def _load_single(cls, path: Path, config: IndexConfig, use_mmap: bool) -> "VectorIndex":
        """Load single-file index."""
        io_flags = faiss.IO_FLAG_MMAP if use_mmap else 0
        faiss_index = faiss.read_index(str(path / "index.faiss"), io_flags)

        # SECURITY: pickle.load executes arbitrary code. These files are only
        # loaded from locally-built artifacts (artifacts/ directory). Do NOT load
        # pickles from untrusted / user-uploaded sources without verification.
        with open(path / "documents.pkl", "rb") as f:
            documents = pickle.load(f)  # noqa: S301

        # Convert old Document format if needed
        documents = cls._ensure_document_type(documents)

        return cls(faiss_index, documents, config)

    @classmethod
    def _load_sharded(
        cls,
        path: Path,
        shard_dirs: list[Path],
        config: IndexConfig,
        use_mmap: bool,
    ) -> "VectorIndex":
        """Load sharded index."""
        io_flags = faiss.IO_FLAG_MMAP if use_mmap else 0

        shard_indexes = []
        shard_offsets = [0]
        total_vectors = 0

        for shard_dir in shard_dirs:
            shard_path = shard_dir / "index.faiss"
            if not shard_path.exists():
                continue

            shard_index = faiss.read_index(str(shard_path), io_flags)
            shard_indexes.append(shard_index)
            total_vectors += shard_index.ntotal
            shard_offsets.append(total_vectors)

        # Load documents
        docs_path = path / "documents.pkl"
        if docs_path.exists():
            with open(docs_path, "rb") as f:
                documents = pickle.load(f)
        else:
            documents = []
            for shard_dir in shard_dirs:
                shard_docs = shard_dir / "documents.pkl"
                if shard_docs.exists():
                    with open(shard_docs, "rb") as f:
                        documents.extend(pickle.load(f))

        documents = cls._ensure_document_type(documents)

        logger.info("Loaded %d shards with %d vectors", len(shard_indexes), total_vectors)

        return cls(
            faiss_index=shard_indexes[0] if shard_indexes else None,
            documents=documents,
            config=config,
            is_sharded=True,
            shard_indexes=shard_indexes,
            shard_offsets=shard_offsets,
        )

    @classmethod
    def _load_config(cls, path: Path) -> IndexConfig:
        """Load config from multiple possible formats.

        Tries in order:
        1. config.json (new format)
        2. retriever_config.json (old format)
        3. Auto-detect from FAISS index
        """
        # Try new format
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return IndexConfig.from_dict(json.load(f))

        # Try old retriever_config format
        old_config_path = path / "retriever_config.json"
        if old_config_path.exists():
            with open(old_config_path) as f:
                old_config = json.load(f)

            # Map old fields to new
            return IndexConfig(
                embedding_model=old_config.get(
                    "embedding_model", old_config.get("model", "unknown")
                ),
                embedding_dim=old_config.get("embedding_dim", old_config.get("dimension", 768)),
                index_type=old_config.get("index_type", old_config.get("faiss_index_type", "flat")),
                n_documents=old_config.get("num_documents", old_config.get("n_documents", 0)),
                chunk_size=old_config.get("chunk_size"),
                chunk_overlap=old_config.get("chunk_overlap"),
            )

        # Auto-detect from FAISS index
        faiss_path = path / "index.faiss"
        if faiss_path.exists():
            temp_index = faiss.read_index(str(faiss_path))
            dim = temp_index.d
            n_docs = temp_index.ntotal

            logger.warning(
                "No config found at %s, auto-detecting: dim=%d, n_docs=%d", path, dim, n_docs
            )

            return IndexConfig(
                embedding_model="unknown",
                embedding_dim=dim,
                index_type="flat",
                n_documents=n_docs,
            )

        # Check for sharded index
        shard_dirs = sorted(
            [d for d in path.iterdir() if d.is_dir() and d.name.startswith("shard_")]
        )
        if shard_dirs:
            first_shard = shard_dirs[0] / "index.faiss"
            if first_shard.exists():
                temp_index = faiss.read_index(str(first_shard))
                dim = temp_index.d

                logger.warning(
                    "Sharded index without config at %s, auto-detecting: dim=%d", path, dim
                )

                return IndexConfig(
                    embedding_model="unknown",
                    embedding_dim=dim,
                    index_type="flat",
                    n_documents=0,
                )

        raise FileNotFoundError(f"No valid index found at {path}")

    @staticmethod
    def _ensure_document_type(documents: list) -> list[Document]:
        """Convert legacy document formats to Document type."""
        if not documents:
            return []

        # Check if already correct type
        if isinstance(documents[0], Document):
            return documents

        # Convert from old format
        converted = []
        for doc in documents:
            if hasattr(doc, "id") and hasattr(doc, "text"):
                converted.append(
                    Document(
                        id=doc.id,
                        text=doc.text,
                        metadata=getattr(doc, "metadata", {}),
                        score=getattr(doc, "score", None),
                    )
                )
            elif isinstance(doc, dict):
                converted.append(Document.from_dict(doc))
            else:
                raise ValueError(f"Unknown document format: {type(doc)}")

        return converted

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        return f"VectorIndex({len(self)} docs, {self.config.index_type}, gpu={self._is_gpu})"
