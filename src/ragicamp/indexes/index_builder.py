"""IndexBuilder - Builds VectorIndex with provider pattern.

Clean separation of concerns:
- Corpus: provides documents
- Chunker: splits documents
- EmbedderProvider: converts text to embeddings (managed lifecycle)
- IndexBuilder: orchestrates the build process

Supports:
- Incremental building (batch by batch)
- Checkpointing for crash recovery
- Memory-efficient streaming
"""

import gc
import json
import pickle
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.core.types import Document
from ragicamp.indexes.vector_index import IndexConfig, VectorIndex
from ragicamp.models.providers import EmbedderProvider

logger = get_logger(__name__)


@dataclass
class BuildCheckpoint:
    """Checkpoint for resuming builds."""

    batch_num: int
    total_docs: int
    total_chunks: int

    def to_dict(self) -> dict:
        return {
            "batch_num": self.batch_num,
            "total_docs": self.total_docs,
            "total_chunks": self.total_chunks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BuildCheckpoint":
        return cls(
            batch_num=data["batch_num"],
            total_docs=data["total_docs"],
            total_chunks=data["total_chunks"],
        )


class IndexBuilder:
    """Builds VectorIndex with clean resource management.

    Usage:
        builder = IndexBuilder(
            embedder_provider=EmbedderProvider(EmbedderConfig("BAAI/bge-large-en")),
            chunker=chunker,
            index_type="hnsw",
        )

        index = builder.build(
            documents=corpus.load(),
            output_path="my_index",
            batch_size=5000,
        )

    The embedder is only loaded during the build, then unloaded.
    """

    def __init__(
        self,
        embedder_provider: EmbedderProvider,
        chunker: Any,  # DocumentChunker
        index_type: str = "hnsw",
        embedding_batch_size: int = 4096,
    ):
        """Initialize builder.

        Args:
            embedder_provider: Provider for embedder (lazy loading)
            chunker: Document chunker
            index_type: FAISS index type (flat, ivf, hnsw)
            embedding_batch_size: Batch size for encoding
        """
        self.embedder_provider = embedder_provider
        self.chunker = chunker
        self.index_type = index_type
        self.embedding_batch_size = embedding_batch_size

    def build(
        self,
        documents: Iterator[Document] | list[Document],
        output_path: Path | str,
        doc_batch_size: int = 5000,
        checkpoint_interval: int = 1,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> VectorIndex:
        """Build index from documents.

        Args:
            documents: Iterator or list of documents
            output_path: Where to save the index
            doc_batch_size: Documents per batch
            checkpoint_interval: Save checkpoint every N batches
            on_batch: Callback(batch_num, total_chunks) after each batch

        Returns:
            Built VectorIndex
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        work_dir = output_path / ".work"
        work_dir.mkdir(exist_ok=True)

        # Check for checkpoint
        checkpoint = self._load_checkpoint(work_dir)
        start_batch = checkpoint.batch_num if checkpoint else 0
        total_docs = checkpoint.total_docs if checkpoint else 0
        total_chunks = checkpoint.total_chunks if checkpoint else 0

        # Load or create FAISS index
        index_path = work_dir / "index.faiss"
        chunks_path = work_dir / "chunks.pkl"

        if checkpoint and index_path.exists():
            logger.info("Resuming from batch %d (%d chunks)", start_batch, total_chunks)
            faiss_index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                all_chunks = pickle.load(f)
            embedding_dim = faiss_index.d
        else:
            faiss_index = None
            all_chunks = []
            embedding_dim = None

        # Load embedder for the build
        logger.info("Loading embedder for build...")

        with self.embedder_provider.load() as embedder:
            if embedding_dim is None:
                embedding_dim = embedder.get_dimension()
                faiss_index = self._create_faiss_index(embedding_dim)

            # Process in batches
            doc_batch = []
            batch_num = start_batch  # Continue numbering from checkpoint
            docs_to_skip = start_batch * doc_batch_size

            logger.info("Building index (batch_size=%d)...", doc_batch_size)

            for doc in documents:
                # Skip already processed docs
                if docs_to_skip > 0:
                    docs_to_skip -= 1
                    continue

                doc_batch.append(doc)

                if len(doc_batch) >= doc_batch_size:
                    batch_num += 1

                    batch_chunks, batch_embeddings = self._process_batch(
                        doc_batch, embedder, batch_num
                    )

                    all_chunks.extend(batch_chunks)
                    if len(batch_embeddings) > 0:
                        faiss_index.add(batch_embeddings)

                    total_docs += len(doc_batch)
                    total_chunks += len(batch_chunks)
                    doc_batch = []

                    if on_batch:
                        on_batch(batch_num, total_chunks)

                    # Checkpoint
                    if batch_num % checkpoint_interval == 0:
                        self._save_checkpoint(
                            work_dir, faiss_index, all_chunks, batch_num, total_docs, total_chunks
                        )

                    gc.collect()

            # Final batch
            if doc_batch:
                batch_num += 1
                batch_chunks, batch_embeddings = self._process_batch(
                    doc_batch, embedder, batch_num, is_final=True
                )

                all_chunks.extend(batch_chunks)
                if len(batch_embeddings) > 0:
                    faiss_index.add(batch_embeddings)

                total_docs += len(doc_batch)
                total_chunks += len(batch_chunks)

        # Embedder is now unloaded
        logger.info("Build complete: %d docs → %d chunks", total_docs, total_chunks)

        # Create config
        config = IndexConfig(
            embedding_model=self.embedder_provider.model_name,
            embedding_dim=embedding_dim,
            index_type=self.index_type,
            n_documents=total_chunks,
            chunk_size=getattr(self.chunker.config, "chunk_size", None),
            chunk_overlap=getattr(self.chunker.config, "chunk_overlap", None),
        )

        # Create index
        index = VectorIndex(
            faiss_index=faiss_index,
            documents=all_chunks,
            config=config,
        )

        # Save to final location
        index.save(output_path)

        # Cleanup work directory
        self._cleanup_work_dir(work_dir)

        return index

    def _process_batch(
        self,
        doc_batch: list[Document],
        embedder,
        batch_num: int,
        is_final: bool = False,
    ) -> tuple[list[Document], np.ndarray]:
        """Process a batch: chunk → embed → normalize."""
        suffix = " (final)" if is_final else ""
        logger.info("[Batch %d] Processing %d docs%s", batch_num, len(doc_batch), suffix)

        # Chunk documents
        t0 = time.time()
        if hasattr(self.chunker, "chunk_documents"):
            chunks = list(self.chunker.chunk_documents(iter(doc_batch), show_progress=False))
        else:
            chunks = []
            for doc in doc_batch:
                doc_chunks = list(self.chunker.chunk(doc))
                chunks.extend(doc_chunks)

        logger.info("  Chunked: %d chunks in %.1fs", len(chunks), time.time() - t0)

        if not chunks:
            return [], np.empty((0, 0), dtype=np.float32)

        # Embed
        t0 = time.time()
        texts = [c.text for c in chunks]
        embeddings = embedder.batch_encode(texts)

        logger.info("  Embedded: %d vectors in %.1fs", len(embeddings), time.time() - t0)

        # Normalize
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        return chunks, embeddings

    def _create_faiss_index(self, dim: int) -> faiss.Index:
        """Create FAISS index based on type."""
        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            return index
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            return faiss.IndexIVFFlat(quantizer, dim, 4096, faiss.METRIC_INNER_PRODUCT)
        else:  # flat
            return faiss.IndexFlatIP(dim)

    def _load_checkpoint(self, work_dir: Path) -> BuildCheckpoint | None:
        """Load checkpoint if exists."""
        path = work_dir / "checkpoint.json"
        if not path.exists():
            return None

        with open(path) as f:
            return BuildCheckpoint.from_dict(json.load(f))

    def _save_checkpoint(
        self,
        work_dir: Path,
        faiss_index: faiss.Index,
        chunks: list[Document],
        batch_num: int,
        total_docs: int,
        total_chunks: int,
    ):
        """Save checkpoint for resume."""
        logger.info("  Saving checkpoint (batch %d)...", batch_num)

        # Save FAISS index
        faiss.write_index(faiss_index, str(work_dir / "index.faiss"))

        # Save chunks
        with open(work_dir / "chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)

        # Save checkpoint metadata
        checkpoint = BuildCheckpoint(batch_num, total_docs, total_chunks)
        with open(work_dir / "checkpoint.json", "w") as f:
            json.dump(checkpoint.to_dict(), f)

    def _cleanup_work_dir(self, work_dir: Path):
        """Clean up work directory after successful build."""
        import shutil

        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass


# =============================================================================
# Convenience function
# =============================================================================


def build_index(
    embedding_model: str,
    documents: Iterator[Document] | list[Document],
    output_path: str | Path,
    chunk_size: int = Defaults.CHUNK_SIZE,
    chunk_overlap: int = Defaults.CHUNK_OVERLAP,
    chunking_strategy: str = "recursive",
    index_type: str = Defaults.FAISS_INDEX_TYPE,
    embedding_backend: str = "vllm",
    doc_batch_size: int = 5000,
    embedding_batch_size: int = 4096,
) -> VectorIndex:
    """Build a vector index from documents.

    Convenience function that creates all required components.

    Args:
        embedding_model: Model name (e.g., "BAAI/bge-large-en-v1.5")
        documents: Documents to index
        output_path: Where to save
        chunk_size: Chunk size in characters
        chunk_overlap: Overlap between chunks
        chunking_strategy: "recursive", "fixed", "sentence"
        index_type: "flat", "ivf", "hnsw"
        embedding_backend: "vllm" or "sentence_transformers"
        doc_batch_size: Documents per batch
        embedding_batch_size: Embeddings per GPU batch

    Returns:
        Built VectorIndex
    """
    from ragicamp.corpus import ChunkConfig, DocumentChunker
    from ragicamp.models.providers import EmbedderConfig, EmbedderProvider

    # Create components
    embedder_provider = EmbedderProvider(
        EmbedderConfig(
            model_name=embedding_model,
            backend=embedding_backend,
        )
    )

    chunker = DocumentChunker(
        ChunkConfig(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )

    builder = IndexBuilder(
        embedder_provider=embedder_provider,
        chunker=chunker,
        index_type=index_type,
        embedding_batch_size=embedding_batch_size,
    )

    return builder.build(
        documents=documents,
        output_path=output_path,
        doc_batch_size=doc_batch_size,
    )
