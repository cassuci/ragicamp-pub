"""Embedding index builder.

Builds dense vector indexes for semantic search.
Supports both sentence-transformers and vLLM embedding backends.

Uses incremental FAISS building for memory efficiency:
- Each batch: chunk → embed → normalize → add to FAISS
- Checkpoints index every N batches for crash recovery
- Never loads all embeddings into RAM at once

Supports checkpointing for resume after crashes.
"""

import gc
import os
import pickle
import time
from typing import Any

# Disable tokenizers parallelism to avoid fork warnings
# Must be set before importing transformers/tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.utils.artifacts import get_artifact_manager

from .checkpoint import CheckpointManager, EmbeddingCheckpoint
from .storage import EmbeddingStorage

logger = get_logger(__name__)

# Maximum document size in characters (larger docs are truncated)
MAX_DOC_CHARS = 100_000

# Checkpoint index every N batches
# Checkpoint every batch - large-scale embedding can take hours per batch
CHECKPOINT_INTERVAL = 1


def build_embedding_index(
    index_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    corpus_config: dict[str, Any],
    chunking_strategy: str = "recursive",
    doc_batch_size: int = 5000,
    embedding_batch_size: int = 64,
    chunk_workers: int = None,  # Kept for API compatibility, not used
    index_type: str = None,
    use_gpu: bool = None,
    nlist: int = None,
    nprobe: int = None,
    embedding_backend: str = "vllm",
    vllm_gpu_memory_fraction: float = Defaults.VLLM_GPU_MEMORY_FRACTION,
) -> str:
    """Build a shared embedding index with batched processing.

    This is the expensive part - chunking + embedding. Once built, multiple
    retrievers can reuse this index.

    Uses incremental FAISS building:
    - Each batch: chunk → embed → normalize → add to FAISS
    - Checkpoints every 5 batches for crash recovery
    - Memory efficient: only one batch in RAM at a time

    Supports resume from checkpoint if process is interrupted.

    Args:
        index_name: Canonical name for this index
        embedding_model: Embedding model to use
        chunk_size: Chunk size in characters
        chunk_overlap: Chunk overlap in characters
        corpus_config: Corpus configuration
        chunking_strategy: Chunking strategy ('recursive', 'fixed', 'sentence', 'paragraph')
        doc_batch_size: Documents to process per batch (default 5000)
        embedding_batch_size: Texts to embed per GPU batch (default 64, increase for more VRAM)
        chunk_workers: Deprecated, kept for API compatibility
        index_type: FAISS index type ('flat', 'ivf', 'ivfpq', 'hnsw'). Default: Defaults.FAISS_INDEX_TYPE
        use_gpu: Whether to use GPU FAISS. Default: Defaults.FAISS_USE_GPU
        nlist: Number of clusters for IVF indexes. Default: Defaults.FAISS_IVF_NLIST
        nprobe: Number of clusters to search. Default: Defaults.FAISS_IVF_NPROBE
        embedding_backend: 'vllm' or 'sentence_transformers'
        vllm_gpu_memory_fraction: GPU memory fraction for vLLM (default 0.9)

    Returns:
        Path to the saved index
    """
    # Apply defaults
    index_type = index_type or Defaults.FAISS_INDEX_TYPE
    use_gpu = use_gpu if use_gpu is not None else Defaults.FAISS_USE_GPU
    nlist = nlist or Defaults.FAISS_IVF_NLIST
    nprobe = nprobe or Defaults.FAISS_IVF_NPROBE

    import faiss
    import numpy as np
    from tqdm import tqdm

    from ragicamp.core.types import Document
    from ragicamp.corpus import ChunkConfig, CorpusConfig, DocumentChunker, WikipediaCorpus
    from ragicamp.models.embedder import create_embedder

    manager = get_artifact_manager()

    # Use all CPU cores for FAISS
    num_threads = os.cpu_count() or 8
    faiss.omp_set_num_threads(num_threads)

    logger.info("=" * 60)
    logger.info("Building shared embedding index: %s", index_name)
    logger.info("  Embedding model: %s", embedding_model)
    logger.info("  Embedding backend: %s", embedding_backend)
    logger.info("  Chunk size: %s, overlap: %s", chunk_size, chunk_overlap)
    logger.info("  Chunking strategy: %s", chunking_strategy)
    logger.info(
        "  Doc batch size: %s, embedding batch size: %s", doc_batch_size, embedding_batch_size
    )
    logger.info("  FAISS index type: %s, GPU: %s, threads: %s", index_type, use_gpu, num_threads)
    if index_type in ("ivf", "ivfpq"):
        logger.info("  IVF params: nlist=%s, nprobe=%s", nlist, nprobe)
    logger.info("=" * 60)

    # Initialize corpus
    corpus_metadata = {}
    if corpus_config.get("wikirank_top_k"):
        corpus_metadata["wikirank_top_k"] = corpus_config["wikirank_top_k"]
    if corpus_config.get("min_chars"):
        corpus_metadata["min_chars"] = corpus_config["min_chars"]
    if "streaming" in corpus_config:
        corpus_metadata["streaming"] = corpus_config["streaming"]
    if corpus_config.get("num_proc"):
        corpus_metadata["num_proc"] = corpus_config["num_proc"]

    corpus_cfg = CorpusConfig(
        name=f"wikipedia_{corpus_config.get('version', 'simple')}",
        source=corpus_config.get("source", "wikimedia/wikipedia"),
        version=corpus_config.get("version", "20231101.simple"),
        max_docs=corpus_config.get("max_docs"),
        metadata=corpus_metadata,
    )
    corpus = WikipediaCorpus(corpus_cfg)

    # Initialize encoder using factory (backend-agnostic)
    logger.info("Loading embedding model (%s)...", embedding_backend)
    encoder = create_embedder(
        model_name=embedding_model,
        backend=embedding_backend,
        gpu_memory_fraction=vllm_gpu_memory_fraction,
        enforce_eager=False,
    )
    embedding_dim = encoder.get_sentence_embedding_dimension()
    logger.info("  Embedder loaded (dim=%s)", embedding_dim)

    # Setup work directory for checkpointing
    work_dir = manager.indexes_dir / ".work" / index_name
    work_dir.mkdir(parents=True, exist_ok=True)
    index_checkpoint_path = work_dir / "index.faiss"

    # Initialize checkpoint manager and storage (for chunks only now)
    checkpoint_mgr = CheckpointManager(work_dir)
    storage = EmbeddingStorage(work_dir)

    # Check for existing checkpoint (resume support)
    checkpoint = checkpoint_mgr.load(EmbeddingCheckpoint)
    if checkpoint and index_checkpoint_path.exists():
        logger.info(
            "  Resuming from checkpoint: batch %s, %s docs",
            checkpoint.batch_num,
            checkpoint.total_docs,
        )
        logger.info("  Loading checkpointed index...")
        index = faiss.read_index(str(index_checkpoint_path))
        logger.info("  Loaded index with %s vectors", index.ntotal)
        start_batch = checkpoint.batch_num
        total_docs = checkpoint.total_docs
        total_chunks = checkpoint.total_chunks
    else:
        # Initialize fresh FAISS index
        if index_type == "flat":
            cpu_index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            cpu_index = faiss.IndexIVFFlat(
                quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
        elif index_type == "ivfpq":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            m = min(32, embedding_dim // 4)
            cpu_index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, 8)
        elif index_type == "hnsw":
            cpu_index = faiss.IndexHNSWFlat(embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
            cpu_index.hnsw.efConstruction = 200
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        index = cpu_index
        start_batch = 0
        total_docs = 0
        total_chunks = 0

    is_trained = index_type in ("flat", "hnsw") or index.is_trained

    # Chunking config
    chunk_config = ChunkConfig(
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunker = DocumentChunker(chunk_config)

    doc_batch = []
    batch_num = start_batch
    docs_to_skip = start_batch * doc_batch_size

    # ==========================================================================
    # INCREMENTAL BUILD: Chunk → Embed → Normalize → Add to FAISS
    # ==========================================================================
    logger.info("Building index incrementally (chunk -> embed -> normalize -> index)...")

    def process_batch(doc_batch: list, batch_num: int, index, is_final: bool = False):
        """Process a single batch: chunk, embed, normalize, add to index."""
        nonlocal total_docs, total_chunks, is_trained

        batch_size = len(doc_batch)
        suffix = " (final)" if is_final else ""
        logger.info("\n  [Batch %s] Processing %s documents%s...", batch_num, batch_size, suffix)

        # Chunking phase
        t_chunk = time.time()
        batch_chunks = []
        truncated = 0

        for doc in tqdm(doc_batch, desc="    Chunking", leave=False, ncols=80):
            text = doc.text
            if len(text) > MAX_DOC_CHARS:
                text = text[:MAX_DOC_CHARS]
                truncated += 1
                doc = Document(id=doc.id, text=text, metadata=doc.metadata)
            doc_chunks = list(chunker.strategy.chunk_document(doc))
            batch_chunks.extend(doc_chunks)

        chunk_elapsed = time.time() - t_chunk
        if truncated > 0:
            logger.info(
                "    %s chunks in %.1fs (truncated %s)", len(batch_chunks), chunk_elapsed, truncated
            )
        else:
            logger.info("    %s chunks in %.1fs", len(batch_chunks), chunk_elapsed)

        total_docs += batch_size
        total_chunks += len(batch_chunks)

        if batch_chunks:
            texts = [c.text for c in batch_chunks]
            logger.info("    Embedding %s chunks...", len(texts))

            # Embed
            embeddings = encoder.encode(
                texts, show_progress_bar=True, batch_size=embedding_batch_size
            )

            # Normalize in-place (guard against zero-norm from empty/whitespace chunks)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            np.divide(embeddings, norms, out=embeddings)

            # Train IVF if needed (first batch only)
            if not is_trained and index_type in ("ivf", "ivfpq"):
                logger.info("    Training IVF index on %s vectors...", len(embeddings))
                index.train(embeddings)
                is_trained = True

            # Add to index
            t_index = time.time()
            index.add(embeddings.astype(np.float32))
            index_elapsed = time.time() - t_index

            logger.info(
                "    Added to index in %.1fs (total: %s vectors)", index_elapsed, index.ntotal
            )

            # Save chunks to disk
            storage.append_chunks(batch_chunks)

            # Cleanup
            del embeddings, texts
            gc.collect()

        logger.info(
            "  Total: %s docs -> %s chunks (index: %s)", total_docs, total_chunks, index.ntotal
        )

        # Checkpoint every N batches
        if batch_num % CHECKPOINT_INTERVAL == 0:
            logger.info("    Saving checkpoint...")
            faiss.write_index(index, str(index_checkpoint_path))
            checkpoint_mgr.save(
                EmbeddingCheckpoint(
                    batch_num=batch_num,
                    total_docs=total_docs,
                    total_chunks=total_chunks,
                    embedding_sizes=[],  # Not used in incremental mode
                )
            )

    try:
        for doc in corpus.load():
            # Skip already processed docs when resuming
            if docs_to_skip > 0:
                docs_to_skip -= 1
                continue

            doc_batch.append(doc)

            if len(doc_batch) >= doc_batch_size:
                batch_num += 1
                process_batch(doc_batch, batch_num, index)
                doc_batch = []

        # Process remaining docs
        if doc_batch:
            batch_num += 1
            process_batch(doc_batch, batch_num, index, is_final=True)
            doc_batch = []

    except Exception as e:
        # Save checkpoint on error
        logger.error("\nError at batch %s: %s", batch_num, e)
        logger.info("    Saving checkpoint before exit...")
        faiss.write_index(index, str(index_checkpoint_path))
        checkpoint_mgr.save(
            EmbeddingCheckpoint(
                batch_num=batch_num - 1,  # Last successful batch
                total_docs=total_docs,
                total_chunks=total_chunks,
                embedding_sizes=[],
            )
        )
        storage.close()
        raise RuntimeError(f"Build failed at batch {batch_num}: {e}") from e

    logger.info(
        "\nBuild complete: %s docs -> %s chunks -> %s vectors",
        total_docs,
        total_chunks,
        index.ntotal,
    )

    # Set nprobe for IVF indexes
    if index_type in ("ivf", "ivfpq") and hasattr(index, "nprobe"):
        index.nprobe = nprobe
        logger.info("  Set nprobe=%s for IVF index", nprobe)

    # Save to final location
    artifact_path = manager.get_embedding_index_path(index_name)
    logger.info("\nSaving to %s...", artifact_path)

    # Save FAISS index
    faiss.write_index(index, str(artifact_path / "index.faiss"))

    # Load and save all chunks
    logger.info("  Saving documents...")
    all_chunks = storage.load_all_chunks()
    with open(artifact_path / "documents.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    logger.info("  Saved %s documents", len(all_chunks))

    # Cleanup work directory
    storage.cleanup()
    checkpoint_mgr.clear()
    if index_checkpoint_path.exists():
        index_checkpoint_path.unlink()

    # Try to remove work dir if empty
    try:
        work_dir.rmdir()
    except OSError:
        pass

    del all_chunks
    gc.collect()

    config = {
        "name": index_name,
        "type": "embedding",
        "embedding_model": embedding_model,
        "embedding_backend": embedding_backend,
        "index_type": index_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunking_strategy": chunking_strategy,
        "corpus_version": corpus_config.get("version"),
        "corpus_max_docs": corpus_config.get("max_docs"),
        "num_documents": total_chunks,
        "embedding_dim": embedding_dim,
        "use_gpu": use_gpu,
        "nlist": nlist,
        "nprobe": nprobe,
    }
    manager.save_json(config, artifact_path / "config.json")

    del index
    encoder.unload()
    gc.collect()

    logger.info("Saved to: %s", artifact_path)
    if use_gpu:
        logger.info("  Index will be loaded to GPU on next use (use_gpu=%s)", use_gpu)
    return str(artifact_path)
