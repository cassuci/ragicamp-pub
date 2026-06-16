"""Hierarchical index builder.

Builds hierarchical indexes with parent-child chunk relationships.

Uses incremental FAISS building for memory efficiency:
- Each batch: chunk → embed children → normalize → add to FAISS
- Checkpoints index every N batches for crash recovery
- Never loads all embeddings into RAM at once

Supports checkpointing for resume after crashes.
"""

import gc
import os
import pickle
import time
from typing import Any

from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.utils.artifacts import get_artifact_manager

from .checkpoint import CheckpointManager, HierarchicalCheckpoint
from .storage import EmbeddingStorage

logger = get_logger(__name__)

# Checkpoint every batch - hierarchical batches can take hours
CHECKPOINT_INTERVAL = 1


def _create_faiss_index(index_type: str, dim: int):
    """Create a FAISS index based on the configured type.

    Args:
        index_type: One of ``"flat"``, ``"hnsw"``, ``"ivf"``.
        dim: Embedding dimension.

    Returns:
        A FAISS index instance.
    """
    import faiss

    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
        logger.info("Created HNSW index (dim=%d, M=32, efConstruction=200)", dim)
        return index
    elif index_type == "ivf":
        quantizer = faiss.IndexFlatIP(dim)
        # nlist will be set properly after training; start with a placeholder
        index = faiss.IndexIVFFlat(quantizer, dim, 4096, faiss.METRIC_INNER_PRODUCT)
        logger.info("Created IVF index (dim=%d, nlist=4096)", dim)
        return index
    else:
        logger.info("Created flat (brute-force) index (dim=%d)", dim)
        return faiss.IndexFlatIP(dim)


def build_hierarchical_index(
    retriever_config: dict[str, Any],
    corpus_config: dict[str, Any],
    doc_batch_size: int = 1000,
    embedding_batch_size: int = 64,
    embedding_backend: str = "vllm",
    vllm_gpu_memory_fraction: float = Defaults.VLLM_GPU_MEMORY_FRACTION,
) -> str:
    """Build a hierarchical index with batched processing.

    Hierarchical indexes have unique parent/child chunking, so they
    can't share indexes with other retrievers.

    Uses incremental FAISS building:
    - Each batch: chunk → embed children → normalize → add to FAISS
    - Checkpoints every 5 batches for crash recovery
    - Memory efficient: only one batch in RAM at a time

    Supports resume from checkpoint if process is interrupted.

    Args:
        retriever_config: Retriever configuration with chunk sizes
        corpus_config: Corpus configuration
        doc_batch_size: Documents to process per batch (default 1000)
        embedding_batch_size: Texts to embed per GPU batch (default 64)
        embedding_backend: 'vllm' or 'sentence_transformers'
        vllm_gpu_memory_fraction: GPU memory fraction for vLLM (default 0.9)

    Returns:
        Retriever name
    """
    import faiss
    import numpy as np

    from ragicamp.core.types import Document
    from ragicamp.corpus import CorpusConfig, WikipediaCorpus
    from ragicamp.models.embedder import create_embedder
    from ragicamp.rag.chunking.hierarchical import HierarchicalChunker

    manager = get_artifact_manager()
    # Use embedding_index for the index name (consistent with dense indexes)
    # Fall back to name if embedding_index not specified
    index_name = retriever_config.get("embedding_index", retriever_config["name"])
    retriever_name = retriever_config["name"]
    embedding_model = retriever_config.get("embedding_model", Defaults.EMBEDDING_MODEL)

    # Get embedding backend from retriever config, fall back to parameter
    backend = retriever_config.get("embedding_backend", embedding_backend)

    parent_chunk_size = retriever_config.get("parent_chunk_size", 1024)
    child_chunk_size = retriever_config.get("child_chunk_size", 256)
    parent_overlap = retriever_config.get("parent_overlap", 100)
    child_overlap = retriever_config.get("child_overlap", 50)
    index_type = retriever_config.get("index_type", "flat")

    # Use all CPU cores for FAISS
    num_threads = os.cpu_count() or 8
    faiss.omp_set_num_threads(num_threads)

    logger.info("=" * 60)
    logger.info("Building hierarchical index: %s", index_name)
    logger.info("  Embedding model: %s", embedding_model)
    logger.info("  Embedding backend: %s", backend)
    logger.info("  Parent chunks: %s", parent_chunk_size)
    logger.info("  Child chunks: %s", child_chunk_size)
    logger.info(
        "  Doc batch size: %s, embedding batch size: %s", doc_batch_size, embedding_batch_size
    )
    logger.info("  FAISS index type: %s", index_type)
    logger.info("  FAISS threads: %s", num_threads)
    logger.info("=" * 60)

    # Build metadata dict for WikiRank filter and other options
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

    # Initialize chunker
    chunker = HierarchicalChunker(
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_overlap,
    )

    # Initialize encoder using factory (backend-agnostic)
    logger.info("Loading embedding model (%s)...", backend)
    encoder = create_embedder(
        model_name=embedding_model,
        backend=backend,
        gpu_memory_fraction=vllm_gpu_memory_fraction,
        enforce_eager=False,
    )
    embedding_dim = encoder.get_sentence_embedding_dimension()
    logger.info("  Embedder loaded (dim=%s)", embedding_dim)

    # Setup work directory for checkpointing and temp storage
    work_dir = manager.indexes_dir / ".work" / index_name
    work_dir.mkdir(parents=True, exist_ok=True)
    index_checkpoint_path = work_dir / "child_index.faiss"

    # Initialize storage and checkpoint manager
    storage = EmbeddingStorage(work_dir)
    checkpoint_mgr = CheckpointManager(work_dir)

    # Mapping file for child->parent relationships
    mapping_path = work_dir / "child_to_parent.pkl"

    # Check for existing checkpoint (resume support)
    checkpoint = checkpoint_mgr.load(HierarchicalCheckpoint)
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
        parent_count = checkpoint.total_parent_chunks
        child_count = checkpoint.total_child_chunks
        mapping_file = open(mapping_path, "ab")
    else:
        # Create fresh index based on configured type
        index = _create_faiss_index(index_type, embedding_dim)
        start_batch = 0
        total_docs = 0
        parent_count = 0
        child_count = 0
        mapping_file = open(mapping_path, "wb")

    doc_batch = []
    batch_num = start_batch
    docs_to_skip = start_batch * doc_batch_size

    # ==========================================================================
    # INCREMENTAL BUILD: Chunk → Embed → Normalize → Add to FAISS
    # ==========================================================================
    logger.info("Building index incrementally (chunk -> embed -> normalize -> index)...")

    try:
        for doc in corpus.load():
            # Skip already processed docs when resuming
            if docs_to_skip > 0:
                docs_to_skip -= 1
                continue

            doc_with_id = Document(
                id=f"wiki_{total_docs}",
                text=doc.text,
                metadata=doc.metadata,
            )
            doc_batch.append(doc_with_id)
            total_docs += 1

            if len(doc_batch) >= doc_batch_size:
                batch_num += 1
                batch_size = len(doc_batch)
                logger.info("\n  [Batch %s] Processing %s documents...", batch_num, batch_size)

                # Chunking phase
                t_chunk = time.time()
                parent_chunks, child_chunks, batch_child_to_parent = chunker.chunk_documents(
                    iter(doc_batch)
                )
                chunk_elapsed = time.time() - t_chunk
                logger.info(
                    "    Chunking (hierarchical) -> %s parents, %s children (%.1fs)",
                    len(parent_chunks),
                    len(child_chunks),
                    chunk_elapsed,
                )

                # Save mapping
                pickle.dump(batch_child_to_parent, mapping_file)
                mapping_file.flush()

                # Save parent chunks (no embedding needed for parents)
                storage.append_chunks(parent_chunks, key="parent")
                parent_count += len(parent_chunks)

                if child_chunks:
                    child_texts = [c.text for c in child_chunks]
                    logger.info("    Embedding %s child chunks...", len(child_texts))

                    # Embed
                    embeddings = encoder.encode(
                        child_texts, show_progress_bar=True, batch_size=embedding_batch_size
                    )

                    # Normalize in-place (guard against zero-norm from empty/whitespace chunks)
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-12)
                    np.divide(embeddings, norms, out=embeddings)

                    # Add to index
                    t_index = time.time()
                    index.add(embeddings.astype(np.float32))
                    index_elapsed = time.time() - t_index
                    logger.info(
                        "    Added to index in %.1fs (total: %s vectors)",
                        index_elapsed,
                        index.ntotal,
                    )

                    # Save child chunks
                    storage.append_chunks(child_chunks, key="child")
                    child_count += len(child_chunks)

                    del child_texts, embeddings
                    gc.collect()

                logger.info(
                    "  Total: %s docs -> %s parents, %s children",
                    total_docs,
                    parent_count,
                    child_count,
                )
                doc_batch = []
                del parent_chunks, child_chunks, batch_child_to_parent
                gc.collect()

                # Checkpoint every N batches
                if batch_num % CHECKPOINT_INTERVAL == 0:
                    logger.info("    Saving checkpoint...")
                    faiss.write_index(index, str(index_checkpoint_path))
                    checkpoint_mgr.save(
                        HierarchicalCheckpoint(
                            batch_num=batch_num,
                            total_docs=total_docs,
                            total_parent_chunks=parent_count,
                            total_child_chunks=child_count,
                            parent_embedding_sizes=[],
                            child_embedding_sizes=[],
                        )
                    )

        # Process remaining docs
        if doc_batch:
            batch_num += 1
            batch_size = len(doc_batch)
            logger.info("\n  [Batch %s] Processing %s documents (final)...", batch_num, batch_size)

            t_chunk = time.time()
            parent_chunks, child_chunks, batch_child_to_parent = chunker.chunk_documents(
                iter(doc_batch)
            )
            chunk_elapsed = time.time() - t_chunk
            logger.info(
                "    Chunking (hierarchical) -> %s parents, %s children (%.1fs)",
                len(parent_chunks),
                len(child_chunks),
                chunk_elapsed,
            )

            pickle.dump(batch_child_to_parent, mapping_file)
            mapping_file.flush()

            storage.append_chunks(parent_chunks, key="parent")
            parent_count += len(parent_chunks)

            if child_chunks:
                child_texts = [c.text for c in child_chunks]
                logger.info("    Embedding %s child chunks...", len(child_texts))
                embeddings = encoder.encode(
                    child_texts, show_progress_bar=True, batch_size=embedding_batch_size
                )

                # Normalize in-place
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                np.divide(embeddings, norms, out=embeddings)

                # Add to index
                t_index = time.time()
                index.add(embeddings.astype(np.float32))
                logger.info("    Added to index in %.1fs", time.time() - t_index)

                storage.append_chunks(child_chunks, key="child")
                child_count += len(child_chunks)

                del child_texts, embeddings

            logger.info(
                "  Total: %s docs -> %s parents, %s children", total_docs, parent_count, child_count
            )
            del doc_batch, parent_chunks, child_chunks, batch_child_to_parent
            gc.collect()

    except Exception as e:
        # Save checkpoint on error
        logger.error("\nError at batch %s: %s", batch_num, e)
        logger.info("    Saving checkpoint before exit...")
        faiss.write_index(index, str(index_checkpoint_path))
        checkpoint_mgr.save(
            HierarchicalCheckpoint(
                batch_num=batch_num - 1,
                total_docs=total_docs,
                total_parent_chunks=parent_count,
                total_child_chunks=child_count,
                parent_embedding_sizes=[],
                child_embedding_sizes=[],
            )
        )
        mapping_file.close()
        storage.close()
        raise RuntimeError(f"Build failed at batch {batch_num}: {e}") from e

    mapping_file.close()
    logger.info(
        "\nBuild complete: %s docs -> %s parents, %s children -> %s vectors",
        total_docs,
        parent_count,
        child_count,
        index.ntotal,
    )

    # ==========================================================================
    # Save to final location
    # ==========================================================================
    logger.info("\nSaving to final location...")
    index_path = manager.get_embedding_index_path(index_name)
    index_path.mkdir(parents=True, exist_ok=True)

    # Load and save all documents
    logger.info("  Saving parent documents...")
    all_parent_docs = storage.load_all_chunks(key="parent")
    with open(index_path / "parent_docs.pkl", "wb") as f:
        pickle.dump(all_parent_docs, f)
    logger.info("    %s parent docs", len(all_parent_docs))

    logger.info("  Saving child documents...")
    all_child_docs = storage.load_all_chunks(key="child")
    with open(index_path / "child_docs.pkl", "wb") as f:
        pickle.dump(all_child_docs, f)
    logger.info("    %s child docs", len(all_child_docs))

    # Build parent ID -> index mapping
    parent_id_to_idx = {}
    for idx, doc in enumerate(all_parent_docs):
        parent_id_to_idx[doc.id] = idx

    # Load and merge child_to_parent mapping
    logger.info("  Saving child-to-parent mapping...")
    child_to_parent = {}
    with open(mapping_path, "rb") as f:
        while True:
            try:
                batch_mapping = pickle.load(f)
                child_to_parent.update(batch_mapping)
            except EOFError:
                break

    with open(index_path / "child_to_parent.pkl", "wb") as f:
        pickle.dump(child_to_parent, f)
    logger.info("    %s mappings", len(child_to_parent))

    # Save FAISS index
    faiss.write_index(index, str(index_path / "child_index.faiss"))

    # Save index config
    index_config = {
        "name": index_name,
        "type": "hierarchical",
        "embedding_model": embedding_model,
        "embedding_backend": backend,
        "index_type": index_type,
        "parent_chunk_size": parent_chunk_size,
        "child_chunk_size": child_chunk_size,
        "parent_overlap": parent_overlap,
        "child_overlap": child_overlap,
        "num_parents": parent_count,
        "num_children": child_count,
        "embedding_dim": embedding_dim,
    }
    manager.save_json(index_config, index_path / "config.json")

    # Create retriever config that references the index
    retriever_path = manager.get_retriever_path(retriever_name)
    retriever_config_data = {
        "name": retriever_name,
        "type": "hierarchical",
        "embedding_index": index_name,  # Reference the index by its name
        "embedding_model": embedding_model,
        "embedding_backend": backend,
        "parent_chunk_size": parent_chunk_size,
        "child_chunk_size": child_chunk_size,
        "num_parents": parent_count,
        "num_children": child_count,
    }
    manager.save_json(retriever_config_data, retriever_path / "config.json")

    # Cleanup work directory
    storage.cleanup()
    checkpoint_mgr.clear()
    if mapping_path.exists():
        mapping_path.unlink()
    if index_checkpoint_path.exists():
        index_checkpoint_path.unlink()

    # Try to remove work dir if empty
    try:
        work_dir.rmdir()
    except OSError:
        pass

    # Clean up
    del all_parent_docs, all_child_docs, child_to_parent, parent_id_to_idx
    del index
    encoder.unload()
    del chunker
    gc.collect()

    logger.info("Saved hierarchical index: %s", index_name)
    return index_name
