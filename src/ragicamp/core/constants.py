"""Constants and enums for RAGiCamp.

Centralizes all magic strings and constants used throughout the codebase.
"""

from enum import Enum


class AgentType(str, Enum):
    """Agent types supported by RAGiCamp."""

    DIRECT_LLM = "direct_llm"
    FIXED_RAG = "fixed_rag"
    ITERATIVE_RAG = "iterative_rag"
    SELF_RAG = "self_rag"

    @classmethod
    def rag_types(cls) -> list["AgentType"]:
        """Get agent types that use retrieval."""
        return [cls.FIXED_RAG, cls.ITERATIVE_RAG, cls.SELF_RAG]

    @classmethod
    def requires_retriever(cls, agent_type: str) -> bool:
        """Check if agent type requires a retriever."""
        return agent_type in [t.value for t in cls.rag_types()]


class MetricType(str, Enum):
    """Metric types supported by RAGiCamp."""

    # Standard metrics
    EXACT_MATCH = "exact_match"
    F1 = "f1"

    # Semantic metrics
    BERTSCORE = "bertscore"
    BLEURT = "bleurt"

    # Context-aware RAG metrics
    FAITHFULNESS = "faithfulness"
    HALLUCINATION = "hallucination"
    CONTEXT_RECALL = "context_recall"
    ANSWER_IN_CONTEXT = "answer_in_context"

    # LLM-based
    LLM_JUDGE = "llm_judge"
    LLM_JUDGE_QA = "llm_judge_qa"

    @classmethod
    def standard_metrics(cls) -> list["MetricType"]:
        """Fast, deterministic metrics."""
        return [cls.EXACT_MATCH, cls.F1]

    @classmethod
    def semantic_metrics(cls) -> list["MetricType"]:
        """Neural-based semantic metrics."""
        return [cls.BERTSCORE, cls.BLEURT]

    @classmethod
    def context_metrics(cls) -> list["MetricType"]:
        """Metrics that require retrieved context."""
        return [
            cls.FAITHFULNESS,
            cls.HALLUCINATION,
            cls.CONTEXT_RECALL,
            cls.ANSWER_IN_CONTEXT,
        ]

    @classmethod
    def llm_metrics(cls) -> list["MetricType"]:
        """LLM-based metrics."""
        return [cls.LLM_JUDGE, cls.LLM_JUDGE_QA]

    @classmethod
    def requires_context(cls, metric_type: str) -> bool:
        """Check if metric requires retrieved context."""
        return metric_type in {m.value for m in cls.context_metrics()}


# === Reranker Model Registry ===
# Single source of truth for reranker short names → HuggingFace model paths.
# Referenced by both RerankerProvider and CrossEncoderReranker.
RERANKER_MODELS: dict[str, str] = {
    "bge": "BAAI/bge-reranker-large",
    "bge-base": "BAAI/bge-reranker-base",
    "bge-v2": "BAAI/bge-reranker-v2-m3",
    "ms-marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ms-marco-large": "cross-encoder/ms-marco-MiniLM-L-12-v2",
}


# === Error Markers ===

# Prefix used by model backends when an API call fails.
# Metrics should detect and exclude these from scoring.
ERROR_PREDICTION_PREFIX = "[ERROR:"


def is_error_prediction(text: str) -> bool:
    """Check if a prediction is an error marker from a failed API call."""
    return text.startswith(ERROR_PREDICTION_PREFIX)


# === Default Values ===


class Defaults:
    """Default configuration values."""

    # Model
    MAX_TOKENS = 256
    TEMPERATURE = 0.7
    TOP_P = 1.0
    MAX_INPUT_LENGTH = 4096  # Max tokenizer input length for HF models

    # Context truncation (characters) for agent prompts
    MAX_CONTEXT_CHARS = 2000
    MAX_VERIFICATION_CONTEXT_CHARS = 3000

    # Retrieval
    TOP_K = 5
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_BACKEND = "sentence_transformers"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    HYBRID_ALPHA = 0.7

    # Evaluation & Execution
    BATCH_SIZE = 8
    MIN_BATCH_SIZE = 1  # Minimum batch size for auto-reduction
    CHECKPOINT_INTERVAL = 50  # Save checkpoint every N items
    EXECUTOR_BATCH_SIZE = 32  # Default for ResilientExecutor

    # Paths
    DATA_DIR = "data"
    OUTPUT_DIR = "outputs"
    ARTIFACTS_DIR = "artifacts"
    CACHE_DIR = "data/datasets"

    # GPU Memory Fractions
    #
    # Agents manage their own model loading strategy:
    # - Simple RAG: batch_retrieve (full GPU) → unload → batch_generate (full GPU)
    # - Interleaved RAG: both models loaded with reduced fractions
    #
    # These are defaults - agents can override based on their strategy.
    VLLM_GPU_MEMORY_FRACTION = 0.9  # General default for vLLM GPU memory
    VLLM_GPU_MEMORY_FRACTION_FULL = 0.95  # When model has exclusive GPU access
    VLLM_GPU_MEMORY_FRACTION_SHARED = 0.45  # When sharing GPU with another model
    VLLM_EMBEDDER_GPU_MEMORY_FRACTION_SHARED = 0.25  # Embedder when sharing

    # FAISS GPU Configuration
    # NOTE: GPU FAISS disabled by default - B200 (Blackwell) not yet supported by faiss-gpu-cu12.
    # Enable once faiss adds Blackwell kernels, or build from source.
    FAISS_USE_GPU = False  # Disable GPU FAISS until Blackwell support added
    FAISS_GPU_MEMORY_FRACTION = 0.35  # Fraction of GPU memory for FAISS when sharing
    FAISS_GPU_TEMP_MEMORY_MB = 512  # Temporary memory for FAISS GPU operations (MB)
    FAISS_INDEX_TYPE = "hnsw"  # Default: flat, ivf, hnsw (hnsw is fastest for CPU)
    FAISS_IVF_NLIST = 4096  # Number of clusters for IVF indexes
    FAISS_IVF_NPROBE = 128  # Number of clusters to search (higher = better recall)
    FAISS_HNSW_M = 32  # HNSW connectivity parameter
    FAISS_HNSW_EF_CONSTRUCTION = 200  # HNSW build-time search depth
    FAISS_HNSW_EF_SEARCH = 128  # HNSW query-time search depth
    FAISS_CPU_THREADS = 0  # 0 = auto-detect (use all available cores)

    # Embedding Model Optimization
    # These settings optimize sentence-transformers inference similar to how vLLM
    # optimizes causal LM inference.
    EMBEDDING_BATCH_SIZE = 256  # Batch size for encoding (tune based on GPU memory)
    EMBEDDING_USE_FP16 = True  # Use FP16 for faster encoding (minimal quality loss)
    EMBEDDING_NORMALIZE = True  # L2 normalize embeddings (required for cosine similarity)
    EMBEDDING_SHOW_PROGRESS = True  # Show progress bar during encoding
    EMBEDDING_USE_FLASH_ATTENTION = True  # Use Flash Attention 2 if available
    EMBEDDING_USE_TORCH_COMPILE = True  # Apply torch.compile() for speedup
    # For production, consider using infinity-emb or HuggingFace TEI server
    # which provide vLLM-style continuous batching for embeddings
