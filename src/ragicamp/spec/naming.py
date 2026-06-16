"""Experiment naming conventions.

Provides functions to generate canonical experiment names from parameters.
Consistent naming is crucial for experiment identification and reproducibility.

Naming scheme: {type}_{model_short}_{dataset}_{hash8}

The hash is a deterministic 8-char hex digest of all behavior-affecting params,
ensuring short, unique, and collision-resistant names.
"""

import hashlib
from typing import Any


def _model_short(model: str) -> str:
    """Extract short model identifier.

    Examples:
        >>> _model_short("vllm:Qwen/Qwen2.5-7B-Instruct")
        'Qwen257BI'
        >>> _model_short("hf:google/gemma-2-2b-it")
        'gemma22bI'
    """
    # Take the last segment after "/" or ":"
    name = model.split("/")[-1] if "/" in model else model.split(":")[-1]
    # Remove common suffixes, collapse punctuation
    name = name.replace("-Instruct", "I").replace("-instruct", "I")
    name = name.replace("-it", "I")
    name = name.replace(".", "")
    name = name.replace("-", "")
    return name[:12]  # Cap length


def _spec_hash(spec_fields: dict[str, Any]) -> str:
    """Deterministic 8-char hash from sorted spec fields."""
    canon = "|".join(f"{k}={v}" for k, v in sorted(spec_fields.items()))
    return hashlib.sha256(canon.encode()).hexdigest()[:8]


def name_direct(model: str, prompt: str, dataset: str) -> str:
    """Generate experiment name for direct (non-RAG) experiments.

    Format: direct_{model_short}_{dataset}_{hash8}

    Args:
        model: Model specification (e.g., 'hf:google/gemma-2-2b-it')
        prompt: Prompt style name
        dataset: Dataset name

    Returns:
        Canonical experiment name

    Example:
        >>> name_direct("vllm:google/gemma-2-2b-it", "concise", "nq")
        'direct_gemma22bI_nq_...'
    """
    ms = _model_short(model)
    h = _spec_hash({"model": model, "prompt": prompt, "dataset": dataset})
    return f"direct_{ms}_{dataset}_{h}"


def name_rag(
    model: str,
    prompt: str,
    dataset: str,
    retriever: str,
    top_k: int,
    query_transform: str = "none",
    reranker: str = "none",
    rrf_k: int | None = None,
    alpha: float | None = None,
    agent_type: str | None = None,
    agent_params: Any | None = None,
) -> str:
    """Generate experiment name for RAG experiments.

    Format: {prefix}_{model_short}_{dataset}_{hash8}

    The prefix encodes the agent type:
    - 'rag' for fixed_rag (default)
    - 'iterative' for iterative_rag
    - 'self' for self_rag

    Args:
        model: Model specification
        prompt: Prompt style name
        dataset: Dataset name
        retriever: Retriever name
        top_k: Number of documents to retrieve
        query_transform: Query transformation type
        reranker: Reranker type
        rrf_k: RRF fusion constant for hybrid retrievers
        alpha: Dense/sparse blend for hybrid retrievers
        agent_type: Agent type (fixed_rag, iterative_rag, self_rag)
        agent_params: Agent-specific parameters (tuple of tuples or dict)

    Returns:
        Canonical experiment name

    Example:
        >>> name_rag("vllm:Qwen/Qwen2.5-7B-Instruct", "concise", "nq",
        ...          "dense_bge_large_512", 5)
        'rag_Qwen257BI_nq_...'
    """
    ms = _model_short(model)

    # Determine prefix from agent type
    prefix = "rag"
    if agent_type and agent_type != "fixed_rag":
        prefix = agent_type.replace("_rag", "")  # "iterative", "self"

    # Normalize agent_params to a stable string for hashing
    if agent_params is not None:
        if isinstance(agent_params, dict):
            ap_str = str(sorted(agent_params.items()))
        elif isinstance(agent_params, tuple):
            ap_str = str(sorted(agent_params))
        else:
            ap_str = str(agent_params)
    else:
        ap_str = "None"

    fields = {
        "model": model,
        "retriever": retriever,
        "top_k": top_k,
        "prompt": prompt,
        "qt": query_transform or "none",
        "rr": reranker or "none",
        "rrf_k": rrf_k,
        "alpha": alpha,
        "agent_type": agent_type,
        "agent_params": ap_str,
    }
    h = _spec_hash(fields)
    return f"{prefix}_{ms}_{dataset}_{h}"
