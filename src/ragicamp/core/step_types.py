"""Standardized step type constants for agent pipeline tracing.

All agents should use these constants instead of ad-hoc strings
to ensure consistent step identification across the codebase.
"""

QUERY_TRANSFORM = "query_transform"
BATCH_ENCODE = "batch_encode"
BATCH_SEARCH = "batch_search"
RERANK = "rerank"
GENERATE = "generate"
BATCH_GENERATE = "batch_generate"
CACHE_HIT = "cache_hit"

# Iterative RAG specific
BATCH_SUFFICIENCY = "batch_sufficiency"
EVALUATE_SUFFICIENCY = "evaluate_sufficiency"
REFINE_QUERY = "refine_query"
BATCH_REFINE = "batch_refine"

# Self RAG specific
BATCH_ASSESS = "batch_assess"
ASSESS_RETRIEVAL = "assess_retrieval"
BATCH_VERIFY = "batch_verify"
VERIFY = "verify"
FALLBACK_GENERATE = "fallback_generate"
BATCH_FALLBACK_GENERATE = "batch_fallback_generate"
