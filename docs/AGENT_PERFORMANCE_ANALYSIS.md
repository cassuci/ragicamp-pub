# Agent Performance Analysis: Provider Load/Unload Bottleneck

> **Created**: 2026-02-11
> **Status**: Implemented (provider ref-counting + agent session wrapping)

## Problem

Optuna trials with complex agent configurations (`self_rag + hyde + reranker + hybrid`)
hit the 3600s timeout. Root cause: **repeated model load/unload cycles**. Each
`with provider.load()` call fully loads/unloads the model from GPU. A single vLLM
generator load takes 30-120s, and worst-case trials do 3-5 generator loads.

## Cost Breakdown: Model Loads Per Agent Type

### Before Optimization

With HyDE query transform + cross-encoder reranker enabled:

| Agent | Generator Loads | Embedder Loads | Reranker Loads | Total |
|-------|----------------|----------------|----------------|-------|
| `fixed_rag` | 2 (HyDE + generate) | 1 | 1 | 4 |
| `self_rag` | 3 (HyDE + assess + generate) | 1 | 1 | 5 |
| `iterative_rag` (2 iters) | 4 (HyDE x2 + sufficiency/refine + generate) | 2 | 2 | 8 |

**Time cost**: 30-120s per load x extra loads = **2-8 minutes wasted per trial**.

### Why So Many Loads?

1. **HyDE transformer** calls `generator_provider.load()` inside `batch_transform()`,
   separate from the agent's own generator load for answer generation.
2. **Reranker** calls `reranker_provider.load()` in `_apply_reranking()`, once per
   iteration in iterative_rag.
3. **Iterative_rag** repeats embed+search+rerank per iteration, plus sufficiency
   checking and refinement each requiring a generator load.
4. **Self_rag** loads the generator three times: assess, HyDE (via query transform),
   and generate+verify.

### After Optimization

| Agent | Generator Loads | Embedder Loads | Reranker Loads | Total |
|-------|----------------|----------------|----------------|-------|
| `fixed_rag` | 1 | 1 | 1 | 3 |
| `self_rag` | 1 | 1 | 1 | 3 |
| `iterative_rag` (2 iters) | 1 | 1 | 1 | 3 |

**Expected saving**: 2-8 minutes per trial (30-120s per avoided load).

## Remaining Secondary Bottlenecks

After eliminating model load/unload cycles, the remaining bottlenecks are CPU-bound
Python loops (seconds, not minutes):

| Bottleneck | File | Impact | Notes |
|---|---|---|---|
| HybridSearcher sequential RRF fusion | `retrievers/hybrid.py:143-183` | Medium | Per-query Python loop for score computation and sorting |
| BM25 sequential search | `indexes/sparse.py:195` | Medium | `rank_bm25` library has no batch API |
| Cache `execute()` vs `executemany()` | `cache/*.py` | Low | Only affects cache-miss writes |
| Query transform result merging | `agents/base.py:299-313` | Low | Per-query dedup+sort, small data |

These were tracked in the engineering backlog (now resolved and archived).

## Solution: Three-Part Optimization

### 1. Provider Ref-Counting (High Impact)

Added `_refcount: int` to `GeneratorProvider`, `EmbedderProvider`, `RerankerProvider`.
On first `load()`, actually loads the model. On subsequent nested `load()` calls,
increments refcount and yields the existing model. On context exit, decrements;
only unloads at refcount 0.

**Files**: `models/providers/generator.py`, `embedder.py`, `reranker.py`

Fully backward-compatible: existing `with provider.load() as gen:` calls work
identically. Nested calls reuse the loaded model.

### 2. Agent Session Wrapping (High Impact)

Each agent's `run()` method now opens provider sessions at the top level:

- **`self_rag`**: Wraps entire run with generator session (assess + HyDE + generate +
  verify all reuse the same loaded generator). Reranker session if configured.
- **`iterative_rag`**: Wraps with embedder + generator + reranker sessions across all
  iterations and final generation. All inner `provider.load()` calls hit the refcount
  path.
- **`fixed_rag`**: Wraps with generator session (for HyDE) and reranker session if
  configured.

**Files**: `agents/self_rag.py`, `iterative_rag.py`, `fixed_rag.py`

### 3. HyDE Iteration Guard (Medium Impact)

In `iterative_rag`, query transform is now only applied on iteration 0. On later
iterations, queries have been LLM-refined and are already document-like, making HyDE
redundant. This saves 1 generator load per additional iteration.

**File**: `agents/iterative_rag.py:293`

**Why iteration 0 only?** HyDE generates a hypothetical answer to convert a short
question into a document-like text for dense retrieval. After iteration 0, the
`_phase_evaluate_and_refine` method produces LLM-refined queries that are already
descriptive and document-like (e.g., "Find information about X that explains Y and
covers Z"). Applying HyDE to these refined queries would generate a hypothetical
answer *of a hypothetical answer* — adding noise rather than signal, while costing
an extra generator call per iteration. The same logic applies to MultiQuery: refined
queries are already specific enough that generating reformulations adds little value.

**Enforcement**: A `logger.debug()` message is emitted when the transform is skipped
on iteration > 0, making the behavior visible in logs. The guard is at
`_phase_embed_and_search()` — any future code that bypasses this method would need
to implement its own guard.

### 4. Retrieval Cache for Transformed Queries (Medium Impact)

Previously, retrieval cache was disabled when `query_transform != "none"`. But cache
keys include the query text, so transformed queries get unique cache entries naturally.
Removed the conditional — cache is now always enabled.

**File**: `factory/agents.py:212-215`

## How HyDE/MultiQuery Multiply Work

Query transformers expand each query into multiple search queries:

1. **HyDE**: Generates a hypothetical answer via LLM, searches with both original
   query and hypothetical answer (2 queries per original).
2. **MultiQuery**: Generates 3 reformulations via LLM, searches with all of them
   plus the original (4 queries per original).

The LLM call inside the transformer is the expensive part — it requires loading the
generator. With ref-counting, the outer session keeps the generator loaded, and the
transformer's inner `provider.load()` call is essentially free (just increments refcount).

## Retrieval Cache Interaction

The retrieval cache (`RetrievalStore`) keys on `(retriever_name, query_text_hash, top_k)`.
When query transforms are enabled:

- **Original queries** always cache-hit on repeat runs.
- **Transformed queries** (HyDE hypothetical answers) are deterministic per original query
  (temperature=0.7 but same seed), so they also cache-hit on repeat runs.
- **Cache key uniqueness** is preserved: `hash("What is DNA?")` differs from
  `hash("DNA is a molecule that carries genetic information...")`.

Previously the cache was disabled for transformed queries under the assumption that
transformed query texts would never repeat. In practice, Optuna trials often reuse the
same (retriever, query_transform, top_k) across different LLM model/prompt combinations,
making cache hits common.
