# How the Optuna Study Works

This document explains the full lifecycle of a RAGiCamp Optuna study — from YAML config to completed experiment results. It's written assuming no prior knowledge of the codebase.

---

## 1. What Is a Study?

A **study** is a batch run of many RAG experiments. The goal is to find the best combination of models, retrievers, prompts, and retrieval strategies for a given set of QA benchmarks (NQ, TriviaQA, HotpotQA).

Each individual experiment answers: *"Given model X, retriever Y, top_k Z, prompt style P, etc. — what F1/EM score do we get?"*

Optuna orchestrates which combinations to try next, using Bayesian optimization (TPE) to focus on promising regions of the search space after learning from prior trials.

---

## 2. Launching a Study

```bash
uv run ragicamp run conf/study/smart_retrieval_slm.yaml
```

This loads the YAML config and calls `cli/study.py:run_study()`, which orchestrates everything:

```
YAML config file
    │
    ▼
run_study()                          # cli/study.py
    ├── 1. validate_config()         # Pydantic schema validation
    ├── 2. ensure_indexes_exist()    # Build FAISS + BM25 indexes if missing
    ├── 3. build_specs()             # Build baseline specs (direct LLM only)
    ├── 4. _run_spec_list()          # Run baseline experiments
    ├── 5. run_optuna_study()        # Run Optuna-driven RAG trials
    └── 6. Print summary
```

When a sampling mode is configured (e.g., `mode: tpe`), the baseline (direct LLM) experiments run as a fixed grid, while all RAG experiments are handled trial-by-trial by Optuna.

---

## 3. The YAML Config

The config defines the full search space. Here's the structure (using `smart_retrieval_slm.yaml`):

```yaml
name: smart_retrieval_slm
datasets: [nq, triviaqa, hotpotqa]
metrics: [f1, exact_match, bertscore, bleurt]
batch_size: 128

models: &slm_models          # 7 models across 3 tiers (tiny/small/medium)
  - vllm:Qwen/Qwen2.5-1.5B-Instruct
  - vllm:google/gemma-2-2b-it
  - vllm:meta-llama/Llama-3.2-3B-Instruct
  # ... (7 total — Phi-3-mini removed due to 4096 ctx limit)

direct:                       # Direct LLM baselines (no retrieval)
  enabled: true
  models: *slm_models         # Same models as RAG
  prompts: [concise, fewshot_3]

rag:
  enabled: true
  models: *slm_models

  sampling:                   # Optuna configuration
    mode: tpe                 # Bayesian optimization
    n_experiments: 1000       # Target trial count
    optimize_metric: f1       # What to maximize
    seed: 42
    trial_timeout: 1800       # Kill after 30 min (was 3600; configs needing >30min are degenerate)
    fixed_dims: [dataset, model]   # Explored uniformly, not optimized
    agent_types: [fixed_rag, iterative_rag, self_rag]
    agent_params:
      iterative_rag:
        max_iterations: [1, 2]  # Dropped 3 (main cause of timeouts)
        stop_on_sufficient: [true]

  retrievers:                 # 7 active retrievers
    - {type: dense,  name: dense_bge_large_512, ...}
    - {type: dense,  name: dense_bge_m3_512, ...}
    - {type: dense,  name: dense_gte_qwen2_1.5b_512, ...}
    - {type: hybrid, name: hybrid_bge_large_bm25_07, alpha: 0.7, ...}
    - {type: hybrid, name: hybrid_gte_qwen2_bm25_07, alpha: 0.7, ...}
    - {type: hierarchical, name: hier_bge_large_2048p_448c, ...}

  top_k_values: [3, 5, 10, 15, 20]
  query_transform: [none, hyde, multiquery]
  rrf_k_values: [20, 40, 60]        # RRF fusion constant (hybrid only)
  fetch_k_multiplier: 5             # fetch_k = top_k * 5 when reranking
  prompts: [concise, concise_strict, concise_json, extractive,
            extractive_quoted, cot, cot_final, fewshot_3, fewshot_1]

  reranker:
    configs:
      - {enabled: false, name: none}
      - {enabled: true, name: bge}
      - {enabled: true, name: bge-v2}
```

### Theoretical Search Space Size

```
7 models × 7 retrievers × 5 top_k × 9 prompts × 3 query_transforms
× 3 rerankers × 3 datasets × 3 agent_types × ... = ~200,000+ combinations
```

Exhaustive grid search is infeasible. Optuna TPE explores this space intelligently.

---

## 4. Index Building (Step 2)

Before any experiments run, all required indexes must exist on disk.

### Dense Indexes (FAISS)
For each dense/hybrid retriever config, a FAISS HNSW index is built:
1. Load Wikipedia corpus (200K articles)
2. Chunk into ~512-token passages
3. Embed all chunks using the retriever's embedding model (vLLM backend)
4. Build FAISS HNSW index with inner-product similarity
5. Save to `artifacts/indexes/{index_name}/`

### Sparse Indexes (BM25/TF-IDF)
For hybrid retrievers, a sparse index is built from the same documents:
1. Load documents from the dense index
2. Tokenize and build BM25Okapi (or TF-IDF vectorizer)
3. Save to `artifacts/sparse_indexes/{sparse_name}/`

### Retriever Configs
For each retriever, a config file is saved to `artifacts/retrievers/{name}/config.json` with the embedding model, backend, index paths, alpha (for hybrid), etc. This is what `Experiment.from_spec()` reads at runtime to know how to load the index.

Indexes are built once and reused across all experiments. Multiple retrievers can share the same dense index (e.g., `hybrid_bge_large_bm25_07` reuses the `dense_bge_large_512` index + adds a BM25 sparse index).

---

## 5. Baseline Experiments (Step 4)

Direct LLM baselines (no retrieval) run as a fixed grid:

```
7 models × 2 prompts × 3 datasets = 42 baseline experiments
```

These provide the "no retrieval" reference score. Each runs in a subprocess (see Section 8).

---

## 6. Optuna Trial Loop (Step 5)

This is the core of the study. Handled by `optimization/optuna_search.py:run_optuna_study()`.

### 6.1 Setup

```python
study = optuna.create_study(
    study_name="smart_retrieval_slm_tpe",
    storage="sqlite:///outputs/smart_retrieval_slm/optuna_study.db",
    sampler=TPESampler(seed=42),
    direction="maximize",        # Maximize F1
    load_if_exists=True,         # Resume if DB exists
)
```

The Optuna study is persisted to SQLite, so you can kill and restart the process and it resumes transparently.

### 6.2 Search Space

Extracted from the YAML config:

| Dimension | Values | Count |
|-----------|--------|-------|
| `model` | 7 vLLM model specs | 7 |
| `retriever` | 7 retriever names | 7 |
| `top_k` | [3, 5, 10, 15, 20] | 5 |
| `prompt` | 9 prompt styles | 9 |
| `query_transform` | [none, hyde, multiquery] | 3 |
| `reranker` | [none, bge, bge-v2] | 3 |
| `dataset` | [nq, triviaqa, hotpotqa] | 3 |
| `agent_type` | [fixed_rag, iterative_rag, self_rag] | 3 |

Plus **conditional dimensions** (only suggested when relevant):

**Hybrid-only** (when retriever type is `hybrid`):
- `rrf_k`: [20, 40, 60] — RRF fusion constant
- `alpha`: [0.3, 0.5, 0.7, 0.9] — Dense/sparse blend weight

**Agent-specific** (conditional on `agent_type`):
- `max_iterations`: [1, 2] (only when `agent_type == iterative_rag`)
- `stop_on_sufficient`: [true] (only when `agent_type == iterative_rag`)
- `retrieval_threshold`: [0.3, 0.5, 0.7] (only when `agent_type == self_rag`)
- `verify_answer`: [true, false] (only when `agent_type == self_rag`)
- `fallback_to_direct`: [true] (only when `agent_type == self_rag`)

### 6.3 Warm-Start from Disk

Before running any new trials, Optuna scans the output directory for completed experiments:

```python
seeded = _seed_from_existing(study, search_space, output_dir, ...)
```

For each experiment with a `metadata.json` and `results.json`:
1. Read the experiment's parameters (model, retriever, top_k, etc.)
2. Validate they fall within the current search space
3. Extract the metric value (F1)
4. Register as a completed trial in the Optuna study

This means TPE has historical data to learn from immediately, rather than starting blind. It also prevents Optuna from re-running experiments that already exist on disk.

### 6.4 Stratified Round-Robin (Fixed Dims)

By default, `dataset` and `model` are treated as "benchmark axes" — they're explored uniformly via round-robin, not optimized by TPE.

**Why?** If TPE optimized dataset choice, it would bias towards easy datasets (TriviaQA >> HotpotQA in F1). If it optimized model choice, it would only try the 9B model. We want equal coverage across all datasets and models.

**How it works:**

```
fixed_dims: [dataset, model]
→ 3 datasets × 7 models = 21 fixed combinations
→ 1000 trials / 21 combos ≈ 47 trials per (dataset, model) pair
```

Each trial epoch:
1. Generate all 21 (dataset, model) combinations
2. Shuffle them randomly
3. For each combo, create a `PartialFixedSampler` that locks those dims
4. TPE suggests the remaining dims (retriever, top_k, prompt, qt, reranker, agent_type)

This ensures every model gets tested on every dataset with roughly equal trial budget.

### 6.5 Single Trial Lifecycle

For each trial, here's what happens:

```
┌─────────────────────────────────────────────────────────────┐
│  Trial #N                                                    │
│                                                              │
│  1. TPE suggests parameters:                                 │
│     model=gemma-2-2b, retriever=hybrid_bge_large_bm25_07,   │
│     top_k=10, prompt=concise, qt=hyde, reranker=bge,         │
│     agent_type=iterative_rag, max_iterations=2, rrf_k=40     │
│                                                              │
│  2. Context feasibility check:                               │
│     estimated_input = (512/4) × 10 + 400 = 1680 tokens      │
│     model_ctx = 8192 - 256 = 7936 → OK                      │
│                                                              │
│  3. Duplicate check:                                         │
│     Does outputs/{exp_name}/results.json exist? → Skip       │
│                                                              │
│  4. Build ExperimentSpec from suggested params                │
│                                                              │
│  5. Run experiment in subprocess:                            │
│     run_spec(spec, use_subprocess=True)                      │
│     → spawns scripts/experiments/run_single_experiment.py    │
│     → timeout: 3600s                                         │
│                                                              │
│  6. Extract metric from results:                             │
│     F1 = 0.4523                                              │
│                                                              │
│  7. Report to Optuna:                                        │
│     return 0.4523  (or raise TrialPruned if failed/timeout)  │
└─────────────────────────────────────────────────────────────┘
```

### 6.6 Context Feasibility Pruning

Before running an experiment, we estimate if the prompt will fit in the model's context window:

```
estimated_input = (chunk_chars / chars_per_token) × top_k + prompt_overhead
max_input = model_context_length - generation_buffer
```

If `estimated_input > max_input`, the trial is pruned immediately. This prevents wasting GPU time on combinations that always crash with "prompt length exceeds model length".

Example: Phi-3-mini (4096 ctx) + top_k=20 + fewshot_3 (1500 overhead) = 4060 tokens > 3840 available → pruned.

### 6.7 How TPE Learns

After each trial, Optuna's Tree-structured Parzen Estimator updates its probabilistic model:

1. **Completed trials** are split into "good" (above median F1) and "bad" (below median)
2. TPE models P(params | good) and P(params | bad) as kernel density estimates
3. Next trial: suggest params that maximize P(params | good) / P(params | bad)
4. This naturally exploits high-performing regions while still exploring

**What TPE optimizes** (not in `fixed_dims`):
- retriever, top_k, prompt, query_transform, reranker, agent_type, rrf_k
- Conditional: max_iterations, stop_on_sufficient (for iterative_rag)

**What TPE does NOT optimize** (in `fixed_dims`):
- model, dataset (explored uniformly via round-robin)

---

## 7. ExperimentSpec: The Single Source of Truth

Every experiment is fully defined by an `ExperimentSpec` frozen dataclass (`spec/experiment.py`):

```python
@dataclass(frozen=True)
class ExperimentSpec:
    name: str                           # "rag_vllm_gemma22bit_dense_bge_large_512_k5_concise_nq"
    exp_type: "direct" | "rag"
    model: str                          # "vllm:google/gemma-2-2b-it"
    dataset: str                        # "nq"
    prompt: str                         # "concise"
    retriever: str | None               # "dense_bge_large_512"
    embedding_index: str | None         # "en_bge_large_en_v1.5_c512_o50"
    sparse_index: str | None            # "en_bge_large_en_v1.5_c512_o50_sparse_bm25"
    top_k: int                          # 5
    fetch_k: int | None                 # 25 (top_k × 5 when reranking)
    query_transform: str | None         # "hyde", "multiquery", or None
    reranker: str | None                # "bge" or None
    reranker_model: str | None          # Full HF model path
    rrf_k: int | None                   # RRF fusion constant (hybrid only)
    batch_size: int                     # 128
    metrics: list[str]                  # ["f1", "exact_match", "bertscore", "bleurt"]
    agent_type: str | None              # "iterative_rag", "self_rag", or None (= fixed_rag)
    agent_params: tuple[tuple[str, Any], ...]  # (("max_iterations", 2), ("stop_on_sufficient", True))
```

The spec is:
- **Serializable**: `spec.to_dict()` → JSON, passed to subprocess
- **Frozen**: Immutable after creation (prevents accidental mutation)
- **Self-contained**: Has everything needed to reproduce the experiment

---

## 8. Experiment Execution (Subprocess Isolation)

Each experiment runs in its own subprocess for **CUDA crash isolation**. If one experiment causes a GPU segfault, only that subprocess dies — the parent (Optuna) continues.

### Flow

```
Parent (Optuna)                          Subprocess
─────────────────                        ──────────────────────
run_spec_subprocess()
  ├── clear GPU memory
  ├── serialize spec to JSON
  ├── spawn subprocess ──────────────→   run_single_experiment.py
  │     stdout piped + tee'd               ├── ExperimentSpec.from_dict()
  │     to experiment.log                  ├── run_spec(use_subprocess=False)
  │                                        │   ├── Phase dispatch:
  │                                        │   │   COMPUTING_METRICS → run_metrics_only()
  │                                        │   │   Other → run_generation()
  │                                        │   │
  │                                        │   └── run_generation():
  │                                        │       ├── Experiment.from_spec()
  │                                        │       │   ├── Create providers (lazy)
  │                                        │       │   ├── Load retriever config
  │                                        │       │   ├── Wrap index in LazySearchBackend
  │                                        │       │   ├── Create agent via AgentFactory
  │                                        │       │   └── Create metrics
  │                                        │       │
  │                                        │       └── exp.run()
  │                                        │           ├── INIT phase
  │                                        │           ├── GENERATING phase
  │                                        │           ├── GENERATED phase
  │                                        │           ├── COMPUTING_METRICS phase
  │                                        │           └── COMPLETE phase
  │                                        │
  ├── wait(timeout=1800s) ◄────────────   exit(0)  or  exit(1)
  └── return "ran" or "failed"
```

### Timeout Handling

If the subprocess doesn't complete within `trial_timeout` seconds (default 1800 = 30 min), the parent kills it and the Optuna trial is pruned.

---

## 9. Inside an Experiment: The 5 Phases

Once `Experiment.run()` is called, execution proceeds through 5 phases:

### Phase 1: INIT
- Create output directory: `outputs/study_name/experiment_name/`
- Save `metadata.json` (full spec fields)
- Export `questions.json` (dataset questions + expected answers)
- Save `state.json` with `phase: "init"`

### Phase 2: GENERATING
This is the heavy phase. The agent processes all questions in batches.

**For a RAG agent (e.g., FixedRAGAgent):**

```
Step 1: Load embedder onto GPU
Step 2: Encode all queries → embeddings (batch)
Step 3: Search index → top-k documents per query
Step 4: Unload embedder, free GPU

Step 5: Load reranker onto GPU (if configured)
Step 6: Rerank fetch_k docs → keep top_k per query
Step 7: Unload reranker, free GPU

Step 8: Load generator (LLM) onto GPU
Step 9: Build prompts (query + retrieved context)
Step 10: Batch generate answers
Step 11: Unload generator, free GPU
```

**Key design: One model on GPU at a time.** The embedder, reranker, and generator each get exclusive GPU access for maximum throughput. GPU memory is cleared between each model.

**Predictions saved incrementally** to `predictions.json` with checkpointing every 50 examples. If the process crashes, it can resume from the last checkpoint.

### Phase 3: GENERATED
Marks all predictions complete. Clears GPU memory.

### Phase 4: COMPUTING_METRICS
Compute all configured metrics on the predictions:
- **F1**: Token-level F1 overlap
- **Exact Match**: Binary exact string match
- **BERTScore**: Neural semantic similarity (uses GPU)
- **BLEURT**: Learned quality metric (uses GPU)

Each metric is computed independently. After each metric completes, `state.json` is updated with `metrics_computed: ["f1", "exact_match", ...]`. This means if BERTScore crashes, F1 and EM are still saved.

### Phase 5: COMPLETE
Save final `results.json` with aggregate metrics and metadata.

### Phase-Aware Resume

If an experiment crashes mid-execution, the state machine enables smart resume:
- Crashed in GENERATING → resume from last checkpoint (skip already-answered questions)
- Crashed in COMPUTING_METRICS → skip model loading entirely, just recompute missing metrics
- Already COMPLETE → skip entirely

This is why the state tracks both the current phase and which metrics have been computed.

---

## 10. Agent Types

Three RAG agent strategies are available:

### fixed_rag (Default)
Standard single-pass RAG:
1. Embed queries
2. Search index → top-k docs
3. (Optional) Rerank
4. Build prompt with context
5. Generate answer

### iterative_rag
Multi-iteration refinement:
1. **Iteration 0**: Same as fixed_rag
2. **Check sufficiency**: Does the LLM think the answer is sufficient?
3. **Iteration 1+**: Refine the query based on what's missing, re-retrieve, re-generate
4. Repeat until sufficient or `max_iterations` reached

Key: Query transforms (HyDE, multiquery) and reranking are applied on **every iteration**, not just the first. This was a recent fix — previously they were silently disabled on iterations > 0, causing TPE to model false cross-effects between parameters.

### self_rag
Adaptive retrieval with self-assessment:
1. First attempt: try to answer without retrieval
2. Self-assess: does the model think it needs more information?
3. If needed: retrieve, generate with context, verify answer

---

## 11. Retrieval Pipeline Details

### Query Transforms
Applied before embedding/search to improve retrieval quality:

- **HyDE** (Hypothetical Document Embeddings): Ask the LLM to generate a hypothetical answer, then use that as the search query. Better semantic match with answer-like passages.
- **MultiQuery**: Generate 3 variations of the original question, search with all of them, merge results.

### Hybrid Search (Dense + Sparse)
Combines FAISS vector search with BM25/TF-IDF keyword search using Reciprocal Rank Fusion:

```
score(doc) = alpha / (rrf_k + rank_dense) + (1 - alpha) / (rrf_k + rank_sparse)
```

- `alpha`: Weight for dense vs sparse (0.7 = dense-heavy)
- `rrf_k`: Fusion constant (controls how much weight lower-ranked docs get)
  - `rrf_k=20`: More aggressive — big gap between ranks 1 and 5
  - `rrf_k=60`: Flatter — ranks are more equal (traditional default)

The `rrf_k` parameter is now exposed to Optuna as a tunable dimension (values: [20, 40, 60]).

### Hierarchical Retrieval
Uses two chunk sizes: small child chunks (448 tokens) for precise search, then returns the larger parent chunks (2048 tokens) for more context. Good for questions that need broader context.

### Reranking
After initial retrieval, a cross-encoder reranker re-scores the top `fetch_k` documents and returns only `top_k`. The reranker sees both the query and full document text, so it can judge relevance more accurately than embedding similarity alone.

`fetch_k = top_k × fetch_k_multiplier` (default 5). So if `top_k=5`, we retrieve 25 docs and rerank down to 5.

### Retrieval Cache
When `query_transform=none`, retrieval results (query → top-k doc IDs + scores) are cached in SQLite. This means:
- Same query + same retriever → instant cache hit (no embedding, no search)
- Different models/prompts with the same retriever/top_k still benefit

**Note (v0.4.1 fix):** The retrieval cache is now enabled for all query transforms. Previously it was disabled for HyDE/multiquery. Transformed queries produce different cache keys so hit rates are lower, but the cache still works. See `docs/AGENT_PERFORMANCE_ANALYSIS.md` Section 4.

### Lazy Index Loading
FAISS index deserialization can take 2-3 minutes for large indexes. The `LazySearchBackend` proxy defers this until the first `batch_search()` call. If the retrieval cache has 100% hits, the index never loads at all.

---

## 12. Output Structure

```
outputs/smart_retrieval_slm/
├── study.log                    # Full terminal output
├── study_meta.json              # Study-level metadata
├── optuna_study.db              # Optuna SQLite (trial history, TPE state)
│
├── direct_gemma22bI_nq_a1b2c3d4/        # Baseline experiment
│   ├── state.json               # Phase tracking
│   ├── metadata.json            # Full spec as JSON (source of truth for params)
│   ├── questions.json           # Dataset questions + expected answers
│   ├── predictions.json         # Per-item predictions + metrics
│   ├── results.json             # Aggregate metrics
│   └── experiment.log           # Subprocess stdout
│
├── rag_gemma22bI_nq_e5f6g7h8/           # RAG experiment (fixed_rag)
│   ├── state.json
│   ├── metadata.json
│   ├── questions.json
│   ├── predictions.json         # Includes per-item metrics + step traces
│   ├── results.json             # {"metrics": {"f1": 0.45, "exact_match": 0.32, ...}}
│   └── experiment.log
│
├── iterative_gemma22bI_nq_i9j0k1l2/     # iterative_rag agent
│   └── ...
│
└── _quarantined/                # Experiments affected by known bugs (if quarantined)
    └── ...
```

### Experiment Naming Convention

Names use a short hash-based format: `{prefix}_{model_short}_{dataset}_{hash8}`

- **prefix**: `direct`, `rag` (fixed_rag), `iterative` (iterative_rag), `self` (self_rag)
- **model_short**: Last path segment, collapsed (e.g., `Qwen2.5-7B-Instruct` -> `Qwen257BI`)
- **dataset**: As-is (e.g., `nq`, `triviaqa`)
- **hash8**: First 8 chars of SHA-256 of all behavior-affecting params (model, retriever, top_k, prompt, query_transform, reranker, rrf_k, alpha, agent_type, agent_params)

The hash ensures uniqueness while keeping names short (~30 chars vs ~100+ chars). Full params are always available in `metadata.json`.

To migrate existing long-named experiments: `python scripts/migrate_naming.py --study outputs/your_study`

---

## 13. Resume and Fault Tolerance

The study is designed for long-running GPU workloads (days/weeks). Multiple layers of resume:

| Level | Mechanism | What it saves |
|-------|-----------|---------------|
| **Optuna** | SQLite DB (`optuna_study.db`) | All trial params + results |
| **Experiment state** | `state.json` per experiment | Current phase + metrics computed |
| **Predictions** | Checkpointing every 50 examples | Partial predictions |
| **GPU isolation** | Subprocess per experiment | Parent survives CUDA crashes |

**To resume a study:** Just re-run the same command. Optuna loads from SQLite, skips completed trials, and continues from where it left off.

---

## 14. Current Design Decisions (for review)

### Things that work well
1. **Subprocess isolation**: CUDA crashes don't kill the study
2. **Phase-aware resume**: Can restart from any point
3. **Stratified round-robin**: Equal trial budget per (dataset, model)
4. **Warm-start seeding**: TPE learns from all prior experiments immediately
5. **Context feasibility pruning**: Saves GPU time on impossible combos
6. **Lazy index loading**: Skips multi-minute deserialization when cache hits

### Things to be aware of
1. **Retrieval cache and query_transform**: As of v0.4.1, the retrieval cache works with all query transforms. HyDE/multiquery experiments have lower cache hit rates (transformed queries differ) but are no longer excluded from caching entirely. HyDE experiments may still be slower due to the extra generation step.

2. **Embedding cache is always active**: Query embeddings are cached in SQLite keyed by `(model_name, sha256(text))`. This cache works across experiments and studies — any experiment using the same embedding model benefits.

3. **Retrieval cache key includes alpha**: When `spec.alpha` is set, the cache key includes it (e.g., `hybrid_bge_large_bm25_a0.70`) to avoid returning stale results for a different blend weight.

---

## 15. Data Flow Diagram

```
                    ┌─────────────┐
                    │  YAML Config │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  run_study() │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
    ┌─────────▼──────┐    │    ┌───────▼─────────┐
    │ ensure_indexes  │    │    │  run_optuna_     │
    │ (FAISS + BM25)  │    │    │  study()         │
    └────────────────┘    │    └───────┬─────────┘
                          │            │
                ┌─────────▼──┐   ┌─────▼──────────────┐
                │  Baselines  │   │  Trial Loop         │
                │ (grid run)  │   │  ┌─────────────┐   │
                └─────────────┘   │  │ TPE suggest  │   │
                                  │  └──────┬──────┘   │
                                  │  ┌──────▼──────┐   │
                                  │  │ Build spec   │   │
                                  │  └──────┬──────┘   │
                                  │  ┌──────▼──────┐   │
                                  │  │ Feasibility  │   │
                                  │  │ check        │   │
                                  │  └──────┬──────┘   │
                                  │  ┌──────▼──────┐   │
                                  │  │ Subprocess   │──────→ GPU
                                  │  │ execution    │   │
                                  │  └──────┬──────┘   │
                                  │  ┌──────▼──────┐   │
                                  │  │ Extract F1   │   │
                                  │  │ → Optuna     │   │
                                  │  └─────────────┘   │
                                  └────────────────────┘
                                           │
                                    ┌──────▼──────┐
                                    │ optuna_     │
                                    │ study.db    │
                                    └─────────────┘
```
