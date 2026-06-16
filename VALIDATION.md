# Public Release Validation

This file records the validation performed while preparing the clean public repository copy.

## Unit tests

Command:

```bash
uv run --with pytest pytest tests/ -x -q
```

Result:

```text
720 passed, 3 warnings
```

## CLI smoke

Command:

```bash
uv run ragicamp --help
```

Result: CLI imports and displays available commands.

## Study dry-run smoke

Command:

```bash
uv run ragicamp run conf/study/public_smoke.yaml --dry-run --limit 1
```

Result:

```text
Total experiments: 1
[DRY RUN] Would run the above experiments
```

## End-to-end DirectLLM + FixedRAG smoke

Command:

```bash
uv run ragicamp run conf/study/public_e2e_smoke.yaml --limit 1 --force
```

Result:

```text
Study Complete: public_e2e_smoke
Baselines — Completed: 2, Failed: 0, Skipped: 0
```

This smoke uses deterministic `mock` providers to avoid GPU/API/model-download
requirements while exercising the real study runner, dataset loading, index
building, retriever-config writing, DirectLLM generation, FixedRAG retrieval,
metric computation, and output writing. It writes standard experiment files under
`outputs/public_e2e_smoke/` during validation; these generated files are not
committed.

## End-to-end Optuna/TPE smoke

Command:

```bash
uv run ragicamp run conf/study/public_e2e_smoke.yaml --limit 1 --force --sample 2 --sample-mode tpe --sample-seed 42
```

Result:

```text
Study Complete: public_e2e_smoke
Baselines — Completed: 1, Failed: 0, Skipped: 0
Optuna — Trials: 2, Pruned: 1
Best trial: 0.0000 (trial #0)
```

This goes one step beyond the fixed-grid smoke by exercising Optuna/TPE trial
creation, RAG trial execution, SQLite study persistence, duplicate-trial pruning,
and best-trial reporting. It also caught and fixed a public-copy search-space bug
where Optuna did not infer retriever choices from `rag.retrievers` when
`rag.retriever_names` was omitted.

## Paper-study config dry-run

Command:

```bash
uv run ragicamp run conf/study/smart_retrieval_slm.yaml --dry-run --limit 1
```

Result: builds the expected direct-baseline spec list and TPE trial plan without loading models or building indexes.

## Index-building smoke

Command:

```bash
uv run ragicamp index --corpus simple --embedding mock --max-docs 5
```

Result: downloads a tiny Simple Wikipedia sample, chunks it, embeds it with the deterministic mock embedder, and builds a local FAISS index successfully.

Generated local artifacts from this smoke are not committed.

## Notes

A full study rerun still requires model downloads, retrieval indexes, GPU resources, and generated outputs. Raw predictions and large artifacts are intentionally excluded from the public repository.
