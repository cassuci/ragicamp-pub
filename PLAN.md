# PLAN — End-to-End Public Pipeline Smoke

This plan complements [`WORK.md`](WORK.md). `WORK.md` tracks task status; this file explains the implementation plan for proving that the public repository can rerun the study pipeline on a small scale.

## Goal

Add and validate a tiny end-to-end smoke path that exercises the same public framework layers used by the SLM study:

1. study YAML parsing;
2. index creation / retriever config creation;
3. DirectLLM experiment execution;
4. FixedRAG experiment execution with retrieval;
5. metrics computation;
6. output writing under `outputs/`.

The smoke must be cheap, deterministic, and not require GPU/API credentials or large model downloads.

## Approach

Add lightweight `mock` providers for public validation:

- `mock:` generator backend returns deterministic short answers.
- `mock` embedder backend returns deterministic hash-based vectors.

Then add:

```text
conf/study/public_e2e_smoke.yaml
```

The config should include:

- one dataset (`nq`), run with `--limit 1`;
- one direct experiment using `mock:answer`;
- one fixed RAG experiment using a tiny dense retriever;
- metrics `f1` and `exact_match` only;
- tiny Simple Wikipedia corpus config for index building.

## Validation Commands

Run from the public repo root:

```bash
uv run --with pytest pytest tests/ -x -q
uv run ragicamp --help
uv run ragicamp run conf/study/public_smoke.yaml --dry-run --limit 1
uv run ragicamp run conf/study/smart_retrieval_slm.yaml --dry-run --limit 1
uv run ragicamp run conf/study/public_e2e_smoke.yaml --limit 1
```

Optional index-only check:

```bash
uv run ragicamp index --corpus simple --embedding mock --max-docs 5
```

## Success Criteria

- Unit tests pass.
- Public dry-runs pass.
- `public_e2e_smoke.yaml` writes complete output directories under `outputs/public_e2e_smoke/`.
- The smoke builds or reuses a tiny local index/retriever config.
- Generated smoke outputs/artifacts are removed before commit.
- [`VALIDATION.md`](VALIDATION.md) is updated with final command results.

## Non-goals

- Do not run the full `smart_retrieval_slm.yaml` study.
- Do not commit generated indexes, caches, outputs, or raw predictions.
- Do not add paper manuscript sources/PDFs to the public repo.
