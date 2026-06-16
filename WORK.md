# WORK — Public RAGiCamp Release

See also: [`PLAN.md`](PLAN.md) for the current end-to-end smoke implementation plan.

Branch/repo: `cassuci/ragicamp-pub` public release workspace

Workspace:

```text
ragicamp-public/
```

## Goal

Prepare a clean public GitHub repository containing the RAGiCamp framework code needed to regenerate SLM/RAG study results and run aggregate analysis, without raw predictions, large artifacts, paper manuscript sources/PDFs, private cloud workflows, or RL/orchestrator material.

Target remote:

```text
https://github.com/cassuci/ragicamp-pub
```

## Current Status

- [x] Created clean-history public workspace from pre-RL base commit `fbce71c`.
- [x] Removed paper manuscript directories/PDFs.
- [x] Removed raw outputs and large artifacts.
- [x] Removed private cloud backup CLI commands and implementation.
- [x] Removed obvious RL/orchestrator code/docs/configs from selected base.
- [x] Added placeholder `outputs/README.md` and `artifacts/README.md`.
- [x] Rewrote public `README.md`.
- [x] Added `VALIDATION.md`.
- [x] Added `PROVENANCE.md` documenting representativeness vs. SLM study run dates.
- [x] Added `conf/study/public_smoke.yaml`.
- [x] Unit tests passed before final doc/script inspection: `720 passed, 3 warnings`.
- [x] Public smoke dry-run passed.
- [x] `smart_retrieval_slm.yaml` dry-run passed.
- [x] Index-building smoke passed after fixing public-copy bugs in `cmd_index` and `IndexBuilder` chunker handling.
- [x] Initial public commit pushed to `https://github.com/cassuci/ragicamp-pub`.

## Active Work

Prove the public repo can rerun the SLM study pipeline in a small, end-to-end way, not only dry-run/spec generation.

## Open Tasks

### P0 — End-to-end SLM/RAG pipeline smoke

Plan: [`PLAN.md`](PLAN.md).

- [x] Add or adapt a tiny RAG smoke config that uses a locally buildable small index (`conf/study/public_e2e_smoke.yaml`).
- [x] Ensure `ragicamp run <smoke-config> --limit 1` executes generation + retrieval + metrics.
- [x] Prefer fake/lightweight local providers if available; deterministic `mock` generator/embedder providers added.
- [x] Confirm DirectLLM path executes.
- [x] Confirm FixedRAG path executes against the tiny index.
- [x] Confirm output files are written under `outputs/` and can be deleted before commit.

### P0 — Re-run validation after smoke fixes

- [x] `uv run --with pytest pytest tests/ -x -q` (`720 passed, 3 warnings`).
- [x] `uv run ragicamp --help`.
- [x] `uv run ragicamp run conf/study/public_smoke.yaml --dry-run --limit 1`.
- [x] `uv run ragicamp run conf/study/smart_retrieval_slm.yaml --dry-run --limit 1`.
- [x] `uv run ragicamp index --corpus simple --embedding mock --max-docs 5`.
- [x] `uv run ragicamp run conf/study/public_e2e_smoke.yaml --limit 1 --force`.
- [x] `uv run ragicamp run conf/study/public_e2e_smoke.yaml --limit 1 --force --sample 2 --sample-mode tpe --sample-seed 42` (Optuna/TPE path).

### P1 — Final public hygiene

- [x] Remove generated `.venv/`, `outputs/*`, `artifacts/cache`, `artifacts/indexes`, `__pycache__` before commit.
- [x] Run large-file scan.
- [x] Run private/internal/RL scan.
- [x] Update `VALIDATION.md` with final smoke results.
- [x] Commit and push final public repo changes (`d40c5ce`).

### P2 — Optional polish

- [ ] Add `LICENSE`.
- [ ] Add `CITATION.cff`.
- [ ] Decide whether notebooks should be kept in full or reduced to analysis utility scripts.

## Provenance conclusion

See [`PROVENANCE.md`](PROVENANCE.md). The public repo is representative of the SLM study framework lineage, but not a byte-for-byte run-date archive: preserved output metadata has timestamps but no git SHA, and the main run spanned 2026-02-10 to 2026-02-21. The selected public base (`fbce71c`, 2026-03-02) is pre-RL and has no core framework or `smart_retrieval_slm.yaml` changes relative to the final-run-adjacent commit (`0fc2341`, 2026-02-20).

## Notes

- The full `conf/study/smart_retrieval_slm.yaml` is expected to require GPU resources, model downloads, and rebuilt retrieval indexes. The public repo should honestly claim that the full study config parses/plans, while a smaller smoke config proves the end-to-end pipeline.
- Do not commit raw predictions or generated indexes.
- Do not modify the private source repo while working here.
