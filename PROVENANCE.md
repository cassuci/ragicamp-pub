# Provenance — Public RAGiCamp Release vs. SLM Study Runs

This public repository is a curated clean-history release of the framework used for the SLM/RAG sensitivity study. It is intended to be **representative and runnable**, not a byte-for-byte forensic export of the private repository at every moment the study was running.

## What we can confirm

The preserved SLM study outputs in the private research workspace record experiment timestamps but not git SHAs. Their `metadata.json` files contain fields such as experiment name, model, dataset, retriever, metrics, duration, and timestamp; they do **not** include `git_commit` / `git_sha`.

Observed output timestamp ranges:

```text
outputs/smart_retrieval_slm_v0: 2026-02-02T15:45:36 → 2026-02-03T00:42:35
outputs/smart_retrieval_slm_v1: 2026-02-03T23:41:55 → 2026-02-10T15:45:29
outputs/smart_retrieval_slm:    2026-02-10T17:33:13 → 2026-02-21T01:14:44
```

The final/main run therefore completed around **2026-02-21**.

The latest private commit before the final output window ended was:

```text
0fc2341 2026-02-20 22:35:39 +0000 docs: add comprehensive analysis summary across all notebooks
```

The clean public release was built from the pre-RL commit:

```text
fbce71c 2026-03-02 00:00:05 +0000 docs: overhaul documentation for research accuracy
```

The first private commit introducing the later RL/orchestrator code was:

```text
88654c7 2026-03-02 00:49:04 +0000 feat: add RL orchestrator for adaptive RAG (Phase 0 + Phase 1)
```

Thus the public base is **pre-RL**, but slightly after the SLM study run finished.

## Why this base was chosen

The public base `fbce71c` was chosen because it is the last clean pre-RL framework point before the later orchestration research was integrated. It avoids publishing unrelated RL/orchestrator code while preserving the SLM/RAG framework and analysis stack.

A comparison between the final-run-adjacent commit (`0fc2341`) and the public base (`fbce71c`) showed no changes to the core framework source or the main SLM study config. The differences were limited to follow-up analysis/docs and one additional focused-study config:

```text
conf/study/focused_investigation.yaml
notebooks/* analysis updates
selected docs updates
```

The main study config was unchanged over that interval:

```text
conf/study/smart_retrieval_slm.yaml
```

## Public-release adjustments

The public release is not a raw archive. It intentionally removes or omits:

- private git history;
- paper manuscript PDFs/sources;
- raw predictions;
- generated outputs;
- retrieval indexes and embedding caches;
- private cloud backup/sync workflows;
- later RL/orchestrator material.

It also adds small public validation helpers that were not part of the original study run:

- deterministic `mock` generator/embedder providers;
- `conf/study/public_smoke.yaml`;
- `conf/study/public_e2e_smoke.yaml`;
- `WORK.md`, `PLAN.md`, and `VALIDATION.md`.

These additions are for public smoke testing only. They do not change the meaning of the original SLM study config.

## Validation in the public release

The public repository has been validated with:

```bash
uv run --with pytest pytest tests/ -x -q
uv run ragicamp --help
uv run ragicamp run conf/study/public_smoke.yaml --dry-run --limit 1
uv run ragicamp run conf/study/smart_retrieval_slm.yaml --dry-run --limit 1
uv run ragicamp run conf/study/public_e2e_smoke.yaml --limit 1 --force
uv run ragicamp run conf/study/public_e2e_smoke.yaml --limit 1 --force --sample 2 --sample-mode tpe --sample-seed 42
```

See [`VALIDATION.md`](VALIDATION.md) for details.

## Bottom line

The public repository is representative of the framework state used for the SLM/RAG study in the following sense:

- it is based on the same pre-RL framework lineage;
- it preserves the main SLM study configuration;
- the core framework source did not change between the final-run-adjacent commit and the selected public base;
- it has been smoke-tested through direct, fixed-RAG, index-building, metrics, output writing, and Optuna/TPE paths.

It is **not** a byte-for-byte reproduction of the private repository on a single run date, because the original run spanned multiple days/commits and the preserved output metadata does not record git SHAs.
