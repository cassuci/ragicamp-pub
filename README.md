# RAGiCamp

RAGiCamp is a Python framework for controlled Retrieval-Augmented Generation (RAG) experiments. It supports configurable datasets, retrievers, query transforms, rerankers, agents, models, metrics, and analysis workflows.

This public release focuses on the framework used for the small-language-model RAG sensitivity study. Raw experiment outputs and per-question predictions are intentionally not included.

## What is included

- RAG experiment runner and configuration builder
- Direct LLM, fixed RAG, iterative RAG, and Self-RAG agents
- Dataset loaders for NQ, TriviaQA, HotpotQA, TechQA, and PubMedQA
- Corpus loading and chunking utilities
- Dense, sparse, hybrid, and hierarchical retrieval support
- Query transforms such as HyDE and multi-query reformulation
- Reranker integration
- Metrics and comparison utilities
- Public smoke configuration for validating installation

## What is not included

- Raw per-question predictions
- Large generated outputs
- FAISS/BM25 index artifacts
- Embedding caches
- Private cloud backup/sync workflows
- Paper manuscript sources/PDFs

Placeholders in `outputs/` and `artifacts/` explain where generated local files will appear.

## Installation

RAGiCamp uses Python 3.10+ and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Check the CLI:

```bash
uv run ragicamp --help
```

## Validate the repository

Run the unit test suite:

```bash
uv run --with pytest pytest tests/ -x -q
```

Validate config parsing/spec generation with the public smoke config:

```bash
uv run ragicamp run conf/study/public_smoke.yaml --dry-run --limit 1
```

Run a tiny end-to-end DirectLLM + FixedRAG pipeline with deterministic mock providers:

```bash
uv run ragicamp run conf/study/public_e2e_smoke.yaml --limit 1 --force
```

This builds a tiny local index, writes a retriever config, runs direct and RAG experiments, computes metrics, and writes standard output files under `outputs/public_e2e_smoke/`.

## Running studies

Example dry run:

```bash
uv run ragicamp run conf/study/smart_retrieval_slm.yaml --dry-run
```

A full study requires model downloads, retrieval indexes, and appropriate GPU resources. Generated outputs are written under `outputs/`; generated indexes and caches are written under `artifacts/`.

## Building indexes

Use the CLI or scripts under `scripts/` to build retrieval indexes. For a no-download model smoke:

```bash
uv run ragicamp index --corpus simple --embedding mock --max-docs 5
```

For a small real embedding model:

```bash
uv run ragicamp index --corpus simple --embedding minilm --max-docs 500
```

Project-specific index scripts are available in `scripts/build_*.py`.

## Analysis

Analysis helpers and notebooks are included for aggregate study analysis. They assume that experiment outputs have been generated locally or provided separately. Raw predictions are not committed to this repository.

## License and citation

Add the final license and citation metadata before making the GitHub repository public.
