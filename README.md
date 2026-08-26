# rag-benchmark-kit

A configurable, benchmarked Retrieval-Augmented Generation pipeline. Every stage,
chunking through generation, is swappable from a YAML file, and a benchmark harness
measures which combination actually answers questions best.

> **Status: under construction.** The benchmark harness and the published results table
> are not built yet. This README is rewritten around real measured numbers once they
> exist. What is described below works today.

## What works today

Ingestion and retrieval over a real corpus: the GDPR (Regulation (EU) 2016/679) and the
EU AI Act (Regulation (EU) 2024/1689), fetched from the EU Publications Office.

```bash
make dev                 # sync dependencies and install the pre-commit hooks
make download-corpus     # fetch and cache the two regulations
make index               # chunk, embed and store them
make query Q="Can a controller charge a fee for a subject access request?"
```

The default pipeline runs entirely on your machine at no cost: local embeddings on CPU
and Ollama for generation, so no API key and no signup are needed. The first run
downloads model weights, which takes a few minutes and several gigabytes.

## Configuration

Every stage is chosen by name in [configs/default.yaml](configs/default.yaml). Changing a
strategy is a one-word edit with no code change:

| Stage | Available |
|---|---|
| corpus | `eu_regulations`, `markdown_docs` |
| chunker | `fixed`, `recursive`, `semantic`, `structural` |
| embedder | `bge_small`, `bge_large`, `e5_base`, `openai` |
| store | `qdrant` |
| retriever | `dense`, `hybrid`, `hybrid_rerank` |
| generator | `cited` |

Secrets and machine-specific settings live in the environment instead. Copy
[.env.example](.env.example) to `.env` as a starting point.

## Development

```bash
make check    # the full gate: ruff, mypy --strict, pytest with a coverage floor
```

## Licence

MIT. See [LICENSE](LICENSE).
