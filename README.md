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

The whole stack, with one command:

```bash
make up      # or: docker compose -f docker/docker-compose.yml up --wait
curl -s localhost:8000/api/v1/queries -H 'Content-Type: application/json' \
  -d '{"question": "Can a controller charge a fee for a subject access request?"}'
```

`up` starts Postgres, Qdrant and Ollama, applies the migrations, pulls the model,
downloads the two regulations and builds the index, then starts the API. Nothing else is
needed before that curl returns a real answer.

**The first run downloads several gigabytes** of model weights and takes a while. They
land in a named volume, so later runs start quickly.

Without Docker:

```bash
make dev                 # sync dependencies and install the pre-commit hooks
make download-corpus     # fetch and cache the two regulations
make index               # chunk, embed and store them
make query Q="Can a controller charge a fee for a subject access request?"
```

The default pipeline runs entirely on your machine at no cost: local embeddings on CPU
and Ollama for generation, so no API key and no signup are needed.

## Benchmarking

The harness is the runner that executes the pipeline repeatedly under different
configurations and records scored results.

```bash
rag-bench benchmark plan                    # what a sweep would run, and what it costs
rag-bench benchmark run                     # run the full grid
rag-bench benchmark run --eval-set data/eval/smoke.jsonl   # fast iteration
rag-bench benchmark resume <run_id>         # continue after an interruption
rag-bench benchmark report <run_id>         # a markdown table, ready to paste
```

Two details worth knowing:

**Indexes are built once per group, not once per configuration.** Chunker and embedder
decide what gets indexed; the retriever does not. The default grid is 24 configurations
over 8 distinct indexes, so grouping on that turns 24 ingestions into 8.

**Runs are resumable.** Results are written as each question is answered, so a grid that
dies at configuration 19 of 24 restarts with `resume` and skips what already finished.

Metrics come in two kinds. `hit_rate`, `mrr`, `abstention_accuracy`, the latency
percentiles and the cost estimate are computed directly: deterministic, free, and
identical on a rerun. The four RAGAS metrics are judged by a language model, so they are
optional, cost a call per question per configuration, and carry the judge's own variance.
Enable them with `uv sync --extra ragas` and `--ragas`.

Everything is also reported per difficulty band, because a configuration that wins
overall but collapses on multi-hop questions is the finding worth reading.

## HTTP API

The pipeline is usable as a service, not just a CLI.

```bash
make serve      # or: uvicorn rag_bench.api.main:app --port 8000
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/queries` | Ask a question, get an answer with citations |
| `GET` | `/api/v1/configurations` | List every registered component, by stage |
| `POST` | `/api/v1/indexes` | Start an index build (202 with a `Location` header) |
| `GET` | `/api/v1/indexes/{id}` | Poll a build's progress |
| `GET` | `/api/v1/benchmark-runs` | List runs, cursor-paginated |
| `GET` | `/api/v1/benchmark-runs/{id}` | One run with its configurations and metrics |
| `GET` | `/health/live`, `/health/ready` | Liveness and readiness |

Interactive docs live at `/docs` and the OpenAPI document at `/openapi.json`, which is
the closest thing this project has to a user interface.

Every failure shares one envelope, so a client writes the handling once:

```json
{"error": {"code": "INDEX_NOT_READY", "message": "...", "details": {}, "request_id": "..."}}
```

The `request_id` is echoed in the `X-Request-ID` header, propagated from the caller when
supplied, and bound to every log line the request produces. Setting `API_KEY` turns on
`X-API-Key` authentication for the write endpoints; leave it unset and the quickstart
keeps working with no setup.

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

## Container

A multi-stage build: `uv sync --frozen --no-dev` produces the environment in a builder
stage, and the runtime image carries the resulting virtualenv, a non-root user and a
`HEALTHCHECK`, but none of the build tooling.

The image is roughly 1.3 GB, most of it CPU PyTorch, whose shared libraries alone are
about 450 MB. That is the cost of local embeddings working with no API key, which is a
deliberate trade rather than an oversight: dropping `sentence-transformers` would fit the
image under 500 MB but would make the free quickstart impossible. Model weights are
mounted from a volume rather than baked in, so a rebuild does not re-download them.

Kubernetes manifests are deliberately absent. This is a benchmark tool that is run, not a
service that is operated, and a Helm chart done well once elsewhere is worth more than one
done twice adequately.

## Licence

MIT. See [LICENSE](LICENSE).
