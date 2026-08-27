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

## Benchmark methodology

### The corpus

The GDPR (Regulation (EU) 2016/679) and the EU AI Act (Regulation (EU) 2024/1689), fetched
from the EU Publications Office. Both are long, densely cross-referential documents where
naive chunking demonstrably fails, because definitions live far from their use sites and
articles reference other articles. That is what makes the differences between strategies
visible rather than noise.

### The evaluation set

[data/eval/gdpr_ai_act_qa.jsonl](data/eval/gdpr_ai_act_qa.jsonl) holds 116 question and
answer pairs: 46 single-hop, 49 multi-hop, 11 definitional and 10 negative. They cite 124
distinct sections, 73 in the GDPR and 51 in the AI Act.

The 10 negative questions are the ones worth having. They ask things the corpus cannot
answer, so `abstention_accuracy` measures whether a configuration invents an answer when
it should decline, which the quality metrics alone will not show.

**How it was built, and what has not happened yet.**

1. Article and paragraph text was extracted from the downloaded corpus programmatically.
2. An LLM (Claude) drafted every pair while reading that extracted text, not from memory.
3. Each pair was checked automatically by `rag-bench eval verify`, which confirms that
   every citation resolves to a real section and measures how much of each answer's
   wording appears in the section it cites.

As of the run on **2026-08-27**: **116 pairs, 0 citation errors, 0 pairs discarded, mean
answer support 94.8%.** Re-run it yourself with `make verify-eval`.

**No human has yet read these pairs.** That step is in the build plan and has not been
done. The automated check proves an answer's wording comes from the article it cites; it
cannot prove the answer is a fair reading of the law, that the cited article is the *best*
authority, or that a multi-hop question needs the hops it claims. Until a domain reader
has been through them, treat any number computed from this set as provisional.

### What the metrics mean

| Metric | Meaning |
|---|---|
| `hit_rate` | Share of answerable questions where retrieval surfaced a cited section |
| `mrr` | Mean reciprocal rank of the first relevant chunk |
| `abstention_accuracy` | Share of negative questions the system correctly declined |
| `faithfulness` | RAGAS: is the answer supported by the retrieved context |
| `answer_relevancy` | RAGAS: does the answer address the question |
| `context_precision` | RAGAS: is the retrieved context free of irrelevant material |
| `context_recall` | RAGAS: did retrieval find what the ground truth needed |
| `p50/p95 latency` | Retrieval and generation timed separately |
| `estimated_cost_usd` | Token counts priced from `configs/pricing.yaml` |

The first three are computed directly: deterministic, free, and identical on a rerun. The
four RAGAS metrics are judged by a language model, so they carry the judge's own variance
and are optional.

## Limitations

An honest list, because the numbers are worth less without it.

- **The evaluation set has not been human-verified.** See above. This is the single
  largest caveat.
- **RAGAS metrics are LLM-judged** and carry their own variance. Two runs of the same
  configuration will not produce identical faithfulness scores.
- **116 questions is a small sample.** Differences of a few percentage points between
  configurations are inside the noise, and on the 10-question smoke set several
  configurations tie outright.
- **Definitional questions are scored loosely.** GDPR Article 4 and AI Act Article 3 hold
  every definition in one article, 8,700 and 17,300 characters respectively, so any chunk
  overlapping them counts as a hit. The other 121 citations are paragraph-level with a
  median span of 473 characters.
- **Results are corpus-specific.** These are two EU regulations in English. Nothing here
  says the same chunker wins on medical notes, source code or customer support tickets.
- **English only.** Both the corpus and the embedding models are English.
- **The published numbers come from one model.** Swapping the generator changes
  faithfulness and abstention behaviour, which is why every run records the provider and
  model it used.

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
