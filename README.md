# rag-benchmark-kit

[![CI](https://github.com/heyitspratik/rag-benchmark-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/heyitspratik/rag-benchmark-kit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/heyitspratik/rag-benchmark-kit/branch/main/graph/badge.svg)](https://codecov.io/gh/heyitspratik/rag-benchmark-kit)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-green.svg)](LICENSE)

**Every stage of a RAG pipeline, swappable from one YAML file, with a benchmark harness
that measures which combination actually answers best.**

Not a chat-with-your-PDF demo. The point is evidence: change `chunker.name` from `fixed`
to `structural`, rerun, and see what it did to hit rate, latency and cost over the GDPR
and the EU AI Act. In the published run below that one change is worth 8 points of hit
rate, measured over a 116 pair evaluation set built from the regulations themselves.

## Results

All 24 configurations, 40 questions each, run on 2026-09-08 with `ollama` / `llama3.2:3b`
at commit `e788566`. Sorted by hit rate, winner emphasised. Full table and the
per-difficulty breakdown in [`results/full_grid_v1/results.md`](results/full_grid_v1/results.md).

| Chunker | Embedder | Retriever | Hit@k | MRR | Abstain | p50 Ret. | p95 Gen. | Cost $ |
|---|---|---|---|---|---|---|---|---|
| **structural** | **bge_large** | **hybrid** | **0.971** | **0.831** | **1.000** | **112** | **31383** | **0.0000** |
| structural | bge_small | hybrid | 0.971 | 0.792 | 1.000 | 22 | 28985 | 0.0000 |
| structural | bge_small | dense | 0.970 | 0.831 | 0.667 | 20 | 31299 | 0.0000 |
| recursive | bge_small | hybrid_rerank | 0.941 | 0.772 | 0.667 | 3809 | 35050 | 0.0000 |
| fixed | bge_large | hybrid | 0.939 | 0.612 | 0.833 | 131 | 34893 | 0.0000 |
| fixed | bge_small | hybrid_rerank | 0.938 | 0.760 | 1.000 | 3967 | 35470 | 0.0000 |
| structural | bge_large | hybrid_rerank | 0.935 | 0.844 | 1.000 | 4210 | 26931 | 0.0000 |
| structural | bge_small | hybrid_rerank | 0.933 | 0.822 | 1.000 | 4104 | 35000 | 0.0000 |
| _15 rows omitted_ | | | | | | | | |
| semantic | bge_large | dense | 0.697 | 0.544 | 1.000 | 189 | 37931 | 0.0000 |

Latencies are milliseconds. Cost is zero because the answers were generated locally, which
is the point of the Ollama default, not a rounding artefact.

### What the run actually says

**The chunker decides the outcome.** Averaged over every embedder and retriever it pairs
with, `structural` scores 0.949 and `semantic` scores 0.768. That 18 point spread is wider
than the embedder axis and the retriever axis put together. Splitting on the regulation's
own article and paragraph boundaries beats inferring boundaries from embedding similarity,
on a corpus that ships real boundaries.

**The bigger embedder is not worth it here.** `bge_large` averages 0.859 against
`bge_small` at 0.853, a difference of half a point, for six times the retrieval latency
(160 ms against 26 ms median). Row two of the table is the practical pick: the same 0.971
hit rate as the winner at a fifth of the retrieval time.

**The rerank rescues weak chunking rather than improving good chunking.** Across the grid
`hybrid_rerank` does lead on average, 0.902 against 0.863 for plain `hybrid`, but it costs
4059 ms of median retrieval against 74 ms, roughly 55 times slower. Look at where the gain
lands. For `semantic`, the worst chunker, the rerank is its best retriever by a clear
margin. For `structural`, the best chunker, plain `hybrid` beats the reranked variant
(0.971 against 0.935). Paying seconds per query to recover from a chunking decision you
could have made correctly for free is the expensive way to solve it.

**Abstention is not free either.** Three configurations that retrieve well still answered
questions the corpus cannot support, scoring 0.667 on the six negative questions. Hit rate
alone would have hidden that.

### Caveats on these numbers

- **A 40-question subset**, not the full 116 pair set, chosen to keep one run near an hour
  on a developer machine. `data/eval/subset_40.jsonl` holds it.
- **`llama3.2:3b` generated every answer.** A 3B model is a floor, not a ceiling. Retrieval
  metrics are model-independent, the abstention and latency figures are not.
- **31 of 960 answers (3.2%) were dropped** for citing a passage number that was never
  supplied. The generator refuses invalid citations rather than silently accepting them, so
  those questions are recorded as errors and excluded. Eight configurations lost none, the
  worst lost 4 of 40. It is spread evenly enough not to favour any configuration.
- **The RAGAS metrics were not measured.** They need a judge model. Enable with
  `uv sync --extra ragas` and `--ragas`.
- **No human has read the evaluation pairs.** See
  [`docs/benchmark-methodology.md`](docs/benchmark-methodology.md).

Reproduce the whole thing:

```bash
make up                                        # the whole stack, one command
make bench                                     # the 24-configuration grid
rag-bench benchmark report <run_id> --write    # writes results/<name>/
```

[`.github/workflows/benchmark.yml`](.github/workflows/benchmark.yml) runs a reduced grid on
a clean machine, so the result is reproducible rather than merely claimed.

## Architecture

```mermaid
flowchart LR
    subgraph ingest [Indexing]
        direction TB
        A[Corpus] --> B(Loader)
        B --> C(Chunker)
        C --> D(Embedder)
        D --> E[(Vector store)]
    end

    subgraph query [Querying]
        direction TB
        Q[Question] --> R(Retriever)
        R --> G(Generator)
        G --> ANS[Answer and citations]
    end

    E -.-> R

    classDef swap fill:#2d5a8c,stroke:#1a3d63,color:#fff
    class B,C,D,E,R,G swap
```

Every shaded stage is chosen by name from a registry. Detail in
[docs/architecture.md](docs/architecture.md).

## Quickstart

Three commands. No API key, no signup, no cost.

```bash
git clone https://github.com/heyitspratik/rag-benchmark-kit && cd rag-benchmark-kit
make up
curl -s localhost:8000/api/v1/queries -H 'Content-Type: application/json' \
  -d '{"question": "Can a controller charge a fee for a subject access request?"}'
```

`make up` starts Postgres, Qdrant and Ollama, applies the migrations, pulls the model,
downloads both regulations and builds the index, then starts the API. Nothing else is
needed before that curl returns a real answer.

> **The first run downloads several gigabytes** of model weights and takes a while. They
> land in a named volume, so later runs start quickly.

Without Docker:

```bash
make dev && make download-corpus && make index
make query Q="Can a controller charge a fee for a subject access request?"
```

The default pipeline runs entirely on your machine: local embeddings on CPU, Ollama for
generation.

## HTTP API

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

Interactive docs at `/docs`, OpenAPI at `/openapi.json`. Every failure shares one
envelope, so a client writes the handling once:

```json
{"error": {"code": "INDEX_NOT_READY", "message": "...", "details": {}, "request_id": "..."}}
```

The `request_id` is echoed in `X-Request-ID`, propagated from the caller when supplied,
and bound to every log line the request produces. Set `API_KEY` to require an `X-API-Key`
header on the write endpoints; leave it unset and the quickstart keeps working.

## Configuration

Everything is chosen by name in [`configs/default.yaml`](configs/default.yaml). Changing a
strategy is a one-word edit with no code change:

```yaml
chunker:
  name: structural          # fixed | recursive | semantic | structural
  params:
    max_chars: 1200
    overlap: 150

embedder:
  name: bge_small           # bge_small | bge_large | e5_base | openai
  params:
    batch_size: 32
    normalize: true

retriever:
  name: hybrid_rerank       # dense | hybrid | hybrid_rerank
  params:
    top_k: 5
    overfetch_multiplier: 4
    bm25_weight: 0.4
    dense_weight: 0.6
```

| Stage | Available | Notes |
|---|---|---|
| corpus | `eu_regulations`, `markdown_docs` | The second proves the pipeline is corpus-agnostic |
| chunker | `fixed`, `recursive`, `semantic`, `structural` | `fixed` is the naive baseline |
| embedder | `bge_small`, `bge_large`, `e5_base`, `openai` | The first three run locally on CPU |
| store | `qdrant`, `pgvector` | Two genuinely different backends behind one interface |
| retriever | `dense`, `hybrid`, `hybrid_rerank` | Hybrid fuses BM25 and vectors by rank |
| generator | `cited` | Validates its own citations and raises on a bad one |

Secrets and machine-specific values live in the environment, never in YAML. Copy
[`.env.example`](.env.example) to `.env`.

## Extending

Adding a chunking strategy is a class and a decorator. Nothing else changes: not the
pipeline, not the config schema, not the benchmark runner, not the API.

```python
# src/rag_bench/components/chunkers/sentence.py
from rag_bench.core.interfaces import BaseChunker
from rag_bench.core.models import Chunk, Document
from rag_bench.core.registry import register_chunker


@register_chunker("sentence")
class SentenceChunker(BaseChunker):
    """One chunk per sentence, skipping fragments."""

    def __init__(self, min_chars: int = 40) -> None:
        self._min_chars = min_chars

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        for piece in document.text.split(". "):
            end = start + len(piece)
            if len(piece.strip()) >= self._min_chars:
                chunks.append(
                    Chunk.create(
                        doc_id=document.id,
                        ordinal=len(chunks),
                        text=piece,
                        char_start=start,
                        char_end=end,
                    )
                )
            start = end + 2
        return chunks
```

Import it in `components/chunkers/__init__.py`, then use it:

```yaml
chunker:
  name: sentence
  params:
    min_chars: 40
```

It now appears in `GET /api/v1/configurations` and can be swept by the benchmark, because
both read the registry rather than a fixed list.

**The one rule.** Set truthful `char_start` and `char_end`. Those offsets are how a chunk
is mapped back to the document sections that ground truth cites, and they are what make
your strategy comparable to every other one.

## Benchmark methodology

Full detail in [docs/benchmark-methodology.md](docs/benchmark-methodology.md). In short:

- **Corpus.** GDPR and EU AI Act from the EU Publications Office. 99 and 113 articles, 872
  numbered paragraphs, parsed with character offsets.
- **Evaluation set.** 116 pairs: 46 single-hop, 49 multi-hop, 11 definitional, 10
  negative. LLM-drafted from the extracted article text, then checked by
  `rag-bench eval verify`, which found 0 citation errors and 94.8% mean answer support on
  2026-08-27. **No human has read them yet.**
- **Metrics.** `hit_rate`, `mrr`, `abstention_accuracy`, latency percentiles and cost are
  computed directly and rerun identically. The four RAGAS metrics are LLM-judged and
  optional.
- **Index reuse.** 24 configurations share 8 indexes, so the corpus is ingested 8 times
  rather than 24.
- **Reproducibility.** Every run records the commit, whether the tree was dirty, and the
  provider and model that generated the answers.

## Limitations

An honest list. The numbers are worth less without it.

- **The evaluation set has not been human-verified.** The automated check proves an
  answer's wording comes from the article it cites, not that it is a fair reading of the
  law. This is the largest caveat here.
- **RAGAS metrics are LLM-judged** and carry the judge's own variance. Two runs of one
  configuration will not produce identical faithfulness scores.
- **116 questions is a small sample.** Differences of a few points sit inside the noise.
  On the 10-question smoke set several configurations tie outright.
- **Definitional questions are scored loosely.** GDPR Article 4 and AI Act Article 3 hold
  every definition in a single article, 8,700 and 17,300 characters, so any overlapping
  chunk counts as a hit. The other 121 citations are paragraph-level, median span 473
  characters.
- **Results are corpus-specific.** Two English EU regulations. Nothing here says the same
  chunker wins on medical notes, source code or support tickets.
- **English only**, corpus and embedding models alike.
- **The image is about 1.3 GB**, most of it CPU PyTorch. Dropping local embeddings would
  fit it under 500 MB and make the free quickstart impossible. A trade, not an oversight.
- **Latency is machine-specific.** Compare configurations within a run, not across
  machines.
- **No Kubernetes manifests.** This is a tool that is run, not a service that is operated.

## Project structure

```
src/rag_bench/
├── core/
│   ├── interfaces/       # One abstract base class per swappable stage
│   ├── registry.py       # Name to implementation, the extension point
│   ├── models.py         # Document, Chunk, ScoredChunk, Answer
│   ├── config.py         # Validated experiment YAML, config fingerprints
│   ├── settings.py       # Environment layer, per-provider credential rules
│   └── llm.py            # One chat-model factory, Ollama health probe
├── components/           # Every implementation, registered on import
│   ├── loaders/          # Corpus download and per-format parsing
│   ├── chunkers/         # fixed, recursive, semantic, structural
│   ├── embedders/        # Local sentence-transformers models, OpenAI
│   ├── stores/           # Qdrant, pgvector
│   ├── retrievers/       # dense, hybrid, hybrid_rerank, RRF fusion
│   └── generators/       # Cited generation and citation validation
├── pipeline/             # Indexer and querier, assembled from config alone
├── benchmark/            # Grid, runner, metrics, RAGAS, reporter, eval verification
├── db/                   # SQLAlchemy models, schema owned by Alembic
├── api/                  # FastAPI app, one error envelope, health checks
└── cli/                  # Typer commands

configs/                  # Committed experiment descriptions
data/eval/                # The evaluation set, committed
alembic/versions/         # Migrations, from the first commit
docker/                   # Multi-stage Dockerfile and the compose stack
docs/                     # Architecture and benchmark methodology
results/                  # Published numbers
```

## Development

```bash
make check              # ruff, mypy --strict, pytest with a coverage floor
make test-integration   # needs Docker for Postgres and Qdrant
make verify-eval        # re-check every evaluation pair against the corpus
```

511 tests, 94% coverage, `mypy --strict` clean. No test requires an API key, a network,
or a running Ollama.

## Contributing

Issues and pull requests are welcome. `make check` must pass, PR titles follow
[conventional commits](https://www.conventionalcommits.org/), and a new component belongs
behind its interface rather than special-cased around it.

## Acknowledgements

- The EU Publications Office, for making both regulations freely available.
- [BAAI](https://huggingface.co/BAAI) for the BGE embedding and reranking models.
- [Qdrant](https://qdrant.tech/), [pgvector](https://github.com/pgvector/pgvector),
  [Ollama](https://ollama.com/), [RAGAS](https://docs.ragas.io/) and
  [uv](https://docs.astral.sh/uv/).

## Licence

MIT. See [LICENSE](LICENSE).
