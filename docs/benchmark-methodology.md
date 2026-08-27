# Benchmark methodology

What is measured, how, and what the numbers cannot tell you.

## The harness

The harness is the runner that executes the pipeline repeatedly under different
configurations and records scored results. One command expands a sweep into every
combination of its axes and runs each one over the whole evaluation set.

```bash
rag-bench benchmark plan     # what it would run, and what that costs
rag-bench benchmark run      # run it
rag-bench benchmark resume <run_id>
rag-bench benchmark report <run_id> --write
```

## The corpus

The GDPR (Regulation (EU) 2016/679) and the EU AI Act (Regulation (EU) 2024/1689),
fetched from the EU Publications Office Cellar API.

The EUR-Lex web pages are not used: they sit behind a JavaScript bot challenge that
returns an empty `202` to any plain HTTP client. Cellar serves the same authenticated
XHTML through ordinary content negotiation, which is what makes `make download-corpus`
work unattended.

Parsing recovers the real structure of each regulation: 99 GDPR articles, 113 AI Act
articles, and 872 numbered paragraphs, each with character offsets into the document.
Downloads are cached with a manifest recording the URL, size and SHA-256 of every file,
so a corpus can be traced to exactly what was fetched.

## The evaluation set

116 pairs in [`data/eval/gdpr_ai_act_qa.jsonl`](../data/eval/gdpr_ai_act_qa.jsonl).

| Difficulty | Count | Meaning |
|---|---|---|
| `single_hop` | 46 | Answered by one paragraph |
| `multi_hop` | 49 | Needs two or more paragraphs or articles |
| `definitional` | 11 | Asks what a defined term means |
| `negative` | 10 | The corpus cannot answer it; the system should decline |

The negatives matter most. Without them a configuration that answers confidently no
matter what looks identical to one that knows when to stop.

### How it was built

1. Article and paragraph text was extracted from the downloaded corpus programmatically.
2. An LLM (Claude) drafted every pair while reading that extracted text, not from memory.
3. `rag-bench eval verify` checked each pair: that every citation resolves to a real
   section, and how much of the answer's wording appears in the section it cites.

Run of **2026-08-27**: 116 pairs, **0 citation errors, 0 pairs discarded, 94.8% mean
answer support**. Reproduce with `make verify-eval`.

### What has not happened

**No human has read these pairs.** The automated check proves an answer's wording comes
from the article it cites. It cannot prove the answer is a fair reading of the law, that
the cited article is the best authority, or that a multi-hop question genuinely needs its
hops. Treat every number derived from this set as provisional until a domain reader has
been through it.

## Metrics

### Computed directly

Deterministic, free, and identical on a rerun.

| Metric | Definition |
|---|---|
| `hit_rate` | Share of answerable questions where at least one retrieved chunk covers a cited section. Negatives are excluded, because they have no correct section and counting them would penalise correct behaviour. |
| `mrr` | Mean of `1 / (rank + 1)` for the first relevant chunk, counting a miss as zero |
| `abstention_accuracy` | Share of the 10 negative questions the system correctly declined |
| `p50` / `p95` latency | Retrieval and generation timed separately, because they are different trade-offs |
| `estimated_cost_usd` | Token counts priced from [`configs/pricing.yaml`](../configs/pricing.yaml). Local models are genuinely free, so they price at zero. |

### Judged by a model

The four RAGAS metrics: `faithfulness`, `answer_relevancy`, `context_precision`,
`context_recall`. They are optional, enabled with `uv sync --extra ragas` and `--ragas`,
because they need a judge model and cost one call per question per configuration.

Use a stronger judge than the model under test. Judging a model with itself measures the
same weaknesses twice.

Everything is also reported per difficulty band. A configuration that wins overall but
collapses on multi-hop questions is the finding worth reading, and the aggregate hides it.

## Index reuse

Chunker and embedder determine what is indexed. The retriever does not. The default grid
is 24 configurations over 8 distinct indexes, so the harness groups on that and ingests
the corpus 8 times instead of 24.

```
4 chunkers x 2 embedders x 3 retrievers = 24 configurations
4 chunkers x 2 embedders                =  8 indexes
```

`rag-bench benchmark plan` prints the saving before you commit to a run.

## Resumability

Results are written as each question is answered, not at the end. A run that dies at
configuration 19 of 24 restarts with `rag-bench benchmark resume <run_id>` and skips what
already finished, rebuilding only the indexes the remaining work needs.

This is not a nicety. The full grid took 66 minutes on a developer machine against the
10-question smoke set alone.

## Reproducibility

Every run records the git commit that produced it, whether the working tree was dirty,
and the provider and model that generated the answers. A results table that cannot be
traced to the code and the model behind it is an assertion, not evidence.

## Threats to validity

- The evaluation set has not been human-verified.
- RAGAS metrics carry the judge's variance; two runs will not agree exactly.
- 116 questions is a small sample. Differences of a few points are inside the noise.
- Definitional questions cite whole definition articles (GDPR Art. 4 is 8,700 characters,
  AI Act Art. 3 is 17,300), so any overlapping chunk counts as a hit. The other 121
  citations are paragraph-level with a median span of 473 characters.
- Two English EU regulations. Nothing here says the same chunker wins on medical notes or
  source code.
- Latency depends on the machine. Compare configurations within a run, not across runs on
  different hardware.
