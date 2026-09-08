# Benchmark: full_grid_v1

- Run ID: `1da80c9c-d656-46e8-bdb3-b8d8218aaea7`
- Commit: `e788566d9adf`
- Eval set: `data/eval/subset_40.jsonl` (40 questions)
- Generated with: `ollama` / `llama3.2:3b`
- Configurations: 24
- Sorted by: **hit_rate**

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
| structural | bge_large | dense | 0.912 | 0.828 | 1.000 | 110 | 34942 | 0.0000 |
| recursive | bge_large | hybrid_rerank | 0.909 | 0.774 | 0.667 | 4015 | 21307 | 0.0000 |
| fixed | bge_large | hybrid_rerank | 0.909 | 0.723 | 1.000 | 4304 | 42989 | 0.0000 |
| fixed | bge_small | hybrid | 0.875 | 0.672 | 0.833 | 20 | 31540 | 0.0000 |
| recursive | bge_large | hybrid | 0.848 | 0.586 | 0.667 | 117 | 23468 | 0.0000 |
| semantic | bge_small | hybrid_rerank | 0.824 | 0.668 | 1.000 | 3980 | 21979 | 0.0000 |
| semantic | bge_large | hybrid_rerank | 0.824 | 0.664 | 1.000 | 6299 | 51912 | 0.0000 |
| recursive | bge_large | dense | 0.806 | 0.568 | 0.833 | 110 | 36344 | 0.0000 |
| semantic | bge_large | hybrid | 0.781 | 0.524 | 0.833 | 253 | 36227 | 0.0000 |
| fixed | bge_small | dense | 0.774 | 0.611 | 0.667 | 17 | 43269 | 0.0000 |
| fixed | bge_large | dense | 0.774 | 0.562 | 0.833 | 115 | 36251 | 0.0000 |
| recursive | bge_small | hybrid | 0.765 | 0.565 | 0.833 | 23 | 25958 | 0.0000 |
| recursive | bge_small | dense | 0.758 | 0.540 | 0.833 | 19 | 31337 | 0.0000 |
| semantic | bge_small | hybrid | 0.750 | 0.545 | 1.000 | 35 | 24278 | 0.0000 |
| semantic | bge_small | dense | 0.735 | 0.490 | 0.833 | 28 | 23629 | 0.0000 |
| semantic | bge_large | dense | 0.697 | 0.544 | 1.000 | 189 | 37931 | 0.0000 |

Not measured in this run: Faith., Ans.Rel., Ctx.Prec., Ctx.Rec.. The RAGAS metrics need a judge model and are enabled with `--ragas`.

## Hit rate by difficulty

| Chunker | Embedder | Retriever | definitional | multi_hop | negative | single_hop |
|---|---|---|---|---|---|---|
| structural | bge_large | hybrid | 0.750 | 1.000 | n/a | 1.000 |
| structural | bge_small | hybrid | 1.000 | 1.000 | n/a | 0.933 |
| structural | bge_small | dense | 1.000 | 1.000 | n/a | 0.933 |
| recursive | bge_small | hybrid_rerank | 0.750 | 1.000 | n/a | 0.933 |
| fixed | bge_large | hybrid | 1.000 | 1.000 | n/a | 0.867 |
| fixed | bge_small | hybrid_rerank | 1.000 | 1.000 | n/a | 0.857 |
| structural | bge_large | hybrid_rerank | 1.000 | 0.933 | n/a | 0.929 |
| structural | bge_small | hybrid_rerank | 1.000 | 0.933 | n/a | 0.923 |
| structural | bge_large | dense | 0.750 | 1.000 | n/a | 0.867 |
| recursive | bge_large | hybrid_rerank | 0.750 | 1.000 | n/a | 0.867 |
| fixed | bge_large | hybrid_rerank | 1.000 | 1.000 | n/a | 0.800 |
| fixed | bge_small | hybrid | 1.000 | 0.929 | n/a | 0.786 |
| recursive | bge_large | hybrid | 1.000 | 0.857 | n/a | 0.800 |
| semantic | bge_small | hybrid_rerank | 1.000 | 0.867 | n/a | 0.733 |
| semantic | bge_large | hybrid_rerank | 1.000 | 0.867 | n/a | 0.733 |
| recursive | bge_large | dense | 1.000 | 0.857 | n/a | 0.692 |
| semantic | bge_large | hybrid | 1.000 | 0.933 | n/a | 0.538 |
| fixed | bge_small | dense | 1.000 | 0.929 | n/a | 0.571 |
| fixed | bge_large | dense | 1.000 | 1.000 | n/a | 0.533 |
| recursive | bge_small | hybrid | 0.750 | 0.800 | n/a | 0.733 |
| recursive | bge_small | dense | 1.000 | 0.733 | n/a | 0.733 |
| semantic | bge_small | hybrid | 1.000 | 0.733 | n/a | 0.692 |
| semantic | bge_small | dense | 1.000 | 0.800 | n/a | 0.600 |
| semantic | bge_large | dense | 1.000 | 0.857 | n/a | 0.467 |
