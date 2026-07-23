# Evaluation harness

Quantitative evaluation is what separates a demo from an engineered system.
This harness reports both **retrieval quality** and **answer quality**.

## Metrics

| Metric | What it measures |
|---|---|
| context_precision@k | fraction of retrieved chunks that are relevant |
| context_recall@k | fraction of relevant chunks that were retrieved |
| hit_rate / MRR | did any relevant chunk surface, and how high |
| answer_f1 | unigram overlap between answer and reference |
| answer_coverage | fraction of reference facts present in the answer |
| faithfulness | fraction of answer tokens grounded in retrieved context (anti-hallucination proxy) |

## Run it

```bash
# start Qdrant + your LLM first (docker compose up)
python -m eval.run_eval --ingest data/raw/federated_learning.pdf
python -m eval.run_eval --eval eval/eval_set.json
```

Results print as a table and are saved to `eval/results.json`.

## A/B the retrieval upgrades

Show the impact of hybrid search + reranking by toggling flags and re-running:

```bash
USE_HYBRID=false USE_RERANKER=false python -m eval.run_eval --eval eval/eval_set.json   # baseline
USE_HYBRID=true  USE_RERANKER=true  python -m eval.run_eval --eval eval/eval_set.json   # upgraded
```

Put the before/after numbers straight into your resume and README.

## Heavier, LLM-judged evaluation (optional)

The built-in metrics need no extra installs. For LLM-graded faithfulness and
answer-relevancy, install `ragas` and adapt `run_eval.py` to feed it
`{question, answer, contexts, ground_truth}` — the harness already collects
exactly those fields.
