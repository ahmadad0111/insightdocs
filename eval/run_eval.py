"""Run the RAG evaluation harness and print a results table.

Usage:
    # 1. Start Qdrant + your LLM, then ingest the eval document:
    python -m eval.run_eval --ingest data/raw/federated_learning.pdf
    # 2. Evaluate:
    python -m eval.run_eval --eval eval/eval_set.json

Produces per-question metrics plus an aggregate summary, and writes
eval/results.json for the README table.
"""
import argparse
import json
import os
import statistics

from eval.metrics import (
    context_precision, context_recall, hit_rate, mrr,
    token_f1, answer_coverage, faithfulness,
)


def _load_pipeline():
    from src.api.dependencies import build_pipeline
    return build_pipeline()


def ingest(path):
    pipe = _load_pipeline()
    result = pipe["ingestion"].ingest(path)
    print(f"Ingested {result['filename']}: {result['num_chunks']} chunks "
          f"(document_id={result['document_id']})")


def evaluate(eval_path):
    pipe = _load_pipeline()
    service = pipe["service"]
    chain = service.rag_chain

    with open(eval_path) as f:
        spec = json.load(f)

    rows = []
    for item in spec["items"]:
        q = item["question"]
        ref = item.get("reference_answer", "")
        rel_pages = item.get("relevant_pages", [])

        chunks = chain._prepare_chunks(q)
        contexts = [c.payload["text"] for c in chunks]
        retrieved_pages = [c.payload.get("page") for c in chunks]

        out = service.query(q)
        answer = out["answer"]

        rows.append({
            "question": q,
            "context_precision": round(context_precision(retrieved_pages, rel_pages), 3),
            "context_recall": round(context_recall(retrieved_pages, rel_pages), 3),
            "hit_rate": round(hit_rate(retrieved_pages, rel_pages), 3),
            "mrr": round(mrr(retrieved_pages, rel_pages), 3),
            "answer_f1": round(token_f1(answer, ref), 3),
            "answer_coverage": round(answer_coverage(answer, ref), 3),
            "faithfulness": round(faithfulness(answer, contexts), 3),
            "latency_ms": out.get("latency_ms"),
        })

    metric_keys = ["context_precision", "context_recall", "hit_rate", "mrr",
                   "answer_f1", "answer_coverage", "faithfulness"]
    summary = {k: round(statistics.mean(r[k] for r in rows), 3) for k in metric_keys}
    summary["avg_latency_ms"] = round(
        statistics.mean(r["latency_ms"] for r in rows if r["latency_ms"]), 1
    ) if any(r["latency_ms"] for r in rows) else None

    _print_table(rows, summary, metric_keys)
    os.makedirs("eval", exist_ok=True)
    with open("eval/results.json", "w") as f:
        json.dump({"config": _config_summary(), "rows": rows, "summary": summary}, f, indent=2)
    print("\nSaved eval/results.json")
    return summary


def _config_summary():
    from src.core.config import Config
    return Config.summary()


def _print_table(rows, summary, metric_keys):
    print("\n=== Per-question metrics ===")
    header = ["question"] + metric_keys
    print(" | ".join(h[:16].ljust(16) for h in header))
    print("-" * (19 * len(header)))
    for r in rows:
        line = [r["question"][:16].ljust(16)] + [f"{r[k]:.3f}".ljust(16) for k in metric_keys]
        print(" | ".join(line))
    print("\n=== Aggregate ===")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", help="Path to a PDF to ingest before evaluating")
    ap.add_argument("--eval", help="Path to an eval_set.json")
    args = ap.parse_args()
    if args.ingest:
        ingest(args.ingest)
    if args.eval:
        evaluate(args.eval)
    if not args.ingest and not args.eval:
        ap.print_help()
