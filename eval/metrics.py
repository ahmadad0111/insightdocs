"""Lightweight, dependency-free RAG metrics.

These are transparent re-implementations of the ideas behind RAGAS so the
harness runs with zero extra installs. For a heavier, LLM-judged evaluation
you can swap in the `ragas` package (see eval/README.md).

Retrieval metrics (need ground-truth relevant chunk ids / pages):
    - context_precision@k
    - context_recall@k
    - hit_rate@k / MRR

Answer metrics (need a reference answer):
    - token_f1  (unigram overlap F1)
    - answer_coverage (fraction of reference tokens present)
"""
import re
from collections import Counter

_WORD = re.compile(r"\w+")


def _toks(text):
    return _WORD.findall((text or "").lower())


# ---------- retrieval ----------
def context_precision(retrieved_ids, relevant_ids):
    if not retrieved_ids:
        return 0.0
    rel = set(relevant_ids)
    hits = sum(1 for r in retrieved_ids if r in rel)
    return hits / len(retrieved_ids)


def context_recall(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 0.0
    rel = set(relevant_ids)
    found = sum(1 for r in set(retrieved_ids) if r in rel)
    return found / len(rel)


def hit_rate(retrieved_ids, relevant_ids):
    rel = set(relevant_ids)
    return 1.0 if any(r in rel for r in retrieved_ids) else 0.0


def mrr(retrieved_ids, relevant_ids):
    rel = set(relevant_ids)
    for i, r in enumerate(retrieved_ids):
        if r in rel:
            return 1.0 / (i + 1)
    return 0.0


# ---------- answer ----------
def token_f1(prediction, reference):
    p, r = Counter(_toks(prediction)), Counter(_toks(reference))
    if not p or not r:
        return 0.0
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def answer_coverage(prediction, reference):
    ref = set(_toks(reference))
    if not ref:
        return 0.0
    pred = set(_toks(prediction))
    return len(ref & pred) / len(ref)


def faithfulness(prediction, contexts):
    """Fraction of answer tokens supported by the retrieved context.

    A cheap proxy for hallucination: a grounded answer reuses context
    vocabulary; an ungrounded one introduces many out-of-context tokens.
    Stopwords are ignored to avoid rewarding filler.
    """
    stop = {"the", "a", "an", "of", "to", "and", "in", "is", "are", "for",
            "on", "with", "that", "this", "it", "as", "by", "be", "or"}
    ctx = set()
    for c in contexts:
        ctx |= set(_toks(c))
    pred = [t for t in _toks(prediction) if t not in stop]
    if not pred:
        return 0.0
    supported = sum(1 for t in pred if t in ctx)
    return supported / len(pred)
