from eval.metrics import (
    context_precision, context_recall, hit_rate, mrr,
    token_f1, answer_coverage, faithfulness,
)


def test_retrieval_metrics():
    retrieved = [2, 3, 9]
    relevant = [2, 3]
    assert context_precision(retrieved, relevant) == 2 / 3
    assert context_recall(retrieved, relevant) == 1.0
    assert hit_rate(retrieved, relevant) == 1.0
    assert mrr([9, 2, 3], relevant) == 0.5   # first hit at rank 2


def test_retrieval_metrics_no_hit():
    assert hit_rate([9, 8], [1, 2]) == 0.0
    assert mrr([9, 8], [1, 2]) == 0.0
    assert context_recall([], [1]) == 0.0


def test_answer_metrics():
    ref = "federated learning trains without sharing data"
    good = "Federated learning trains a model without sharing raw data."
    bad = "Bananas are yellow fruits."
    assert token_f1(good, ref) > token_f1(bad, ref)
    assert answer_coverage(good, ref) > 0.6
    assert answer_coverage(bad, ref) == 0.0


def test_faithfulness_rewards_grounded_answers():
    contexts = ["Federated learning aggregates client updates using a Hessian weighting scheme."]
    grounded = "It aggregates client updates with Hessian weighting."
    hallucinated = "It uses quantum blockchain satellites."
    assert faithfulness(grounded, contexts) > faithfulness(hallucinated, contexts)
    assert faithfulness(grounded, contexts) >= 0.6
