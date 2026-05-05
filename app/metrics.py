"""
Evaluation metrics for ranked retrieval.

Metrics implemented
-------------------
  precision@k     – fraction of top-k that are relevant
  recall@k        – fraction of all relevant items retrieved in top-k
  ap@k            – average precision at k (area under precision-recall curve)
  ndcg@k          – normalized discounted cumulative gain
  spearman_rho    – Spearman rank correlation between predicted and gold ranking
  mrr             – mean reciprocal rank
  
All functions follow a consistent signature:
    metric_fn(relevant_ids: set[str], ranked_ids: list[str], k: int) → float
"""
from __future__ import annotations
import math
import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def precision_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if not ranked_ids or k <= 0:
        return 0.0
    top_k = ranked_ids[:k]
    hits  = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def recall_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    """Fraction of all relevant items that appear in top-k."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = ranked_ids[:k]
    hits  = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def average_precision_at_k(
    relevant_ids: set[str], ranked_ids: list[str], k: int
) -> float:
    """Average precision at k (AP@k)."""
    if not relevant_ids or k <= 0:
        return 0.0
    hits   = 0
    score  = 0.0
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant_ids:
            hits  += 1
            score += hits / i
    return score / min(len(relevant_ids), k)


def dcg_at_k(relevance: list[float], k: int) -> float:
    """Discounted cumulative gain at k."""
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevance[:k])
    )


def ndcg_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    """Normalized DCG at k. Assumes binary relevance."""
    if not relevant_ids or k <= 0:
        return 0.0
    pred_rel  = [1.0 if rid in relevant_ids else 0.0 for rid in ranked_ids[:k]]
    ideal_rel = sorted(pred_rel, reverse=True)
    actual_dcg = dcg_at_k(pred_rel, k)
    ideal_dcg  = dcg_at_k(ideal_rel, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def mean_reciprocal_rank(
    relevant_ids: set[str], ranked_ids: list[str]
) -> float:
    """MRR – reciprocal of the rank of the first relevant item."""
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def spearman_rho(gold_ranks: dict[str, int], pred_ranks: dict[str, int]) -> float:
    """
    Spearman rank correlation between gold and predicted rankings.

    Parameters
    ----------
    gold_ranks : {resume_id: gold_rank}  (lower = better)
    pred_ranks : {resume_id: predicted_rank}

    Returns
    -------
    float in [-1, 1]; 1 = perfect correlation
    """
    common = sorted(set(gold_ranks) & set(pred_ranks))
    if len(common) < 2:
        return float("nan")
    g = np.array([gold_ranks[k] for k in common], dtype=float)
    p = np.array([pred_ranks[k] for k in common], dtype=float)
    d_sq = (g - p) ** 2
    n = len(common)
    rho = 1 - (6 * d_sq.sum()) / (n * (n ** 2 - 1))
    return float(rho)


def f1_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    p = precision_at_k(relevant_ids, ranked_ids, k)
    r = recall_at_k(relevant_ids, ranked_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate(
    relevant_ids: set[str],
    ranked_ids: list[str],
    k_values: list[int] | None = None,
    gold_ranks: dict[str, int] | None = None,
) -> dict:
    """
    Compute all metrics at multiple k values.

    Returns
    -------
    dict with keys like "precision@5", "recall@10", "ndcg@10", "mrr", etc.
    """
    k_values = k_values or [1, 3, 5, 10, 20]
    metrics: dict[str, float] = {}

    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(relevant_ids, ranked_ids, k)
        metrics[f"recall@{k}"]    = recall_at_k(relevant_ids, ranked_ids, k)
        metrics[f"ap@{k}"]        = average_precision_at_k(relevant_ids, ranked_ids, k)
        metrics[f"ndcg@{k}"]      = ndcg_at_k(relevant_ids, ranked_ids, k)
        metrics[f"f1@{k}"]        = f1_at_k(relevant_ids, ranked_ids, k)

    metrics["mrr"] = mean_reciprocal_rank(relevant_ids, ranked_ids)

    if gold_ranks is not None:
        pred_rank_dict = {rid: i for i, rid in enumerate(ranked_ids)}
        metrics["spearman_rho"] = spearman_rho(gold_ranks, pred_rank_dict)

    return {k: round(v, 4) for k, v in metrics.items()}


def batch_evaluate(
    queries: list[dict],   # each: {"relevant_ids": set, "ranked_ids": list}
    k_values: list[int] | None = None,
) -> dict:
    """
    Average metrics over multiple queries.
    queries: list of dicts with keys "relevant_ids" and "ranked_ids"
    """
    if not queries:
        return {}
    all_metrics: list[dict] = []
    for q in queries:
        m = evaluate(
            relevant_ids=q["relevant_ids"],
            ranked_ids=q["ranked_ids"],
            k_values=k_values,
            gold_ranks=q.get("gold_ranks"),
        )
        all_metrics.append(m)

    # average across queries
    keys = all_metrics[0].keys()
    avg  = {
        k: round(np.mean([m[k] for m in all_metrics if k in m]), 4)
        for k in keys
    }
    avg["num_queries"] = len(queries)
    return avg
