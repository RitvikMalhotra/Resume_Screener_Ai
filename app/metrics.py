"""
Phase 2 – Ranking Metrics

Metrics implemented
-------------------
  precision@k     - fraction of top-k that are relevant
  recall@k        - fraction of all relevant items in top-k
  ap@k            - average precision at k
  ndcg@k          - normalized discounted cumulative gain
  f1@k            - harmonic mean of precision and recall
  mrr             - mean reciprocal rank
  spearman_rho    - rank correlation between predicted and gold ranking

Interview talking point:
  "We use NDCG@K as our primary metric because it's position-aware —
   a relevant candidate at rank 1 contributes more than one at rank 5.
   Precision@K tells recruiters exactly what to expect: if P@3=0.67,
   2 of the top 3 shown candidates will be genuinely suitable."
"""
from __future__ import annotations
import math
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Core metric functions ──────────────────────────────────────────────────

def precision_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not ranked_ids or k <= 0:
        return 0.0
    hits = sum(1 for rid in ranked_ids[:k] if rid in relevant_ids)
    return hits / k


def recall_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not relevant_ids or k <= 0:
        return 0.0
    hits = sum(1 for rid in ranked_ids[:k] if rid in relevant_ids)
    return hits / len(relevant_ids)


def average_precision_at_k(
    relevant_ids: set[str], ranked_ids: list[str], k: int
) -> float:
    if not relevant_ids or k <= 0:
        return 0.0
    hits, score = 0, 0.0
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant_ids:
            hits  += 1
            score += hits / i
    return score / min(len(relevant_ids), k)


def dcg_at_k(relevance: list[float], k: int) -> float:
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevance[:k])
    )


def ndcg_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not relevant_ids or k <= 0:
        return 0.0
    pred_rel  = [1.0 if rid in relevant_ids else 0.0 for rid in ranked_ids[:k]]
    ideal_rel = sorted(pred_rel, reverse=True)
    actual    = dcg_at_k(pred_rel, k)
    ideal     = dcg_at_k(ideal_rel, k)
    return actual / ideal if ideal > 0 else 0.0


def mean_reciprocal_rank(relevant_ids: set[str], ranked_ids: list[str]) -> float:
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def f1_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    p = precision_at_k(relevant_ids, ranked_ids, k)
    r = recall_at_k(relevant_ids, ranked_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def spearman_rho(
    gold_ranks: dict[str, int],
    pred_ranks: dict[str, int],
) -> float:
    common = sorted(set(gold_ranks) & set(pred_ranks))
    if len(common) < 2:
        return float("nan")
    g    = np.array([gold_ranks[k] for k in common], dtype=float)
    p    = np.array([pred_ranks[k] for k in common], dtype=float)
    d_sq = (g - p) ** 2
    n    = len(common)
    return float(1 - (6 * d_sq.sum()) / (n * (n ** 2 - 1)))


# ── Composite evaluation ───────────────────────────────────────────────────

def evaluate(
    relevant_ids: set[str],
    ranked_ids: list[str],
    k_values: Optional[list[int]] = None,
    gold_ranks: Optional[dict[str, int]] = None,
) -> dict:
    """
    Compute all metrics at multiple k values.

    Returns flat dict:
      {"precision@5": 0.8, "recall@5": 0.6, "ndcg@5": 0.91, "mrr": 1.0, ...}
    """
    k_values = k_values or [1, 3, 5, 10]
    metrics: dict[str, float] = {}

    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(relevant_ids, ranked_ids, k)
        metrics[f"recall@{k}"]    = recall_at_k(relevant_ids, ranked_ids, k)
        metrics[f"ap@{k}"]        = average_precision_at_k(relevant_ids, ranked_ids, k)
        metrics[f"ndcg@{k}"]      = ndcg_at_k(relevant_ids, ranked_ids, k)
        metrics[f"f1@{k}"]        = f1_at_k(relevant_ids, ranked_ids, k)

    metrics["mrr"] = mean_reciprocal_rank(relevant_ids, ranked_ids)

    if gold_ranks is not None:
        pred_rank_dict      = {rid: i for i, rid in enumerate(ranked_ids)}
        metrics["spearman_rho"] = spearman_rho(gold_ranks, pred_rank_dict)

    return {k: round(v, 4) for k, v in metrics.items()}


def batch_evaluate(
    queries: list[dict],
    k_values: Optional[list[int]] = None,
) -> dict:
    """
    Average metrics over multiple queries.

    Each query dict: {"relevant_ids": set, "ranked_ids": list}
    """
    if not queries:
        return {}
    all_metrics = [
        evaluate(
            relevant_ids = q["relevant_ids"],
            ranked_ids   = q["ranked_ids"],
            k_values     = k_values,
            gold_ranks   = q.get("gold_ranks"),
        )
        for q in queries
    ]
    keys = all_metrics[0].keys()
    avg  = {
        k: round(float(np.mean([m[k] for m in all_metrics if k in m])), 4)
        for k in keys
    }
    avg["num_queries"] = len(queries)
    return avg


# ── Quick eval helper ──────────────────────────────────────────────────────

def evaluate_with_report(
    relevant_ids: set[str],
    ranked_ids: list[str],
    k_values: Optional[list[int]] = None,
    primary_k: int = 5,
) -> dict:
    """
    Run evaluate() and attach a full MetricsReport.
    Returns dict with keys: "raw", "report"
    """
    from app.metrics_report import build_report

    k_values   = k_values or [1, 3, 5, 10]
    raw        = evaluate(relevant_ids, ranked_ids, k_values)
    report     = build_report(
        raw_metrics = raw,
        n_relevant  = len(relevant_ids),
        n_ranked    = len(ranked_ids),
        primary_k   = primary_k,
    )
    return {
        "raw":    raw,
        "report": report.to_dict(),
    }