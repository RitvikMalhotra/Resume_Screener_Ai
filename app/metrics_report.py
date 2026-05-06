"""
Phase 2 – Ranking Metrics Reporter

Wraps raw metric values into human-readable report with:
  - Grade (A/B/C/D/F) per metric
  - Plain English interpretation
  - Overall system verdict

Interview talking point:
  "We evaluate ranking quality using NDCG@K and Precision@K rather
   than accuracy because order matters — surfacing the best candidate
   at rank 1 is worth more than surfacing them at rank 5."
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ── Grading thresholds ─────────────────────────────────────────────────────

_GRADES = [
    (0.90, "A", "Excellent"),
    (0.75, "B", "Good"),
    (0.55, "C", "Fair"),
    (0.35, "D", "Poor"),
    (0.00, "F", "Very Poor"),
]


def grade(score: float) -> tuple[str, str]:
    """Return (letter_grade, label) for a score in [0, 1]."""
    for threshold, letter, label in _GRADES:
        if score >= threshold:
            return letter, label
    return "F", "Very Poor"


# ── Metric descriptions ────────────────────────────────────────────────────

_DESCRIPTIONS = {
    "precision": (
        "Of the top-{k} results shown, what fraction are actually relevant? "
        "High precision = fewer irrelevant candidates shown to the recruiter."
    ),
    "recall": (
        "Of all relevant candidates in the pool, what fraction appear in top-{k}? "
        "High recall = fewer good candidates missed."
    ),
    "ndcg": (
        "Normalized Discounted Cumulative Gain at {k}. "
        "Rewards ranking relevant candidates higher. "
        "1.0 = perfect ranking order."
    ),
    "mrr": (
        "Mean Reciprocal Rank. "
        "How high is the first relevant candidate ranked? "
        "1.0 = always at position 1."
    ),
    "ap": (
        "Average Precision at {k}. "
        "Area under the precision-recall curve. "
        "Penalizes gaps between relevant results."
    ),
}


# ── Report dataclasses ─────────────────────────────────────────────────────

@dataclass
class MetricEntry:
    name: str
    k: Optional[int]
    value: float
    grade: str
    grade_label: str
    description: str

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "k":           self.k,
            "value":       self.value,
            "grade":       self.grade,
            "grade_label": self.grade_label,
            "description": self.description,
        }


@dataclass
class MetricsReport:
    entries: list[MetricEntry]
    overall_grade: str
    overall_label: str
    verdict: str
    n_relevant: int
    n_ranked: int
    primary_k: int

    def to_dict(self) -> dict:
        return {
            "overall_grade": self.overall_grade,
            "overall_label": self.overall_label,
            "verdict":       self.verdict,
            "n_relevant":    self.n_relevant,
            "n_ranked":      self.n_ranked,
            "primary_k":     self.primary_k,
            "metrics":       [e.to_dict() for e in self.entries],
        }

    def summary(self) -> dict:
        """Flat dict of just name→value for quick API responses."""
        return {e.name: e.value for e in self.entries}


# ── Builder ────────────────────────────────────────────────────────────────

def build_report(
    raw_metrics: dict[str, float],
    n_relevant: int,
    n_ranked: int,
    primary_k: int = 5,
) -> MetricsReport:
    """
    Convert raw metrics dict → full MetricsReport.

    Parameters
    ----------
    raw_metrics : output of app.metrics.evaluate()
    n_relevant  : number of ground-truth relevant items
    n_ranked    : total items ranked
    primary_k   : k value used for overall grade (default 5)
    """
    entries: list[MetricEntry] = []

    for key, val in raw_metrics.items():
        if key in ("num_queries",):
            continue

        # parse "precision@5" → ("precision", 5)
        if "@" in key:
            metric_type, k_str = key.split("@")
            k = int(k_str)
        else:
            metric_type = key
            k = None

        desc_template = _DESCRIPTIONS.get(metric_type, "Ranking metric.")
        desc = desc_template.replace("{k}", str(k)) if k else desc_template

        g, gl = grade(val)
        entries.append(MetricEntry(
            name        = key,
            k           = k,
            value       = round(val, 4),
            grade       = g,
            grade_label = gl,
            description = desc,
        ))

    # overall grade = NDCG@primary_k if available, else mean of all
    ndcg_key = f"ndcg@{primary_k}"
    if ndcg_key in raw_metrics:
        overall_score = raw_metrics[ndcg_key]
    else:
        vals = [v for k, v in raw_metrics.items() if k != "num_queries"]
        overall_score = sum(vals) / len(vals) if vals else 0.0

    og, ol = grade(overall_score)

    # human verdict
    if overall_score >= 0.90:
        verdict = "The ranking system is performing excellently. Top candidates are surfaced reliably."
    elif overall_score >= 0.75:
        verdict = "Good ranking quality. Most relevant candidates appear near the top."
    elif overall_score >= 0.55:
        verdict = "Fair performance. Some relevant candidates are ranked lower than ideal."
    elif overall_score >= 0.35:
        verdict = "Poor ranking. Consider retraining the reranker or improving the embedding model."
    else:
        verdict = "Very poor ranking. The system is not reliably identifying relevant candidates."

    return MetricsReport(
        entries       = entries,
        overall_grade = og,
        overall_label = ol,
        verdict       = verdict,
        n_relevant    = n_relevant,
        n_ranked      = n_ranked,
        primary_k     = primary_k,
    )