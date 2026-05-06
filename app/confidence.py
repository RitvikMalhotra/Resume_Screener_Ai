"""
Phase 3 – Confidence Scoring

Assigns a confidence level to each ranked result based on:
  1. Absolute final_score value
  2. Gap between this result and the next (separation)
  3. Score distribution across all results (spread)

Confidence levels
-----------------
  HIGH       - strong signal, large gap from next result
  MEDIUM     - reasonable signal, moderate separation
  LOW        - weak signal, close to neighboring scores
  UNCERTAIN  - scores too close to distinguish reliably

Interview talking point:
  "We don't just return scores — we flag when the model is uncertain.
   If the top-2 candidates have scores 0.91 and 0.89, we label both
   UNCERTAIN because a 0.02 gap is within noise. The recruiter should
   review both rather than trusting the ranking blindly."
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


# ── Thresholds ─────────────────────────────────────────────────────────────

# Minimum final_score to be considered meaningful
_SCORE_THRESHOLDS = {
    "high":      0.70,
    "medium":    0.40,
    "low":       0.15,
}

# Minimum gap to the next result to be considered clearly separated
_GAP_THRESHOLD = 0.08

# If std dev of all scores is below this, everything is "uncertain"
_SPREAD_THRESHOLD = 0.05


@dataclass
class ConfidenceResult:
    level: str          # "high" | "medium" | "low" | "uncertain"
    label: str          # human-readable
    reason: str         # plain English explanation
    score_gap: float    # gap to next result (0 if last)
    is_clear_winner: bool

    def to_dict(self) -> dict:
        return {
            "level":          self.level,
            "label":          self.label,
            "reason":         self.reason,
            "score_gap":      round(self.score_gap, 4),
            "is_clear_winner": self.is_clear_winner,
        }


def _level_label(level: str) -> str:
    return {
        "high":      "High Confidence",
        "medium":    "Medium Confidence",
        "low":       "Low Confidence",
        "uncertain": "Uncertain",
    }.get(level, "Unknown")


def score_confidence(final_scores: list[float]) -> list[ConfidenceResult]:
    """
    Given a list of final_scores in rank order (best first),
    return a ConfidenceResult for each.
    """
    if not final_scores:
        return []

    n      = len(final_scores)
    arr    = np.array(final_scores)
    spread = float(np.std(arr)) if n > 1 else 1.0

    results: list[ConfidenceResult] = []

    for i, score in enumerate(final_scores):
        gap = float(arr[i] - arr[i + 1]) if i < n - 1 else 0.0

        # low spread → all scores similar → uncertain across the board
        if spread < _SPREAD_THRESHOLD:
            level  = "uncertain"
            reason = (
                f"All scores are within a narrow band (std={spread:.3f}). "
                "The model cannot reliably distinguish these candidates."
            )

        elif score >= _SCORE_THRESHOLDS["high"] and gap >= _GAP_THRESHOLD:
            level  = "high"
            reason = (
                f"Strong score ({score:.2f}) with a clear gap "
                f"of {gap:.2f} to the next candidate."
            )

        elif score >= _SCORE_THRESHOLDS["medium"]:
            if gap < _GAP_THRESHOLD and i < n - 1:
                level  = "uncertain"
                reason = (
                    f"Score is reasonable ({score:.2f}) but the gap to the "
                    f"next candidate is only {gap:.2f} — too close to be certain."
                )
            else:
                level  = "medium"
                reason = (
                    f"Moderate score ({score:.2f}). "
                    "Candidate is likely relevant but not a standout match."
                )

        elif score >= _SCORE_THRESHOLDS["low"]:
            level  = "low"
            reason = (
                f"Weak score ({score:.2f}). "
                "Candidate has limited alignment with the job description."
            )

        else:
            level  = "uncertain"
            reason = (
                f"Very low score ({score:.2f}). "
                "Result may be a false positive from retrieval stage."
            )

        is_clear_winner = (
            i == 0
            and score >= _SCORE_THRESHOLDS["high"]
            and gap >= _GAP_THRESHOLD * 2
        )

        results.append(ConfidenceResult(
            level          = level,
            label          = _level_label(level),
            reason         = reason,
            score_gap      = gap,
            is_clear_winner = is_clear_winner,
        ))

    return results