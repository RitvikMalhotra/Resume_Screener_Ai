"""
Phase 3 – Keyword Fallback Heuristic

When the embedding + reranker pipeline produces low-confidence results,
this module provides a keyword-overlap fallback score as a safety net.

Why this matters
----------------
  LLMs and embeddings can fail on:
    - Very short resumes (< 100 words)
    - Domain-specific jargon not in training data
    - Resumes in non-standard formats

  In these cases, simple keyword overlap is more reliable than
  a confused embedding model.

Scoring method
--------------
  1. Extract keywords from JD (nouns, skills, tools)
  2. Count how many appear in the resume
  3. Normalize by total JD keywords → overlap score [0, 1]
  4. Weighted by keyword importance (title keywords > generic words)

Interview talking point:
  "Our fallback uses TF-IDF-weighted keyword overlap. If the neural
   pipeline scores a candidate below 0.15, we re-score using keyword
   matching and take the max of the two. This prevents good candidates
   with unconventional resume formats from being filtered out."
"""
from __future__ import annotations
import re
import string
from dataclasses import dataclass


# ── Common stop words to ignore ────────────────────────────────────────────

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "we", "our", "you", "your", "they",
    "their", "this", "that", "these", "those", "it", "its", "as", "up",
    "experience", "years", "work", "working", "team", "ability", "strong",
    "good", "excellent", "great", "looking", "seeking", "join", "role",
    "position", "candidate", "required", "preferred", "plus", "bonus",
}

# High-value technical keywords get a score multiplier
_HIGH_VALUE_PATTERNS = [
    r'\b(python|pytorch|tensorflow|jax)\b',
    r'\b(llm|llms|gpt|bert|llama|mistral|claude)\b',
    r'\b(lora|qlora|rlhf|sft|peft|rag)\b',
    r'\b(faiss|pinecone|weaviate|chroma|qdrant)\b',
    r'\b(docker|kubernetes|k8s|aws|gcp|azure)\b',
    r'\b(transformer|transformers|attention|fine.tun)\b',
    r'\b(fastapi|flask|django|uvicorn)\b',
    r'\b(sql|postgresql|mysql|mongodb|redis)\b',
    r'\b(git|github|ci.cd|devops|mlops)\b',
    r'\b(react|typescript|javascript|nodejs)\b',
]


def _extract_keywords(text: str) -> dict[str, float]:
    """
    Extract keywords from text with importance weights.
    Returns {keyword: weight} where weight >= 1.0
    """
    text_lower = text.lower()

    # remove punctuation except hyphens (keep "fine-tuning")
    cleaned = re.sub(r'[^\w\s\-]', ' ', text_lower)
    tokens  = [
        t.strip('-').strip()
        for t in cleaned.split()
        if t.strip('-') and t.strip('-') not in _STOP_WORDS and len(t) > 2
    ]

    keywords: dict[str, float] = {}
    for token in tokens:
        keywords[token] = keywords.get(token, 1.0)

    # boost high-value technical terms
    for pattern in _HIGH_VALUE_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            word = match.group(0).strip()
            if word in keywords:
                keywords[word] = max(keywords[word], 2.5)
            else:
                keywords[word] = 2.5

    return keywords


@dataclass
class FallbackScore:
    score: float                    # normalized overlap [0, 1]
    matched_keywords: list[str]     # which keywords matched
    total_jd_keywords: int
    coverage_pct: float             # % of JD keywords found in resume
    used_as_fallback: bool          # True if this replaced neural score

    def to_dict(self) -> dict:
        return {
            "score":             round(self.score, 4),
            "matched_keywords":  self.matched_keywords[:20],  # cap at 20
            "total_jd_keywords": self.total_jd_keywords,
            "coverage_pct":      round(self.coverage_pct, 1),
            "used_as_fallback":  self.used_as_fallback,
        }


class KeywordFallback:
    """
    Keyword-overlap fallback scorer.

    Usage
    -----
    fb = KeywordFallback(threshold=0.15)
    score = fb.score(jd_text, resume_text)

    if fb.should_use_fallback(neural_score):
        final = fb.blend(neural_score, fallback_score.score)
    """

    def __init__(self, threshold: float = 0.15, blend_alpha: float = 0.4):
        """
        Parameters
        ----------
        threshold   : neural scores below this trigger fallback
        blend_alpha : weight of neural score in blended result
                      final = alpha * neural + (1-alpha) * keyword
        """
        self.threshold   = threshold
        self.blend_alpha = blend_alpha

    def should_use_fallback(self, neural_score: float) -> bool:
        return neural_score < self.threshold

    def score(
        self,
        jd_text: str,
        resume_text: str,
        neural_score: float = 0.0,
    ) -> FallbackScore:
        """
        Compute keyword overlap score between JD and resume.
        """
        jd_keywords     = _extract_keywords(jd_text)
        resume_keywords = _extract_keywords(resume_text)

        if not jd_keywords:
            return FallbackScore(
                score=0.0, matched_keywords=[],
                total_jd_keywords=0, coverage_pct=0.0,
                used_as_fallback=False,
            )

        matched: list[str] = []
        weighted_hits  = 0.0
        total_weight   = sum(jd_keywords.values())

        for kw, weight in jd_keywords.items():
            # check exact match or substring (e.g. "fine-tun" matches "fine-tuning")
            if kw in resume_keywords or any(kw in rk for rk in resume_keywords):
                matched.append(kw)
                weighted_hits += weight

        raw_score    = weighted_hits / total_weight if total_weight > 0 else 0.0
        coverage_pct = 100 * len(matched) / len(jd_keywords)

        # cap at 1.0
        score = min(raw_score, 1.0)

        used = self.should_use_fallback(neural_score)
        return FallbackScore(
            score             = score,
            matched_keywords  = sorted(matched),
            total_jd_keywords = len(jd_keywords),
            coverage_pct      = coverage_pct,
            used_as_fallback  = used,
        )

    def blend(self, neural_score: float, keyword_score: float) -> float:
        """
        Blend neural and keyword scores.
        Only called when neural_score < threshold.
        """
        return self.blend_alpha * neural_score + (1 - self.blend_alpha) * keyword_score


def apply_fallback(
    jd_text: str,
    results: list[dict],
    fallback: KeywordFallback | None = None,
) -> list[dict]:
    """
    Apply fallback scoring to a list of result dicts.
    Each dict must have keys: resume_text, final_score.
    Mutates and returns the list.
    """
    fb = fallback or KeywordFallback()

    for r in results:
        neural_score = r.get("final_score", 0.0)
        if fb.should_use_fallback(neural_score):
            fs = fb.score(jd_text, r.get("resume_text", ""), neural_score)
            blended = fb.blend(neural_score, fs.score)
            r["final_score"]    = round(blended, 4)
            r["fallback"]       = fs.to_dict()
            r["fallback_used"]  = True
        else:
            r["fallback_used"]  = False
            r["fallback"]       = None

    return results