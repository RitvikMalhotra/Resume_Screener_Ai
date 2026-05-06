"""
Phase 3 – Input Validator + Edge Case Handler

Catches bad inputs before they reach the pipeline:
  - Empty or whitespace-only resume text
  - Duplicate resume IDs
  - Very short resumes (< 50 words) — flagged, not rejected
  - Very long resumes — truncation warning
  - Empty job description
  - Suspiciously similar resumes (possible duplicates)

Returns structured warnings alongside results so the
caller knows exactly what was wrong without crashing.

Interview talking point:
  "We validate inputs at the API boundary and return structured
   warnings rather than crashing. A resume with 10 words doesn't
   get rejected — it gets a low_content warning and a reduced
   confidence score. The system degrades gracefully."
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


# ── Thresholds ─────────────────────────────────────────────────────────────

MIN_WORDS_WARNING  = 50     # warn if resume has fewer than this
MIN_WORDS_ERROR    = 5      # reject if resume has fewer than this
MAX_CHARS_WARNING  = 10000  # warn if resume exceeds this
MIN_JD_WORDS       = 20     # minimum words in job description
MAX_RESUME_SIMILARITY = 0.85  # jaccard threshold for duplicate detection


# ── Warning types ──────────────────────────────────────────────────────────

@dataclass
class ValidationWarning:
    resume_id: Optional[str]
    code: str       # machine-readable code
    severity: str   # "error" | "warning" | "info"
    message: str    # human-readable

    def to_dict(self) -> dict:
        return {
            "resume_id": self.resume_id,
            "code":      self.code,
            "severity":  self.severity,
            "message":   self.message,
        }


@dataclass
class ValidationResult:
    is_valid: bool
    warnings: list[ValidationWarning] = field(default_factory=list)
    cleaned_resumes: list[dict] = field(default_factory=list)
    rejected_ids: list[str] = field(default_factory=list)

    def has_errors(self) -> bool:
        return any(w.severity == "error" for w in self.warnings)

    def to_dict(self) -> dict:
        return {
            "is_valid":     self.is_valid,
            "warnings":     [w.to_dict() for w in self.warnings],
            "rejected_ids": self.rejected_ids,
            "n_warnings":   len(self.warnings),
            "n_errors":     sum(1 for w in self.warnings if w.severity == "error"),
        }


# ── Helpers ────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def _char_count(text: str) -> int:
    return len(text)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity between two texts."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union        = tokens_a | tokens_b
    return len(intersection) / len(union)


# ── Validator ──────────────────────────────────────────────────────────────

class InputValidator:
    """
    Validates job description + resume batch before pipeline execution.

    Usage
    -----
    validator = InputValidator()
    result = validator.validate(jd_text, resumes)

    if result.has_errors():
        # return 422 with result.to_dict()
    else:
        # proceed with result.cleaned_resumes
        # attach result.warnings to response
    """

    def __init__(
        self,
        min_words_warning: int  = MIN_WORDS_WARNING,
        min_words_error: int    = MIN_WORDS_ERROR,
        max_chars_warning: int  = MAX_CHARS_WARNING,
        check_duplicates: bool  = True,
    ):
        self.min_words_warning  = min_words_warning
        self.min_words_error    = min_words_error
        self.max_chars_warning  = max_chars_warning
        self.check_duplicates   = check_duplicates

    def validate(
        self,
        jd_text: str,
        resumes: list[dict],   # each: {"resume_id": str, "text": str}
    ) -> ValidationResult:
        warnings: list[ValidationWarning] = []
        cleaned:  list[dict] = []
        rejected: list[str]  = []

        # ── validate JD ───────────────────────────────────────────────────
        jd_warnings = self._validate_jd(jd_text)
        warnings.extend(jd_warnings)

        jd_errors = [w for w in jd_warnings if w.severity == "error"]
        if jd_errors:
            return ValidationResult(
                is_valid        = False,
                warnings        = warnings,
                cleaned_resumes = [],
                rejected_ids    = [],
            )

        # ── validate resumes ──────────────────────────────────────────────
        seen_ids: dict[str, int] = {}

        for resume in resumes:
            rid  = resume.get("resume_id", "").strip()
            text = resume.get("text", "").strip()

            resume_warnings, should_reject = self._validate_resume(rid, text, seen_ids)
            warnings.extend(resume_warnings)

            if should_reject:
                rejected.append(rid)
            else:
                seen_ids[rid] = len(cleaned)
                cleaned.append({
                    "resume_id": rid,
                    "text":      text,
                    "metadata":  resume.get("metadata", {}),
                    "word_count": _word_count(text),
                })

        # ── check for near-duplicate resumes ─────────────────────────────
        if self.check_duplicates and len(cleaned) > 1:
            dup_warnings = self._check_duplicates(cleaned)
            warnings.extend(dup_warnings)

        is_valid = len(cleaned) > 0

        return ValidationResult(
            is_valid        = is_valid,
            warnings        = warnings,
            cleaned_resumes = cleaned,
            rejected_ids    = rejected,
        )

    def _validate_jd(self, jd_text: str) -> list[ValidationWarning]:
        warnings = []
        if not jd_text.strip():
            warnings.append(ValidationWarning(
                resume_id = None,
                code      = "JD_EMPTY",
                severity  = "error",
                message   = "Job description is empty.",
            ))
            return warnings

        wc = _word_count(jd_text)
        if wc < MIN_JD_WORDS:
            warnings.append(ValidationWarning(
                resume_id = None,
                code      = "JD_TOO_SHORT",
                severity  = "warning",
                message   = (
                    f"Job description has only {wc} words. "
                    "Short JDs reduce ranking accuracy. "
                    "Consider adding requirements and responsibilities."
                ),
            ))
        return warnings

    def _validate_resume(
        self,
        rid: str,
        text: str,
        seen_ids: dict[str, int],
    ) -> tuple[list[ValidationWarning], bool]:
        warnings    = []
        should_reject = False

        # missing ID
        if not rid:
            warnings.append(ValidationWarning(
                resume_id = None,
                code      = "MISSING_ID",
                severity  = "error",
                message   = "Resume is missing a resume_id.",
            ))
            should_reject = True
            return warnings, should_reject

        # duplicate ID
        if rid in seen_ids:
            warnings.append(ValidationWarning(
                resume_id = rid,
                code      = "DUPLICATE_ID",
                severity  = "error",
                message   = f"Duplicate resume_id '{rid}'. Second occurrence rejected.",
            ))
            should_reject = True
            return warnings, should_reject

        # empty text
        if not text:
            warnings.append(ValidationWarning(
                resume_id = rid,
                code      = "EMPTY_TEXT",
                severity  = "error",
                message   = f"Resume '{rid}' has empty text and will be rejected.",
            ))
            should_reject = True
            return warnings, should_reject

        wc = _word_count(text)
        cc = _char_count(text)

        # too short to be useful
        if wc < self.min_words_error:
            warnings.append(ValidationWarning(
                resume_id = rid,
                code      = "TOO_SHORT_REJECT",
                severity  = "error",
                message   = (
                    f"Resume '{rid}' has only {wc} words (minimum {self.min_words_error}). "
                    "Too little content for meaningful ranking."
                ),
            ))
            should_reject = True
            return warnings, should_reject

        # short but not rejected
        if wc < self.min_words_warning:
            warnings.append(ValidationWarning(
                resume_id = rid,
                code      = "LOW_CONTENT",
                severity  = "warning",
                message   = (
                    f"Resume '{rid}' has only {wc} words. "
                    "Short resumes may score lower than their true fit. "
                    "Confidence will be reduced."
                ),
            ))

        # very long resume
        if cc > self.max_chars_warning:
            warnings.append(ValidationWarning(
                resume_id = rid,
                code      = "TRUNCATED",
                severity  = "info",
                message   = (
                    f"Resume '{rid}' has {cc} characters and will be "
                    f"truncated to {self.max_chars_warning} for embedding."
                ),
            ))

        return warnings, should_reject

    def _check_duplicates(
        self, resumes: list[dict]
    ) -> list[ValidationWarning]:
        warnings = []
        n = len(resumes)
        for i in range(n):
            for j in range(i + 1, n):
                sim = _jaccard_similarity(resumes[i]["text"], resumes[j]["text"])
                if sim >= MAX_RESUME_SIMILARITY:
                    warnings.append(ValidationWarning(
                        resume_id = resumes[i]["resume_id"],
                        code      = "NEAR_DUPLICATE",
                        severity  = "warning",
                        message   = (
                            f"Resumes '{resumes[i]['resume_id']}' and "
                            f"'{resumes[j]['resume_id']}' are {sim:.0%} similar. "
                            "They may be duplicate submissions."
                        ),
                    ))
        return warnings