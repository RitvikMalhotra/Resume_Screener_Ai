"""
Resume + job description preprocessing.

Handles:
  - PDF / DOCX / plain-text input
  - HTML stripping, unicode normalization
  - Section-aware cleaning
  - Truncation to a safe token budget
"""
from __future__ import annotations
import re
import unicodedata
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ── optional heavy parsers (graceful degradation) ──────────────────────────
try:
    import fitz  # PyMuPDF
    _PYMUPDF = True
except ImportError:
    _PYMUPDF = False
    logger.warning("PyMuPDF not found – PDF parsing disabled")

try:
    from docx import Document as DocxDocument
    _DOCX = True
except ImportError:
    _DOCX = False
    logger.warning("python-docx not found – DOCX parsing disabled")

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False


# ── regex patterns ──────────────────────────────────────────────────────────
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_MULTI_SPACE   = re.compile(r"[ \t]{2,}")
_RE_BULLET        = re.compile(r"^[\s•◦▸▪‣●○\-\*]+", re.MULTILINE)
_RE_EMAIL         = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_RE_PHONE         = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_RE_URL           = re.compile(r"https?://\S+|www\.\S+")

# section headers commonly found in resumes
_SECTION_HEADERS = re.compile(
    r"(EXPERIENCE|EDUCATION|SKILLS|PROJECTS|CERTIFICATIONS|SUMMARY|OBJECTIVE"
    r"|PUBLICATIONS|AWARDS|LANGUAGES|REFERENCES|WORK HISTORY|EMPLOYMENT)",
    re.IGNORECASE,
)


class ResumePreprocessor:
    """
    Stateless transformer: raw bytes / str → clean text ready for embedding.

    Usage
    -----
    p = ResumePreprocessor(max_chars=8000, anonymize=False)
    text = p.process(raw_bytes, filename="resume.pdf")
    """

    def __init__(self, max_chars: int = 8000, anonymize: bool = False):
        self.max_chars = max_chars
        self.anonymize = anonymize  # strip PII before embedding

    # ── public API ─────────────────────────────────────────────────────────

    def process(
        self,
        content: str | bytes,
        filename: str = "",
    ) -> str:
        """Full pipeline: parse → clean → truncate."""
        raw_text = self._parse(content, filename)
        clean    = self._clean(raw_text)
        if self.anonymize:
            clean = self._strip_pii(clean)
        return clean[: self.max_chars]

    def process_job_description(self, jd: str) -> str:
        """Lighter cleaning for the job description."""
        clean = self._clean(jd)
        return clean[: self.max_chars]

    # ── parsing ────────────────────────────────────────────────────────────

    def _parse(self, content: str | bytes, filename: str) -> str:
        suffix = Path(filename).suffix.lower() if filename else ""

        if isinstance(content, str):
            return content

        # bytes path
        if suffix == ".pdf":
            return self._parse_pdf(content)
        if suffix in (".docx", ".doc"):
            return self._parse_docx(content)
        # fallback: assume utf-8 text
        return content.decode("utf-8", errors="replace")

    def _parse_pdf(self, data: bytes) -> str:
        if not _PYMUPDF:
            raise RuntimeError("PyMuPDF required for PDF parsing. pip install pymupdf")
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        return "\n".join(pages)

    def _parse_docx(self, data: bytes) -> str:
        if not _DOCX:
            raise RuntimeError("python-docx required. pip install python-docx")
        import io
        doc = DocxDocument(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    # ── cleaning ───────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        # strip HTML
        if _BS4 and "<" in text:
            text = BeautifulSoup(text, "html.parser").get_text(separator="\n")

        # unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # remove non-printable control chars (keep newlines/tabs)
        text = "".join(
            ch for ch in text
            if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
        )

        # normalize bullets to dashes
        text = _RE_BULLET.sub("- ", text)

        # collapse whitespace
        text = _RE_MULTI_SPACE.sub(" ", text)
        text = _RE_MULTI_NEWLINE.sub("\n\n", text)

        return text.strip()

    def _strip_pii(self, text: str) -> str:
        text = _RE_EMAIL.sub("[EMAIL]", text)
        text = _RE_PHONE.sub("[PHONE]", text)
        text = _RE_URL.sub("[URL]", text)
        return text


def extract_sections(text: str) -> dict[str, str]:
    """
    Split resume text into labelled sections.
    Returns dict like {"EXPERIENCE": "...", "EDUCATION": "...", ...}
    Useful for section-weighted scoring.
    """
    parts: dict[str, str] = {}
    current_label = "HEADER"
    current_lines: list[str] = []

    for line in text.splitlines():
        m = _SECTION_HEADERS.match(line.strip())
        if m:
            if current_lines:
                parts[current_label] = "\n".join(current_lines).strip()
            current_label = m.group(0).upper()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        parts[current_label] = "\n".join(current_lines).strip()

    return parts
