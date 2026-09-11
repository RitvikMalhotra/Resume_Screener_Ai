"""
Central client for the hosted LLM (NVIDIA API catalog).

Every AI feature in this app goes through here. It exists because the raw
chat-completions response needs a lot more defensive handling than a naive
`data["choices"][0]["message"]["content"]` gives you:

  - Reasoning models return `content: null` and put their output in
    `reasoning_content` when they exhaust max_tokens mid-thought.
  - Reasoning output is wrapped in <think>…</think>, and the closing tag is
    missing when the response was truncated.
  - `content` is sometimes a list of parts rather than a string.
  - Models asked for JSON wrap it in markdown fences, add a prose preamble,
    or get cut off before the closing brace.

Anything unrecoverable raises LLMError, which carries a message that is safe
to show a user -- callers should never surface a raw Python exception.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

API_KEY = os.getenv("NVIDIA_API_KEY", "")
API_URL = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
MODEL   = os.getenv("NVIDIA_MODEL", "meta/muse-glimmer-30b")

# Reasoning models spend most of their budget inside <think>. Starving them is
# what makes `content` come back null, so these are deliberately generous.
DEFAULT_MAX_TOKENS = 1200
JSON_MAX_TOKENS    = 2000

# This model is slow: a ~2000-token JSON answer measured 30-45s in production.
# The ceiling is the serverless execution limit, since overrunning that surfaces
# as a gateway error this app can't catch and degrade from -- so prefer one long
# attempt over several short ones (see _should_retry: timeouts aren't retried,
# because a slow model stays slow and a second attempt just burns the budget).
REQUEST_TIMEOUT_S = float(os.getenv("NVIDIA_TIMEOUT", "45"))
MAX_ATTEMPTS      = int(os.getenv("NVIDIA_MAX_ATTEMPTS", "2"))


class LLMError(RuntimeError):
    """Raised with a user-safe message when the model can't produce a usable answer."""


def is_configured() -> bool:
    return bool(API_KEY)


# ── Response parsing ────────────────────────────────────────────────────────

def _content_to_text(content: Any) -> str:
    """`content` is usually a string, but some APIs return a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
        return "".join(parts)
    return ""


def _extract_message_text(data: dict) -> str:
    """Pull usable text out of a chat-completions payload, or return ''."""
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}

    text = _content_to_text(message.get("content"))
    if text.strip():
        return text

    # Reasoning models put everything here when they run out of room for a
    # final answer, so it's the best available signal rather than nothing.
    return _content_to_text(message.get("reasoning_content"))


def strip_thinking(text: str) -> str:
    """
    Remove reasoning blocks. Handles three shapes:
      <think>…</think>answer   → answer
      …</think>answer          → answer   (opening tag lost to truncation)
      <think>… (no close)      → ''       (truncated mid-thought, no answer yet)
    """
    if not text:
        return ""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]

    # An unterminated opener means everything after it is reasoning, not answer.
    if "<think>" in text:
        text = text.split("<think>", 1)[0]

    return text.strip()


# ── JSON extraction ─────────────────────────────────────────────────────────

def _close_truncated(snippet: str) -> Optional[str]:
    """Best-effort repair of JSON cut off by max_tokens: close what's still open."""
    stack: list[str] = []
    in_string = False
    escaped = False

    for ch in snippet:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()

    if not stack:
        return None

    repaired = snippet
    if in_string:
        repaired += '"'
    # Drop a dangling "key": or trailing comma before closing.
    repaired = re.sub(r",\s*$", "", repaired)
    repaired = re.sub(r',\s*"[^"]*"\s*:\s*$', "", repaired)
    repaired = re.sub(r'"[^"]*"\s*:\s*$', "", repaired)
    repaired = re.sub(r",\s*$", "", repaired)

    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def _loads(text: str) -> Any:
    """
    Parse with strict=False so literal newlines and tabs inside strings are
    accepted. Models routinely emit multi-line values (an email body, say) with
    real newlines rather than \\n escapes, which strict JSON rejects outright.
    """
    return json.loads(text, strict=False)


def extract_json(raw: str) -> Any:
    """Pull a JSON value out of model output that may be fenced, prefixed, or truncated."""
    if not raw or not raw.strip():
        raise LLMError("The model returned an empty response.")

    text = raw.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        # Unterminated fence (truncated response).
        text = re.sub(r"^```(?:json)?\s*", "", text).strip()

    try:
        return _loads(text)
    except json.JSONDecodeError:
        pass

    start = min(
        (i for i in (text.find("{"), text.find("[")) if i != -1),
        default=-1,
    )
    if start == -1:
        raise LLMError("The model did not return JSON.")

    candidate = text[start:]

    try:
        return _loads(candidate)
    except json.JSONDecodeError:
        pass

    # Trailing prose after a complete object: decode just the first value.
    try:
        value, _ = json.JSONDecoder(strict=False).raw_decode(candidate)
        return value
    except json.JSONDecodeError:
        pass

    repaired = _close_truncated(candidate)
    if repaired:
        try:
            return _loads(repaired)
        except json.JSONDecodeError:
            pass

    raise LLMError("The model returned malformed JSON.")


# ── Calling ─────────────────────────────────────────────────────────────────

async def call_llm(
    prompt: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.3,
    system: Optional[str] = None,
    timeout: Optional[float] = None,
    attempts: Optional[int] = None,
) -> str:
    """Run a prompt and return non-empty text, or raise LLMError."""
    if not is_configured():
        raise LLMError("AI features are not configured on this deployment (missing NVIDIA_API_KEY).")

    timeout  = REQUEST_TIMEOUT_S if timeout is None else timeout
    attempts = MAX_ATTEMPTS if attempts is None else attempts

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = "unknown error"

    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.post(API_URL, headers=headers, json=payload)

            if res.status_code == 401:
                raise LLMError("The AI provider rejected the API key (401). Check NVIDIA_API_KEY.")
            if res.status_code == 404:
                raise LLMError(f"The AI provider does not recognise the model '{MODEL}' (404). Check NVIDIA_MODEL.")
            if res.status_code == 429 or res.status_code >= 500:
                last_error = f"provider returned {res.status_code}"
                logger.warning("LLM attempt %d/%d failed: %s", attempt, attempts, last_error)
                if attempt < attempts:
                    await asyncio.sleep(0.6 * attempt)
                    continue
                raise LLMError(f"The AI provider is unavailable right now ({res.status_code}). Try again shortly.")
            if res.status_code >= 400:
                raise LLMError(f"The AI provider rejected the request ({res.status_code}).")

            text = strip_thinking(_extract_message_text(res.json()))
            if text:
                return text

            # Empty almost always means the reasoning ate the whole budget.
            last_error = "model returned no usable content"
            logger.warning(
                "LLM attempt %d/%d returned empty content (max_tokens=%d)",
                attempt, attempts, max_tokens,
            )
            if attempt < attempts:
                payload["max_tokens"] = min(int(payload["max_tokens"] * 1.75), 4096)
                continue
            raise LLMError("The model returned an empty response. Try again or raise NVIDIA max tokens.")

        except httpx.TimeoutException:
            # Not retried: the model being slow isn't transient, and a second
            # attempt would just run out the serverless execution budget too.
            logger.warning("LLM attempt %d/%d timed out after %.0fs", attempt, attempts, timeout)
            raise LLMError(
                f"The AI provider took longer than {timeout:.0f}s to respond. Try again shortly."
            )
        except httpx.HTTPError as exc:
            last_error = str(exc)
            logger.warning("LLM attempt %d/%d transport error: %s", attempt, attempts, exc)
            if attempt < attempts:
                continue
            raise LLMError("Could not reach the AI provider.")

    raise LLMError(f"The AI request failed: {last_error}.")


async def call_llm_json(
    prompt: str,
    *,
    max_tokens: int = JSON_MAX_TOKENS,
    temperature: float = 0.1,
    system: Optional[str] = None,
    timeout: Optional[float] = None,
    attempts: Optional[int] = None,
) -> Any:
    """Run a prompt that must return JSON, and parse it defensively."""
    raw = await call_llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        timeout=timeout,
        attempts=attempts,
    )
    return extract_json(raw)
