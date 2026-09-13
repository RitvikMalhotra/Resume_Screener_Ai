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
import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Callable, Optional

import httpx

logger = logging.getLogger(__name__)

API_KEY = os.getenv("NVIDIA_API_KEY", "")
API_URL = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions")

# Default chosen by benchmarking the provider's catalogue on this app's own
# prompts: it answers a skill-extraction prompt in ~4s where meta/muse-glimmer-30b
# needed ~108s for the identical answer, mostly because it honours the
# thinking toggle below and muse-glimmer ignores it.
MODEL = os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-flash-0731")

# These models reason before answering, and the reasoning is ~85% of everything
# they generate: 1011 completion tokens versus a 230-character answer. Turning it
# off returned the same answer from 67 tokens. Latency here is dominated by token
# count, so this is the single biggest lever on how fast the AI features feel.
# Models that don't support the flag ignore it; the few that reject it outright
# are handled by retrying without it.
DISABLE_THINKING = os.getenv("NVIDIA_DISABLE_THINKING", "true").lower() == "true"

# Reasoning models spend most of their budget inside <think>. Starving them is
# what makes `content` come back null, so these are deliberately generous.
DEFAULT_MAX_TOKENS = 1200
JSON_MAX_TOKENS    = 2000

# This provider's latency is wildly variable rather than uniformly slow:
# three identical requests measured 4.3s, 8.7s and 52.9s, with the same token
# counts and no rate-limit headers, and some runs exceed two minutes. Since a
# slow response is bad luck in the queue rather than a property of the request,
# several short attempts beat one long one -- a retry re-draws and usually
# lands fast, where a single long wait just rides out the bad draw.
# Measured successful responses land at 0.9s, 1.4s, 5.6s and 16.2s, so 18s is
# past the point where waiting longer is still likely to pay off -- a request
# still open then has drawn a bad queue slot and is usually minutes away.
# 5 x 18s = 90s of draws instead of 3 x 40s, which spent the entire serverless
# budget riding out two bad draws. TOTAL_BUDGET_S stays under the ~120s
# execution limit so a total failure returns our own message, not a 502.
REQUEST_TIMEOUT_S = float(os.getenv("NVIDIA_TIMEOUT", "18"))
MAX_ATTEMPTS      = int(os.getenv("NVIDIA_MAX_ATTEMPTS", "5"))
TOTAL_BUDGET_S    = float(os.getenv("NVIDIA_TOTAL_BUDGET", "95"))

# Slow calls here are not slow, they are stuck: measured on the live site,
# the long waits were 18s+2s and 18s+18s+1s -- a request that never answered,
# the full timeout, then a retry that answered in a second or two. Healthy
# short-JSON calls land in 0.8-3s. So once a call has run past this point, a
# duplicate is sent and whichever answers first wins, instead of sitting out
# the rest of the timeout.
HEDGE_AFTER_S = float(os.getenv("NVIDIA_HEDGE_AFTER", "5"))


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

async def _post_once(payload: dict, headers: dict, timeout: float) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.post(API_URL, headers=headers, json=payload)


async def _post_hedged(payload: dict, headers: dict, timeout: float, hedge_after: float) -> httpx.Response:
    """
    Send the request; if it hasn't answered after `hedge_after` seconds, send one
    duplicate and return the first 200. At most two copies are ever in flight:
    the provider throttles concurrency per key, and firing more measured slower.
    """
    first = asyncio.create_task(_post_once(dict(payload), headers, timeout))
    if hedge_after <= 0 or hedge_after >= timeout:
        return await first

    done, _ = await asyncio.wait({first}, timeout=hedge_after)
    if done:
        return first.result()

    logger.info("LLM call still open after %.1fs; sending a hedge request", hedge_after)
    # Only the rest of the original window, so an attempt where both copies
    # hang ends exactly when it would have without hedging.
    second = asyncio.create_task(_post_once(dict(payload), headers, timeout - hedge_after))
    pending = {first, second}
    last_res: Optional[httpx.Response] = None
    last_exc: Optional[BaseException] = None
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    res = task.result()
                except Exception as exc:
                    last_exc = exc
                    continue
                if res.status_code == 200:
                    return res
                last_res = res
        if last_res is not None:
            return last_res
        raise last_exc
    finally:
        for task in (first, second):
            if not task.done():
                task.cancel()


async def call_llm(
    prompt: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.3,
    system: Optional[str] = None,
    timeout: Optional[float] = None,
    attempts: Optional[int] = None,
    hedge_after: Optional[float] = None,
) -> str:
    """Run a prompt and return non-empty text, or raise LLMError."""
    if not is_configured():
        raise LLMError("AI features are not configured on this deployment (missing NVIDIA_API_KEY).")

    timeout     = REQUEST_TIMEOUT_S if timeout is None else timeout
    attempts    = MAX_ATTEMPTS if attempts is None else attempts
    hedge_after = HEDGE_AFTER_S if hedge_after is None else hedge_after

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
    if DISABLE_THINKING:
        payload["chat_template_kwargs"] = {"thinking": False}
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = "unknown error"
    deadline = time.monotonic() + min(timeout * attempts, TOTAL_BUDGET_S)

    for attempt in range(1, attempts + 1):
        remaining = deadline - time.monotonic()
        if attempt > 1 and remaining < 10:
            logger.warning("LLM giving up: %.0fs left of the total budget", remaining)
            break

        try:
            res = await _post_hedged(payload, headers, min(timeout, max(remaining, 5)), hedge_after)

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
                # Some models reject the thinking toggle rather than ignoring
                # it; drop it and try once more before giving up.
                if "chat_template_kwargs" in payload:
                    last_error = f"provider returned {res.status_code} for the thinking toggle"
                    logger.warning(
                        "Provider returned %s; retrying without the thinking toggle", res.status_code
                    )
                    payload.pop("chat_template_kwargs")
                    continue
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
            # Retried on purpose: latency here is queue luck, not a property of
            # the request, so a fresh attempt usually draws a much faster one.
            last_error = "timed out"
            logger.warning("LLM attempt %d/%d timed out after %.0fs", attempt, attempts, timeout)
            if attempt < attempts:
                continue
            raise LLMError(
                "The AI provider is responding too slowly right now. Try again in a moment."
            )
        except httpx.HTTPError as exc:
            last_error = str(exc)
            logger.warning("LLM attempt %d/%d transport error: %s", attempt, attempts, exc)
            if attempt < attempts:
                continue
            raise LLMError("Could not reach the AI provider.")

    raise LLMError(f"The AI request failed: {last_error}.")


async def diagnose(
    prompt: str = "Reply with the single word: ok",
    max_tokens: int = 400,
    system: Optional[str] = None,
    no_think: bool = False,
    timeout: float = 120.0,
    model: Optional[str] = None,
) -> dict:
    """
    Self-test for the AI dependency: does one controlled call and reports what
    came back, including token usage and finish_reason. Exists because latency
    on a reasoning model is dominated by how many tokens it generates before
    answering, which is otherwise invisible from the outside.
    """
    if not is_configured():
        return {"ok": False, "error": "NVIDIA_API_KEY is not set"}

    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    payload = {
        "model": model or MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": False,
    }
    if no_think:
        # Some NIM reasoning models expose a toggle for the thinking phase.
        payload["chat_template_kwargs"] = {"thinking": False}
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            res = await client.post(API_URL, headers=headers, json=payload)
    except Exception as exc:
        return {"ok": False, "elapsed_s": round(time.perf_counter() - t0, 1),
                "error": f"{type(exc).__name__}: {exc}"}

    elapsed = round(time.perf_counter() - t0, 1)
    if res.status_code != 200:
        return {"ok": False, "elapsed_s": elapsed, "http_status": res.status_code,
                "error": res.text[:300]}

    # Provider-side throttling shows up here long before it's obvious from
    # latency alone.
    rate_headers = {
        k: v for k, v in res.headers.items()
        if "ratelimit" in k.lower() or "retry-after" in k.lower() or "x-request-id" == k.lower()
    }

    data = res.json()
    choice  = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content   = _content_to_text(message.get("content"))
    reasoning = _content_to_text(message.get("reasoning_content"))
    answer    = strip_thinking(content or reasoning)

    return {
        "ok": bool(answer),
        "elapsed_s": elapsed,
        "model": model or MODEL,
        "max_tokens_requested": max_tokens,
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "rate_headers": rate_headers,
        "answer_preview": answer[:200],
    }


async def list_models() -> dict:
    """Ask the provider which models this key can actually use."""
    if not is_configured():
        return {"ok": False, "error": "NVIDIA_API_KEY is not set"}

    url = API_URL.split("/chat/completions")[0].rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.get(url, headers={"Authorization": f"Bearer {API_KEY}"})
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if res.status_code != 200:
        return {"ok": False, "http_status": res.status_code, "error": res.text[:300], "url": url}

    data = res.json()
    ids = sorted(m.get("id", "") for m in data.get("data", []))
    return {"ok": True, "count": len(ids), "models": ids, "current": MODEL}


async def call_llm_json(
    prompt: str,
    *,
    max_tokens: int = JSON_MAX_TOKENS,
    temperature: float = 0.1,
    system: Optional[str] = None,
    timeout: Optional[float] = None,
    attempts: Optional[int] = None,
    hedge_after: Optional[float] = None,
    cache: bool = False,
    valid: Optional[Callable[[Any], bool]] = None,
) -> Any:
    """
    Run a prompt that must return JSON, and parse it defensively.

    cache=True reuses a stored answer for an identical model+prompt. Only for
    analysis, where the same resume and JD should give the same answer anyway.
    `valid` gates what is stored, so one malformed answer can't be replayed forever.
    """
    key = _cache_key(system, prompt) if cache else None
    if key:
        hit = await _cache_get(key)
        if hit is not None and (valid is None or valid(hit)):
            return hit

    raw = await call_llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        timeout=timeout,
        attempts=attempts,
        hedge_after=hedge_after,
    )
    result = extract_json(raw)
    if key and (valid is None or valid(result)):
        await _cache_put(key, result)
    return result


def _cache_key(system: Optional[str], prompt: str) -> Optional[str]:
    from app import db
    if not db.is_configured():
        return None
    return hashlib.sha256(f"{MODEL}\x00{system or ''}\x00{prompt}".encode()).hexdigest()


# The cache is an optimisation: a database problem must never fail the AI call.
async def _cache_get(key: str) -> Any:
    from app import db
    try:
        return await asyncio.to_thread(db.get_ai_cache, key)
    except Exception as exc:
        logger.warning("AI cache read failed: %s", exc)
        return None


async def _cache_put(key: str, value: Any) -> None:
    from app import db
    try:
        await asyncio.to_thread(db.put_ai_cache, key, value)
    except Exception as exc:
        logger.warning("AI cache write failed: %s", exc)
