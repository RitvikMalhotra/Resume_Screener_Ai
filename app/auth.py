"""
Password hashing + JWT session tokens.

Replaces Supabase Auth. Tokens are bearer tokens returned in the response body
and sent back as `Authorization: Bearer <token>` -- simpler than cookie-based
sessions for a static-file frontend, at the cost of the token being readable
by any JS on the page (accepted tradeoff here; there's no XSS-sensitive
third-party script running on these pages).
"""
from __future__ import annotations
import logging
import os
import time
from typing import Optional

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

JWT_SECRET    = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days

try:
    import bcrypt
    _BCRYPT = True
except ImportError:
    _BCRYPT = False
    logger.warning("bcrypt not installed -- signup/login disabled")

try:
    import jwt as _pyjwt
    _JWT = True
except ImportError:
    _JWT = False
    logger.warning("PyJWT not installed -- signup/login disabled")


def auth_configured() -> bool:
    return _BCRYPT and _JWT and bool(JWT_SECRET)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_token(user_id: int, email: str) -> str:
    payload = {"sub": str(user_id), "email": email, "exp": int(time.time()) + JWT_TTL_SECONDS}
    return _pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _extract_bearer_token(request: Request) -> Optional[str]:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    return header[len("Bearer "):].strip()


def get_user_id_from_request(request: Request) -> Optional[int]:
    """Best-effort auth: returns the user id if a valid token is present, else None."""
    if not auth_configured():
        return None
    token = _extract_bearer_token(request)
    if not token:
        return None
    try:
        payload = _pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return int(payload["sub"])
    except _pyjwt.PyJWTError:
        return None


def require_user_id(request: Request) -> int:
    """Strict auth for endpoints that must have a logged-in user."""
    user_id = get_user_id_from_request(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user_id
