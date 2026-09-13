"""
Postgres access for user accounts + screening history.

Replaces the Supabase-backed auth/data layer the frontend used to talk to
directly. A plain psycopg connection per call is used deliberately -- Neon's
injected DATABASE_URL already routes through PgBouncer, so pooling is handled
upstream and there's no long-lived process to keep a pool warm across
serverless invocations.
"""
from __future__ import annotations
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
    _PSYCOPG = True
except ImportError:
    _PSYCOPG = False
    logger.warning("psycopg not installed -- auth/database features disabled")


def is_configured() -> bool:
    return bool(DATABASE_URL) and _PSYCOPG


def get_connection():
    if not is_configured():
        raise RuntimeError("DATABASE_URL is not set (or psycopg isn't installed)")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def init_schema() -> None:
    """Idempotently create the tables this app needs. Safe to call on every cold start."""
    if not is_configured():
        return
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id               SERIAL PRIMARY KEY,
                email            TEXT UNIQUE NOT NULL,
                password_hash    TEXT NOT NULL,
                full_name        TEXT,
                plan             TEXT NOT NULL DEFAULT 'free',
                screenings_used  INTEGER NOT NULL DEFAULT 0,
                screenings_limit INTEGER NOT NULL DEFAULT 5,
                created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS screenings (
                id              SERIAL PRIMARY KEY,
                user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                jd_snippet      TEXT,
                candidate_count INTEGER,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS payment_orders (
                order_id   TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                amount     INTEGER NOT NULL,
                status     TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_cache (
                key        TEXT PRIMARY KEY,
                value      JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        conn.commit()
    logger.info("Database schema ready")


_PUBLIC_USER_COLUMNS = "id, email, full_name, plan, screenings_used, screenings_limit, created_at"


def create_user(email: str, password_hash: str, full_name: str) -> dict[str, Any]:
    with get_connection() as conn:
        row = conn.execute(
            f"INSERT INTO users (email, password_hash, full_name) VALUES (%s, %s, %s) "
            f"RETURNING {_PUBLIC_USER_COLUMNS}",
            (email, password_hash, full_name),
        ).fetchone()
        conn.commit()
    return row


def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, email, password_hash, full_name, plan, screenings_used, screenings_limit, created_at "
            "FROM users WHERE email = %s",
            (email,),
        ).fetchone()


def get_user_by_id(user_id: int) -> Optional[dict[str, Any]]:
    with get_connection() as conn:
        return conn.execute(
            f"SELECT {_PUBLIC_USER_COLUMNS} FROM users WHERE id = %s",
            (user_id,),
        ).fetchone()


def create_payment_order(order_id: str, user_id: int, amount: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO payment_orders (order_id, user_id, amount) VALUES (%s, %s, %s)",
            (order_id, user_id, amount),
        )
        conn.commit()


def claim_payment_order(order_id: str, user_id: int) -> bool:
    """
    Mark an order paid, but only if it belongs to this user and is still
    pending. The conditional UPDATE is what makes this safe: a replayed
    payment (or one pointed at someone else's account) matches no row and
    returns False rather than granting a second upgrade.
    """
    with get_connection() as conn:
        row = conn.execute(
            "UPDATE payment_orders SET status = 'paid' "
            "WHERE order_id = %s AND user_id = %s AND status = 'pending' "
            "RETURNING order_id",
            (order_id, user_id),
        ).fetchone()
        conn.commit()
    return row is not None


def upgrade_to_pro(user_id: int) -> dict[str, Any]:
    with get_connection() as conn:
        row = conn.execute(
            f"UPDATE users SET plan = 'pro', screenings_limit = 999999 WHERE id = %s "
            f"RETURNING {_PUBLIC_USER_COLUMNS}",
            (user_id,),
        ).fetchone()
        conn.commit()
    return row


def record_screening(user_id: int, jd_snippet: str, candidate_count: int) -> dict[str, Any]:
    """Insert a screening row and bump the user's usage counter. Returns the updated user."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO screenings (user_id, jd_snippet, candidate_count) VALUES (%s, %s, %s)",
            (user_id, jd_snippet[:200], candidate_count),
        )
        row = conn.execute(
            f"UPDATE users SET screenings_used = screenings_used + 1 WHERE id = %s "
            f"RETURNING {_PUBLIC_USER_COLUMNS}",
            (user_id,),
        ).fetchone()
        conn.commit()
    return row


def get_ai_cache(key: str) -> Optional[Any]:
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM ai_cache WHERE key = %s", (key,)).fetchone()
    return row["value"] if row else None


def put_ai_cache(key: str, value: Any) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO ai_cache (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING",
            (key, Jsonb(value)),
        )
        conn.commit()


def list_screenings(user_id: int, limit: int = 10) -> list[dict[str, Any]]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT jd_snippet, candidate_count, created_at FROM screenings "
            "WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        ).fetchall()
