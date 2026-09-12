"""
Razorpay order creation and payment verification.

The previous flow used Razorpay's quick checkout: the browser picked the
amount, paid, and then simply asked the API to mark the account Pro. Nothing
tied the upgrade to a real payment, so any logged-in user could POST to the
upgrade endpoint and get Pro for free -- and could have paid ₹1 for it by
editing the amount in devtools.

This module moves both decisions server-side. The order (and its price) is
created here, and an upgrade is only honoured when Razorpay's HMAC signature
over that order verifies against the key secret, which never reaches the
browser.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import uuid

import httpx

logger = logging.getLogger(__name__)

KEY_ID     = os.getenv("RAZORPAY_KEY_ID", "")
KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
ORDERS_URL = "https://api.razorpay.com/v1/orders"

# Price lives here, not in the client, so it can't be tampered with.
PRO_PLAN_AMOUNT_PAISE = int(os.getenv("PRO_PLAN_AMOUNT_PAISE", "99900"))  # ₹999
CURRENCY = "INR"

REQUEST_TIMEOUT_S = 20.0


class PaymentError(RuntimeError):
    """Raised with a user-safe message when an order or verification fails."""


def is_configured() -> bool:
    return bool(KEY_ID and KEY_SECRET)


async def create_order(user_id: int) -> dict:
    """Create a Razorpay order for the Pro plan. Returns the order payload."""
    if not is_configured():
        raise PaymentError("Payments are not configured on this deployment (missing RAZORPAY_KEY_ID/RAZORPAY_KEY_SECRET).")

    payload = {
        "amount": PRO_PLAN_AMOUNT_PAISE,
        "currency": CURRENCY,
        "receipt": f"pro-{user_id}-{uuid.uuid4().hex[:12]}",
        "notes": {"user_id": str(user_id), "plan": "pro"},
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
            res = await client.post(ORDERS_URL, auth=(KEY_ID, KEY_SECRET), json=payload)
    except httpx.HTTPError as exc:
        logger.warning("Razorpay order request failed: %s", exc)
        raise PaymentError("Could not reach the payment provider. Try again shortly.")

    if res.status_code == 401:
        raise PaymentError("The payment provider rejected the API credentials (401).")
    if res.status_code >= 400:
        logger.warning("Razorpay order rejected (%s): %s", res.status_code, res.text[:300])
        raise PaymentError(f"The payment provider rejected the order ({res.status_code}).")

    order = res.json()
    if not order.get("id"):
        raise PaymentError("The payment provider returned no order id.")
    return order


def verify_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """
    Razorpay signs `{order_id}|{payment_id}` with the key secret (HMAC-SHA256).
    Recomputing it here is what proves the payment actually happened, rather
    than trusting the browser's word for it.
    """
    if not is_configured() or not (order_id and payment_id and signature):
        return False

    expected = hmac.new(
        KEY_SECRET.encode("utf-8"),
        f"{order_id}|{payment_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature)
