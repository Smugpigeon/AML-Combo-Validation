"""Auth primitives: password hashing, JWT sessions, API key minting."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt

from app.config import get_settings


API_KEY_PREFIX = "ac_live_"
API_KEY_RANDOM_BYTES = 32   # 256-bit entropy


# ---------------------------------------------------------------------------
# Passwords (bcrypt)
# ---------------------------------------------------------------------------


def hash_password(plaintext: str) -> str:
    return bcrypt.hashpw(plaintext.encode(), bcrypt.gensalt()).decode()


def verify_password(plaintext: str, stored_hash: str) -> bool:
    try:
        return bcrypt.checkpw(plaintext.encode(), stored_hash.encode())
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# JWT (for web session cookies)
# ---------------------------------------------------------------------------


def create_session_token(user_id: str) -> str:
    s = get_settings()
    payload = {
        "sub": str(user_id),
        "iat": datetime.now(tz=timezone.utc),
        "exp": (datetime.now(tz=timezone.utc)
                + timedelta(hours=s.JWT_EXPIRATION_HOURS)),
    }
    return jwt.encode(payload, s.JWT_SECRET, algorithm=s.JWT_ALGORITHM)


def decode_session_token(token: str) -> Optional[str]:
    """Return user_id string if valid, else None."""
    s = get_settings()
    try:
        payload = jwt.decode(token, s.JWT_SECRET, algorithms=[s.JWT_ALGORITHM])
        return payload.get("sub")
    except jwt.InvalidTokenError:
        return None


# ---------------------------------------------------------------------------
# API keys (ac_live_<32 bytes base64>)
# ---------------------------------------------------------------------------


def mint_api_key() -> tuple[str, str, str]:
    """Generate a new API key.

    Returns (plaintext_key, prefix, sha256_hash). Only plaintext is shown
    ONCE to the user; only the hash is stored server-side.
    """
    raw = secrets.token_urlsafe(API_KEY_RANDOM_BYTES)
    plaintext = API_KEY_PREFIX + raw
    prefix = plaintext[:16]  # "ac_live_" + first 8 chars of raw
    key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
    return plaintext, prefix, key_hash


def hash_api_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode()).hexdigest()


def api_key_matches(plaintext: str, stored_hash: str) -> bool:
    """Constant-time comparison of sha256(plaintext) and stored_hash."""
    return hmac.compare_digest(hashlib.sha256(plaintext.encode()).hexdigest(),
                                stored_hash)
