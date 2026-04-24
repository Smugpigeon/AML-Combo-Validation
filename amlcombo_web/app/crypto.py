"""Symmetric encryption for user LLM API keys (BYOK).

We use Fernet (AES-128-CBC + HMAC-SHA256) with a single server-side key
derived from `FERNET_KEY` env var. All ciphertext is self-authenticated
and timestamped, so we can reject keys that were encrypted under an old
rotated server key.

Security model:
  - Server admin has physical access to FERNET_KEY and could decrypt any
    stored user key.
  - Database dump alone is NOT sufficient to recover user keys (key is
    kept separate, not in DB).
  - Users are told up-front: "we don't promise zero-knowledge. If you
    need that, self-host or proxy through your own LLM gateway."
"""

from __future__ import annotations

from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings


@lru_cache
def _fernet() -> Fernet:
    key = get_settings().FERNET_KEY
    if isinstance(key, str):
        key = key.encode()
    return Fernet(key)


def encrypt_llm_key(plaintext: str) -> str:
    """Encrypt plaintext to a urlsafe-base64 Fernet token."""
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_llm_key(ciphertext: str) -> str:
    """Decrypt a token back to plaintext. Raises ValueError on tamper/
    wrong-key/malformed input (we wrap Fernet's InvalidToken for clarity)."""
    try:
        return _fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken as e:
        raise ValueError("Could not decrypt LLM key (tampered or wrong server key)") from e
