"""Test BYOK LLM key storage + encryption."""

from __future__ import annotations

import pytest


@pytest.fixture
def auth_client(client, request):
    email = f"{request.node.name}@example.com"
    r = client.post("/auth/signup",
                     json={"email": email, "password": "password1"})
    assert r.status_code == 200, r.text
    return client


def test_add_openai_key_stores_encrypted(auth_client):
    fake = "sk-" + "a" * 50
    r = auth_client.post(
        "/api/v1/llm-keys",
        json={"provider": "openai", "plaintext_key": fake, "label": "test"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["provider"] == "openai"
    assert body["key_suffix"].endswith(fake[-4:])
    # Plaintext must NOT appear in the listing response
    assert fake not in r.text


def test_add_anthropic_key_validates_prefix(auth_client):
    r = auth_client.post(
        "/api/v1/llm-keys",
        json={"provider": "anthropic", "plaintext_key": "sk-wrong-prefix-" + "a"*30},
    )
    assert r.status_code == 400
    assert "sk-ant-" in r.json()["detail"]


def test_add_openai_key_validates_prefix(auth_client):
    r = auth_client.post(
        "/api/v1/llm-keys",
        json={"provider": "openai", "plaintext_key": "pk-" + "a"*30},
    )
    assert r.status_code == 400


def test_reject_unknown_provider(auth_client):
    r = auth_client.post(
        "/api/v1/llm-keys",
        json={"provider": "gemini", "plaintext_key": "sk-" + "a"*30},
    )
    assert r.status_code == 400


def test_list_and_revoke_llm_key(auth_client):
    fake = "sk-ant-" + "b" * 60
    r = auth_client.post(
        "/api/v1/llm-keys",
        json={"provider": "anthropic", "plaintext_key": fake, "label": "anth"},
    )
    key_id = r.json()["id"]

    r = auth_client.get("/api/v1/llm-keys")
    assert r.status_code == 200
    assert len(r.json()) == 1

    r = auth_client.delete(f"/api/v1/llm-keys/{key_id}")
    assert r.status_code == 204


def test_encryption_roundtrip_module():
    """Verify Fernet can encrypt + decrypt → same plaintext."""
    from app.crypto import encrypt_llm_key, decrypt_llm_key
    pt = "sk-test-roundtrip-" + "x" * 30
    ct = encrypt_llm_key(pt)
    assert ct != pt                              # ciphertext differs
    assert decrypt_llm_key(ct) == pt             # roundtrip OK


def test_decrypt_rejects_tampered_ciphertext():
    from app.crypto import decrypt_llm_key
    with pytest.raises(ValueError):
        decrypt_llm_key("gAAAAAB-not-valid-ciphertext")
