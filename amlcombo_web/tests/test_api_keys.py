"""Test the API-key minting + use flow."""

from __future__ import annotations

import pytest


@pytest.fixture
def auth_client(client, request):
    """Client with a logged-in session cookie.

    Uses a per-test unique email so tests don't collide on the shared
    in-memory database.
    """
    email = f"{request.node.name}@example.com"
    r = client.post("/auth/signup",
                     json={"email": email, "password": "password1"})
    assert r.status_code == 200, r.text
    return client


def test_create_list_revoke_api_key(auth_client):
    # Create
    r = auth_client.post("/api/v1/keys", json={"label": "my-laptop"})
    assert r.status_code == 201
    body = r.json()
    assert body["plaintext_key"].startswith("ac_live_")
    assert body["prefix"].startswith("ac_live_")
    key = body["plaintext_key"]

    # List (should show one active)
    r = auth_client.get("/api/v1/keys")
    assert r.status_code == 200
    keys = r.json()
    assert len(keys) == 1
    assert keys[0]["label"] == "my-laptop"
    assert keys[0]["revoked_at"] is None

    # Revoke
    r = auth_client.delete(f"/api/v1/keys/{keys[0]['id']}")
    assert r.status_code == 204
    r = auth_client.get("/api/v1/keys?include_revoked=true")
    assert r.json()[0]["revoked_at"] is not None

    # Revoked key should no longer authenticate (use a fresh client w/o cookie)
    from fastapi.testclient import TestClient
    from app.main import app
    fresh = TestClient(app)
    r = fresh.get(
        "/api/v1/patients",
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 401


def test_api_key_authenticates_requests(auth_client):
    r = auth_client.post("/api/v1/keys", json={"label": "api"})
    key = r.json()["plaintext_key"]

    # Use the key to call /api/v1/patients without the cookie session
    # (we use a fresh client to drop the session cookie)
    from fastapi.testclient import TestClient
    from app.main import app
    fresh = TestClient(app)
    r = fresh.get("/api/v1/patients",
                  headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    assert r.json() == []  # no submissions yet
