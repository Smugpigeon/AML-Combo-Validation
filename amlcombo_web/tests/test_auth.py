"""Signup / login / logout flow tests."""

from __future__ import annotations


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_landing_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"AML" in r.content


def test_signup_json_creates_user_and_sets_cookie(client):
    r = client.post(
        "/auth/signup",
        json={"email": "a@example.com", "password": "password1",
              "full_name": "Alice", "institution": "UCSF"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["email"] == "a@example.com"
    # Session cookie set
    assert "session" in r.cookies or "session" in r.headers.get("set-cookie", "")


def test_signup_duplicate_rejects(client):
    client.post("/auth/signup",
                json={"email": "b@example.com", "password": "password1"})
    r = client.post("/auth/signup",
                     json={"email": "b@example.com", "password": "password1"})
    assert r.status_code == 409


def test_signup_short_password_rejected(client):
    r = client.post("/auth/signup",
                     json={"email": "short@example.com", "password": "short"})
    # Pydantic's min_length=8 catches this with 422
    assert r.status_code in (400, 422)


def test_login_correct_credentials(client):
    client.post("/auth/signup",
                json={"email": "c@example.com", "password": "password1"})
    r = client.post("/auth/login",
                     json={"email": "c@example.com", "password": "password1"})
    assert r.status_code == 200
    assert r.json()["email"] == "c@example.com"


def test_login_wrong_password_rejected(client):
    client.post("/auth/signup",
                json={"email": "d@example.com", "password": "password1"})
    r = client.post("/auth/login",
                     json={"email": "d@example.com", "password": "wrong"})
    assert r.status_code == 401


def test_dashboard_requires_auth(client):
    """Unauthenticated request to /dashboard → 401."""
    r = client.get("/dashboard", follow_redirects=False)
    assert r.status_code == 401
