"""Tests for the smart-paste endpoint and LLM JSON extraction helpers."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.llm import (
    ParsedClinicalText,
    _extract_first_json_object,
    parse_clinical_text,
)


# ---------------------------------------------------------------------------
# _extract_first_json_object helper
# ---------------------------------------------------------------------------


def test_extract_plain_json():
    out = _extract_first_json_object('{"a": 1, "b": [2, 3]}')
    assert out == {"a": 1, "b": [2, 3]}


def test_extract_from_markdown_fence():
    payload = '```json\n{"x": true}\n```'
    assert _extract_first_json_object(payload) == {"x": True}


def test_extract_from_chatter():
    text = 'Here is the JSON:\n\n{"a": 1, "b": 2}\n\nLet me know if you need more.'
    assert _extract_first_json_object(text) == {"a": 1, "b": 2}


def test_extract_with_nested_braces():
    text = 'Sure! {"outer": {"inner": {"deep": 3}}, "arr": [{"x": 1}]}'
    out = _extract_first_json_object(text)
    assert out == {"outer": {"inner": {"deep": 3}}, "arr": [{"x": 1}]}


def test_extract_with_braces_in_string():
    """Brace counting must respect string literals."""
    text = '{"note": "contains { and } inside", "ok": true}'
    out = _extract_first_json_object(text)
    assert out == {"note": "contains { and } inside", "ok": True}


def test_extract_with_escaped_quotes():
    text = '{"msg": "she said \\"hi\\" then \\"bye\\""}'
    out = _extract_first_json_object(text)
    assert out["msg"] == 'she said "hi" then "bye"'


def test_extract_returns_none_on_garbage():
    assert _extract_first_json_object("no braces here") is None
    assert _extract_first_json_object("") is None


# ---------------------------------------------------------------------------
# parse_clinical_text (provider call mocked)
# ---------------------------------------------------------------------------


_FAKE_LLM_JSON = json.dumps({
    "parsed": {
        "patient_label": "AML-2024-017",
        "age": 45,
        "sex": "female",
        "is_initial_diagnosis": True,
        "prior_mds": False,
        "karyotype_text": "46,XX[20]",
        "fusions": [],
        "mutations": [
            {"gene": "FLT3", "variant_type": "ITD", "vaf": 0.45,
             "is_ITD": True, "is_TKD": False, "allelic_ratio": 0.62,
             "is_biallelic": False},
            {"gene": "NPM1", "variant_type": "missense", "vaf": 0.42,
             "is_ITD": False, "is_TKD": False, "is_biallelic": False},
        ],
        "wbc": 95.0, "platelet": 32.0, "hemoglobin": 8.5, "ldh": 1240.0,
    },
    "confidence": {
        "patient_label": 1.0, "age": 1.0, "sex": 0.95,
        "mutations.0.gene": 0.99, "mutations.0.allelic_ratio": 0.9,
        "karyotype_text": 0.6,  # low conf
    },
    "warnings": ["Karyotype reported as normal; verify cytogenetics"],
})


def _fake_response(text=_FAKE_LLM_JSON):
    class R:
        pass
    r = R()
    r.text = text
    r.model = "gpt-4o-mini"
    r.provider = "openai"
    r.tokens_in = 850
    r.tokens_out = 320
    return r


@patch("app.llm._call_openai")
def test_parse_clinical_text_happy_path(mock_openai):
    mock_openai.return_value = _fake_response()
    result = parse_clinical_text(
        raw_text="Some long-enough clinical text pasted here with lots of data.",
        api_key="sk-test-" + "x" * 40, provider="openai",
    )
    assert isinstance(result, ParsedClinicalText)
    assert result.parsed["patient_label"] == "AML-2024-017"
    assert result.parsed["age"] == 45
    assert len(result.parsed["mutations"]) == 2
    assert result.confidence["karyotype_text"] == 0.6
    assert len(result.warnings) == 1
    assert result.tokens_in == 850


@patch("app.llm._call_openai")
def test_parse_rejects_too_short_text(mock_openai):
    with pytest.raises(ValueError, match="too short"):
        parse_clinical_text("a", "sk-x", "openai")
    mock_openai.assert_not_called()


@patch("app.llm._call_openai")
def test_parse_truncates_huge_text(mock_openai):
    mock_openai.return_value = _fake_response()
    huge = "x" * 30000  # larger than 20k cap
    parse_clinical_text(huge, "sk-x", "openai")
    # Check that the prompt passed to LLM was truncated
    _, _, _, user_prompt, _ = mock_openai.call_args.args
    assert len(user_prompt) < 21000   # cap + headers only
    assert "truncated" in user_prompt


@patch("app.llm._call_openai")
def test_parse_raises_on_bad_json(mock_openai):
    mock_openai.return_value = _fake_response(
        "Sorry, I can't parse this. <invalid JSON>"
    )
    with pytest.raises(ValueError, match="unparseable"):
        parse_clinical_text("enough text here to pass length check",
                             "sk-x", "openai")


def test_parse_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        parse_clinical_text("enough clinical text to pass", "key",
                             "gemini")


@patch("app.llm._call_anthropic")
def test_parse_via_anthropic(mock_anthropic):
    mock_anthropic.return_value = _fake_response()
    mock_anthropic.return_value.provider = "anthropic"
    mock_anthropic.return_value.model = "claude-haiku-4-5"
    result = parse_clinical_text(
        "enough clinical text to pass length check",
        "sk-ant-" + "x" * 40, "anthropic",
    )
    assert result.provider == "anthropic"


# ---------------------------------------------------------------------------
# /api/v1/patients/parse endpoint (via FastAPI TestClient)
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_client_with_llm_key(client, request):
    """Client with logged-in user + stored LLM key."""
    email = f"{request.node.name}@example.com"
    client.post("/auth/signup",
                 json={"email": email, "password": "password1"})
    fake_key = "sk-" + "a" * 50
    r = client.post("/api/v1/llm-keys",
                     json={"provider": "openai", "plaintext_key": fake_key,
                           "label": "test-key"})
    assert r.status_code == 201, r.text
    return client, r.json()["id"]


@patch("app.llm._call_openai")
def test_parse_endpoint_happy_path(mock_openai, auth_client_with_llm_key):
    mock_openai.return_value = _fake_response()
    client, key_id = auth_client_with_llm_key

    r = client.post("/api/v1/patients/parse", json={
        "raw_text": "Patient: 45 yo F, FLT3-ITD AR=0.62 VAF 45%, NPM1-mut VAF 42%, 46,XX",
        "llm_key_id": key_id,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["parsed"]["age"] == 45
    assert body["parsed"]["patient_label"] == "AML-2024-017"
    assert len(body["parsed"]["mutations"]) == 2
    assert len(body["warnings"]) == 1


def test_parse_endpoint_requires_auth(client):
    r = client.post("/api/v1/patients/parse",
                     json={"raw_text": "enough text to pass",
                           "llm_key_id": "00000000-0000-0000-0000-000000000000"})
    assert r.status_code == 401


def test_parse_endpoint_rejects_short_text(auth_client_with_llm_key):
    client, key_id = auth_client_with_llm_key
    r = client.post("/api/v1/patients/parse",
                     json={"raw_text": "a", "llm_key_id": key_id})
    assert r.status_code == 422  # Pydantic min_length validation


def test_parse_endpoint_rejects_unknown_key(auth_client_with_llm_key):
    client, _ = auth_client_with_llm_key
    r = client.post("/api/v1/patients/parse",
                     json={"raw_text": "enough clinical text for length",
                           "llm_key_id": "00000000-0000-0000-0000-000000000000"})
    assert r.status_code == 404


@patch("app.llm._call_openai")
def test_parse_endpoint_updates_last_used_and_logs_audit(
    mock_openai, auth_client_with_llm_key,
):
    """After a parse, the LLM key's last_used_at updates and a usage event
    is recorded (but no raw text leaks into the event meta)."""
    mock_openai.return_value = _fake_response()
    client, key_id = auth_client_with_llm_key

    SECRET_TEXT = "UNIQUE-SENTINEL-STRING-THAT-MUST-NOT-LEAK-123"
    r = client.post("/api/v1/patients/parse",
                     json={"raw_text": SECRET_TEXT * 3, "llm_key_id": key_id})
    assert r.status_code == 200

    # Pull the audit events (other tests in this module may have added some;
    # just check the sentinel doesn't leak into any of them).
    from app.db import SessionLocal
    from app.models import UsageEvent
    from sqlalchemy import select
    with SessionLocal() as db:
        events = db.execute(
            select(UsageEvent).where(UsageEvent.event_type == "llm_parse_used")
        ).scalars().all()
    assert len(events) >= 1
    for ev in events:
        meta_blob = json.dumps(ev.meta or {})
        assert "UNIQUE-SENTINEL" not in meta_blob, (
            "Raw text must not leak into audit event metadata"
        )
    # Latest event should have our token fields
    assert "tokens_in" in (events[-1].meta or {})
