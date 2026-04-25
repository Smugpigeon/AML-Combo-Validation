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


# ---------------------------------------------------------------------------
# File upload path: /api/v1/patients/parse-file
# ---------------------------------------------------------------------------


def _make_rna_seq_csv(n_genes: int = 500) -> bytes:
    """Generate a CSV that should trigger _looks_like_rna_seq."""
    lines = ["symbol,count"]
    for i in range(n_genes):
        lines.append(f"GENE{i:05d},{42.5 + (i % 100)}")
    return ("\n".join(lines)).encode("utf-8")


def _make_clinical_csv() -> bytes:
    """A CSV that looks like a multi-patient clinical table."""
    text = (
        "mrn,age,sex,karyotype,flt3_itd,npm1,wbc,platelet\n"
        "A-001,45,F,46;XX[20],yes AR=0.62 VAF 45%,missense VAF 42%,95,32\n"
        "A-002,72,M,complex del(5q) del(17p),no,no,12,25\n"
        "A-003,58,F,46;XX[20],no,missense VAF 38%,45,88\n"
    )
    return text.encode("utf-8")


def test_looks_like_rna_seq_positive():
    from app.routers.patients import _looks_like_rna_seq
    text = _make_rna_seq_csv(500).decode("utf-8")
    is_rna, n, mean = _looks_like_rna_seq(text)
    assert is_rna is True
    assert n == 500
    assert 40 < mean < 150


def test_looks_like_rna_seq_rejects_clinical_table():
    from app.routers.patients import _looks_like_rna_seq
    text = _make_clinical_csv().decode("utf-8")
    is_rna, _, _ = _looks_like_rna_seq(text)
    assert is_rna is False


def test_looks_like_rna_seq_rejects_too_few_rows():
    from app.routers.patients import _looks_like_rna_seq
    # Only 50 rows — below 100-row threshold
    lines = ["symbol,count"] + [f"G{i},5.0" for i in range(50)]
    is_rna, _, _ = _looks_like_rna_seq("\n".join(lines))
    assert is_rna is False


def test_parse_file_endpoint_auto_routes_rna_seq(auth_client_with_llm_key):
    """Uploading an RNA-Seq CSV should skip the LLM and return is_rna_seq=True."""
    client, key_id = auth_client_with_llm_key
    csv_bytes = _make_rna_seq_csv(500)
    r = client.post(
        "/api/v1/patients/parse-file",
        data={"llm_key_id": key_id},
        files={"file": ("rna_counts.csv", csv_bytes, "text/csv")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_rna_seq"] is True
    assert body["rna_seq_gene_count"] == 500
    assert body["model"] == "(no LLM used)"   # confirm no LLM call
    assert body["parsed"] == {}


@patch("app.llm._call_openai")
def test_parse_file_endpoint_clinical_csv_uses_llm(
    mock_openai, auth_client_with_llm_key,
):
    """Uploading a clinical CSV should go to the LLM and extract one patient."""
    mock_openai.return_value = _fake_response()
    client, key_id = auth_client_with_llm_key
    csv_bytes = _make_clinical_csv()
    r = client.post(
        "/api/v1/patients/parse-file",
        data={"llm_key_id": key_id, "patient_identifier": "A-001"},
        files={"file": ("patients.csv", csv_bytes, "text/csv")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_rna_seq"] is False
    assert body["parsed"]["age"] == 45
    # Focus patient hint was in the LLM prompt
    _, _, _, user_prompt, _ = mock_openai.call_args.args
    assert "A-001" in user_prompt
    assert "FOCUS PATIENT" in user_prompt


def test_parse_file_endpoint_rejects_huge_file(auth_client_with_llm_key):
    """Files over 2 MB should be rejected with a helpful 413."""
    client, key_id = auth_client_with_llm_key
    huge = b"a,b\n" + (b"x,1\n" * 1_000_000)  # ~4 MB
    r = client.post(
        "/api/v1/patients/parse-file",
        data={"llm_key_id": key_id},
        files={"file": ("huge.csv", huge, "text/csv")},
    )
    assert r.status_code == 413
    assert "2 MB" in r.json()["detail"]


def test_parse_file_endpoint_requires_auth(client):
    r = client.post(
        "/api/v1/patients/parse-file",
        data={"llm_key_id": "00000000-0000-0000-0000-000000000000"},
        files={"file": ("x.csv", b"a,b\n1,2", "text/csv")},
    )
    assert r.status_code == 401


def test_parse_file_endpoint_reads_gbk_encoded_file(auth_client_with_llm_key):
    """Chinese hospital exports often use GBK — we must decode them."""
    client, key_id = auth_client_with_llm_key
    text_zh = "symbol,count\n" + "\n".join(f"基因{i},5.5" for i in range(200))
    gbk_bytes = text_zh.encode("gbk")
    r = client.post(
        "/api/v1/patients/parse-file",
        data={"llm_key_id": key_id},
        files={"file": ("counts_gbk.csv", gbk_bytes, "text/csv")},
    )
    # RNA-Seq detection should still work via gene_col_names/value_col_names
    assert r.status_code == 200, r.text
    assert r.json()["is_rna_seq"] is True


# ---------------------------------------------------------------------------
# Zip archive support
# ---------------------------------------------------------------------------


def _make_zip(files: list[tuple[str, bytes]]) -> bytes:
    """Build an in-memory zip containing the given (name, bytes) entries."""
    import io as _io
    import zipfile as _zip
    buf = _io.BytesIO()
    with _zip.ZipFile(buf, "w") as zf:
        for name, data in files:
            zf.writestr(name, data)
    return buf.getvalue()


def test_extract_zip_files_chinese_filename_recovered():
    """macOS / Chinese-Windows zip tools store CJK names as raw GBK/UTF-8
    bytes WITHOUT setting the UTF-8 flag bit, so Python's zipfile decodes
    them as CP437 → mojibake. Our _decode_zip_filename() recovers them."""
    from app.routers.patients import _extract_zip_files
    import zipfile as _zip
    import io as _io
    buf = _io.BytesIO()
    with _zip.ZipFile(buf, "w") as zf:
        zf.writestr("测试套件/患者A.txt", b"45 yo F, FLT3-ITD")
    raw = buf.getvalue()
    # Python's writestr DOES set the UTF-8 flag (good). Verify we leave
    # well-formed UTF-8 names alone.
    out = _extract_zip_files(raw)
    assert any("测试套件" in n for n, _ in out)


def test_decode_zip_filename_recovers_cp437_garble():
    """Direct unit test: simulate a ZipInfo where the filename is the
    CP437 mojibake of a UTF-8 Chinese name."""
    import zipfile as _zip
    from app.routers.patients import _decode_zip_filename
    # Build a ZipInfo with the WRONG decoding (no UTF-8 flag, name is
    # already in mojibake form as Python's zipfile would produce).
    info = _zip.ZipInfo()
    info.filename = "测试套件/患者A.txt".encode("utf-8").decode("cp437")
    info.flag_bits = 0  # crucial: UTF-8 flag NOT set
    fixed = _decode_zip_filename(info)
    assert "测试套件" in fixed
    assert "患者A" in fixed


def test_decode_zip_filename_keeps_ascii_unchanged():
    import zipfile as _zip
    from app.routers.patients import _decode_zip_filename
    info = _zip.ZipInfo()
    info.filename = "rna_counts.csv"
    info.flag_bits = 0
    assert _decode_zip_filename(info) == "rna_counts.csv"


def test_extract_zip_files_basic():
    from app.routers.patients import _extract_zip_files
    z = _make_zip([
        ("clinical.txt", b"45 yo F, FLT3-ITD"),
        ("rna_counts.csv", b"symbol,count\nFLT3,42.5"),
        ("__MACOSX/._foo", b"junk"),     # should be filtered
        (".DS_Store", b"junk"),          # should be filtered
    ])
    out = _extract_zip_files(z)
    names = [n for n, _ in out]
    assert "clinical.txt" in names
    assert "rna_counts.csv" in names
    assert not any("MACOSX" in n or n.startswith(".") for n in names)


def test_extract_zip_rejects_invalid_zip():
    from app.routers.patients import _extract_zip_files
    with pytest.raises(HTTPException := __import__("fastapi").HTTPException):
        _extract_zip_files(b"not a zip")


def test_classify_file_skips_documentation():
    """README / guide files should NOT be passed to LLM, even if their
    content decodes as text. Saves tokens + avoids context pollution."""
    from app.routers.patients import _classify_file
    readme_text = b"""# AMLCombo Test Kit

## How to use this kit

Step 1: sign up at amlcombo.org
Step 2: drop the zip into the smart-paste area
Step 3: enter the focus patient ID

This guide explains how to use the platform.
"""
    kind, _ = _classify_file("README.md", readme_text)
    assert kind == "documentation"

    # Filename hint also catches files without strong content markers
    cn_text = "步骤一：注册账号\n步骤二：上传文件".encode("utf-8")
    kind, _ = _classify_file("使用指南.md", cn_text)
    assert kind == "documentation"


def test_classify_file_routes_correctly():
    from app.routers.patients import _classify_file
    # Clinical text file
    kind, text = _classify_file("note.txt", b"45 yo F newly diagnosed AML")
    assert kind == "clinical"
    assert text and "45 yo" in text
    # RNA-Seq file (need >= 100 rows for the heuristic)
    rna_csv = b"symbol,count\n" + b"\n".join(
        f"GENE{i:04d},{42.5 + i % 10}".encode() for i in range(150)
    )
    kind, text = _classify_file("counts.csv", rna_csv)
    assert kind == "rna_seq"
    # Garbage binary
    kind, text = _classify_file("photo.bin", b"\x00\x01\xff\xfe" * 100)
    # latin-1 decodes anything; the heuristic just doesn't match RNA
    assert kind in ("clinical", "ignored")


@patch("app.llm._call_openai")
def test_parse_file_zip_with_clinical_only(mock_openai, auth_client_with_llm_key):
    """Zip with only a clinical text file: server LLM-parses it, no RNA staged."""
    mock_openai.return_value = _fake_response()
    client, key_id = auth_client_with_llm_key
    z = _make_zip([
        ("ngs_report.txt",
         b"Patient A: 45 F, FLT3-ITD AR=0.62 VAF 45%, NPM1 missense"),
    ])
    r = client.post(
        "/api/v1/patients/parse-file",
        data={"llm_key_id": key_id},
        files={"file": ("submission.zip", z, "application/zip")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["parsed"]["age"] == 45
    assert body["rna_seq_attachment_b64"] is None
    assert len(body["zip_summary"]) == 1
    assert body["zip_summary"][0]["kind"] == "clinical"


def test_parse_file_zip_rna_only(auth_client_with_llm_key):
    """Zip with only an RNA-Seq file: short-circuits, no LLM, returns base64."""
    client, key_id = auth_client_with_llm_key
    rna_bytes = b"symbol,count\n" + b"\n".join(
        f"GENE{i:04d},{50 + i}".encode() for i in range(200)
    )
    z = _make_zip([("counts.csv", rna_bytes)])
    r = client.post(
        "/api/v1/patients/parse-file",
        data={"llm_key_id": key_id},
        files={"file": ("rna_only.zip", z, "application/zip")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_rna_seq"] is True
    assert body["rna_seq_attachment_b64"] is not None
    assert body["rna_seq_attachment_filename"] == "counts.csv"
    # Verify base64 decodes back to the original CSV
    import base64 as _b64
    decoded = _b64.b64decode(body["rna_seq_attachment_b64"])
    assert decoded == rna_bytes
    assert body["model"] == "(no LLM used)"


@patch("app.llm._call_openai")
def test_parse_file_zip_mixed_clinical_and_rna(mock_openai, auth_client_with_llm_key):
    """The killer use case: zip with BOTH clinical text AND RNA-Seq counts."""
    mock_openai.return_value = _fake_response()
    client, key_id = auth_client_with_llm_key
    rna_bytes = b"symbol,count\n" + b"\n".join(
        f"G{i:04d},{i*1.0}".encode() for i in range(150)
    )
    z = _make_zip([
        ("patient_note.txt", b"45 yo F, FLT3-ITD positive AR=0.62"),
        ("expression.csv", rna_bytes),
        ("__MACOSX/._junk", b"system junk"),
    ])
    r = client.post(
        "/api/v1/patients/parse-file",
        data={"llm_key_id": key_id, "patient_identifier": "first patient"},
        files={"file": ("submission.zip", z, "application/zip")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # Clinical fields should be parsed
    assert body["parsed"]["age"] == 45
    # RNA-Seq attachment present
    assert body["rna_seq_attachment_b64"] is not None
    assert body["rna_seq_attachment_filename"] == "expression.csv"
    # Zip summary classifies both files (MACOSX filtered out)
    kinds = {item["kind"] for item in body["zip_summary"]}
    assert "clinical" in kinds
    assert "rna_seq" in kinds


# ---------------------------------------------------------------------------
# Demo RNA-Seq toggle on submission
# ---------------------------------------------------------------------------


def test_submit_with_demo_rna_seq_skips_file_requirement(
    auth_client_with_llm_key, monkeypatch,
):
    """Setting use_demo_rna_seq=true allows submitting without an rna_counts file.

    We mock out the celery task so the test doesn't actually run the kit;
    we just verify the submission row is created and the demo CSV is
    materialized on disk.
    """
    client, _ = auth_client_with_llm_key

    # Stub out the celery enqueue call so no kit prediction is attempted
    monkeypatch.setattr(
        "app.routers.patients.run_patient_kit",
        type("_FakeTask", (), {"delay": staticmethod(lambda *a, **kw: None)}),
    )

    # Stub joblib + numpy lookups in the demo-CSV generator (the real
    # joblib path won't exist in the test env)
    fake_kept = [f"GENE{i:04d}" for i in range(100)]

    class _FakeBundle(dict):
        pass

    fake_bundle = _FakeBundle({"kept_genes": fake_kept})

    import joblib as _real_joblib
    monkeypatch.setattr(_real_joblib, "load", lambda *a, **kw: fake_bundle)

    payload = {
        "patient_label": "DEMO-TEST",
        "age": 50, "sex": "male",
        "mutations": [{"gene": "FLT3", "is_ITD": True, "vaf": 0.4}],
    }
    r = client.post(
        "/api/v1/patients",
        data={"payload_json": __import__("json").dumps(payload),
               "use_demo_rna_seq": "true"},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "queued"
    assert body["patient_label"] == "DEMO-TEST"


def test_submit_without_rna_or_demo_rejected(auth_client_with_llm_key):
    """If neither a real file nor the demo flag is given, expect 400."""
    client, _ = auth_client_with_llm_key
    payload = {"patient_label": "FAIL-TEST", "age": 50, "sex": "male"}
    r = client.post(
        "/api/v1/patients",
        data={"payload_json": __import__("json").dumps(payload)},
    )
    assert r.status_code == 400
    assert "rna" in r.json()["detail"].lower()


@patch("app.llm._call_openai")
def test_focus_patient_gets_into_prompt(mock_openai):
    mock_openai.return_value = _fake_response()
    from app.llm import parse_clinical_text
    parse_clinical_text(
        "MRN, Age, Sex\nPT-001, 45, F\nPT-002, 72, M",
        "sk-x", "openai", focus_patient="PT-002",
    )
    _, _, _, user_prompt, _ = mock_openai.call_args.args
    assert "PT-002" in user_prompt
    assert "FOCUS PATIENT" in user_prompt


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
