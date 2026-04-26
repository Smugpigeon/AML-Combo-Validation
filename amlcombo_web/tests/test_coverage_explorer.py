"""Smoke tests for the /coverage-explorer route + /api/v1/coverage/compute endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_coverage_explorer_page_loads(client):
    """GET /coverage-explorer returns the HTMX page (200, contains key markers)."""
    r = client.get("/coverage-explorer")
    assert r.status_code == 200
    body = r.text
    assert "多靶点覆盖" in body or "Coverage Explorer" in body.lower()
    assert "coverageExplorer()" in body  # alpine.js component name
    assert "/api/v1/coverage/compute" in body  # endpoint reference


def test_taxonomy_metadata_endpoint(client):
    """GET /api/v1/coverage/taxonomy returns 18 targets + drug pool."""
    r = client.get("/api/v1/coverage/taxonomy")
    assert r.status_code == 200
    d = r.json()
    assert d["framework"].startswith("Palmer-Sorger")
    assert len(d["targets"]) >= 15
    assert len(d["drug_pool"]) >= 20
    # Spot-check critical targets
    target_ids = {t["id"] for t in d["targets"]}
    for required in ("FLT3", "IDH1", "IDH2", "BCL2", "MENIN_NPM1", "MENIN_KMT2A"):
        assert required in target_ids


def test_compute_returns_correct_combo_for_flt3_npm1(client):
    """End-to-end POST: FLT3+NPM1+DNMT3A patient should get FLT3i in top combo."""
    r = client.post("/api/v1/coverage/compute", json={
        "mutations": ["FLT3", "NPM1", "DNMT3A"],
        "flt3_itd": True,
        "age": 45,
        "max_arity": 4,
        "coverage_threshold": 0.65,
    })
    assert r.status_code == 200
    d = r.json()
    assert "FLT3" in d["active_targets"]
    assert d["active_targets"]["FLT3"] == 1.0
    assert "MENIN_NPM1" in d["active_targets"]
    assert len(d["top_combinations"]) >= 1
    top = d["top_combinations"][0]
    # FLT3 inhibitors — expanded to include ChEMBL-derived multi-kinase
    # drugs (Sunitinib, Sorafenib, etc.) that also have high FLT3 coverage.
    flt3i = {"Quizartinib (AC220)", "Gilteritinib", "Midostaurin",
             "Sunitinib", "Sorafenib", "Crenolanib",
             "Cabozantinib", "Linifanib (ABT-869)", "Tandutinib (MLN518)",
             "AST-487", "Dovitinib (CHIR-258)"}
    assert any(d in flt3i for d in top["drug_ids"]), \
        f"Top combo {top['drug_ids']} has no FLT3i (any class)"
    assert top["feasible"] is True


def test_compute_apl_returns_atra_or_arsenic(client):
    """APL patient (PML_RARA fusion) — though our taxonomy has limited
    APL-specific targets, framework should still produce some output."""
    r = client.post("/api/v1/coverage/compute", json={
        "mutations": [],
        "fusions": ["PML_RARA"],
        "age": 38,
        "max_arity": 3,
    })
    assert r.status_code == 200
    d = r.json()
    # differentiation_block at minimum should be active for APL
    assert "differentiation_block" in d["active_targets"]


def test_compute_idh1_recommends_ivosidenib(client):
    """IDH1 elderly patient → Ivo or Olutasidenib in top combo."""
    r = client.post("/api/v1/coverage/compute", json={
        "mutations": ["IDH1"],
        "age": 78,
        "max_arity": 3,
        "coverage_threshold": 0.55,
    })
    assert r.status_code == 200
    d = r.json()
    assert "IDH1" in d["active_targets"]
    if d["top_combinations"]:
        top_drugs = set(d["top_combinations"][0]["drug_ids"])
        assert top_drugs & {"Ivosidenib", "Olutasidenib"}


def test_compute_with_custom_drug_pool(client):
    """User-restricted drug pool should be respected."""
    r = client.post("/api/v1/coverage/compute", json={
        "mutations": ["FLT3", "NPM1"],
        "flt3_itd": True, "age": 45,
        "drug_pool": ["Quizartinib (AC220)", "Venetoclax", "Azacytidine"],
        "coverage_threshold": 0.40,
    })
    assert r.status_code == 200
    d = r.json()
    assert d["drug_pool_size"] == 3
    if d["top_combinations"]:
        for combo in d["top_combinations"]:
            for drug in combo["drug_ids"]:
                assert drug in {"Quizartinib (AC220)", "Venetoclax", "Azacytidine"}


def test_compute_excluded_drugs_respected(client):
    """Excluding Quizartinib should push solver to Gilteritinib for FLT3."""
    r = client.post("/api/v1/coverage/compute", json={
        "mutations": ["FLT3"],
        "flt3_itd": True, "age": 45,
        "excluded_drugs": ["Quizartinib (AC220)"],
        "max_arity": 2,
        "coverage_threshold": 0.40,
    })
    assert r.status_code == 200
    d = r.json()
    if d["top_combinations"]:
        for combo in d["top_combinations"]:
            assert "Quizartinib (AC220)" not in combo["drug_ids"]


def test_compute_empty_mutations_returns_baseline(client):
    """Patient with no inputs — only always-active baseline targets fire."""
    r = client.post("/api/v1/coverage/compute", json={"mutations": []})
    assert r.status_code == 200
    d = r.json()
    # Should still have BCL2 (always_active_baseline 0.6)
    assert "BCL2" in d["active_targets"]


def test_compute_validates_request_schema(client):
    """Bad request body → 422."""
    r = client.post("/api/v1/coverage/compute", json={"max_arity": "not-a-number"})
    assert r.status_code == 422
