"""Tests for the ChEMBL → 18-axis coverage pipeline (Phase B.2)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from combo_val.coverage.chembl_coverage import (
    TaxonomyMap, aggregate_drug_coverage, coverage_vector_18d,
    load_taxonomy_map,
)


def test_taxonomy_map_loads_with_18_axes():
    tm = load_taxonomy_map()
    assert len(tm.axis_to_uniprots) >= 15
    # Spot-check key axes
    assert "FLT3" in tm.axis_to_uniprots
    assert "P36888" in tm.axis_to_uniprots["FLT3"]      # FLT3 UniProt
    assert "BCL2" in tm.axis_to_uniprots
    assert "P10415" in tm.axis_to_uniprots["BCL2"]      # BCL2 UniProt
    assert "MENIN_NPM1" in tm.axis_to_uniprots
    assert "O00255" in tm.axis_to_uniprots["MENIN_NPM1"]  # MEN1


def test_pIC50_normalization_bins():
    tm = load_taxonomy_map()
    # Per the YAML thresholds
    assert tm.normalize_pIC50(9.5) == 1.0
    assert tm.normalize_pIC50(8.5) == 0.85
    assert tm.normalize_pIC50(7.5) == 0.65
    assert tm.normalize_pIC50(6.5) == 0.40
    assert tm.normalize_pIC50(5.5) == 0.10
    assert tm.normalize_pIC50(0.0) == 0.10


def test_uniprot_to_axes_inverse_map():
    tm = load_taxonomy_map()
    # MEN1 belongs to BOTH MENIN_KMT2A and MENIN_NPM1 (same drug class
    # serves both axes via different patient indications)
    axes_for_men1 = tm.uniprot_to_axes.get("O00255", [])
    assert "MENIN_KMT2A" in axes_for_men1
    assert "MENIN_NPM1" in axes_for_men1


def test_aggregate_drug_coverage_with_synthetic_csv(tmp_path):
    """Build a tiny synthetic ChEMBL CSV and verify the aggregator
    produces the expected coverage vector (max(normalize) over UniProts)."""
    csv_path = tmp_path / "chembl_test.csv"
    rows = [
        # Drug A: FLT3 strong (pIC50 9.5 → cov 1.0), BCL2 weak (pIC50 5.0 → 0.10)
        {"beataml_drug_id": "DrugA", "chembl_id": "C1",
         "target_chembl_id": "T1", "target_uniprot": "P36888",
         "target_name": "FLT3", "standard_type": "IC50",
         "standard_value_nM": "0.3", "pIC50": "9.5", "assay_chembl_id": "A1"},
        {"beataml_drug_id": "DrugA", "chembl_id": "C1",
         "target_chembl_id": "T2", "target_uniprot": "P10415",
         "target_name": "BCL2", "standard_type": "IC50",
         "standard_value_nM": "10000", "pIC50": "5.0", "assay_chembl_id": "A2"},
        # Drug B: BCL2 strong + MCL1 weak (BH3 mimetic profile)
        {"beataml_drug_id": "DrugB", "chembl_id": "C2",
         "target_chembl_id": "T2", "target_uniprot": "P10415",
         "target_name": "BCL2", "standard_type": "IC50",
         "standard_value_nM": "1.0", "pIC50": "9.0", "assay_chembl_id": "A3"},
        {"beataml_drug_id": "DrugB", "chembl_id": "C2",
         "target_chembl_id": "T3", "target_uniprot": "Q07820",
         "target_name": "MCL1", "standard_type": "IC50",
         "standard_value_nM": "300", "pIC50": "6.5", "assay_chembl_id": "A4"},
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cov = aggregate_drug_coverage(chembl_csv=csv_path)
    assert "DrugA" in cov
    assert cov["DrugA"]["FLT3"] == 1.0     # pIC50 9.5 → bin 9.0+
    assert cov["DrugA"].get("BCL2", 0) == 0.10
    assert "DrugB" in cov
    assert cov["DrugB"]["BCL2"] == 1.0     # pIC50 9.0 → bin 9.0+
    assert cov["DrugB"]["MCL1"] == 0.40    # pIC50 6.5 → bin 6.0


def test_max_aggregation_when_multiple_assays_for_same_target(tmp_path):
    """Multiple bioactivity records for same (drug, UniProt) → use MAX pIC50."""
    csv_path = tmp_path / "test.csv"
    rows = [
        {"beataml_drug_id": "X", "chembl_id": "C", "target_chembl_id": "T",
         "target_uniprot": "P36888", "target_name": "FLT3",
         "standard_type": "IC50", "standard_value_nM": "100",
         "pIC50": "7.0", "assay_chembl_id": "A1"},
        {"beataml_drug_id": "X", "chembl_id": "C", "target_chembl_id": "T",
         "target_uniprot": "P36888", "target_name": "FLT3",
         "standard_type": "IC50", "standard_value_nM": "5",
         "pIC50": "8.3", "assay_chembl_id": "A2"},  # higher affinity assay wins
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cov = aggregate_drug_coverage(chembl_csv=csv_path)
    # max(7.0, 8.3) = 8.3 → bin [8.0, 9.0) → 0.85
    assert cov["X"]["FLT3"] == 0.85


def test_coverage_vector_18d_returns_correct_shape():
    cov_map = {"X": {"FLT3": 1.0, "BCL2": 0.85}}
    axis_order = ["FLT3", "IDH1", "BCL2", "MCL1"]
    v = coverage_vector_18d("X", cov_map, axis_order)
    assert v.shape == (4,)
    assert v[0] == 1.0     # FLT3
    assert v[1] == 0.0     # IDH1 (missing → 0)
    assert v[2] == 0.85    # BCL2
    assert v[3] == 0.0     # MCL1 (missing → 0)


def test_unknown_drug_returns_zero_vector():
    v = coverage_vector_18d("NoSuchDrug", {}, ["FLT3", "BCL2"])
    assert np.allclose(v, [0.0, 0.0])


def test_aggregator_skips_records_without_uniprot(tmp_path):
    csv_path = tmp_path / "t.csv"
    rows = [
        {"beataml_drug_id": "X", "chembl_id": "C", "target_chembl_id": "T",
         "target_uniprot": "", "target_name": "?",
         "standard_type": "IC50", "standard_value_nM": "1",
         "pIC50": "9.0", "assay_chembl_id": "A"},
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cov = aggregate_drug_coverage(chembl_csv=csv_path)
    # Drug X exists but has no axis-relevant target → empty inner dict
    assert cov.get("X", {}) == {}


def test_aggregator_handles_missing_csv_gracefully():
    cov = aggregate_drug_coverage(chembl_csv="/nonexistent/path.csv")
    assert cov == {}
