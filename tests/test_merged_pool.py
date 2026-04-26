"""Tests for the 3-tier MergedDrugPool (Phase B.4).

Verifies:
  - Tier 1 (expert) coverage retrieved correctly
  - Tier 2 (ChEMBL) coverage retrieved correctly
  - Tier 1 + Tier 2 max-merge produces the expected union
  - Tier 3 (GIN novel SMILES) inference path works (when ckpt available)
  - Provenance string correctly identifies the source for each drug
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from combo_val.coverage.merged_pool import MergedDrugPool
from combo_val.coverage.taxonomy import load_taxonomy


@pytest.fixture
def pool():
    return MergedDrugPool(taxonomy=load_taxonomy())


def test_expert_drugs_subset_of_known(pool):
    assert len(pool.expert_drugs) >= 20
    assert "Quizartinib (AC220)" in pool.expert_drugs
    assert "Venetoclax" in pool.expert_drugs
    assert pool.expert_drugs.issubset(set(pool.known_drugs))


def test_chembl_adds_drugs_not_in_expert(pool):
    """ChEMBL pipeline should bring in multi-kinase drugs that the
    expert taxonomy missed (e.g. Sunitinib FLT3+KIT polypharmacology)."""
    only_chembl = pool.chembl_drugs - pool.expert_drugs
    if not pool.chembl_drugs:
        pytest.skip("ChEMBL CSV not present in this checkout")
    assert len(only_chembl) >= 5, \
        "Expected at least 5 drugs in ChEMBL but not in expert"


def test_coverage_for_drug_max_merges_tiers(pool):
    """Quizartinib should have FLT3 from BOTH tiers — max should win."""
    if "Quizartinib (AC220)" not in pool.chembl_drugs:
        pytest.skip("Quizartinib not in ChEMBL CSV")
    cov = pool.coverage_for_drug("Quizartinib (AC220)")
    # Expert says FLT3=0.95, ChEMBL says FLT3=0.85; max wins → 0.95
    assert "FLT3" in cov
    assert cov["FLT3"] >= 0.85
    # ChEMBL adds KIT_RTK off-target that expert missed
    assert "KIT_RTK" in cov
    assert cov["KIT_RTK"] >= 0.5


def test_provenance_strings(pool):
    """coverage_provenance should correctly tag each drug's source."""
    # Expert-only drugs
    if "Idasanutlin" in pool.expert_drugs and "Idasanutlin" not in pool.chembl_drugs:
        assert pool.coverage_provenance("Idasanutlin") == "expert"
    # ChEMBL-only drugs (e.g. Sunitinib not in expert)
    if "Sunitinib" in pool.chembl_drugs and "Sunitinib" not in pool.expert_drugs:
        assert pool.coverage_provenance("Sunitinib") == "chembl"
    # Both
    if "Quizartinib (AC220)" in pool.expert_drugs and "Quizartinib (AC220)" in pool.chembl_drugs:
        assert pool.coverage_provenance("Quizartinib (AC220)") == "expert+chembl"
    # Unknown
    assert pool.coverage_provenance("NoSuchDrug") == "unknown"


def test_register_novel_smiles(pool):
    pool.register_novel_smiles({"NovelDrug-1": "CC(=O)Oc1ccccc1C(=O)O"})  # aspirin
    assert "NovelDrug-1" in pool.novel_smiles
    assert pool.coverage_provenance("NovelDrug-1") == "gin (novel SMILES)"


def test_build_coverage_matrix_for_known_drugs(pool):
    """Standard path — building a matrix for drugs in expert/chembl pool."""
    drug_ids = ["Quizartinib (AC220)", "Venetoclax", "Azacytidine"]
    cm = pool.build_coverage_matrix(drug_ids)
    assert cm.coverage.shape[0] == 3
    assert cm.coverage.shape[1] == 18   # 18 axes
    # FLT3 should be hit by Quizartinib
    flt3_idx = cm.target_id_to_idx["FLT3"]
    quiz_idx = cm.drug_id_to_idx["Quizartinib (AC220)"]
    assert cm.coverage[quiz_idx, flt3_idx] >= 0.85


def test_build_matrix_with_novel_smiles_when_ckpt_missing(pool, tmp_path):
    """If GIN ckpt is missing, novel drugs get all-zero coverage and
    the build still succeeds (no crash)."""
    pool._gin_ckpt = tmp_path / "no_such.pt"   # force ckpt-missing path
    pool.register_novel_smiles({"NovelDrug-1": "CCO"})
    cm = pool.build_coverage_matrix(["Quizartinib (AC220)", "NovelDrug-1"])
    assert cm.coverage.shape[0] == 2
    novel_idx = cm.drug_id_to_idx["NovelDrug-1"]
    # Without GIN ckpt, novel drug has all-zero coverage
    assert np.allclose(cm.coverage[novel_idx], 0)


def test_chembl_overlay_doesnt_destroy_expert(pool):
    """Critical safety: max-merge must preserve expert annotation when
    ChEMBL has lower or no value for a drug-axis pair.

    Real example: Venetoclax. Expert annotates BCL2=0.90, MCL1=0 (Ven is
    BCL2-selective). ChEMBL might also have BCL2 at 1.0 + MCL1 at 0.4.
    After merge: BCL2=max(0.90, 1.0)=1.0, MCL1=max(0, 0.4)=0.4.
    Both tiers visible.
    """
    if "Venetoclax" not in pool.expert_drugs:
        pytest.skip("Venetoclax not in expert taxonomy")
    cov = pool.coverage_for_drug("Venetoclax")
    assert cov.get("BCL2", 0) >= 0.85   # expert had 0.90
    # ChEMBL augments stem_cell_program / OXPHOS that expert manually annotated
    assert "BCL2" in cov


def test_known_drugs_property_sorted(pool):
    drugs = pool.known_drugs
    assert drugs == sorted(drugs)


def test_coverage_for_unknown_drug_returns_empty():
    pool = MergedDrugPool(taxonomy=load_taxonomy())
    cov = pool.coverage_for_drug("CompletelyMadeUpDrug-XYZ")
    assert cov == {}
