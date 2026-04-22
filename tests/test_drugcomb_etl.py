"""DrugComb ETL tests with synthetic chunks — no 1.4 GB file needed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from combo_val.data.drugcomb_etl import (
    DrugCombConfig,
    _build_cell_line_lookup,
    _detect_schema,
    _normalize_cell_line,
    run_drugcomb_etl,
)


# ---------------------------------------------------------------------------
# Cell line normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("HL-60", "HL-60"),
        ("HL60", "HL-60"),
        ("hl 60", "HL-60"),
        ("MV4-11", "MV4-11"),
        ("MV-4-11", "MV4-11"),
        ("MV4;11", "MV4-11"),
        ("MOLM-13", "MOLM-13"),
        ("MOLM13", "MOLM-13"),
        ("KASUMI-1", "KASUMI-1"),
        ("OCI-AML3", "OCI-AML3"),
        ("OCI AML 3", "OCI-AML3"),
        ("THP-1", "THP-1"),
        ("K562", None),          # CML, not AML
        ("A549", None),          # lung, not AML
        ("", None),
        (None, None),
    ],
)
def test_normalize_cell_line(raw, expected):
    assert _normalize_cell_line(raw) == expected


def test_build_cell_line_lookup_covers_all_aliases():
    lookup = _build_cell_line_lookup()
    # All canonical + aliases should resolve
    assert lookup["hl60"] == "HL-60"
    assert lookup["mv411"] == "MV4-11"
    assert lookup["ociaml3"] == "OCI-AML3"


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------


def test_schema_detect_standard():
    cfg = DrugCombConfig()
    # Simulated DrugComb v1.5 schema
    df = pd.DataFrame(
        {
            "block_id": [1, 2],
            "drug_row": ["Venetoclax", "Quizartinib"],
            "drug_col": ["Azacitidine", "Sorafenib"],
            "cell_line_name": ["HL-60", "MV4-11"],
            "synergy_loewe": [5.0, -2.0],
            "synergy_bliss": [3.0, -1.0],
            "synergy_zip": [4.0, -1.5],
            "synergy_hsa": [6.0, -2.5],
            "study_name": ["ONEIL", "ONEIL"],
        }
    )
    schema = _detect_schema(df, cfg)
    assert schema["drug1"] == "drug_row"
    assert schema["drug2"] == "drug_col"
    assert schema["cell_line"] == "cell_line_name"
    assert schema["synergy_loewe"] == "synergy_loewe"
    assert schema["synergy_hsa"] == "synergy_hsa"
    assert schema["block_id"] == "block_id"
    assert schema["study"] == "study_name"


def test_schema_detect_missing_required():
    cfg = DrugCombConfig()
    df = pd.DataFrame(
        {
            "drug_row": ["Venetoclax"],
            # no drug_col, no cell_line → should raise
        }
    )
    with pytest.raises(ValueError, match="missing required columns"):
        _detect_schema(df, cfg)


# ---------------------------------------------------------------------------
# End-to-end ETL on synthetic fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_drugcomb_file(tmp_path):
    """Build a tiny DrugComb-style CSV covering AML + non-AML cell lines."""
    rows = [
        # Row AML + both drugs in BeatAML vocab → should survive strict
        {"block_id": 1, "drug_row": "Venetoclax", "drug_col": "Azacytidine",
         "cell_line_name": "HL-60", "synergy_loewe": 10.0, "synergy_bliss": 8.0,
         "synergy_zip": 9.0, "synergy_hsa": 11.0, "study_name": "ONEIL"},
        # Row AML + quizartinib code (AC220) + sorafenib → should survive via paren_alias
        {"block_id": 2, "drug_row": "AC220", "drug_col": "Sorafenib",
         "cell_line_name": "MV4-11", "synergy_loewe": -3.0, "synergy_bliss": -2.0,
         "synergy_zip": -2.5, "synergy_hsa": -3.5, "study_name": "ONEIL"},
        # Row AML + ABT-199 (manual alias for venetoclax) → should survive via manual
        {"block_id": 3, "drug_row": "ABT-199", "drug_col": "Imatinib",
         "cell_line_name": "MOLM-13", "synergy_loewe": 0.0, "synergy_bliss": 0.5,
         "synergy_zip": 0.2, "synergy_hsa": -0.1, "study_name": "DRUGCOMB"},
        # Row NON-AML (K562, CML) → should be filtered out
        {"block_id": 4, "drug_row": "Venetoclax", "drug_col": "Imatinib",
         "cell_line_name": "K562", "synergy_loewe": 1.0, "synergy_bliss": 1.0,
         "synergy_zip": 1.0, "synergy_hsa": 1.0, "study_name": "ONEIL"},
        # Row AML but one drug missing from BeatAML vocab → loose path only
        {"block_id": 5, "drug_row": "Venetoclax", "drug_col": "5-Fluorouracil",
         "cell_line_name": "KASUMI-1", "synergy_loewe": 2.0, "synergy_bliss": 1.5,
         "synergy_zip": 1.8, "synergy_hsa": 2.2, "study_name": "ONEIL"},
        # Row AML both drugs missing → neither path
        {"block_id": 6, "drug_row": "AZ628", "drug_col": "XYZ-123",
         "cell_line_name": "U937", "synergy_loewe": 3.0, "synergy_bliss": 2.0,
         "synergy_zip": 2.5, "synergy_hsa": 3.5, "study_name": "ONEIL"},
    ]
    df = pd.DataFrame(rows)
    f = tmp_path / "synthetic_drugcomb.csv"
    df.to_csv(f, index=False)
    return f


@pytest.fixture
def synthetic_beataml_vocab_file(tmp_path):
    df = pd.DataFrame({
        "patient_id": [1, 2, 3, 4],
        "drug_id": ["Venetoclax", "Quizartinib (AC220)", "Sorafenib", "Imatinib"],
        "auc": [100.0, 150.0, 120.0, 200.0],
        "ic50": [1.0, 2.0, 1.5, 3.0],
    })
    f = tmp_path / "beataml_long.csv"
    df.to_csv(f, index=False)
    return f


def test_etl_end_to_end_synthetic(synthetic_drugcomb_file, synthetic_beataml_vocab_file, tmp_path):
    # Need 'Azacytidine' in the mock vocab too for row 1 to strict-match.
    # Update vocab file:
    df = pd.read_csv(synthetic_beataml_vocab_file)
    df = pd.concat([df, pd.DataFrame([{"patient_id": 5, "drug_id": "Azacytidine",
                                        "auc": 80, "ic50": 0.5}])], ignore_index=True)
    df.to_csv(synthetic_beataml_vocab_file, index=False)

    out_dir = tmp_path / "canonical"
    cfg = DrugCombConfig(
        input_path=synthetic_drugcomb_file,
        beataml_long_path=synthetic_beataml_vocab_file,
        out_dir=out_dir,
        chunk_size=2,  # force multiple chunks
    )
    manifest = run_drugcomb_etl(cfg)

    # Check output files exist
    assert (out_dir / "drugcomb_aml_pairs.csv").exists()
    assert (out_dir / "drugcomb_aml_pairs_any_match.csv").exists()
    assert (out_dir / "drugcomb_drug_alignment.csv").exists()
    assert (out_dir / "drugcomb_filter_manifest.json").exists()

    # Strict pairs: 3 rows expected
    # (Venetoclax+Azacytidine / HL-60, AC220+Sorafenib / MV4-11, ABT-199→Venetoclax + Imatinib / MOLM-13)
    strict = pd.read_csv(out_dir / "drugcomb_aml_pairs.csv")
    assert len(strict) == 3, f"Expected 3 strict pairs, got {len(strict)}: {strict['cell_line_id'].tolist()}"
    assert set(strict["cell_line_id"]) == {"HL-60", "MV4-11", "MOLM-13"}

    # Any-match: 4 rows (strict 3 + one-sided KASUMI-1 row)
    loose = pd.read_csv(out_dir / "drugcomb_aml_pairs_any_match.csv")
    assert len(loose) == 4
    assert "KASUMI-1" in set(loose["cell_line_id"])

    # K562 row (non-AML) must not appear anywhere
    assert "K562" not in set(strict["cell_line_id"])
    assert "K562" not in set(loose["cell_line_id"])

    # Drug alignment has the expected sources
    alignment = pd.read_csv(out_dir / "drugcomb_drug_alignment.csv")
    sources = set(alignment["match_source"].unique())
    assert "no_match" in sources  # 5-Fluorouracil, AZ628, XYZ-123
    assert "manual" in sources    # ABT-199

    # Manifest stats look right
    assert manifest["n_pairs_strict_both_mapped"] == 3
    assert manifest["n_pairs_any_mapped"] == 4
    assert manifest["aml_rows_kept"] == 5  # 6 total - 1 K562
