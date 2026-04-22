"""Tests for drug name normalizer + aligner.

No DrugComb file needed — uses synthetic BeatAML drug vocab.
"""

from __future__ import annotations

import pytest

from combo_val.data.drug_name_normalizer import (
    DrugNameAligner,
    normalize_drug_name,
    parenthesized_aliases,
)


# ---------------------------------------------------------------------------
# normalize_drug_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Venetoclax", "venetoclax"),
        ("  Venetoclax  ", "venetoclax"),
        ("Quizartinib (AC220)", "quizartinib"),
        ("(R)-Crizotinib", "crizotinib"),
        ("(±)-Nilotinib", "nilotinib"),
        ("17-AAG (Tanespimycin)", "17-aag"),
        ("Dovitinib (CHIR-258)", "dovitinib"),
        ("Lapatinib Ditosylate", "lapatinib"),               # ditosylate stripped (DrugComb salt form)
        ("Pazopanib hydrochloride", "pazopanib"),            # hydrochloride stripped
        ("Navelbine ditartrate", "navelbine"),               # ditartrate stripped
        ("Imatinib Mesylate", "imatinib"),
        ("", ""),
        (None, ""),
        ("Cytarabine hydrate", "cytarabine"),
    ],
)
def test_normalize_drug_name(raw, expected):
    assert normalize_drug_name(raw) == expected


# ---------------------------------------------------------------------------
# parenthesized_aliases
# ---------------------------------------------------------------------------


def test_parenthesized_aliases_extracts_both():
    aliases = parenthesized_aliases("Quizartinib (AC220)")
    assert "quizartinib" in aliases
    assert "ac220" in aliases
    assert len(aliases) == 2  # no duplicates


def test_parenthesized_aliases_no_parens():
    aliases = parenthesized_aliases("Venetoclax")
    assert aliases == ["venetoclax"]


def test_parenthesized_aliases_empty():
    assert parenthesized_aliases("") == []
    assert parenthesized_aliases(None) == []


# ---------------------------------------------------------------------------
# DrugNameAligner
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_beataml_vocab():
    # Subset of real BeatAML drugs including some tricky cases
    return [
        "Venetoclax",
        "Quizartinib (AC220)",        # parenthesized alias in source
        "17-AAG (Tanespimycin)",      # hyphenated + alias
        "Dovitinib (CHIR-258)",
        "Imatinib",
        "Sorafenib",
        "Azacytidine",                # BeatAML spelling variant
        "Ivosidenib",
        "Enasidenib",
        "Midostaurin",
        "Gilteritinib",
    ]


def test_exact_match(mock_beataml_vocab):
    aligner = DrugNameAligner(mock_beataml_vocab)
    assert aligner.align("Venetoclax") == ("Venetoclax", "exact")
    assert aligner.align("venetoclax") == ("Venetoclax", "exact")
    assert aligner.align("VENETOCLAX") == ("Venetoclax", "exact")


def test_paren_alias_to_stem(mock_beataml_vocab):
    """'Quizartinib' in DrugComb should map to 'Quizartinib (AC220)' in BeatAML."""
    aligner = DrugNameAligner(mock_beataml_vocab)
    canonical, source = aligner.align("Quizartinib")
    assert canonical == "Quizartinib (AC220)"
    # Could be 'exact' if 'quizartinib' was indexed as alias, 'paren_alias' otherwise
    assert source in ("exact", "paren_alias")


def test_paren_alias_to_code(mock_beataml_vocab):
    """'AC220' in DrugComb should also map to 'Quizartinib (AC220)' via code alias."""
    aligner = DrugNameAligner(mock_beataml_vocab)
    canonical, source = aligner.align("AC220")
    assert canonical == "Quizartinib (AC220)"


def test_manual_alias_abt199_to_venetoclax(mock_beataml_vocab):
    """ABT-199 is a research code for venetoclax — goes through MANUAL_ALIASES."""
    aligner = DrugNameAligner(mock_beataml_vocab)
    canonical, source = aligner.align("ABT-199")
    assert canonical == "Venetoclax"
    assert source == "manual"


def test_spelling_variant(mock_beataml_vocab):
    """BeatAML uses 'Azacytidine' (y), others 'Azacitidine' (i). Manual alias maps."""
    aligner = DrugNameAligner(mock_beataml_vocab)
    canonical, source = aligner.align("Azacitidine")
    assert canonical == "Azacytidine"
    assert source == "manual"


def test_fuzzy_match(mock_beataml_vocab):
    """Typo → fuzzy match."""
    aligner = DrugNameAligner(mock_beataml_vocab)
    canonical, source = aligner.align("Venetoklax")  # typo
    assert canonical == "Venetoclax"
    assert source == "fuzzy"


def test_no_match(mock_beataml_vocab):
    aligner = DrugNameAligner(mock_beataml_vocab)
    canonical, source = aligner.align("5-Fluorouracil")  # not in vocab
    assert canonical is None
    assert source == "no_match"


def test_empty_name(mock_beataml_vocab):
    aligner = DrugNameAligner(mock_beataml_vocab)
    assert aligner.align("") == (None, "no_match")
    assert aligner.align(None) == (None, "no_match")


def test_align_many_preserves_input_keys(mock_beataml_vocab):
    aligner = DrugNameAligner(mock_beataml_vocab)
    out = aligner.align_many(["Venetoclax", "ABT-199", "AC220", "NotInVocab"])
    assert set(out.keys()) == {"Venetoclax", "ABT-199", "AC220", "NotInVocab"}
    assert out["Venetoclax"][0] == "Venetoclax"
    assert out["ABT-199"][0] == "Venetoclax"
    assert out["AC220"][0] == "Quizartinib (AC220)"
    assert out["NotInVocab"][0] is None
