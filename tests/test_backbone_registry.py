"""Tests for the kit backbone registry — make sure all 6 backbones load
and produce consistent output shapes, and that fallback to 'mlp' works
cleanly when checkpoints are missing."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from combo_val.clinical.demo_kit_run import _synthetic_rna_counts
from combo_val.clinical.kit_predict import (
    BACKBONE_REGISTRY,
    predict_for_patient,
)
from combo_val.clinical.kit_schema import KitInput, KitOutput, MutationCall


@pytest.fixture
def flt3_patient_inputs():
    """FLT3-ITD + NPM1 patient — canonical combo should be FLT3i + BCL2i."""
    bundle_path = Path("data/canonical/beataml_rna_preprocessor.joblib")
    if not bundle_path.exists():
        pytest.skip("preprocessor bundle missing")
    bundle = joblib.load(bundle_path)
    rna = _synthetic_rna_counts(bundle["kept_genes"], "young_flt3")
    kit = KitInput(
        patient_id="TEST-FLT3",
        mutations=[
            MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.62),
            MutationCall(gene="NPM1"),
        ],
        karyotype_text="46,XX[20]",
        wbc=95.0, platelet=32.0, hemoglobin=8.5, ldh=1240.0,
        alt=28.0, ast=35.0, albumin=3.5,
        age=45, sex="female", is_initial_diagnosis=True,
    )
    return rna, kit


def test_registry_has_expected_6_backbones():
    """The registry should list all 6 documented backbones."""
    expected = {"mlp", "st-v2", "st-v3-bliss", "st-v3-distill",
                "st-v3-186pair", "mlp+synergy"}
    assert set(BACKBONE_REGISTRY.keys()) == expected


def test_registry_entries_well_formed():
    """Each backbone entry must have kind, checkpoint, label, description."""
    required = {"kind", "checkpoint", "label", "description"}
    for name, spec in BACKBONE_REGISTRY.items():
        missing = required - set(spec.keys())
        assert not missing, f"'{name}' missing fields: {missing}"
        assert spec["kind"] in {"mlp", "st", "mlp+synergy"}, f"'{name}' bad kind"


def test_unknown_backbone_raises(flt3_patient_inputs):
    rna, kit = flt3_patient_inputs
    with pytest.raises(ValueError, match="Unknown backbone"):
        predict_for_patient(rna, kit, backbone="no-such-backbone")


def test_mlp_default_still_works(flt3_patient_inputs):
    """Calling with no backbone kwarg uses the 'mlp' default."""
    rna, kit = flt3_patient_inputs
    out = predict_for_patient(rna, kit, top_k=3)
    assert isinstance(out, KitOutput)
    assert len(out.top_combinations) <= 3
    assert out.top_combinations[0]["layer3_backbone"] == \
        BACKBONE_REGISTRY["mlp"]["label"]


def test_mlp_top1_for_flt3_is_canonical(flt3_patient_inputs):
    """With the mlp backbone, a FLT3-mut patient should get an FLT3i+BCL2i
    pair at top-1 (or in the top-3) — this is the clinical ground truth
    the default backbone is expected to honor."""
    rna, kit = flt3_patient_inputs
    out = predict_for_patient(rna, kit, backbone="mlp", top_k=5)
    top_pairs = [
        f"{c['drug1']} + {c['drug2']}"
        for c in out.top_combinations
    ]
    canonical = {
        "Gilteritinib + Venetoclax", "Venetoclax + Gilteritinib",
        "Quizartinib (AC220) + Venetoclax", "Venetoclax + Quizartinib (AC220)",
    }
    assert any(p in canonical for p in top_pairs), \
        f"No canonical FLT3i+BCL2i pair in top-5: {top_pairs}"


@pytest.mark.parametrize("backbone", list(BACKBONE_REGISTRY.keys()))
def test_each_backbone_loads_and_runs(flt3_patient_inputs, backbone):
    """Every backbone with a present checkpoint should run end-to-end
    and produce a valid KitOutput. Skip with message if checkpoint missing."""
    rna, kit = flt3_patient_inputs
    ckpt_path = Path(BACKBONE_REGISTRY[backbone]["checkpoint"])
    if not ckpt_path.exists():
        pytest.skip(f"'{backbone}' checkpoint missing at {ckpt_path}")
    if backbone == "mlp+synergy":
        syn_path = Path(BACKBONE_REGISTRY[backbone]["synergy_checkpoint"])
        if not syn_path.exists():
            pytest.skip(f"'{backbone}' synergy head missing at {syn_path}")

    out = predict_for_patient(rna, kit, backbone=backbone, top_k=3)
    assert isinstance(out, KitOutput)
    # All top combos carry the backbone label so clinicians can audit
    for c in out.top_combinations:
        assert "layer3_backbone" in c
        assert c["layer3_backbone"] == BACKBONE_REGISTRY[backbone]["label"]


def test_fallback_to_mlp_when_checkpoint_missing(flt3_patient_inputs, tmp_path, monkeypatch):
    """If a backbone's checkpoint is absent, the predictor should warn and
    fall back to 'mlp' rather than crashing."""
    rna, kit = flt3_patient_inputs

    # Monkey-patch the registry for a FAKE backbone name pointing to nonexistent file
    fake_backbone = "fake-missing"
    monkeypatch.setitem(BACKBONE_REGISTRY, fake_backbone, {
        "kind": "st",
        "checkpoint": str(tmp_path / "nonexistent_checkpoint.pt"),
        "label": "FakeBackbone",
        "description": "intentionally broken for test",
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = predict_for_patient(rna, kit, backbone=fake_backbone, top_k=2)
    assert isinstance(out, KitOutput)
    # The top combo should carry the MLP label (fallback activated)
    assert out.top_combinations[0]["layer3_backbone"] == \
        BACKBONE_REGISTRY["mlp"]["label"]
    # A warning should have been emitted
    assert any("fallback" in str(w.message).lower() or
               "missing" in str(w.message).lower() for w in caught)


def test_legacy_prefer_set_transformer_still_works(flt3_patient_inputs):
    """Back-compat: prefer_set_transformer=True should map to backbone='st-v2'."""
    rna, kit = flt3_patient_inputs
    ckpt = Path(BACKBONE_REGISTRY["st-v2"]["checkpoint"])
    if not ckpt.exists():
        pytest.skip("st-v2 checkpoint not present")
    out = predict_for_patient(rna, kit, prefer_set_transformer=True, top_k=2)
    assert out.top_combinations[0]["layer3_backbone"] == \
        BACKBONE_REGISTRY["st-v2"]["label"]
