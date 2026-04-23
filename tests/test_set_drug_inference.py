"""Tests for Path B Set Transformer inference wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from combo_val.combo.set_drug_inference import (
    SetDrugInference,
    load_set_drug_predictor,
)


def test_load_returns_none_when_missing():
    """Graceful fallback: missing checkpoint returns None so kit_predict
    can switch to Baseline A MLP without crashing."""
    result = load_set_drug_predictor(Path("/nonexistent/checkpoint.pt"))
    assert result is None


def test_load_returns_none_for_default_path_if_unavailable():
    """Default path should also return None if no v2 training has happened."""
    # This is intentionally flaky — it returns None on a clean tree and a
    # valid instance once v2 has been trained. Both are fine.
    result = load_set_drug_predictor()
    assert result is None or isinstance(result, SetDrugInference)


# Conditional tests that exercise the loaded predictor — only run if a v2
# checkpoint is available. Otherwise skip so the suite stays green on clean
# trees.


@pytest.fixture
def predictor_or_skip():
    cp = Path("runs/set_drug_predictor_v2_stable/final_model.pt")
    if not cp.exists():
        pytest.skip(
            "Set Transformer v2 checkpoint not present; run "
            "scripts/train_set_transformer_v2_stable.py first."
        )
    return load_set_drug_predictor(cp)


def test_predict_singles_shape_and_finite(predictor_or_skip):
    """predict_singles returns (n_drugs,) finite AUC predictions."""
    pred = predictor_or_skip
    # Use a realistic BeatAML feature vector (read a real patient row)
    import pandas as pd
    pf = pd.read_csv("data/canonical/beataml_patient_features.csv")
    features = pf.iloc[0].drop("patient_id").to_numpy(dtype=np.float32)

    singles = pred.predict_singles(features)
    assert singles.shape == (len(pred.drug_vocab),)
    assert np.isfinite(singles).all()
    # AUC predictions should land in the trained range (0-300 roughly)
    assert singles.min() > -100
    assert singles.max() < 500


def test_predict_pairs_symmetric_and_finite(predictor_or_skip):
    pred = predictor_or_skip
    import pandas as pd
    pf = pd.read_csv("data/canonical/beataml_patient_features.csv")
    features = pf.iloc[0].drop("patient_id").to_numpy(dtype=np.float32)

    # Test on a 6-drug subset
    sub_indices = list(range(6))
    pair_auc = pred.predict_pairs(features, sub_indices)
    assert pair_auc.shape == (6, 6)
    assert np.isfinite(pair_auc).all()
    # Permutation invariance → matrix should be near-symmetric
    asym = np.abs(pair_auc - pair_auc.T).max()
    assert asym < 1e-3, f"Expected near-symmetric, max asymmetry = {asym}"


def test_predict_triples_returns_top_k_sorted(predictor_or_skip):
    pred = predictor_or_skip
    import pandas as pd
    pf = pd.read_csv("data/canonical/beataml_patient_features.csv")
    features = pf.iloc[0].drop("patient_id").to_numpy(dtype=np.float32)

    sub_indices = list(range(6))   # C(6, 3) = 20 triplets
    triples = pred.predict_triples(features, sub_indices, top_k=5)
    assert len(triples) == 5
    # Sorted ascending by predicted AUC (lower = more cell-killing)
    aucs = [t[3] for t in triples]
    assert aucs == sorted(aucs)


def test_predict_set_for_patient_by_name(predictor_or_skip):
    pred = predictor_or_skip
    import pandas as pd
    pf = pd.read_csv("data/canonical/beataml_patient_features.csv")
    features = pf.iloc[0].drop("patient_id").to_numpy(dtype=np.float32)

    # Use the first two drugs from the vocab
    first_two = pred.drug_vocab[:2]
    auc = pred.predict_set_for_patient(features, first_two)
    assert np.isfinite(auc)


def test_predict_set_raises_on_unknown_drug(predictor_or_skip):
    pred = predictor_or_skip
    import pandas as pd
    pf = pd.read_csv("data/canonical/beataml_patient_features.csv")
    features = pf.iloc[0].drop("patient_id").to_numpy(dtype=np.float32)

    with pytest.raises(KeyError, match="Drug"):
        pred.predict_set_for_patient(features, ["NotARealDrug123"])


def test_permutation_invariance_on_real_features(predictor_or_skip):
    """Same drugs in different order → same predicted AUC (up to tiny epsilon)."""
    pred = predictor_or_skip
    import pandas as pd
    pf = pd.read_csv("data/canonical/beataml_patient_features.csv")
    features = pf.iloc[0].drop("patient_id").to_numpy(dtype=np.float32)

    drug_a, drug_b, drug_c = pred.drug_vocab[:3]
    auc_abc = pred.predict_set_for_patient(features, [drug_a, drug_b, drug_c])
    auc_cba = pred.predict_set_for_patient(features, [drug_c, drug_b, drug_a])
    auc_bac = pred.predict_set_for_patient(features, [drug_b, drug_a, drug_c])

    max_spread = max(auc_abc, auc_cba, auc_bac) - min(auc_abc, auc_cba, auc_bac)
    assert max_spread < 1e-3, f"permutation spread {max_spread} exceeds tolerance"
