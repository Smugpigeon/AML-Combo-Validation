from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from combo_val.virtual_cell.beataml_pilot import (
    BeatAMLVirtualCellModel,
    ModelConfig,
    fit_blend_alpha,
    load_and_prepare_beataml,
    make_patient_split,
)


def test_patient_split_is_deterministic_and_disjoint() -> None:
    patient_ids = [str(index) for index in range(100)]
    first = make_patient_split(patient_ids, seed=42)
    second = make_patient_split(patient_ids, seed=42)
    assert first == second
    assert not (set(first.train) & set(first.validation))
    assert not (set(first.train) & set(first.test))
    assert not (set(first.validation) & set(first.test))
    assert set(first.train) | set(first.validation) | set(first.test) == set(patient_ids)


def test_virtual_cell_model_shapes() -> None:
    config = ModelConfig(n_features=12, n_drugs=7, latent_dim=8)
    model = BeatAMLVirtualCellModel(config)
    prediction, latent = model(torch.randn(5, 12), torch.tensor([0, 1, 2, 3, 4]))
    assert prediction.shape == (5,)
    assert latent.shape == (5, 8)


def test_preparation_keeps_patient_rows_in_one_split(tmp_path) -> None:
    patients = np.arange(20)
    feature_frame = pd.DataFrame(
        {
            "patient_id": patients,
            "rna_pc_01": patients.astype(float),
            "mut_TP53": (patients % 2).astype(float),
        }
    )
    response_rows = []
    for patient_id in patients:
        for drug_id in ["A", "B", "C"]:
            response_rows.append(
                {
                    "patient_id": patient_id,
                    "drug_id": drug_id,
                    "auc": float(patient_id + ord(drug_id)),
                }
            )
    feature_path = tmp_path / "features.csv"
    response_path = tmp_path / "responses.csv"
    feature_frame.to_csv(feature_path, index=False)
    pd.DataFrame(response_rows).to_csv(response_path, index=False)

    prepared = load_and_prepare_beataml(feature_path, response_path, split_seed=7)
    split_counts = prepared.responses.groupby("patient_id")["split"].nunique()
    assert split_counts.max() == 1
    assert np.isfinite(prepared.features_scaled).all()
    assert np.isfinite(prepared.responses["target_z"]).all()


def test_blend_alpha_recovers_useful_residual() -> None:
    baseline = np.array([10.0, 10.0, 10.0])
    neural = np.array([8.0, 10.0, 12.0])
    observed = np.array([9.0, 10.0, 11.0])
    assert fit_blend_alpha(observed, baseline, neural) == 0.5

