"""Sanity check: run predict_for_patient with each of the 6 backbones and show
top-1 pair + predicted AUC, demonstrating the backbone registry works."""

from __future__ import annotations

import warnings

import joblib

from combo_val.clinical.demo_kit_run import _synthetic_rna_counts
from combo_val.clinical.kit_predict import BACKBONE_REGISTRY, predict_for_patient
from combo_val.clinical.kit_schema import KitInput, MutationCall


def main():
    warnings.filterwarnings("ignore")
    bundle = joblib.load("data/canonical/beataml_rna_preprocessor.joblib")
    kept = bundle["kept_genes"]
    rna = _synthetic_rna_counts(kept, "young_flt3")
    kit = KitInput(
        patient_id="SMOKE-FLT3",
        mutations=[
            MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.62),
            MutationCall(gene="NPM1"),
        ],
        karyotype_text="46,XX[20]",
        wbc=95.0, platelet=32.0, hemoglobin=8.5, ldh=1240.0,
        alt=28.0, ast=35.0, albumin=3.5,
        age=45, sex="female", is_initial_diagnosis=True,
    )

    print(f"{'backbone':<20s}  {'top-1 pair':<50s}  {'AUC':>6s}  {'mech':>6s}  label")
    print("-" * 130)
    for bname in BACKBONE_REGISTRY:
        try:
            out = predict_for_patient(rna, kit, backbone=bname, top_k=3)
            top1 = out.top_combinations[0]
            pair = f"{top1['drug1']} + {top1['drug2']}"
            print(
                f"{bname:<20s}  {pair:<50s}  {top1['predicted_combo_auc']:>6.1f}  "
                f"{top1['mech_score']:>+6.2f}  {top1['layer3_backbone']}"
            )
        except Exception as e:
            print(f"{bname:<20s}  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
