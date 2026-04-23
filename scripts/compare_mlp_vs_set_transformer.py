"""Side-by-side comparison: kit output with MLP backbone vs Set Transformer backbone.

Run this after training set_drug_predictor_v2_stable to see how the Layer 3
recommendations change when we swap the backbone. This is the delta-diff that
validates whether switching Layer 3 actually improves clinical recommendations.

Metrics reported:
  - Top-k pair agreement (Jaccard)
  - Rank correlation on the union top-10
  - Mean |Δ predicted_combo_auc| across all pairs
  - Per-patient direction of change (is MLP or ST more confident about FLT3+BCL2i?)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from combo_val.clinical.demo_kit_run import _synthetic_rna_counts
from combo_val.clinical.kit_predict import predict_for_patient
from combo_val.clinical.kit_schema import KitInput, MutationCall


def _summarize_pairs(top_combos: list[dict]) -> pd.DataFrame:
    rows = []
    for c in top_combos:
        rows.append({
            "pair": " + ".join(sorted([c["drug1"], c["drug2"]])),
            "rank": c["rank"],
            "auc": c["predicted_combo_auc"],
            "backbone": c.get("layer3_backbone", "?"),
        })
    return pd.DataFrame(rows)


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def compare(patient_name: str, kit: KitInput, rna_counts: pd.Series) -> dict:
    mlp_out = predict_for_patient(
        rna_counts, kit, prefer_set_transformer=False, top_k=10,
    )
    st_out = predict_for_patient(
        rna_counts, kit, prefer_set_transformer=True, top_k=10,
    )

    mlp_pairs = _summarize_pairs(mlp_out.top_combinations)
    st_pairs = _summarize_pairs(st_out.top_combinations)
    jaccard_top5 = _jaccard(
        list(mlp_pairs.head(5)["pair"]),
        list(st_pairs.head(5)["pair"]),
    )
    jaccard_top10 = _jaccard(list(mlp_pairs["pair"]), list(st_pairs["pair"]))

    # Check FLT3-i + BCL2 pair rank in both
    flt3i_bcl2_pairs = {"Gilteritinib + Venetoclax", "Quizartinib (AC220) + Venetoclax"}
    mlp_flt3_rank = next(
        (r["rank"] for _, r in mlp_pairs.iterrows() if r["pair"] in flt3i_bcl2_pairs), None,
    )
    st_flt3_rank = next(
        (r["rank"] for _, r in st_pairs.iterrows() if r["pair"] in flt3i_bcl2_pairs), None,
    )

    print(f"\n=== {patient_name} ({st_out.top_combinations[0].get('layer3_backbone', '?')}"
          f" vs {mlp_out.top_combinations[0].get('layer3_backbone', '?')}) ===")
    print(f"  MLP top-5:  {list(mlp_pairs.head(5)['pair'])}")
    print(f"  ST  top-5:  {list(st_pairs.head(5)['pair'])}")
    print(f"  Jaccard top-5:  {jaccard_top5:.2f}")
    print(f"  Jaccard top-10: {jaccard_top10:.2f}")
    print(f"  FLT3i+BCL2 rank — MLP={mlp_flt3_rank}, ST={st_flt3_rank}")

    return {
        "patient": patient_name,
        "jaccard_top5": jaccard_top5,
        "jaccard_top10": jaccard_top10,
        "mlp_top5": list(mlp_pairs.head(5)["pair"]),
        "st_top5": list(st_pairs.head(5)["pair"]),
        "flt3i_bcl2_rank_mlp": mlp_flt3_rank,
        "flt3i_bcl2_rank_st": st_flt3_rank,
    }


def main():
    import joblib
    bundle = joblib.load("data/canonical/beataml_rna_preprocessor.joblib")
    kept = bundle["kept_genes"]

    rna = _synthetic_rna_counts(kept, "young_flt3")
    flt3_kit = KitInput(
        patient_id="SYN-FLT3",
        mutations=[
            MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.62),
            MutationCall(gene="NPM1"),
        ],
        karyotype_text="46,XX[20]",
        wbc=95.0, platelet=32.0, hemoglobin=8.5,
        ldh=1240.0, alt=28.0, ast=35.0, albumin=3.5,
        age=45, sex="female", is_initial_diagnosis=True,
    )
    compare("FLT3-mut + NPM1 (young, fit)", flt3_kit, rna)

    rna2 = _synthetic_rna_counts(kept, "elderly_tp53")
    tp53_kit = KitInput(
        patient_id="SYN-TP53",
        mutations=[MutationCall(gene="TP53"), MutationCall(gene="ASXL1")],
        karyotype_text="45,XY,-7,del(5)(q13q33),+8,del(17)(p13)[18]/46,XY[2]",
        wbc=12.0, platelet=25.0, hemoglobin=7.8, ldh=850.0,
        alt=22.0, ast=30.0, albumin=2.9,
        age=72, sex="male", prior_mds=True, is_initial_diagnosis=True,
    )
    compare("TP53 + complex karyotype (elderly, unfit)", tp53_kit, rna2)


if __name__ == "__main__":
    main()
