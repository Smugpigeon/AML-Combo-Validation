"""Variant: ST v3 + mechanism-prior injection at inference time.

Subtract (mech_prior_scale × mech_score) from each pair's predicted AUC
before ranking. Same trick MLP+mech-prior Layer 3 uses. Tests whether
v3's Bliss-grounded absolute prediction PLUS mechanism-aware relative
ranking gives the best of both layers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from combo_val.combo.set_drug_inference import load_set_drug_predictor
from combo_val.combo.mechanism_prior import compute_combo_mech_scores


V3_CHECKPOINT = Path("runs/set_drug_predictor_v3_bliss/final_model.pt")
OUT = Path("runs/set_drug_predictor_v3_bliss")

CLINICAL = (
    "Venetoclax", "Azacytidine", "Cytarabine", "Midostaurin",
    "Quizartinib (AC220)", "Gilteritinib", "Sorafenib", "Crenolanib",
    "Dasatinib", "Nilotinib", "Imatinib", "Ruxolitinib (INCB018424)",
    "Trametinib (GSK1120212)", "Selumetinib (AZD6244)",
    "Alisertib (MLN8237)", "Crizotinib (PF-2341066)",
)


def score_st_v3_plus_mech(predictor, features_row, feature_cols, drug_subset,
                          pid, mech_scale=30.0):
    """Return pair AUC matrix under ST v3 + mechanism prior."""
    drug_indices = [predictor.drug_to_int[d] for d in drug_subset]
    pair_auc = predictor.predict_pairs(features_row, drug_indices)

    pf_df = pd.DataFrame([features_row], columns=feature_cols, index=[pid])
    mech = compute_combo_mech_scores(pf_df, drug_subset)[0]   # (n_d, n_d)
    return pair_auc - mech_scale * mech, mech


def main():
    predictor = load_set_drug_predictor(V3_CHECKPOINT)
    assert predictor is not None
    pf = pd.read_csv("data/canonical/beataml_patient_features.csv").set_index("patient_id")
    feat_cols = list(pf.columns)
    drug_subset = [d for d in CLINICAL if d in predictor.drug_to_int]

    flt3_pids = pf[pf["mut_FLT3"] == 1].head(50).index
    tri_i, tri_j = np.triu_indices(len(drug_subset), k=1)

    ranks_v3 = []
    ranks_hybrid = []
    jaccards = []
    for pid in flt3_pids:
        features = pf.loc[pid].to_numpy(dtype=np.float32)

        # Plain v3
        dm = [predictor.drug_to_int[d] for d in drug_subset]
        v3_matrix = predictor.predict_pairs(features, dm)
        v3_pairs = v3_matrix[tri_i, tri_j]
        v3_order = np.argsort(v3_pairs)
        v3_top5 = {tuple(sorted([drug_subset[tri_i[k]], drug_subset[tri_j[k]]]))
                   for k in v3_order[:5]}

        # Hybrid (v3 + mech prior 30)
        hybrid_matrix, _ = score_st_v3_plus_mech(
            predictor, features, feat_cols, drug_subset, pid,
        )
        hybrid_pairs = hybrid_matrix[tri_i, tri_j]
        hybrid_order = np.argsort(hybrid_pairs)
        hybrid_top5 = {tuple(sorted([drug_subset[tri_i[k]], drug_subset[tri_j[k]]]))
                       for k in hybrid_order[:5]}

        canonical = ("Gilteritinib", "Venetoclax")

        # Find canonical's rank in each
        def _rank(order, pair_tuples_by_order):
            for r, k in enumerate(order, start=1):
                if tuple(sorted([drug_subset[tri_i[k]], drug_subset[tri_j[k]]])) == canonical:
                    return r
            return None

        ranks_v3.append(_rank(v3_order, None))
        ranks_hybrid.append(_rank(hybrid_order, None))
        jaccards.append(len(v3_top5 & hybrid_top5) / max(len(v3_top5 | hybrid_top5), 1))

    report = {
        "canonical_pair": " + ".join(canonical),
        "n_flt3_patients": len(flt3_pids),
        "v3_plain_median_rank": int(np.median([r for r in ranks_v3 if r])),
        "v3_plain_frac_top5": round(np.mean([r <= 5 for r in ranks_v3 if r]), 3),
        "v3_hybrid_median_rank": int(np.median([r for r in ranks_hybrid if r])),
        "v3_hybrid_frac_top5": round(np.mean([r <= 5 for r in ranks_hybrid if r]), 3),
        "v3_hybrid_frac_top3": round(np.mean([r <= 3 for r in ranks_hybrid if r]), 3),
        "mean_jaccard_v3_vs_hybrid": round(float(np.mean(jaccards)), 3),
        "note": "Hybrid subtracts 30 * mech_score from v3 pair AUC before ranking.",
    }
    (OUT / "hybrid_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
