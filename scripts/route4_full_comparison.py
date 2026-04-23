"""Route 4 full comparison: 6 approaches scored on the same BeatAML cohort.

Produces the canonical table the user wanted for cross-fork comparison.

For each BeatAML FLT3-mut patient (n=179), compute combo AUC for all pairs
under each approach and report:

  - Pooled Pearson/Spearman between approaches (agreement structure)
  - FLT3i+BCL2i pair rank (rank of Gilt+Ven / Quiz+Ven)
  - Jaccard top-5 vs MLP+mech-prior as reference
  - % patients with Gilt+Ven in top-5

APPROACHES:
  A. MLP only           — Baseline A single-drug AUC, no combo modeling
  B. MLP + mech prior   — current kit default Layer 3
  C. MLP + Route 4 syn  — NEW: MLP single + learned synergy from Route 4
  D. ST v2 only         — Set Transformer direct pair prediction
  E. ST v2 + mech prior — ST single + mech prior synergy
  F. ST v2 + Route 4    — NEW: ST single + learned synergy from Route 4
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from combo_val.baselines.single_drug_mlp import SingleDrugMLP, SingleDrugMLPConfig
from combo_val.clinical.kit_predict import CLINICALLY_RELEVANT_AML_DRUGS
from combo_val.combo.mechanism_prior import compute_combo_mech_scores
from combo_val.combo.set_drug_inference import load_set_drug_predictor
from combo_val.combo.synergy_inference import load_synergy_head


OUT_DIR = Path("runs/route4_comparison")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def _load_mlp_predictions() -> pd.DataFrame:
    """Load cached Baseline A predictions for all 613 BeatAML patients × 165 drugs."""
    p = Path("runs/baseline_single_drug_mlp/predictions_all_patients_all_drugs.csv")
    if p.exists():
        return pd.read_csv(p, index_col=0)
    # Fallback: compute them
    raise FileNotFoundError(f"Need MLP predictions at {p}")


def _build_st_single_predictions(st_predictor, patient_features_raw: np.ndarray,
                                  ) -> np.ndarray:
    """Run ST v2 forward over all 165 drugs for one patient."""
    return st_predictor.predict_singles(patient_features_raw)


def compare():
    print("[compare] loading data + models")
    patients = pd.read_csv("data/canonical/beataml_patient_features.csv").set_index("patient_id")
    flt3_mut_patients = patients[patients["mut_FLT3"] == 1]
    print(f"[compare] FLT3-mut patients: {len(flt3_mut_patients)}")

    mlp_preds = _load_mlp_predictions()  # (n_patients, n_drugs)

    # Drug filter
    filt_drugs = [d for d in CLINICALLY_RELEVANT_AML_DRUGS if d in mlp_preds.columns]
    print(f"[compare] clinical filter drugs in MLP: {len(filt_drugs)}/{len(CLINICALLY_RELEVANT_AML_DRUGS)}")

    # ST v2 predictor
    st_pred = load_set_drug_predictor()
    print(f"[compare] ST v2 available: {st_pred is not None}")

    # Route 4 synergy head
    syn_head = load_synergy_head()
    print(f"[compare] Route 4 synergy head available: {syn_head is not None}")

    # Synergy matrix for the filter (patient-independent from Route 4)
    filt_drug_indices_in_st = [st_pred.drug_to_int[d] for d in filt_drugs]
    route4_synergy = syn_head.predict_synergy_matrix(filt_drug_indices_in_st)  # (n_f, n_f)
    print(f"[compare] Route 4 synergy matrix: shape={route4_synergy.shape}, "
          f"mean={route4_synergy.mean():.2f}, std={route4_synergy.std():.2f}")

    # Mech-prior scores for FLT3-mut cohort
    mech_scores_all = compute_combo_mech_scores(flt3_mut_patients, filt_drugs)
    print(f"[compare] mech_scores: shape={mech_scores_all.shape}, "
          f"mean={mech_scores_all.mean():.2f}")

    # Evaluate 6 approaches on each FLT3-mut patient
    n_d = len(filt_drugs)
    tri_i, tri_j = np.triu_indices(n_d, k=1)
    n_pairs = len(tri_i)
    print(f"[compare] {n_pairs} pairs over {n_d} drugs")

    # Find canonical FLT3i+BCL2i pairs' indices in the filter
    canonical_pairs = []
    flt3i = ["Quizartinib (AC220)", "Gilteritinib", "Midostaurin", "Sorafenib", "Crenolanib"]
    for f in flt3i:
        if f in filt_drugs and "Venetoclax" in filt_drugs:
            i = filt_drugs.index(f)
            j = filt_drugs.index("Venetoclax")
            lo, hi = min(i, j), max(i, j)
            canonical_pairs.append((lo, hi, f + " + Venetoclax"))
    canonical_tri_mask = np.array([
        any((t_i, t_j) == (c[0], c[1]) for c in canonical_pairs)
        for t_i, t_j in zip(tri_i, tri_j)
    ])
    canonical_tri_idx = np.where(canonical_tri_mask)[0]
    print(f"[compare] canonical FLT3i+Ven pair tri-indices: {canonical_tri_idx}")

    # Preload MLP pred for FLT3-mut patients
    flt3_ids = flt3_mut_patients.index.astype(int).tolist()
    mlp_flt3 = mlp_preds.loc[flt3_ids, filt_drugs].to_numpy()

    # Compute each approach's pair scores: (n_flt3_patients, n_pairs)
    results = {}

    # === A. MLP only (pure additive) ===
    pair_auc_mlp_only = (
        0.5 * (mlp_flt3[:, tri_i] + mlp_flt3[:, tri_j])
    )
    results["A_MLP_only"] = pair_auc_mlp_only

    # === B. MLP + mech prior (current kit default) ===
    mech_scores_filt = mech_scores_all[:, tri_i, tri_j]
    pair_auc_mlp_mech = pair_auc_mlp_only - 30.0 * mech_scores_filt
    results["B_MLP_plus_mech_prior"] = pair_auc_mlp_mech

    # === C. MLP + Route 4 synergy ===
    # Lower synergy_loewe = more synergistic = lower combo AUC
    route4_syn_pairs = route4_synergy[tri_i, tri_j]
    pair_auc_mlp_r4 = pair_auc_mlp_only + route4_syn_pairs
    results["C_MLP_plus_Route4"] = pair_auc_mlp_r4

    # === D. ST v2 only (direct pair prediction) ===
    st_single_flt3 = np.array([
        st_pred.predict_singles(
            flt3_mut_patients.loc[int(pid)].to_numpy(dtype=np.float32),
        )[filt_drug_indices_in_st]
        for pid in flt3_ids
    ])
    st_pair_flt3 = np.array([
        st_pred.predict_pairs(
            flt3_mut_patients.loc[int(pid)].to_numpy(dtype=np.float32),
            filt_drug_indices_in_st,
        )[tri_i, tri_j]
        for pid in flt3_ids
    ])
    results["D_ST_only"] = st_pair_flt3

    # === E. ST v2 + mech prior ===
    pair_auc_st_mech = st_pair_flt3 - 30.0 * mech_scores_filt
    results["E_ST_plus_mech_prior"] = pair_auc_st_mech

    # === F. ST v2 + Route 4 synergy ===
    st_pair_additive = 0.5 * (st_single_flt3[:, tri_i] + st_single_flt3[:, tri_j])
    pair_auc_st_r4 = st_pair_additive + route4_syn_pairs
    results["F_ST_plus_Route4"] = pair_auc_st_r4

    # ========== METRICS per approach ==========
    print("\n[compare] evaluating 6 approaches on FLT3-mut cohort (n="
          f"{len(flt3_ids)}, {n_pairs} pairs each)")

    summary = {}
    for name, pair_auc in results.items():
        # Rank of each canonical FLT3i+Ven pair (lower AUC = better = lower rank)
        ranks_per_patient = np.argsort(np.argsort(pair_auc, axis=1), axis=1) + 1
        canonical_ranks = ranks_per_patient[:, canonical_tri_idx]  # (n_pat, n_canonical)
        # Best rank across all canonical FLT3i+Ven combos
        best_canonical_rank = canonical_ranks.min(axis=1)
        # Is best-canonical in top-5?
        in_top5 = (best_canonical_rank <= 5).mean()

        # Top-5 pair indices per patient
        top5_idx = np.argsort(pair_auc, axis=1)[:, :5]
        # Jaccard against MLP+mech-prior baseline
        baseline_top5 = np.argsort(results["B_MLP_plus_mech_prior"], axis=1)[:, :5]
        jaccards = []
        for p_i in range(len(flt3_ids)):
            a = set(top5_idx[p_i].tolist())
            b = set(baseline_top5[p_i].tolist())
            jaccards.append(len(a & b) / len(a | b) if (a | b) else 0.0)
        mean_jaccard = float(np.mean(jaccards))

        summary[name] = {
            "median_best_FLT3i_Ven_rank": int(np.median(best_canonical_rank)),
            "mean_best_FLT3i_Ven_rank": round(float(np.mean(best_canonical_rank)), 1),
            "pct_FLT3i_Ven_top5": round(100 * float(in_top5), 1),
            "mean_jaccard_top5_vs_B": round(mean_jaccard, 3),
            "mean_pair_auc": round(float(pair_auc.mean()), 1),
        }
        print(f"\n  {name}")
        for k, v in summary[name].items():
            print(f"    {k}: {v}")

    # Cross-approach pooled Spearman
    print("\n[compare] Cross-approach agreement (pooled Spearman on pair rankings)")
    methods = list(results.keys())
    pooled_spearman = pd.DataFrame(
        index=methods, columns=methods, dtype=float,
    )
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i <= j:
                r1 = results[m1].ravel()
                r2 = results[m2].ravel()
                s, _ = spearmanr(r1, r2)
                pooled_spearman.loc[m1, m2] = round(s, 3)
                pooled_spearman.loc[m2, m1] = round(s, 3)
    print(pooled_spearman.to_string())
    pooled_spearman.to_csv(OUT_DIR / "pooled_spearman_matrix.csv")

    # Save summary
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )

    # Pretty comparison table
    rows = []
    for name, m in summary.items():
        rows.append({"approach": name, **m})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "comparison_table.csv", index=False)
    print("\n[compare] FINAL COMPARISON TABLE")
    print(table.to_string(index=False))

    return summary, table


if __name__ == "__main__":
    compare()
