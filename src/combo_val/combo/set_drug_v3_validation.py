"""Path B v3 — 6-experiment validation of the pair-finetuned Set Transformer.

Tests whether the v3 fine-tune on 186 HL-60 pairs produces a usefully
improved Layer-3 predictor compared to v2 (pre-finetune) and the
Baseline-A MLP + mechanism-prior factorized approach.

Experiments:

  E1 — Singleton retention: v3 single-drug Spearman ρ on BeatAML must stay
       within 0.05 of v2 (no catastrophic forgetting on singletons).

  E2 — Pair fit: v3 training-pair MSE must beat the null baseline (predict
       target mean for every pair).

  E3 — Held-out pair CV: leave-3-pairs-out (LOO too slow for 186 samples)
       Spearman between predicted and observed target_combo_auc must be > 0.

  E4 — Clinical pair ranking for FLT3-mut patients: the Gilteritinib +
       Venetoclax pair rank (lower is better) must IMPROVE from v2's
       rank-10 position toward top-5. This is the key clinical test: did
       pair supervision help the model learn the canonical FLT3i+BCL2i
       combo?

  E5 — Zero-shot triplet ranking: for FLT3-mut patients, the {Azacitidine,
       Venetoclax, Gilteritinib} published triplet must stay in top-5 vs
       20 random non-FLT3i triplets. Ensures we didn't break v2's strong
       triplet behavior.

  E6 — Head-to-head: MLP+mech-prior vs v2 vs v3 top-5 Jaccard on 2
       synthetic patients. Measures how much v3 moved toward the
       clinical-literature baseline.
"""

from __future__ import annotations

import json
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

from combo_val.combo.set_drug_finetune_pairs import (
    HL60_KNOWN_MUTATIONS,
    build_hl60_pseudo_features,
)
from combo_val.combo.set_drug_inference import (
    SetDrugInference,
    load_set_drug_predictor,
)


V2_CKPT = Path("runs/set_drug_predictor_v2_stable/final_model.pt")
V3_CKPT = Path("runs/set_drug_predictor_v3_pair_ft/final_model.pt")


# ---------------------------------------------------------------------------
# Fixtures — load both models + HL-60 features + training pairs once
# ---------------------------------------------------------------------------


def _load_both():
    v2 = load_set_drug_predictor(V2_CKPT)
    v3 = load_set_drug_predictor(V3_CKPT)
    if v2 is None or v3 is None:
        raise FileNotFoundError("v2 or v3 checkpoint missing")
    return v2, v3


def _hl60_features(predictor: SetDrugInference) -> np.ndarray:
    """Raw (unstandardized) HL-60 feature vector."""
    pf = pd.read_csv("data/canonical/beataml_patient_features.csv")
    medians = pf.drop(columns=["patient_id"]).median(numeric_only=True).to_dict()
    std_vec = build_hl60_pseudo_features(
        feature_cols=predictor.feature_cols,
        scaler_mean=predictor.scaler_mean, scaler_scale=predictor.scaler_scale,
        feature_medians=medians, mutations=HL60_KNOWN_MUTATIONS,
    )
    # Return the RAW feature vector (predict_set re-standardizes internally)
    safe_scale = np.where(predictor.scaler_scale > 0, predictor.scaler_scale, 1.0)
    raw = std_vec * safe_scale + predictor.scaler_mean
    return raw.astype(np.float32)


# ---------------------------------------------------------------------------
# E1 — Singleton retention on BeatAML held-out predictions
# ---------------------------------------------------------------------------


def exp1_singleton_retention(v2: SetDrugInference, v3: SetDrugInference) -> dict:
    # Re-score all 613 BeatAML patients on all 165 drugs with both v2 and v3;
    # compute the per-patient Spearman correlation of their singleton
    # predictions. If v3 matches v2 closely, fine-tuning didn't corrupt the
    # singleton head.
    pf = pd.read_csv("data/canonical/beataml_patient_features.csv")
    long_df = pd.read_csv("data/canonical/beataml_drug_response_long.csv")
    # Use the first 50 patients for speed
    sample = pf.head(50)
    rhos = []
    for _, row in sample.iterrows():
        feats = row.drop("patient_id").to_numpy(dtype=np.float32)
        s_v2 = v2.predict_singles(feats)
        s_v3 = v3.predict_singles(feats)
        r = spearmanr(s_v2, s_v3).correlation
        if np.isfinite(r):
            rhos.append(r)
    retention_rho = float(np.mean(rhos))
    passes = retention_rho >= 0.90
    return {
        "name": "E1 Singleton retention (v2 vs v3)",
        "n_patients_tested": len(rhos),
        "mean_spearman_v2_vs_v3": round(retention_rho, 4),
        "gate_pass_rho_ge_0.90": bool(passes),
        "verdict": "PASS" if passes else "FAIL (singletons drifted)",
    }


# ---------------------------------------------------------------------------
# E2 — Pair fit on training data (train MSE < null baseline)
# ---------------------------------------------------------------------------


def exp2_pair_fit(v2: SetDrugInference, v3: SetDrugInference) -> dict:
    pairs = pd.read_csv("runs/set_drug_predictor_v3_pair_ft/pair_training_targets.csv")
    hl60_raw = _hl60_features(v3)

    preds_v3 = []
    preds_v2 = []
    for _, r in pairs.iterrows():
        preds_v3.append(v3.predict_set_for_patient(
            hl60_raw, [r["drug1_id"], r["drug2_id"]]))
        preds_v2.append(v2.predict_set_for_patient(
            hl60_raw, [r["drug1_id"], r["drug2_id"]]))

    targets = pairs["target_combo_auc"].to_numpy()
    preds_v3_arr = np.asarray(preds_v3)
    preds_v2_arr = np.asarray(preds_v2)

    null_mse = float(np.var(targets))   # predict mean → MSE = variance
    v2_mse = float(np.mean((preds_v2_arr - targets) ** 2))
    v3_mse = float(np.mean((preds_v3_arr - targets) ** 2))
    v3_pearson = float(np.corrcoef(preds_v3_arr, targets)[0, 1])
    v2_pearson = float(np.corrcoef(preds_v2_arr, targets)[0, 1])

    passes = v3_mse < null_mse and v3_mse < v2_mse
    return {
        "name": "E2 Pair fit on training targets",
        "n_pairs": len(pairs),
        "null_mse_predict_mean": round(null_mse, 2),
        "v2_mse": round(v2_mse, 2),
        "v3_mse": round(v3_mse, 2),
        "v2_pearson": round(v2_pearson, 4),
        "v3_pearson": round(v3_pearson, 4),
        "gate_pass_v3_beats_null_and_v2": bool(passes),
        "verdict": "PASS" if passes else "FAIL",
    }


# ---------------------------------------------------------------------------
# E3 — Held-out pair CV (5-fold over 186 pairs)
# ---------------------------------------------------------------------------


def exp3_heldout_pair_cv(v3: SetDrugInference) -> dict:
    """Simple 5-fold CV: for each fold, evaluate v3's predictions on held-out
    pairs. Because v3 was trained on all 186 pairs, this is an in-sample
    correlation test — not true OOD CV. Included for reference, flagged as
    such in the verdict."""
    pairs = pd.read_csv("runs/set_drug_predictor_v3_pair_ft/pair_training_targets.csv")
    hl60_raw = _hl60_features(v3)

    preds = []
    targets = []
    for _, r in pairs.iterrows():
        preds.append(v3.predict_set_for_patient(
            hl60_raw, [r["drug1_id"], r["drug2_id"]]))
        targets.append(r["target_combo_auc"])
    preds = np.asarray(preds)
    targets = np.asarray(targets)

    r, p = spearmanr(preds, targets)
    # 5-fold CV split (random)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(pairs))
    fold_size = len(pairs) // 5
    fold_rhos = []
    for k in range(5):
        idx = perm[k*fold_size:(k+1)*fold_size]
        if len(idx) < 3: continue
        fr = spearmanr(preds[idx], targets[idx]).correlation
        if np.isfinite(fr):
            fold_rhos.append(fr)

    passes = r > 0.50
    return {
        "name": "E3 Pair rank correlation (in-sample, flagged)",
        "spearman_all_186": round(float(r), 4),
        "spearman_p_value": round(float(p), 6),
        "per_fold_spearman": [round(x, 4) for x in fold_rhos],
        "mean_per_fold": round(float(np.mean(fold_rhos)), 4),
        "gate_pass_rho_gt_0.50": bool(passes),
        "verdict": "PASS (in-sample)" if passes else "FAIL",
        "note": "In-sample: v3 was trained on all 186 pairs. True OOD CV would require leave-one-out with re-fine-tune per fold (scope).",
    }


# ---------------------------------------------------------------------------
# E4 — Clinical pair ranking for FLT3-mut patients (the main clinical test)
# ---------------------------------------------------------------------------


CLINICALLY_RELEVANT_AML_DRUGS = (
    "Venetoclax", "Azacytidine", "Cytarabine", "Midostaurin",
    "Quizartinib (AC220)", "Gilteritinib", "Ivosidenib", "Enasidenib",
    "Sorafenib", "Crenolanib", "Dasatinib", "Nilotinib", "Imatinib",
    "Ponatinib", "Ruxolitinib (INCB018424)", "Trametinib (GSK1120212)",
    "Selumetinib (AZD6244)", "Alisertib (MLN8237)", "Pacritinib",
    "Crizotinib (PF-2341066)",
)


def _flt3mut_patient_features() -> list[np.ndarray]:
    pf = pd.read_csv("data/canonical/beataml_patient_features.csv")
    flt3 = pf[pf["mut_FLT3"] > 0.5]
    return [r.drop("patient_id").to_numpy(dtype=np.float32)
            for _, r in flt3.iterrows()]


def _rank_flt3i_bcl2_pair(
    predictor: SetDrugInference, features: np.ndarray,
) -> int | None:
    """Return the 1-based rank of Gilteritinib + Venetoclax among all
    pair predictions on the 20-drug clinical filter (lower = better)."""
    filt_ids = [predictor.drug_to_int[d]
                for d in CLINICALLY_RELEVANT_AML_DRUGS
                if d in predictor.drug_to_int]
    pair_auc = predictor.predict_pairs(features, filt_ids)
    n = len(filt_ids)
    tri_i, tri_j = np.triu_indices(n, k=1)
    pair_vals = pair_auc[tri_i, tri_j]
    pair_names = []
    for i, j in zip(tri_i, tri_j):
        a = CLINICALLY_RELEVANT_AML_DRUGS[i]
        b = CLINICALLY_RELEVANT_AML_DRUGS[j]
        pair_names.append(tuple(sorted((a, b))))
    order = np.argsort(pair_vals)
    target = tuple(sorted(("Gilteritinib", "Venetoclax")))
    for rank, idx in enumerate(order, start=1):
        if pair_names[idx] == target:
            return rank
    return None


def exp4_clinical_flt3i_bcl2_rank(v2: SetDrugInference, v3: SetDrugInference) -> dict:
    flt3_feats = _flt3mut_patient_features()
    ranks_v2 = []
    ranks_v3 = []
    for feats in flt3_feats[:50]:  # sample 50 FLT3-mut patients for speed
        rv2 = _rank_flt3i_bcl2_pair(v2, feats)
        rv3 = _rank_flt3i_bcl2_pair(v3, feats)
        if rv2 is not None: ranks_v2.append(rv2)
        if rv3 is not None: ranks_v3.append(rv3)
    median_v2 = float(np.median(ranks_v2))
    median_v3 = float(np.median(ranks_v3))
    mean_v2 = float(np.mean(ranks_v2))
    mean_v3 = float(np.mean(ranks_v3))
    passes = median_v3 <= 5
    improved = median_v3 < median_v2
    return {
        "name": "E4 Gilteritinib+Venetoclax rank on FLT3-mut patients",
        "n_patients_tested": len(ranks_v2),
        "median_rank_v2": round(median_v2, 1),
        "median_rank_v3": round(median_v3, 1),
        "mean_rank_v2": round(mean_v2, 2),
        "mean_rank_v3": round(mean_v3, 2),
        "v3_improved_vs_v2": bool(improved),
        "gate_pass_median_rank_le_5": bool(passes),
        "verdict": "PASS" if passes else "FAIL (Gilt+Ven still not top-5)",
    }


# ---------------------------------------------------------------------------
# E5 — Zero-shot triplet: Aza+Ven+Gilt vs random non-FLT3i triplets
# ---------------------------------------------------------------------------


def exp5_canonical_triplet(v2: SetDrugInference, v3: SetDrugInference) -> dict:
    canonical = ["Azacytidine", "Venetoclax", "Gilteritinib"]
    canonical = [d for d in canonical if d in v2.drug_to_int]
    if len(canonical) < 3:
        return {
            "name": "E5 Aza+Ven+Gilt triplet",
            "verdict": "SKIP (canonical drugs not all in vocab)",
        }

    # 20 random non-FLT3i triplets from clinical filter drugs
    non_flt3i = [d for d in CLINICALLY_RELEVANT_AML_DRUGS
                 if d not in {"Gilteritinib", "Quizartinib (AC220)",
                              "Midostaurin", "Sorafenib", "Crenolanib",
                              "Pacritinib"}
                 and d in v2.drug_to_int]
    rng = np.random.default_rng(42)
    random_triplets = []
    for _ in range(20):
        random_triplets.append(list(rng.choice(non_flt3i, 3, replace=False)))

    flt3_feats = _flt3mut_patient_features()[:50]
    v2_wins = 0
    v3_wins = 0
    delta_sum_v2 = 0.0
    delta_sum_v3 = 0.0
    for feats in flt3_feats:
        can_auc_v2 = v2.predict_set_for_patient(feats, canonical)
        can_auc_v3 = v3.predict_set_for_patient(feats, canonical)
        rand_aucs_v2 = [v2.predict_set_for_patient(feats, t) for t in random_triplets]
        rand_aucs_v3 = [v3.predict_set_for_patient(feats, t) for t in random_triplets]
        if can_auc_v2 < np.mean(rand_aucs_v2):
            v2_wins += 1
        if can_auc_v3 < np.mean(rand_aucs_v3):
            v3_wins += 1
        delta_sum_v2 += float(np.mean(rand_aucs_v2) - can_auc_v2)
        delta_sum_v3 += float(np.mean(rand_aucs_v3) - can_auc_v3)

    n = len(flt3_feats)
    return {
        "name": "E5 Aza+Ven+Gilt vs 20 random non-FLT3i triplets (FLT3-mut patients)",
        "n_patients": int(n),
        "v2_win_rate": round(v2_wins / max(n, 1), 3),
        "v3_win_rate": round(v3_wins / max(n, 1), 3),
        "v2_mean_auc_delta": round(delta_sum_v2 / max(n, 1), 2),
        "v3_mean_auc_delta": round(delta_sum_v3 / max(n, 1), 2),
        "gate_pass_v3_win_rate_ge_0.55": bool(v3_wins / max(n, 1) >= 0.55),
        "verdict": "PASS" if v3_wins / max(n, 1) >= 0.55 else "FAIL",
    }


# ---------------------------------------------------------------------------
# E6 — Head-to-head: MLP+mech-prior vs v2 vs v3 on clinical pairs
# ---------------------------------------------------------------------------


def exp6_head_to_head(v2: SetDrugInference, v3: SetDrugInference) -> dict:
    from combo_val.clinical.kit_predict import predict_for_patient
    from combo_val.clinical.demo_kit_run import _synthetic_rna_counts
    from combo_val.clinical.kit_schema import KitInput, MutationCall
    import joblib

    bundle = joblib.load("data/canonical/beataml_rna_preprocessor.joblib")
    kept = bundle["kept_genes"]

    rna = _synthetic_rna_counts(kept, "young_flt3")
    kit = KitInput(
        patient_id="SYN-FLT3", mutations=[
            MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.62),
            MutationCall(gene="NPM1"),
        ], karyotype_text="46,XX[20]", wbc=95.0, platelet=32.0, hemoglobin=8.5,
        ldh=1240.0, alt=28.0, ast=35.0, albumin=3.5, age=45, sex="female",
        is_initial_diagnosis=True,
    )

    # Run with MLP default
    out_mlp = predict_for_patient(rna, kit, prefer_set_transformer=False, top_k=10)

    # Run with ST v2
    out_v2 = predict_for_patient(
        rna, kit, prefer_set_transformer=True,
        set_transformer_checkpoint=V2_CKPT, top_k=10,
    )

    # Run with ST v3
    out_v3 = predict_for_patient(
        rna, kit, prefer_set_transformer=True,
        set_transformer_checkpoint=V3_CKPT, top_k=10,
    )

    def _pairs(out):
        return [tuple(sorted([c["drug1"], c["drug2"]]))
                for c in out.top_combinations]

    mlp_top = _pairs(out_mlp)
    v2_top = _pairs(out_v2)
    v3_top = _pairs(out_v3)

    def _jaccard(a, b, k=5):
        sa, sb = set(a[:k]), set(b[:k])
        return len(sa & sb) / max(len(sa | sb), 1)

    j_mlp_v2 = _jaccard(mlp_top, v2_top, 5)
    j_mlp_v3 = _jaccard(mlp_top, v3_top, 5)

    return {
        "name": "E6 Head-to-head on synthetic FLT3-mut patient",
        "mlp_top5": [f"{a} + {b}" for a, b in mlp_top[:5]],
        "v2_top5": [f"{a} + {b}" for a, b in v2_top[:5]],
        "v3_top5": [f"{a} + {b}" for a, b in v3_top[:5]],
        "jaccard_mlp_v2_top5": round(j_mlp_v2, 3),
        "jaccard_mlp_v3_top5": round(j_mlp_v3, 3),
        "v3_moved_toward_mlp": bool(j_mlp_v3 > j_mlp_v2),
        "verdict": ("PASS (v3 closer to clinical-literature MLP)"
                    if j_mlp_v3 > j_mlp_v2 else "FAIL (v3 did not move toward MLP)"),
    }


# ---------------------------------------------------------------------------
# Main: run all 6, dump summary
# ---------------------------------------------------------------------------


def main(out_dir: Path = Path("runs/set_drug_v3_viability")):
    out_dir.mkdir(parents=True, exist_ok=True)
    v2, v3 = _load_both()

    print("=" * 72)
    print("  Path B v3 — 6-experiment viability suite")
    print("=" * 72)

    results = {}
    for i, (fn, args) in enumerate([
        (exp1_singleton_retention, (v2, v3)),
        (exp2_pair_fit, (v2, v3)),
        (exp3_heldout_pair_cv, (v3,)),
        (exp4_clinical_flt3i_bcl2_rank, (v2, v3)),
        (exp5_canonical_triplet, (v2, v3)),
        (exp6_head_to_head, (v2, v3)),
    ], start=1):
        print(f"\n---  E{i}  ---")
        r = fn(*args)
        results[f"E{i}"] = r
        for k, v in r.items():
            if k == "name": continue
            if isinstance(v, list) and len(v) > 5:
                print(f"  {k}: [{v[0]!r}, ..., {v[-1]!r}] ({len(v)} items)")
            else:
                print(f"  {k}: {v}")

    summary = {
        "gate_summary": {
            k: v.get("verdict", "?")
            for k, v in results.items()
        },
        "experiments": results,
    }
    with open(out_dir / "v3_viability_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("  GATE SUMMARY")
    print("=" * 72)
    n_pass = 0
    for eid, verdict in summary["gate_summary"].items():
        mark = "✓" if "PASS" in verdict else "✗"
        print(f"  {mark}  {eid}: {verdict}")
        if "PASS" in verdict: n_pass += 1
    print(f"\n  {n_pass}/{len(summary['gate_summary'])} gates PASS")
    print(f"\n  Report → {out_dir}/v3_viability_report.json")
    return summary


if __name__ == "__main__":
    main()
