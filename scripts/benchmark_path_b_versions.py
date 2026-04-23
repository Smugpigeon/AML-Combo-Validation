"""Comprehensive benchmark: 6 metrics × 5 baselines for Path B comparison.

Baselines compared:
  1. Baseline A MLP + mechanism prior + factorized combo  (current kit default)
  2. Path A alone (clonal-coverage Bliss-IDA)
  3. ST v1 (pre-stability-fix, fold-3 collapse)  [if checkpoint exists]
  4. ST v2 (stability fix, single-drug only)
  5. ST v3 (v2 + Path A distillation)              [this work]

Metrics (all patient-level, averaged over 179 FLT3-mut BeatAML patients
unless otherwise stated):
  M1. Single-drug 5-fold CV ρ           — gate ≥ 0.65
  M2. FLT3i+BCL2i canonical pair rank   — lower = more clinically aligned
  M3. Jaccard top-5 with MLP+mech-prior — how much does pair ranking agree
  M4. Spearman with Path A coverage     — does it align with biology
  M5. Zero-shot T5-style triplet test   — Δ (Gilt+Ven+Aza vs random triplet)
  M6. Bliss consistency rate            — |obs - Bliss_pred| within 2σ band

Output:
  runs/path_b_benchmark/results.json
  runs/path_b_benchmark/comparison_table.csv
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from combo_val.combo.clonal_coverage import (
    CLONE_ARCHETYPES,
    build_drug_clone_coverage,
    build_patient_clone_matrix,
    rank_top_combos_per_patient,
    score_combo_for_patient,
)
from combo_val.combo.mechanism_prior import compute_combo_mech_scores
from combo_val.combo.set_drug_inference import SetDrugInference, load_set_drug_predictor
from combo_val.baselines.single_drug_mlp import SingleDrugMLP, SingleDrugMLPConfig


CLINICAL_DRUG_FILTER = (
    "Venetoclax", "Azacytidine", "Cytarabine", "Midostaurin",
    "Quizartinib (AC220)", "Gilteritinib", "Ivosidenib", "Enasidenib",
    "Sorafenib", "Crenolanib", "Dasatinib", "Nilotinib", "Imatinib",
    "Ponatinib", "Ruxolitinib (INCB018424)", "Trametinib (GSK1120212)",
    "Selumetinib (AZD6244)", "Alisertib (MLN8237)", "Pacritinib",
    "Crizotinib (PF-2341066)",
)

CANONICAL_FLT3_PAIRS = {
    "Gilteritinib + Venetoclax",
    "Quizartinib (AC220) + Venetoclax",
}


# ---------------------------------------------------------------------------
# Baseline scorers (all return pair_auc ndarray (n, n) for the 20-drug panel)
# ---------------------------------------------------------------------------


def _load_mlp(checkpoint_path: Path) -> tuple[SingleDrugMLP, dict]:
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    from dataclasses import fields as _fields
    cfg_dict = ckpt["cfg"]
    kwargs = {f.name: cfg_dict[f.name] for f in _fields(SingleDrugMLPConfig)
              if f.name in cfg_dict}
    cfg = SingleDrugMLPConfig(**kwargs)
    model = SingleDrugMLP(
        n_patient_features=ckpt["n_patient_features"],
        n_drugs=ckpt["n_drugs"], cfg=cfg,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def score_mlp_plus_mech(
    patient_feats_df: pd.DataFrame,          # (n_patients, n_feat) unscaled
    drug_filter: tuple[str, ...],
    mech_prior_scale: float = 30.0,
) -> np.ndarray:
    """Returns (n_patients, n_drugs, n_drugs) predicted combo AUC."""
    model, ckpt = _load_mlp(Path("runs/baseline_single_drug_mlp/final_model.pt"))
    feat_cols = ckpt["feature_cols"]
    drug_vocab = ckpt["drug_vocab"]
    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
    safe_scale = np.where(scaler_scale > 0, scaler_scale, 1.0)

    mask = np.array([d in drug_filter for d in drug_vocab])
    keep_idx = np.where(mask)[0]
    drug_vocab_filt = [drug_vocab[i] for i in keep_idx]

    pf_aligned = patient_feats_df[feat_cols].to_numpy(dtype=np.float32)
    standardized = (pf_aligned - scaler_mean) / safe_scale
    n_p = standardized.shape[0]
    n_d = len(drug_vocab_filt)
    combo_auc = np.zeros((n_p, n_d, n_d), dtype=np.float32)

    # Mech prior for all patients (vectorized)
    mech_scores = compute_combo_mech_scores(
        patient_feats_df[feat_cols], list(drug_vocab_filt),
    )  # (n_p, n_d, n_d)

    # Per-patient MLP predictions
    for pi in range(n_p):
        pf_t = torch.tensor(standardized[pi], dtype=torch.float32)
        with torch.no_grad():
            pred_auc = model.predict_all_drugs_for_patient(pf_t, torch.device("cpu")).numpy()
        pred_auc_filt = pred_auc[keep_idx]
        combo_auc[pi] = (
            0.5 * (pred_auc_filt[:, None] + pred_auc_filt[None, :])
            - mech_prior_scale * mech_scores[pi]
        )
    return combo_auc, drug_vocab_filt


def score_path_a(patient_feats_df: pd.DataFrame, drug_filter: tuple[str, ...]) -> np.ndarray:
    """Returns (n_patients, n_drugs, n_drugs) with Path A coverage as NEGATIVE
    AUC (so higher coverage ⇒ lower "AUC-equivalent" — comparable to combo_auc).
    Actually returns coverage directly; callers should handle sign."""
    drugs = list(drug_filter)
    patient_clones = build_patient_clone_matrix(patient_feats_df)
    drug_clone_cov = build_drug_clone_coverage(drugs).to_numpy(dtype=np.float64)
    clones_arr = patient_clones.to_numpy(dtype=np.float64)
    n_p, n_d = clones_arr.shape[0], len(drugs)
    coverage = np.zeros((n_p, n_d, n_d), dtype=np.float32)
    for i in range(n_d):
        for j in range(n_d):
            if i == j:
                continue
            cov_i = np.clip(drug_clone_cov[i], 0.0, 1.0)
            cov_j = np.clip(drug_clone_cov[j], 0.0, 1.0)
            pair_cov = 1.0 - (1.0 - cov_i) * (1.0 - cov_j)
            present = (clones_arr > 0).astype(np.float64)
            weights = clones_arr * present
            total_w = weights.sum(axis=1)
            num = (weights * pair_cov[None, :]).sum(axis=1)
            coverage[:, i, j] = np.where(total_w > 0, num / total_w, 0.0)
    return coverage, drugs


def score_set_transformer(
    patient_feats_df: pd.DataFrame,
    drug_filter: tuple[str, ...],
    checkpoint_path: Path,
) -> np.ndarray:
    """Returns (n_patients, n_drugs, n_drugs) predicted combo AUC from ST."""
    st = load_set_drug_predictor(checkpoint_path)
    if st is None:
        raise FileNotFoundError(f"checkpoint missing: {checkpoint_path}")

    # Align patient features to ST's expected schema
    feat_cols = st.feature_cols
    pf_aligned = patient_feats_df[feat_cols].to_numpy(dtype=np.float32)

    mask = np.array([d in drug_filter for d in st.drug_vocab])
    keep_idx = np.where(mask)[0]
    drug_vocab_filt = [st.drug_vocab[i] for i in keep_idx]
    st_indices = [st.drug_to_int[d] for d in drug_vocab_filt]

    n_p = pf_aligned.shape[0]
    n_d = len(drug_vocab_filt)
    combo_auc = np.zeros((n_p, n_d, n_d), dtype=np.float32)

    for pi in range(n_p):
        pair_matrix = st.predict_pairs(pf_aligned[pi], st_indices)
        combo_auc[pi] = 0.5 * (pair_matrix + pair_matrix.T)  # force symmetry
    return combo_auc, drug_vocab_filt


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def m2_flt3_canonical_rank(
    pair_auc: np.ndarray, drug_vocab_filt: list[str],
    flt3_mut_mask: np.ndarray,
) -> dict:
    """Mean rank of canonical FLT3i+BCL2i pairs among FLT3-mut patients."""
    n_d = len(drug_vocab_filt)
    tri_i, tri_j = np.triu_indices(n_d, k=1)
    pair_names = [
        " + ".join(sorted([drug_vocab_filt[i], drug_vocab_filt[j]]))
        for i, j in zip(tri_i, tri_j)
    ]
    is_canonical = np.array([n in CANONICAL_FLT3_PAIRS for n in pair_names])
    best_rank_list = []
    for pi in np.where(flt3_mut_mask)[0]:
        vals = pair_auc[pi, tri_i, tri_j]
        # Lower AUC = better. Handle score fields that use "higher = better"
        # by caller-provided sign; here we assume AUC (lower better).
        order = np.argsort(vals)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        canonical_ranks = ranks[is_canonical] + 1  # 1-indexed
        best_rank_list.append(int(canonical_ranks.min()))
    return {
        "mean_best_canonical_rank": round(float(np.mean(best_rank_list)), 2),
        "median_best_canonical_rank": int(np.median(best_rank_list)),
        "n_flt3_mut": int(flt3_mut_mask.sum()),
    }


def m3_jaccard_vs_reference(
    pair_auc: np.ndarray, reference_auc: np.ndarray,
    drug_vocab_filt: list[str], top_k: int = 5,
) -> float:
    """Mean Jaccard(top-k pairs) between `pair_auc` ranking and reference."""
    n_d = len(drug_vocab_filt)
    tri_i, tri_j = np.triu_indices(n_d, k=1)
    pair_names = np.array([
        " + ".join(sorted([drug_vocab_filt[i], drug_vocab_filt[j]]))
        for i, j in zip(tri_i, tri_j)
    ])
    jaccards = []
    for pi in range(pair_auc.shape[0]):
        a = set(pair_names[np.argsort(pair_auc[pi, tri_i, tri_j])[:top_k]])
        b = set(pair_names[np.argsort(reference_auc[pi, tri_i, tri_j])[:top_k]])
        jaccards.append(len(a & b) / max(len(a | b), 1))
    return round(float(np.mean(jaccards)), 3)


def m4_spearman_with_path_a(
    pair_auc: np.ndarray, path_a_coverage: np.ndarray,
    drug_vocab_filt: list[str],
) -> float:
    """Mean Spearman correlation between -pair_auc and Path A coverage across patients."""
    n_d = len(drug_vocab_filt)
    tri_i, tri_j = np.triu_indices(n_d, k=1)
    rhos = []
    for pi in range(pair_auc.shape[0]):
        aucs = pair_auc[pi, tri_i, tri_j]
        covs = path_a_coverage[pi, tri_i, tri_j]
        if np.std(aucs) < 1e-6 or np.std(covs) < 1e-6:
            continue
        r = spearmanr(-aucs, covs).correlation
        if not np.isnan(r):
            rhos.append(r)
    return round(float(np.mean(rhos)), 3)


def m5_zero_shot_triplet(
    score_fn, patient_feats_df, drug_filter, flt3_mut_mask,
    canonical_triplet=("Gilteritinib", "Venetoclax", "Azacytidine"),
    n_random_triplets: int = 20,
    random_state: int = 42,
    valid_drugs: list[str] | None = None,
) -> dict:
    """T5-style: does canonical triplet beat random triplets for FLT3-mut?

    Returns Δ (mean AUC difference: random − canonical, positive = canonical wins)
    plus win-rate fraction of patients.

    score_fn must accept a list of drug NAMES (for a single patient) and return
    a predicted AUC. We support ST, MLP, and Path A here.

    valid_drugs : optional restriction — random triplets sampled only from here
        (use to exclude drugs not in a specific backbone's vocab).
    """
    rng = np.random.default_rng(random_state)
    drugs = list(drug_filter) if valid_drugs is None else list(valid_drugs)
    non_flt3_drugs = [d for d in drugs if "flt3" not in d.lower()
                      and "Gilt" not in d and "Quiz" not in d and "Mido" not in d
                      and "Sorafenib" != d and "Crenolanib" != d and "Ponatinib" != d
                      and "Pacritinib" != d]
    if len(non_flt3_drugs) < 3:
        return {"mean_delta_random_minus_canonical": float("nan"),
                "win_rate": float("nan"), "n_flt3_mut": 0,
                "error": "insufficient non-FLT3 drugs in valid_drugs"}
    flt3_patient_rows = np.where(flt3_mut_mask)[0]
    deltas = []
    wins = 0
    for pi in flt3_patient_rows:
        canonical_auc = score_fn(patient_feats_df.iloc[pi], list(canonical_triplet))
        random_aucs = []
        for _ in range(n_random_triplets):
            rand_triplet = list(rng.choice(non_flt3_drugs, size=3, replace=False))
            random_aucs.append(score_fn(patient_feats_df.iloc[pi], rand_triplet))
        delta = np.mean(random_aucs) - canonical_auc
        deltas.append(delta)
        if delta > 0:
            wins += 1
    return {
        "mean_delta_random_minus_canonical": round(float(np.mean(deltas)), 3),
        "win_rate": round(float(wins / len(flt3_patient_rows)), 3),
        "n_flt3_mut": int(len(flt3_patient_rows)),
    }


def m6_bliss_consistency(
    pair_auc: np.ndarray, single_auc: np.ndarray,
    drug_vocab_filt: list[str],
    scale: float = 300.0, tolerance: float = 50.0,
) -> float:
    """Fraction of pair predictions within `tolerance` AUC of Bliss theoretical.

    Bliss: expected_pair = A + B - A*B/scale
    """
    n_d = len(drug_vocab_filt)
    tri_i, tri_j = np.triu_indices(n_d, k=1)
    n_in_bounds = 0
    total = 0
    for pi in range(pair_auc.shape[0]):
        s = single_auc[pi]  # (n_d,)
        for i, j in zip(tri_i, tri_j):
            bliss = s[i] + s[j] - s[i] * s[j] / scale
            if abs(pair_auc[pi, i, j] - bliss) <= tolerance:
                n_in_bounds += 1
            total += 1
    return round(float(n_in_bounds / total), 3)


# ---------------------------------------------------------------------------
# Single-drug helpers (needed for M6 Bliss)
# ---------------------------------------------------------------------------


def get_mlp_single_auc(patient_feats_df: pd.DataFrame, drug_filter: tuple[str, ...]) -> np.ndarray:
    model, ckpt = _load_mlp(Path("runs/baseline_single_drug_mlp/final_model.pt"))
    feat_cols = ckpt["feature_cols"]
    drug_vocab = ckpt["drug_vocab"]
    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
    safe_scale = np.where(scaler_scale > 0, scaler_scale, 1.0)
    mask = np.array([d in drug_filter for d in drug_vocab])
    keep_idx = np.where(mask)[0]

    pf_aligned = patient_feats_df[feat_cols].to_numpy(dtype=np.float32)
    standardized = (pf_aligned - scaler_mean) / safe_scale

    out = np.zeros((pf_aligned.shape[0], len(keep_idx)), dtype=np.float32)
    for pi in range(pf_aligned.shape[0]):
        pf_t = torch.tensor(standardized[pi], dtype=torch.float32)
        with torch.no_grad():
            pred_auc = model.predict_all_drugs_for_patient(pf_t, torch.device("cpu")).numpy()
        out[pi] = pred_auc[keep_idx]
    return out


def get_st_single_auc(
    patient_feats_df: pd.DataFrame, drug_filter: tuple[str, ...], checkpoint_path: Path,
) -> np.ndarray:
    st = load_set_drug_predictor(checkpoint_path)
    if st is None:
        raise FileNotFoundError(str(checkpoint_path))
    feat_cols = st.feature_cols
    pf_aligned = patient_feats_df[feat_cols].to_numpy(dtype=np.float32)
    mask = np.array([d in drug_filter for d in st.drug_vocab])
    keep_idx = np.where(mask)[0]
    out = np.zeros((pf_aligned.shape[0], len(keep_idx)), dtype=np.float32)
    for pi in range(pf_aligned.shape[0]):
        singles = st.predict_singles(pf_aligned[pi])
        out[pi] = singles[keep_idx]
    return out


# ---------------------------------------------------------------------------
# ST-based triplet score (for M5)
# ---------------------------------------------------------------------------


def _make_st_triplet_scorer(checkpoint_path: Path, feat_cols: list[str]):
    st = load_set_drug_predictor(checkpoint_path)
    if st is None:
        raise FileNotFoundError(str(checkpoint_path))
    # Align feat_cols
    def scorer(patient_row: pd.Series, drug_names: list[str]) -> float:
        features = patient_row[st.feature_cols].to_numpy(dtype=np.float32)
        return st.predict_set_for_patient(features, drug_names)
    return scorer


def _make_mlp_triplet_scorer(feat_cols: list[str]):
    """For MLP: triplet "AUC" = mean single-drug AUC across the 3 drugs."""
    model, ckpt = _load_mlp(Path("runs/baseline_single_drug_mlp/final_model.pt"))
    feat_cols_ckpt = ckpt["feature_cols"]
    drug_vocab = ckpt["drug_vocab"]
    drug_to_int = {d: i for i, d in enumerate(drug_vocab)}
    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
    safe_scale = np.where(scaler_scale > 0, scaler_scale, 1.0)

    def scorer(patient_row: pd.Series, drug_names: list[str]) -> float:
        pf = patient_row[feat_cols_ckpt].to_numpy(dtype=np.float32)
        standardized = (pf - scaler_mean) / safe_scale
        pf_t = torch.tensor(standardized, dtype=torch.float32)
        with torch.no_grad():
            pred_auc = model.predict_all_drugs_for_patient(pf_t, torch.device("cpu")).numpy()
        vals = [pred_auc[drug_to_int[d]] for d in drug_names if d in drug_to_int]
        return float(np.mean(vals))
    return scorer


def _make_path_a_triplet_scorer(drug_filter: tuple[str, ...]):
    """Path A gives coverage (higher = better); negate to be "AUC-like"."""
    drugs = list(drug_filter)
    drug_clone_cov = build_drug_clone_coverage(drugs).to_numpy(dtype=np.float64)
    drug_to_idx = {d: i for i, d in enumerate(drugs)}

    def scorer(patient_row: pd.Series, drug_names: list[str]) -> float:
        patient_clones = build_patient_clone_matrix(
            pd.DataFrame([patient_row]),
        ).iloc[0].to_numpy(dtype=np.float64)
        idxs = [drug_to_idx[d] for d in drug_names if d in drug_to_idx]
        if len(idxs) < len(drug_names):
            return 1.0  # missing drug → worst "AUC" to indicate cannot score
        cov_mat = drug_clone_cov[idxs]  # (N, n_clones)
        cov, _ = score_combo_for_patient(patient_clones, cov_mat)
        return float(1.0 - cov)  # lower = better
    return scorer


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------


def _load_single_drug_cv(ckpt_path: Path) -> dict:
    cv_path = ckpt_path.parent / "cv_metrics.json"
    if not cv_path.exists():
        return {"cv_mean_per_patient_spearman": None, "cv_mean_mae": None}
    d = json.loads(cv_path.read_text())
    o = d.get("overall_cv", {})
    return {
        "cv_mean_per_patient_spearman": o.get("cv_mean_per_patient_spearman"),
        "cv_mean_mae": o.get("cv_mean_mae"),
    }


def run_benchmark(out_dir: Path = Path("runs/path_b_benchmark")) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load BeatAML patients
    pf = pd.read_csv("data/canonical/beataml_patient_features.csv").set_index("patient_id")
    flt3_mut_mask = (pf["mut_FLT3"] == 1).to_numpy()
    print(f"[bench] {pf.shape[0]} patients, {flt3_mut_mask.sum()} FLT3-mut")

    drug_filter = CLINICAL_DRUG_FILTER

    # ---- Compute reference signals ----
    print("[bench] scoring MLP + mech-prior (reference baseline) ...")
    mlp_mech_combo, mlp_drug_vocab = score_mlp_plus_mech(pf.reset_index(), drug_filter)

    print("[bench] scoring Path A clonal coverage ...")
    path_a_cov, _ = score_path_a(pf, drug_filter)
    # Convert coverage (higher=better) → AUC-like (lower=better) via 1 - coverage
    path_a_auc_like = 1.0 - path_a_cov

    # MLP single-drug AUC for Bliss baseline
    print("[bench] scoring MLP single-drug AUC (for Bliss M6) ...")
    mlp_singles = get_mlp_single_auc(pf.reset_index(), drug_filter)

    results = {}

    # ===========================================================
    # Baseline 1: MLP + mech-prior
    # ===========================================================
    print("\n=== Baseline 1: MLP + mech-prior ===")
    b1_cv = _load_single_drug_cv(Path("runs/baseline_single_drug_mlp/final_model.pt"))
    r1 = {
        "name": "MLP + mech-prior",
        "m1_single_drug_cv_rho": b1_cv["cv_mean_per_patient_spearman"],
        "m1_single_drug_cv_mae": b1_cv["cv_mean_mae"],
        "m2_flt3_canonical_rank": m2_flt3_canonical_rank(
            mlp_mech_combo, mlp_drug_vocab, flt3_mut_mask,
        ),
        "m3_jaccard_top5_vs_mlp": 1.0,  # self-reference
        "m4_spearman_with_path_a": m4_spearman_with_path_a(
            mlp_mech_combo, path_a_cov, mlp_drug_vocab,
        ),
        "m5_zero_shot_triplet": m5_zero_shot_triplet(
            _make_mlp_triplet_scorer(list(pf.columns)), pf, drug_filter, flt3_mut_mask,
        ),
        "m6_bliss_consistency": m6_bliss_consistency(
            mlp_mech_combo, mlp_singles, mlp_drug_vocab,
        ),
    }
    results["mlp_mech_prior"] = r1
    print(json.dumps(r1, indent=2))

    # ===========================================================
    # Baseline 2: Path A alone
    # ===========================================================
    print("\n=== Baseline 2: Path A clonal coverage ===")
    r2 = {
        "name": "Path A (clonal coverage)",
        "m1_single_drug_cv_rho": None,  # Path A doesn't do single-drug AUC prediction
        "m1_single_drug_cv_mae": None,
        "m2_flt3_canonical_rank": m2_flt3_canonical_rank(
            path_a_auc_like, mlp_drug_vocab, flt3_mut_mask,
        ),
        "m3_jaccard_top5_vs_mlp": m3_jaccard_vs_reference(
            path_a_auc_like, mlp_mech_combo, mlp_drug_vocab,
        ),
        "m4_spearman_with_path_a": 1.0,  # self
        "m5_zero_shot_triplet": m5_zero_shot_triplet(
            _make_path_a_triplet_scorer(drug_filter), pf, drug_filter, flt3_mut_mask,
        ),
        "m6_bliss_consistency": None,  # different scale
    }
    results["path_a"] = r2
    print(json.dumps(r2, indent=2))

    # ===========================================================
    # Baseline 3: ST v2 (single-drug only)
    # ===========================================================
    v2_path = Path("runs/set_drug_predictor_v2_stable/final_model.pt")
    if v2_path.exists():
        print("\n=== Baseline 3: ST v2 (single-drug only) ===")
        # ST v2 may have dropped some drugs from its vocab during training
        # (e.g., Enasidenib, Ponatinib missing due to BeatAML coverage).
        # Filter drug_filter to what's actually in this ST vocab.
        st_v2 = load_set_drug_predictor(v2_path)
        st_v2_valid_drugs = [d for d in drug_filter if d in st_v2.drug_to_int]
        print(f"  ST v2 vocab has {len(st_v2_valid_drugs)}/{len(drug_filter)} clinical drugs")

        st_v2_combo, _ = score_set_transformer(pf.reset_index(), drug_filter, v2_path)
        st_v2_singles = get_st_single_auc(pf.reset_index(), drug_filter, v2_path)
        v2_cv = _load_single_drug_cv(v2_path)
        r3 = {
            "name": "ST v2 (single-drug only)",
            "m1_single_drug_cv_rho": v2_cv["cv_mean_per_patient_spearman"],
            "m1_single_drug_cv_mae": v2_cv["cv_mean_mae"],
            "m2_flt3_canonical_rank": m2_flt3_canonical_rank(
                st_v2_combo, mlp_drug_vocab, flt3_mut_mask,
            ),
            "m3_jaccard_top5_vs_mlp": m3_jaccard_vs_reference(
                st_v2_combo, mlp_mech_combo, mlp_drug_vocab,
            ),
            "m4_spearman_with_path_a": m4_spearman_with_path_a(
                st_v2_combo, path_a_cov, mlp_drug_vocab,
            ),
            "m5_zero_shot_triplet": m5_zero_shot_triplet(
                _make_st_triplet_scorer(v2_path, list(pf.columns)),
                pf, drug_filter, flt3_mut_mask,
                valid_drugs=st_v2_valid_drugs,
            ),
            "m6_bliss_consistency": m6_bliss_consistency(
                st_v2_combo, st_v2_singles, mlp_drug_vocab,
            ),
        }
        results["st_v2"] = r3
        print(json.dumps(r3, indent=2))

    # ===========================================================
    # Baseline 4: ST v3 (Path A distilled)
    # ===========================================================
    v3_path = Path("runs/set_drug_predictor_v3_distilled/final_model.pt")
    if v3_path.exists():
        print("\n=== Baseline 4: ST v3 (distilled) ===")
        st_v3 = load_set_drug_predictor(v3_path)
        st_v3_valid_drugs = [d for d in drug_filter if d in st_v3.drug_to_int]
        print(f"  ST v3 vocab has {len(st_v3_valid_drugs)}/{len(drug_filter)} clinical drugs")

        st_v3_combo, _ = score_set_transformer(pf.reset_index(), drug_filter, v3_path)
        st_v3_singles = get_st_single_auc(pf.reset_index(), drug_filter, v3_path)
        v3_cv = _load_single_drug_cv(v3_path)
        r4 = {
            "name": "ST v3 (Path A distilled)",
            "m1_single_drug_cv_rho": v3_cv["cv_mean_per_patient_spearman"],
            "m1_single_drug_cv_mae": v3_cv["cv_mean_mae"],
            "m2_flt3_canonical_rank": m2_flt3_canonical_rank(
                st_v3_combo, mlp_drug_vocab, flt3_mut_mask,
            ),
            "m3_jaccard_top5_vs_mlp": m3_jaccard_vs_reference(
                st_v3_combo, mlp_mech_combo, mlp_drug_vocab,
            ),
            "m4_spearman_with_path_a": m4_spearman_with_path_a(
                st_v3_combo, path_a_cov, mlp_drug_vocab,
            ),
            "m5_zero_shot_triplet": m5_zero_shot_triplet(
                _make_st_triplet_scorer(v3_path, list(pf.columns)),
                pf, drug_filter, flt3_mut_mask,
                valid_drugs=st_v3_valid_drugs,
            ),
            "m6_bliss_consistency": m6_bliss_consistency(
                st_v3_combo, st_v3_singles, mlp_drug_vocab,
            ),
        }
        results["st_v3_distilled"] = r4
        print(json.dumps(r4, indent=2))

    # ---- Save ----
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # CSV comparison table
    rows = []
    for key, r in results.items():
        row = {"model": key, "description": r["name"]}
        for m in ["m1_single_drug_cv_rho", "m1_single_drug_cv_mae",
                  "m3_jaccard_top5_vs_mlp", "m4_spearman_with_path_a",
                  "m6_bliss_consistency"]:
            row[m] = r.get(m)
        if "m2_flt3_canonical_rank" in r and isinstance(r["m2_flt3_canonical_rank"], dict):
            row["m2_mean_canonical_rank"] = r["m2_flt3_canonical_rank"]["mean_best_canonical_rank"]
        if "m5_zero_shot_triplet" in r and isinstance(r["m5_zero_shot_triplet"], dict):
            row["m5_triplet_delta"] = r["m5_zero_shot_triplet"]["mean_delta_random_minus_canonical"]
            row["m5_triplet_win_rate"] = r["m5_zero_shot_triplet"]["win_rate"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "comparison_table.csv", index=False)

    print(f"\n[bench] DONE. Results saved to {out_dir}")
    return results


if __name__ == "__main__":
    run_benchmark()
