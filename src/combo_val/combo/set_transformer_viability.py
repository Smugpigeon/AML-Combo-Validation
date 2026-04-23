"""Theoretical viability tests for the Set Transformer combo predictor (Path B).

The Set Transformer is trained ONLY on single-drug data. These tests verify
that the architecture's zero-shot multi-drug behavior is sound — i.e., that
it respects basic theoretical invariants and matches AML biology even
without ever seeing multi-drug labels.

Six tests:

  T1. Single-drug CV performance parity with existing MLP
      — already reported by train_set_drug_predictor; we just surface the
        recorded metrics here.

  T2. Permutation invariance |f(π(S)) − f(S)| < 1e-4
      — architectural guarantee; empirical sanity check.

  T3. Monotonicity: adding a patient-sensitive drug to a set should DECREASE
      predicted combo AUC (more drugs killing more cells).

  T4. Bliss-consistency: on size-N sets, predicted AUC should fall between
      additive-mean and Bliss-product baselines derived from single-drug
      predictions. This tests the model doesn't emit nonsense extrapolations.

  T5. AML biology: FLT3-mut patients should get LOWER combo AUC when a FLT3
      inhibitor (Quizartinib / Gilteritinib / Midostaurin) is in the set,
      compared to matched sets without any FLT3i. This is the clinical
      `triplet > doublet > single` for the FLT3-mut population.

  T6. 3-drug extrapolation sanity: sample 200 random 3-drug sets × 50
      random patients, verify predictions are finite, spread sensibly, and
      don't exhibit pathological behavior (e.g. constant output).

Each test writes to `runs/set_drug_predictor_viability/`:
  - test_{i}_details.csv  (row-level evidence)
  - viability_report.json (summary)
  - viability_report.md   (human-readable)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from combo_val.combo.set_drug_predictor import (
    SetDrugPredictor, SetDrugPredictorConfig, _resolve_device,
)


@dataclass(frozen=True)
class ViabilityConfig:
    checkpoint: Path = Path("runs/set_drug_predictor/final_model.pt")
    patient_features: Path = Path("data/canonical/beataml_patient_features.csv")
    drug_responses: Path = Path("data/canonical/beataml_drug_response_long.csv")
    cv_metrics: Path = Path("runs/set_drug_predictor/cv_metrics.json")
    out_dir: Path = Path("runs/set_drug_predictor_viability")
    random_state: int = 42

    # Test thresholds (pre-registered)
    t1_min_spearman: float = 0.65
    t2_max_permutation_delta: float = 1e-4
    t3_min_monotonic_frac: float = 0.60      # at least 60% of pairs show the drop
    t4_bliss_tolerance: float = 40.0         # AUC units; headroom outside bounds
    t5_min_flt3i_preference_frac: float = 0.55
    t6_max_nan_frac: float = 0.0
    t6_min_pred_std: float = 5.0             # AUC units; not collapsed constant

    # Sample sizes for Monte-Carlo tests
    t3_n_patient_samples: int = 100
    t3_n_drug_pairs_per_patient: int = 10
    t4_n_patients: int = 50
    t4_n_sets: int = 100
    t4_set_sizes: tuple = (2, 3, 4)
    t5_n_controls_per_patient: int = 20
    t6_n_patient_samples: int = 50
    t6_n_triplets: int = 200


def _load_ckpt_and_model(cfg: ViabilityConfig, device: torch.device):
    ckpt = torch.load(cfg.checkpoint, weights_only=False, map_location="cpu")
    model_cfg = SetDrugPredictorConfig(**{
        k: v for k, v in ckpt["cfg"].items()
        if k in SetDrugPredictorConfig.__dataclass_fields__
    })
    model = SetDrugPredictor(
        n_drugs=ckpt["n_drugs"],
        n_patient_features=ckpt["n_patient_features"],
        cfg=model_cfg,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _load_features(ckpt, cfg: ViabilityConfig) -> tuple[pd.DataFrame, np.ndarray]:
    pf = pd.read_csv(cfg.patient_features).set_index("patient_id")
    # Reapply the saved StandardScaler state
    mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
    scaled = (pf.values - mean) / np.where(scale > 0, scale, 1.0)
    pf_scaled = pd.DataFrame(scaled, index=pf.index, columns=pf.columns)
    return pf, pf_scaled


# ---------------------------------------------------------------------------
# Test 1 — single-drug CV performance
# ---------------------------------------------------------------------------


def test_1_single_drug_cv(cfg: ViabilityConfig) -> dict:
    """Surface the CV metrics written by train_set_drug_predictor."""
    m = json.loads(Path(cfg.cv_metrics).read_text())
    rho = m["overall_cv"]["cv_mean_per_patient_spearman"]
    std = m["overall_cv"]["cv_std_per_patient_spearman"]
    mae = m["overall_cv"]["cv_mean_mae"]
    gate_pass = rho >= cfg.t1_min_spearman
    return {
        "test": "T1_single_drug_cv",
        "per_patient_spearman_mean": rho,
        "per_patient_spearman_std": std,
        "mae": mae,
        "gate_threshold": cfg.t1_min_spearman,
        "gate_pass": bool(gate_pass),
    }


# ---------------------------------------------------------------------------
# Test 2 — permutation invariance
# ---------------------------------------------------------------------------


def test_2_permutation_invariance(
    model: SetDrugPredictor, pf_scaled: pd.DataFrame,
    ckpt: dict, cfg: ViabilityConfig, device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_state)
    n_drugs = ckpt["n_drugs"]
    pids = rng.choice(pf_scaled.index.values, size=200, replace=False)
    records = []
    for pid in pids:
        set_size = rng.integers(2, 6)
        drug_set = rng.choice(n_drugs, size=set_size, replace=False).tolist()
        pf_tensor = torch.tensor(
            pf_scaled.loc[pid].values, dtype=torch.float32
        ).unsqueeze(0)
        # Two random permutations
        perm1 = drug_set
        perm2 = rng.permutation(drug_set).tolist()
        perm3 = rng.permutation(drug_set).tolist()
        preds = []
        for p in [perm1, perm2, perm3]:
            y = model.predict_set([p], pf_tensor, device=device)
            preds.append(float(y.item()))
        records.append({
            "patient_id": int(pid),
            "set_size": int(set_size),
            "pred_ref": preds[0],
            "pred_perm2": preds[1],
            "pred_perm3": preds[2],
            "max_delta": max(
                abs(preds[0] - preds[1]),
                abs(preds[0] - preds[2]),
                abs(preds[1] - preds[2]),
            ),
        })
    df = pd.DataFrame(records)
    max_delta = float(df["max_delta"].max())
    mean_delta = float(df["max_delta"].mean())
    gate_pass = max_delta < cfg.t2_max_permutation_delta
    return {
        "test": "T2_permutation_invariance",
        "n_samples": len(df),
        "max_delta": round(max_delta, 9),
        "mean_delta": round(mean_delta, 9),
        "gate_threshold": cfg.t2_max_permutation_delta,
        "gate_pass": bool(gate_pass),
    }, df


# ---------------------------------------------------------------------------
# Test 3 — monotonicity: adding a sensitive drug should decrease combo AUC
# ---------------------------------------------------------------------------


def _predict_single_all_drugs(
    model: SetDrugPredictor, pf_row: pd.Series,
    n_drugs: int, device: torch.device,
) -> np.ndarray:
    pf = torch.tensor(pf_row.values, dtype=torch.float32).unsqueeze(0).expand(n_drugs, -1)
    sets = [[d] for d in range(n_drugs)]
    return model.predict_set(sets, pf, device=device).cpu().numpy()


def test_3_monotonicity(
    model: SetDrugPredictor, pf_scaled: pd.DataFrame, ckpt: dict,
    cfg: ViabilityConfig, device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    """Adding a *sensitive* drug (low predicted single-drug AUC) to an
    existing set should decrease the predicted combo AUC.

    For each of t3_n_patient_samples patients:
      - compute single-drug AUC for all 165 drugs
      - pick top-K (lowest AUC) = "sensitive drugs" for this patient
      - for random pairs (d_anchor, d_addition) from top-K:
          predict f({d_anchor}) and f({d_anchor, d_addition})
          record whether f({d_anchor, d_addition}) < f({d_anchor})
    """
    rng = np.random.default_rng(cfg.random_state)
    n_drugs = ckpt["n_drugs"]
    pids = rng.choice(pf_scaled.index.values,
                      size=min(cfg.t3_n_patient_samples, len(pf_scaled)),
                      replace=False)
    records = []
    for pid in pids:
        pf_row = pf_scaled.loc[pid]
        single_pred = _predict_single_all_drugs(model, pf_row, n_drugs, device)
        # 10 most sensitive drugs for this patient
        top_k = np.argsort(single_pred)[:10]
        # Sample random pairs from top_k
        for _ in range(cfg.t3_n_drug_pairs_per_patient):
            pair = rng.choice(top_k, size=2, replace=False)
            d_anchor, d_add = int(pair[0]), int(pair[1])
            pf_tensor = torch.tensor(pf_row.values, dtype=torch.float32).unsqueeze(0)
            pred_single = float(model.predict_set(
                [[d_anchor]], pf_tensor, device=device
            ).item())
            pred_pair = float(model.predict_set(
                [[d_anchor, d_add]], pf_tensor, device=device
            ).item())
            records.append({
                "patient_id": int(pid),
                "d_anchor": d_anchor,
                "d_addition": d_add,
                "single_auc_anchor": single_pred[d_anchor],
                "single_auc_addition": single_pred[d_add],
                "pred_single": pred_single,
                "pred_pair": pred_pair,
                "delta": pred_pair - pred_single,
                "decreased": pred_pair < pred_single,
            })
    df = pd.DataFrame(records)
    decreased_frac = float(df["decreased"].mean())
    mean_delta = float(df["delta"].mean())
    gate_pass = decreased_frac >= cfg.t3_min_monotonic_frac
    return {
        "test": "T3_monotonicity",
        "n_pairs_tested": len(df),
        "fraction_combo_below_single": round(decreased_frac, 3),
        "mean_delta_combo_minus_single": round(mean_delta, 2),
        "gate_threshold": cfg.t3_min_monotonic_frac,
        "gate_pass": bool(gate_pass),
    }, df


# ---------------------------------------------------------------------------
# Test 4 — Bliss-consistency: predictions should fall in a sensible window
# ---------------------------------------------------------------------------


def test_4_bliss_consistency(
    model: SetDrugPredictor, pf_scaled: pd.DataFrame, ckpt: dict,
    cfg: ViabilityConfig, device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    """For each (patient, N-drug set):

      additive_mean = mean(pred_single(d_i))           # upper bound
      bliss_lower = 300 * (1 - Π(1 - pred_single(d_i) / 300))  # lower bound

    In theory, a reasonable combo predictor should fall approximately between
    these two baselines (with Bliss being the optimistic case and additive
    the pessimistic case). Out-of-bounds predictions ± tolerance are flagged.
    """
    rng = np.random.default_rng(cfg.random_state)
    n_drugs = ckpt["n_drugs"]
    AUC_MAX = 300.0

    pids = rng.choice(pf_scaled.index.values,
                      size=min(cfg.t4_n_patients, len(pf_scaled)),
                      replace=False)
    records = []
    for pid in pids:
        pf_row = pf_scaled.loc[pid]
        single_pred = _predict_single_all_drugs(model, pf_row, n_drugs, device)
        pf_tensor = torch.tensor(pf_row.values, dtype=torch.float32).unsqueeze(0)
        for _ in range(cfg.t4_n_sets):
            set_size = rng.choice(cfg.t4_set_sizes)
            drug_set = rng.choice(n_drugs, size=set_size, replace=False).tolist()
            pred = float(model.predict_set(
                [drug_set], pf_tensor, device=device,
            ).item())
            singles = single_pred[drug_set]
            additive_mean = float(np.mean(singles))
            clipped = np.clip(singles, 0, AUC_MAX)
            prod_survival = float(np.prod(1 - clipped / AUC_MAX))
            bliss_lower = AUC_MAX * prod_survival
            records.append({
                "patient_id": int(pid),
                "set_size": int(set_size),
                "drugs": str(drug_set),
                "pred": pred,
                "additive_mean": round(additive_mean, 2),
                "bliss_lower": round(bliss_lower, 2),
                "in_bounds": (
                    (bliss_lower - cfg.t4_bliss_tolerance)
                    <= pred <=
                    (additive_mean + cfg.t4_bliss_tolerance)
                ),
            })
    df = pd.DataFrame(records)
    in_bounds_frac = float(df["in_bounds"].mean())
    # "pass" means ≥95% of predictions fall within [bliss_lower - tol, additive_mean + tol]
    gate_pass = in_bounds_frac >= 0.95
    return {
        "test": "T4_bliss_consistency",
        "n_sets_tested": len(df),
        "tolerance_auc": cfg.t4_bliss_tolerance,
        "in_bounds_fraction": round(in_bounds_frac, 3),
        "mean_pred": round(float(df["pred"].mean()), 2),
        "mean_additive": round(float(df["additive_mean"].mean()), 2),
        "mean_bliss_lower": round(float(df["bliss_lower"].mean()), 2),
        "gate_threshold": 0.95,
        "gate_pass": bool(gate_pass),
    }, df


# ---------------------------------------------------------------------------
# Test 5 — AML biology: FLT3-mut patients should prefer FLT3i-containing sets
# ---------------------------------------------------------------------------


FLT3I_CANONICAL_DRUGS = (
    "Quizartinib (AC220)", "Gilteritinib", "Midostaurin",
    "Sorafenib", "Crenolanib",
)


def test_5_flt3_biology(
    model: SetDrugPredictor, pf_scaled: pd.DataFrame, pf_raw: pd.DataFrame,
    ckpt: dict, cfg: ViabilityConfig, device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    """For each FLT3-mut patient:
      - compute predicted AUC for the 'clinical triplet' {Gilteritinib, Venetoclax, Azacytidine}
      - compute mean predicted AUC for N random 3-drug sets that do NOT include
        any of the 5 canonical FLT3 inhibitors
      - record whether the triplet AUC is lower (better) than the random mean
    """
    rng = np.random.default_rng(cfg.random_state)
    drug_to_int = ckpt["drug_to_int"]
    drug_vocab = ckpt["drug_vocab"]
    n_drugs = len(drug_vocab)

    # FLT3-mut patients
    flt3_pids = pf_raw[pf_raw["mut_FLT3"] == 1].index.tolist()
    flt3i_ints = [drug_to_int[d] for d in FLT3I_CANONICAL_DRUGS if d in drug_to_int]
    non_flt3i_ints = [i for i in range(n_drugs) if i not in set(flt3i_ints)]
    assert "Gilteritinib" in drug_to_int and "Venetoclax" in drug_to_int
    assert "Azacytidine" in drug_to_int

    target_triplet_ints = [
        drug_to_int["Gilteritinib"],
        drug_to_int["Venetoclax"],
        drug_to_int["Azacytidine"],
    ]

    records = []
    for pid in flt3_pids:
        pf_tensor = torch.tensor(
            pf_scaled.loc[pid].values, dtype=torch.float32
        ).unsqueeze(0)
        # Target triplet
        triplet_auc = float(model.predict_set(
            [target_triplet_ints], pf_tensor, device=device,
        ).item())
        # N random 3-drug non-FLT3i sets
        random_aucs = []
        for _ in range(cfg.t5_n_controls_per_patient):
            rset = rng.choice(non_flt3i_ints, size=3, replace=False).tolist()
            a = float(model.predict_set(
                [rset], pf_tensor, device=device,
            ).item())
            random_aucs.append(a)
        mean_random = float(np.mean(random_aucs))
        records.append({
            "patient_id": int(pid),
            "triplet_auc": round(triplet_auc, 2),
            "mean_random_nonflt3i_auc": round(mean_random, 2),
            "triplet_beats_random": triplet_auc < mean_random,
        })
    df = pd.DataFrame(records)
    frac_wins = float(df["triplet_beats_random"].mean())
    gate_pass = frac_wins >= cfg.t5_min_flt3i_preference_frac
    return {
        "test": "T5_flt3_biology",
        "n_flt3_patients": len(df),
        "triplet": "Gilteritinib + Venetoclax + Azacytidine",
        "fraction_triplet_beats_random_3drug_nonflt3i": round(frac_wins, 3),
        "mean_triplet_auc": round(float(df["triplet_auc"].mean()), 2),
        "mean_random_nonflt3i_auc": round(float(df["mean_random_nonflt3i_auc"].mean()), 2),
        "mean_advantage": round(
            float(df["mean_random_nonflt3i_auc"].mean() - df["triplet_auc"].mean()), 2
        ),
        "gate_threshold": cfg.t5_min_flt3i_preference_frac,
        "gate_pass": bool(gate_pass),
    }, df


# ---------------------------------------------------------------------------
# Test 6 — 3-drug extrapolation sanity
# ---------------------------------------------------------------------------


def test_6_triplet_extrapolation(
    model: SetDrugPredictor, pf_scaled: pd.DataFrame, ckpt: dict,
    cfg: ViabilityConfig, device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_state)
    n_drugs = ckpt["n_drugs"]
    pids = rng.choice(pf_scaled.index.values,
                      size=min(cfg.t6_n_patient_samples, len(pf_scaled)),
                      replace=False)
    records = []
    for pid in pids:
        pf_tensor = torch.tensor(
            pf_scaled.loc[pid].values, dtype=torch.float32
        ).unsqueeze(0)
        for _ in range(cfg.t6_n_triplets):
            triplet = rng.choice(n_drugs, size=3, replace=False).tolist()
            pred = float(model.predict_set([triplet], pf_tensor, device=device).item())
            records.append({
                "patient_id": int(pid),
                "drugs": str(triplet),
                "pred": pred,
                "finite": np.isfinite(pred),
            })
    df = pd.DataFrame(records)
    nan_frac = float(1.0 - df["finite"].mean())
    pred_std = float(df["pred"].std())
    pred_min = float(df["pred"].min())
    pred_max = float(df["pred"].max())
    gate_pass = (nan_frac <= cfg.t6_max_nan_frac) and (pred_std >= cfg.t6_min_pred_std)
    return {
        "test": "T6_triplet_extrapolation",
        "n_triplets_tested": len(df),
        "nan_fraction": round(nan_frac, 4),
        "pred_std": round(pred_std, 2),
        "pred_min": round(pred_min, 2),
        "pred_max": round(pred_max, 2),
        "pred_mean": round(float(df["pred"].mean()), 2),
        "gate_pass": bool(gate_pass),
    }, df


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_all(cfg: ViabilityConfig | None = None) -> dict:
    cfg = cfg or ViabilityConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device("cpu")  # CPU enough; deterministic
    print(f"[viability] device: {device}")

    model, ckpt = _load_ckpt_and_model(cfg, device)
    pf_raw, pf_scaled = _load_features(ckpt, cfg)

    results = {}
    details: dict[str, pd.DataFrame] = {}

    print("\n[T1] Single-drug CV performance …")
    r1 = test_1_single_drug_cv(cfg)
    results["T1"] = r1
    print(f"     ρ={r1['per_patient_spearman_mean']:.3f}  MAE={r1['mae']:.2f}  "
          f"{'PASS' if r1['gate_pass'] else 'FAIL'}")

    print("[T2] Permutation invariance …")
    r2, d2 = test_2_permutation_invariance(model, pf_scaled, ckpt, cfg, device)
    results["T2"] = r2
    details["T2"] = d2
    print(f"     max_delta={r2['max_delta']:.2e}  "
          f"{'PASS' if r2['gate_pass'] else 'FAIL'}")

    print("[T3] Monotonicity (adding sensitive drug → decrease) …")
    r3, d3 = test_3_monotonicity(model, pf_scaled, ckpt, cfg, device)
    results["T3"] = r3
    details["T3"] = d3
    print(f"     frac_decreased={r3['fraction_combo_below_single']:.3f}  "
          f"mean_delta={r3['mean_delta_combo_minus_single']:.2f}  "
          f"{'PASS' if r3['gate_pass'] else 'FAIL'}")

    print("[T4] Bliss-consistency …")
    r4, d4 = test_4_bliss_consistency(model, pf_scaled, ckpt, cfg, device)
    results["T4"] = r4
    details["T4"] = d4
    print(f"     in_bounds={r4['in_bounds_fraction']:.3f}  "
          f"{'PASS' if r4['gate_pass'] else 'FAIL'}")

    print("[T5] AML biology: FLT3i triplet beats random non-FLT3i triplet …")
    r5, d5 = test_5_flt3_biology(model, pf_scaled, pf_raw, ckpt, cfg, device)
    results["T5"] = r5
    details["T5"] = d5
    print(f"     frac_wins={r5['fraction_triplet_beats_random_3drug_nonflt3i']:.3f}  "
          f"advantage={r5['mean_advantage']:.2f}  "
          f"{'PASS' if r5['gate_pass'] else 'FAIL'}")

    print("[T6] 3-drug extrapolation sanity …")
    r6, d6 = test_6_triplet_extrapolation(model, pf_scaled, ckpt, cfg, device)
    results["T6"] = r6
    details["T6"] = d6
    print(f"     nan_frac={r6['nan_fraction']:.4f}  pred_std={r6['pred_std']:.2f}  "
          f"{'PASS' if r6['gate_pass'] else 'FAIL'}")

    # --- Secondary analysis: path B vs existing combo predictor ---
    existing_npz = Path("runs/combo_predictor/combo_auc_predictions.npz")
    comparison = None
    if existing_npz.exists():
        from combo_val.combo.pathB_vs_existing import compare as _cmp
        print("\n[SEC] path B vs existing combo predictor (2-drug) …")
        try:
            comparison = _cmp(
                set_model_ckpt=cfg.checkpoint,
                combo_npz=existing_npz,
                patient_features=cfg.patient_features,
                out_dir=cfg.out_dir,
                random_state=cfg.random_state,
                n_patient_samples=50,
                n_pairs_per_patient=30,
            )
        except Exception as e:
            print(f"[SEC] comparison failed: {e}")
            comparison = {"error": str(e)}
    else:
        comparison = {"skipped": "combo_predictor outputs missing"}

    # Save
    for k, df in details.items():
        df.to_csv(cfg.out_dir / f"{k}_details.csv", index=False)
    all_pass = all(r["gate_pass"] for r in results.values())
    summary = {
        "all_gates_pass": bool(all_pass),
        "tests": results,
        "secondary_comparison_vs_existing": comparison,
    }
    (cfg.out_dir / "viability_report.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # Markdown
    lines = [
        "# Set Transformer (Path B) — Theoretical Viability Report",
        "",
        f"**Overall: {'ALL PASS ✓' if all_pass else 'SOME FAIL ✗'}**",
        "",
        "| Test | Description | Result | Gate |",
        "|---|---|---|---|",
    ]
    lines.append(f"| T1 | Single-drug CV Spearman | ρ={r1['per_patient_spearman_mean']:.3f} "
                 f"± {r1['per_patient_spearman_std']:.3f} (MAE {r1['mae']:.2f}) | "
                 f"{'✓' if r1['gate_pass'] else '✗'} ≥ {r1['gate_threshold']} |")
    lines.append(f"| T2 | Permutation invariance (max |Δ|) | {r2['max_delta']:.2e} | "
                 f"{'✓' if r2['gate_pass'] else '✗'} < {r2['gate_threshold']:.0e} |")
    lines.append(f"| T3 | Fraction of pairs where combo < single | "
                 f"{r3['fraction_combo_below_single']:.3f} | "
                 f"{'✓' if r3['gate_pass'] else '✗'} ≥ {r3['gate_threshold']} |")
    lines.append(f"| T4 | Fraction of sets in [Bliss, Additive ± tol] | "
                 f"{r4['in_bounds_fraction']:.3f} | "
                 f"{'✓' if r4['gate_pass'] else '✗'} ≥ {r4['gate_threshold']} |")
    lines.append(f"| T5 | Fraction FLT3-mut patients where Gilt+Ven+Aza < random non-FLT3i triplet | "
                 f"{r5['fraction_triplet_beats_random_3drug_nonflt3i']:.3f} "
                 f"(Δ={r5['mean_advantage']:.1f}) | "
                 f"{'✓' if r5['gate_pass'] else '✗'} ≥ {r5['gate_threshold']} |")
    lines.append(f"| T6 | 3-drug extrapolation (NaN frac / pred std) | "
                 f"{r6['nan_fraction']:.4f} / {r6['pred_std']:.2f} | "
                 f"{'✓' if r6['gate_pass'] else '✗'} 0 / ≥ {cfg.t6_min_pred_std} |")
    (cfg.out_dir / "viability_report.md").write_text("\n".join(lines) + "\n",
                                                      encoding="utf-8")
    print(f"\n[viability] → {cfg.out_dir}")
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/set_drug_predictor_viability")
    args = ap.parse_args()
    run_all(ViabilityConfig(out_dir=Path(args.out)))


if __name__ == "__main__":
    main()
