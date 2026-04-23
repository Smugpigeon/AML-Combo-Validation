"""Validate model predictions against REAL clinical induction response.

Scope (Route B, strict):
  Test whether Baseline A's per-patient predicted AUC carries signal about
  each patient's actual response to induction chemotherapy. This is the
  missing validation against real clinical outcomes that Week 4's all-
  in-silico head-to-head did not provide.

Cohort filter:
  - diseaseStageAtSpecimenCollection == "Initial Diagnosis"  (sample was
    drawn BEFORE treatment; ex-vivo AUC is not contaminated by prior therapy)
  - responseToInductionTx in {"Complete Response", "Refractory"}  (unambiguous
    clinical outcome; drop "Unknown", "Complete Response w/ incomplete recovery",
    and NaN)
  - Patient must exist in patient_features.csv (has the full 80-dim vector)

Primary endpoints:
  1. Cytarabine-AUC predictor for CR vs Refractory
     - 742/750 BeatAML patients with responseToInductionTx got
       "Standard Chemotherapy" ≈ 7+3 (cytarabine + anthracycline).
       Cytarabine is the only classical induction drug in the 165-drug panel.
     - Hypothesis: low predicted Cyt-AUC (ex-vivo sensitive) → higher CR rate.
     - Metrics: ROC-AUC (binary label = 1 if CR), point-biserial r, Mann-Whitney U.

  2. Best-single-drug predictor as a general "chemo-sensitivity" signal
     - min over drugs of predicted AUC per patient.
     - Same metrics. Tests if the model captures broader sensitivity beyond cyt.

  3. Combo-benefit (Δ) vs CR within FLT3-mut subgroup
     - For FLT3-mut patients (the precision-combo population from Week 4),
       test if the combo-predicted Δ correlates with worse induction response
       to standard chemo (i.e., patients predicted to benefit from combo are
       the ones who fail standard induction, so combo approach has headroom).

Outputs:
  runs/induction_validation/
    ├── cohort.csv                      # 1 row / patient: pred_auc_cyt, pred_best_single,
    │                                     combo_delta, true_cr, flt3_mut, eln, …
    ├── summary.json                    # all metrics + CIs + p-values
    └── figure_calibration.png          # ROC + calibration curve
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class InductionValidationConfig:
    clinical_xlsx: Path = Path("data/raw/BeatAML2.0/beataml_clinical.xlsx")
    patient_features_path: Path = Path("data/canonical/beataml_patient_features.csv")
    baseline_pred_path: Path = Path(
        "runs/baseline_single_drug_mlp/predictions_all_patients_all_drugs.csv"
    )
    combo_auc_npz_path: Path = Path("runs/combo_predictor/combo_auc_predictions.npz")
    out_dir: Path = Path("runs/induction_validation")
    bootstrap_n: int = 2000
    random_state: int = 42


def _bootstrap_roc_auc(y_true: np.ndarray, y_score: np.ndarray,
                       n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Bootstrap ROC-AUC with 95% CI. y_score here is a "lower-is-better"
    predicted AUC, so we negate before feeding to roc_auc_score (which
    expects higher = positive class)."""
    obs = float(roc_auc_score(y_true, -y_score))
    boots = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue  # skip degenerate bootstrap
        boots.append(roc_auc_score(y_true[idx], -y_score[idx]))
    if not boots:
        return obs, float("nan"), float("nan")
    return obs, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def build_cohort(cfg: InductionValidationConfig) -> pd.DataFrame:
    """Filter BeatAML to the clean induction-validation cohort."""
    clin = pd.read_excel(cfg.clinical_xlsx)
    pf = pd.read_csv(cfg.patient_features_path)
    baseline = pd.read_csv(cfg.baseline_pred_path, index_col=0)

    # Filter to initial-diagnosis specimens with unambiguous CR / Refractory
    keep_resp = {"Complete Response", "Refractory"}
    c = clin[
        (clin["diseaseStageAtSpecimenCollection"] == "Initial Diagnosis")
        & (clin["responseToInductionTx"].isin(keep_resp))
    ].copy()
    print(f"[ind-val] after stage + response filter: {len(c)} specimens")

    # One row per patient. If a patient has multiple Initial Diagnosis
    # specimens, keep the first (arbitrary; conservative).
    c = c.drop_duplicates(subset="dbgap_subject_id", keep="first")
    print(f"[ind-val] after one-per-patient dedup: {len(c)} patients")

    # Must be in feature table
    c = c[c["dbgap_subject_id"].isin(pf["patient_id"])]
    print(f"[ind-val] with full 80-dim features: {len(c)} patients")

    # Must be in baseline predictions
    base_ids = set(baseline.index.astype(int))
    c = c[c["dbgap_subject_id"].isin(base_ids)]
    print(f"[ind-val] with baseline predictions: {len(c)} patients")

    # Build the row
    c = c.rename(columns={"dbgap_subject_id": "patient_id"})
    c["true_cr"] = (c["responseToInductionTx"] == "Complete Response").astype(int)

    # Attach predicted AUC columns
    for d in ["Cytarabine", "Venetoclax", "Sorafenib", "Midostaurin",
             "Quizartinib (AC220)", "Gilteritinib", "Azacytidine"]:
        col = f"pred_auc_{d.split(' ')[0].replace('(', '').replace(')', '')}"
        if d in baseline.columns:
            c[col] = c["patient_id"].map(lambda pid: baseline.loc[int(pid), d])

    # Best single drug AUC (min over all drugs)
    c["pred_best_single"] = c["patient_id"].map(
        lambda pid: baseline.loc[int(pid)].min()
    )
    c["pred_best_single_drug"] = c["patient_id"].map(
        lambda pid: baseline.loc[int(pid)].idxmin()
    )

    # Attach mutation + ELN from feature table
    pf_keyed = pf.set_index("patient_id")
    for col in ["mut_FLT3", "mut_NPM1", "mut_IDH1", "mut_IDH2", "mut_TP53",
                "clin_eln_ordinal", "clin_age"]:
        if col in pf_keyed.columns:
            c[col] = c["patient_id"].map(lambda pid: pf_keyed.loc[int(pid), col])

    # Attach combo Δ if available
    if cfg.combo_auc_npz_path.exists():
        npz = np.load(cfg.combo_auc_npz_path, allow_pickle=True)
        combo_auc = npz["combo_auc"]
        patient_ids = npz["patient_ids"]
        pid_to_idx = {int(p): i for i, p in enumerate(patient_ids)}
        n_d = combo_auc.shape[1]
        tri_i, tri_j = np.triu_indices(n_d, k=1)
        best_combos = {}
        for pid in c["patient_id"]:
            if int(pid) in pid_to_idx:
                pi = pid_to_idx[int(pid)]
                best_combos[int(pid)] = float(combo_auc[pi, tri_i, tri_j].min())
        c["pred_best_combo"] = c["patient_id"].map(lambda pid: best_combos.get(int(pid), np.nan))
        c["combo_delta"] = c["pred_best_single"] - c["pred_best_combo"]

    return c


def _report_binary_validator(y_true: np.ndarray, y_score_low_is_positive: np.ndarray,
                              name: str, n_boot: int,
                              rng: np.random.Generator) -> dict:
    """Summarize how well y_score predicts CR (y_true==1).
    y_score_low_is_positive: predicted AUC — LOW means the patient is
    ex-vivo sensitive, so we expect LOW score → HIGH CR rate.
    """
    mask = np.isfinite(y_score_low_is_positive) & np.isfinite(y_true)
    y, s = y_true[mask], y_score_low_is_positive[mask]
    if len(y) < 10 or y.sum() < 3 or (y == 0).sum() < 3:
        return {"name": name, "n": int(len(y)), "status": "too_few_samples"}
    auc, lo, hi = _bootstrap_roc_auc(y, s, n_boot, rng)
    r_pb, p_pb = stats.pointbiserialr(y, -s)  # neg-AUC so higher = CR
    cr_mean = s[y == 1].mean()
    ref_mean = s[y == 0].mean()
    u_stat, u_p = stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="less")
    # less: CR patients' predicted AUC < Refractory patients' predicted AUC
    return {
        "name": name,
        "n": int(len(y)),
        "n_cr": int(y.sum()),
        "n_ref": int((y == 0).sum()),
        "roc_auc": round(auc, 4),
        "roc_auc_ci95": [round(lo, 4), round(hi, 4)],
        "pointbiserial_r": round(float(r_pb), 4),
        "pointbiserial_p": round(float(p_pb), 4),
        "mean_pred_auc_CR": round(float(cr_mean), 2),
        "mean_pred_auc_Refractory": round(float(ref_mean), 2),
        "mannwhitneyU_one_sided_p": round(float(u_p), 4),
    }


def run_induction_validation(cfg: InductionValidationConfig | None = None) -> dict:
    cfg = cfg or InductionValidationConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.random_state)

    cohort = build_cohort(cfg)
    cohort_path = cfg.out_dir / "cohort.csv"
    cohort.to_csv(cohort_path, index=False)
    print(f"[ind-val] cohort saved: {cohort_path}  shape={cohort.shape}")

    y = cohort["true_cr"].to_numpy()
    print(f"[ind-val] CR rate: {y.mean():.1%}  (n_cr={int(y.sum())}, "
          f"n_ref={int((1-y).sum())})")

    # --- Primary validator A: Cytarabine predicted AUC
    print("\n=== A. Cytarabine predicted AUC → CR ===")
    a = _report_binary_validator(
        y, cohort["pred_auc_Cytarabine"].to_numpy(),
        "cytarabine_predicted_auc", cfg.bootstrap_n, rng,
    )
    print(json.dumps(a, indent=2))

    # --- Primary validator B: Best single drug
    print("\n=== B. Best single drug predicted AUC → CR ===")
    b = _report_binary_validator(
        y, cohort["pred_best_single"].to_numpy(),
        "best_single_predicted_auc", cfg.bootstrap_n, rng,
    )
    print(json.dumps(b, indent=2))

    # --- Primary validator C: FLT3-mut subgroup combo-Δ
    print("\n=== C. FLT3-mut subgroup: combo-Δ → CR ===")
    if "combo_delta" in cohort.columns:
        flt3 = cohort[cohort["mut_FLT3"] == 1]
        print(f"[ind-val] FLT3-mut cohort: n={len(flt3)}")
        if len(flt3) >= 10:
            y_f = flt3["true_cr"].to_numpy()
            # Higher combo_delta = combo wins more = we expect LOWER CR to
            # standard induction (because these are the refractory-ish ones
            # who would benefit from combo). So negate to make the score
            # "higher = CR". Actually — NO, it depends on interpretation.
            # We test BOTH directions.
            delta = flt3["combo_delta"].to_numpy()
            c_pos_direction = _report_binary_validator(
                y_f, -delta, "combo_delta_high_means_CR_flt3mut", cfg.bootstrap_n, rng,
            )
            c_neg_direction = _report_binary_validator(
                y_f, delta, "combo_delta_low_means_CR_flt3mut", cfg.bootstrap_n, rng,
            )
            print("  If COMBO-BENEFITING patients also CR on standard:")
            print(json.dumps(c_pos_direction, indent=2))
            print("  If COMBO-BENEFITING patients are REFRACTORY on standard (more interesting):")
            print(json.dumps(c_neg_direction, indent=2))
        else:
            c_pos_direction = {"name": "combo_delta_flt3mut", "status": "too_few_flt3mut"}
            c_neg_direction = None
    else:
        c_pos_direction = c_neg_direction = None

    # --- Secondary: Venetoclax-AUC in the VEN-AZA subgroup (if any)
    print("\n=== D. Venetoclax predicted AUC → CR (overall) ===")
    d = _report_binary_validator(
        y, cohort["pred_auc_Venetoclax"].to_numpy(),
        "venetoclax_predicted_auc", cfg.bootstrap_n, rng,
    )
    print(json.dumps(d, indent=2))

    # --- Oracle comparison: observed ex-vivo AUC
    #   This tells us the UPPER BOUND of what the Cytarabine-based
    #   ex-vivo signal could possibly predict. If oracle ROC-AUC is 0.55,
    #   our model's 0.53 isn't bad. If oracle is 0.75, we're leaving
    #   a lot on the table.
    print("\n=== E. Oracle: OBSERVED ex-vivo Cytarabine AUC → CR ===")
    long = pd.read_csv("data/canonical/beataml_drug_response_long.csv")
    cyt_obs = long[long["drug_id"] == "Cytarabine"].groupby("patient_id")["auc"].mean()
    cohort["observed_auc_Cytarabine"] = cohort["patient_id"].map(
        lambda p: cyt_obs.get(int(p), np.nan)
    )
    e = _report_binary_validator(
        y, cohort["observed_auc_Cytarabine"].to_numpy(),
        "cytarabine_OBSERVED_auc_oracle", cfg.bootstrap_n, rng,
    )
    print(json.dumps(e, indent=2))

    # --- F. OS as continuous outcome: does predicted AUC correlate
    #       with overall survival? This is less noisy than binary CR.
    print("\n=== F. Overall survival continuous outcome ===")
    clin_raw = pd.read_excel(cfg.clinical_xlsx)
    os_map = clin_raw.drop_duplicates(subset="dbgap_subject_id", keep="first")
    os_map = os_map.set_index("dbgap_subject_id")[["overallSurvival", "vitalStatus"]]
    cohort["os_days"] = cohort["patient_id"].map(
        lambda p: os_map.loc[int(p), "overallSurvival"]
        if int(p) in os_map.index else np.nan)
    cohort["deceased"] = cohort["patient_id"].map(
        lambda p: (os_map.loc[int(p), "vitalStatus"] == "Dead")
        if int(p) in os_map.index else False
    ).astype(int)

    f_results: dict = {}
    for score_col, descr in [
        ("pred_best_single", "best single predicted AUC"),
        ("pred_auc_Cytarabine", "cytarabine predicted AUC"),
        ("pred_auc_Venetoclax", "venetoclax predicted AUC"),
        ("combo_delta", "combo Δ (higher = combo wins more)"),
    ]:
        if score_col not in cohort.columns:
            continue
        sub = cohort[["os_days", "deceased", score_col]].dropna()
        if len(sub) < 30:
            continue
        r, p_sp = stats.spearmanr(sub[score_col], sub["os_days"])
        # Median split test: top-half vs bottom-half OS
        med = sub[score_col].median()
        top = sub[sub[score_col] >= med]["os_days"].to_numpy()
        bot = sub[sub[score_col] < med]["os_days"].to_numpy()
        u_stat, u_p = stats.mannwhitneyu(top, bot, alternative="two-sided")
        f_results[score_col] = {
            "n": int(len(sub)),
            "spearman_r": round(float(r), 3),
            "spearman_p": round(float(p_sp), 4),
            "median_split_top_OS_days": round(float(np.median(top)), 1),
            "median_split_bot_OS_days": round(float(np.median(bot)), 1),
            "mw_p": round(float(u_p), 4),
            "descr": descr,
        }
        print(f"  {descr:40s}  ρ={r:.3f} (p={p_sp:.3f})  "
              f"top-half OS={np.median(top):.0f}d vs bot-half={np.median(bot):.0f}d "
              f"(p={u_p:.3f})")

    # --- G. Stratify by treatment type — standard chemo only
    print("\n=== G. Restrict to Standard-Chemotherapy recipients ===")
    std_ids = set(
        clin_raw[clin_raw["typeInductionTx"] == "Standard Chemotherapy"]["dbgap_subject_id"]
    )
    std_cohort = cohort[cohort["patient_id"].isin(std_ids)].copy()
    y_std = std_cohort["true_cr"].to_numpy()
    print(f"[ind-val] std-chemo subgroup: n={len(std_cohort)}  CR rate={y_std.mean():.1%}")
    g = _report_binary_validator(
        y_std, std_cohort["pred_best_single"].to_numpy(),
        "best_single_pred_AUC_stdchemo_only", cfg.bootstrap_n, rng,
    )
    print(json.dumps(g, indent=2))

    summary = {
        "cfg": asdict(cfg),
        "cohort_size": int(len(cohort)),
        "cr_rate": round(float(y.mean()), 3),
        "validators": {
            "A_cytarabine_predicted": a,
            "B_best_single_predicted": b,
            "C_combo_delta_flt3mut_high_means_CR": c_pos_direction,
            "C_combo_delta_flt3mut_low_means_CR": c_neg_direction,
            "D_venetoclax_predicted": d,
            "E_cytarabine_OBSERVED_oracle": e,
            "F_OS_continuous": f_results,
            "G_stdchemo_only_best_single": g,
        },
    }
    (cfg.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # Save updated cohort with observed AUC column
    cohort.to_csv(cohort_path, index=False)

    print(f"\n[ind-val] DONE → {cfg.out_dir}")
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clinical", default="data/raw/BeatAML2.0/beataml_clinical.xlsx")
    ap.add_argument("--features", default="data/canonical/beataml_patient_features.csv")
    ap.add_argument("--baseline", default="runs/baseline_single_drug_mlp/predictions_all_patients_all_drugs.csv")
    ap.add_argument("--combo", default="runs/combo_predictor/combo_auc_predictions.npz")
    ap.add_argument("--out", default="runs/induction_validation")
    args = ap.parse_args()
    cfg = InductionValidationConfig(
        clinical_xlsx=Path(args.clinical),
        patient_features_path=Path(args.features),
        baseline_pred_path=Path(args.baseline),
        combo_auc_npz_path=Path(args.combo),
        out_dir=Path(args.out),
    )
    run_induction_validation(cfg)


if __name__ == "__main__":
    main()
