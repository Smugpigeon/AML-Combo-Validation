"""Week 5 — Independent-cohort validation on TCGA-LAML (n=173).

We can't directly transfer Baseline A's single-drug MLP because TCGA and
BeatAML fit independent PCAs (different spaces). However, two components
of our predictor transfer cleanly:

  1. Mechanism prior — uses mutation features + curated drug-mechanism CSV.
     Mutation axes (mut_FLT3, mut_NPM1, …) are the same schema across cohorts,
     so the prior computes the same way.

  2. Learned synergy residual — depends only on drug IDs, not patient features.

Week 5 analyses:

  A. **Recommendation reproducibility**: Does the mech-prior-only top combo
     for TCGA FLT3-mut patients match what it was for BeatAML FLT3-mut
     patients (Quizartinib + Venetoclax)?

  B. **Biological validation via OS**: Compare overall survival between
     TCGA patients flagged as "combo-benefiting" by the mech prior vs those
     flagged as "combo-agnostic." Hypothesis: combo-benefiting patients
     have different OS than combo-agnostic ones — direction is informative
     either way, but we specifically expect worse OS in the driver-positive
     group (more aggressive AML biology).

  C. **Clinically-relevant top combos**: Same filter as Week 4. Does the
     combo predictor produce the same top-pair distribution on TCGA as
     on BeatAML?

Note: TCGA patients largely received conventional 7+3 induction, so we CAN'T
test "recommended combo → better outcome" because no one got the
recommended combo. Week 5 is about consistency of RECOMMENDATION LOGIC,
not outcome alignment.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from combo_val.combo.mechanism_prior import (
    BEATAML_TO_MECH_ID,
    compute_combo_mech_scores,
    diagnostics as mech_diagnostics,
    patient_target_deficit,
)
from combo_val.validation.head_to_head import CLINICALLY_RELEVANT_AML_DRUGS


@dataclass(frozen=True)
class TCGAValidationConfig:
    tcga_features_path: Path = Path("data/canonical/tcga_laml_patient_features.csv")
    tcga_clinical_path: Path = Path("data/canonical/tcga_laml_clinical.csv")
    combo_auc_npz_path: Path = Path("runs/combo_predictor/combo_auc_predictions.npz")
    out_dir: Path = Path("runs/tcga_validation")
    # How much weight to give the BeatAML-trained synergy pair-residual when
    # combining with the patient-specific mech prior. The mech prior values
    # are roughly {0, 1, 2}; synergy is roughly [-50, +15]. We normalize the
    # synergy to a similar scale.
    synergy_weight: float = 0.05


def _select_top_combos(mech_scores: np.ndarray, drug_names: list[str],
                        top_k: int = 5) -> list[dict]:
    """For each patient, return top-k pairs by mech_score.

    mech_scores: (n_patients, n_drugs, n_drugs) symmetric
    Returns a list of dicts suitable for writing to CSV.
    """
    n_p, n_d, _ = mech_scores.shape
    tri_i, tri_j = np.triu_indices(n_d, k=1)
    out = []
    for p in range(n_p):
        pair_scores = mech_scores[p, tri_i, tri_j]
        if pair_scores.max() <= 0:
            # No positive-mech pair for this patient; skip top-k but record 1 row
            out.append({"patient_id_idx": p, "rank": 1, "d1_idx": -1, "d2_idx": -1,
                        "mech_score": 0.0})
            continue
        top_k_idx = np.argsort(pair_scores)[::-1][:top_k]
        for rank, k in enumerate(top_k_idx):
            if pair_scores[k] <= 0:
                break
            out.append({
                "patient_id_idx": p, "rank": rank + 1,
                "d1_idx": int(tri_i[k]), "d2_idx": int(tri_j[k]),
                "mech_score": float(pair_scores[k]),
            })
    return out


def run_tcga_validation(cfg: TCGAValidationConfig | None = None) -> dict:
    cfg = cfg or TCGAValidationConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    pf = pd.read_csv(cfg.tcga_features_path).set_index("patient_id")
    clin = pd.read_csv(cfg.tcga_clinical_path).rename(columns={"patientId": "patient_id"})
    clin = clin.set_index("patient_id")
    clin["os_months"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
    clin["deceased"] = clin["OS_STATUS"].astype(str).str.startswith("1").astype(int)

    print(f"[tcga-val] TCGA patients: {len(pf)}")
    print(f"[tcga-val] FLT3-mut: {int(pf['mut_FLT3'].sum())}")
    print(f"[tcga-val] NPM1-mut: {int(pf['mut_NPM1'].sum())}")
    print(f"[tcga-val] IDH1-mut: {int(pf['mut_IDH1'].sum())}")
    print(f"[tcga-val] IDH2-mut: {int(pf['mut_IDH2'].sum())}")

    # Drug universe: the clinical filter (matches Week 4 clinical filter)
    drug_names = list(CLINICALLY_RELEVANT_AML_DRUGS)
    print(f"[tcga-val] drug universe: {len(drug_names)} clinically-relevant AML drugs")

    diag = mech_diagnostics(pf, drug_names)
    print(f"[tcga-val] mech diagnostics: {diag}")

    # Compute mechanism-prior combo scores (patient, d1, d2)
    mech_scores = compute_combo_mech_scores(pf, drug_names)
    print(f"[tcga-val] mech_scores shape: {mech_scores.shape}  "
          f"overall max: {mech_scores.max():.3f}  "
          f"positive entries: {(mech_scores > 0).sum():,}")

    # Week 5 uses mech-prior-only scoring. Synergy transfer was attempted but
    # amplified a single BeatAML-ALMANAC pair (Cytarabine+Ruxolitinib with
    # strongly negative residual) to dominate every TCGA patient. We prefer
    # to report the top-5 mechanism-prior combos per TCGA patient and measure
    # overlap with BeatAML recommendations at the SET level, not by forcing
    # a single winner that can be decided by argsort ties.
    print("[tcga-val] scoring by mechanism prior only (no synergy transfer)")

    # A: Top combo per patient (by mech score)
    top_rows = _select_top_combos(mech_scores, drug_names, top_k=3)
    topdf = pd.DataFrame(top_rows)
    topdf["patient_id"] = topdf["patient_id_idx"].map(lambda i: pf.index[i])
    topdf["drug1"] = topdf["d1_idx"].map(lambda i: drug_names[i] if i >= 0 else None)
    topdf["drug2"] = topdf["d2_idx"].map(lambda i: drug_names[i] if i >= 0 else None)
    topdf = topdf[["patient_id", "rank", "drug1", "drug2", "mech_score"]]
    topdf.to_csv(cfg.out_dir / "tcga_top_combos.csv", index=False)

    # Report TOP-K coverage: how many TCGA patients have each candidate pair
    # in their top-k (k=3) mechanism-prior picks? More stable than rank-1
    # which depends on argsort tie-breaking.
    top_all_ranks = topdf.dropna(subset=["drug1"])
    top_pairs_tcga = (
        top_all_ranks.apply(lambda r: tuple(sorted([r["drug1"], r["drug2"]])), axis=1)
        .value_counts().head(12)
    )

    # B: OS analysis by mech engagement
    #   Use the driver-mutation-derived deficit directly; anything > 0 means
    #   the patient has at least one targetable driver. This is independent
    #   of argsort or tiebreaker noise.
    deficit = patient_target_deficit(pf)
    mech_pos = (deficit.sum(axis=1) > 0).to_numpy()
    # Patient-level: align with clinical
    mech_pos_series = pd.Series(mech_pos, index=pf.index, name="mech_positive")
    merged = clin.join(mech_pos_series, how="inner")
    n_pos = int(merged["mech_positive"].sum())
    n_neg = int(len(merged) - n_pos)
    print(f"[tcga-val] OS cohorts: driver+ = {n_pos}, driver- = {n_neg}")

    # Log-rank test (simple survival comparison)
    def _surv_stats(sub: pd.DataFrame) -> dict:
        sub = sub.dropna(subset=["os_months"])
        return {
            "n": int(len(sub)),
            "median_os": round(float(sub["os_months"].median()), 2),
            "mean_os": round(float(sub["os_months"].mean()), 2),
            "deceased_rate": round(float(sub["deceased"].mean()), 3),
        }

    os_stats = {
        "driver_positive": _surv_stats(merged[merged["mech_positive"]]),
        "driver_negative": _surv_stats(merged[~merged["mech_positive"]]),
    }

    # Mann-Whitney on OS months (censored values treated naively; not formal
    # log-rank but adequate for this level of Week 5 analysis)
    pos_os = merged[merged["mech_positive"]]["os_months"].dropna().values
    neg_os = merged[~merged["mech_positive"]]["os_months"].dropna().values
    if len(pos_os) > 0 and len(neg_os) > 0:
        u_stat, u_pval = stats.mannwhitneyu(pos_os, neg_os, alternative="two-sided")
        os_stats["mannwhitney_p"] = round(float(u_pval), 4)
    else:
        os_stats["mannwhitney_p"] = None

    # FLT3-specific OS
    flt3_pos = pf["mut_FLT3"] == 1
    flt3_clin = clin.loc[flt3_pos[flt3_pos].index.intersection(clin.index)]
    flt3_neg_clin = clin.loc[(~flt3_pos)[~flt3_pos].index.intersection(clin.index)]
    os_stats["FLT3_mutant"] = _surv_stats(flt3_clin)
    os_stats["FLT3_wildtype"] = _surv_stats(flt3_neg_clin)

    # C: Do the top-pair recommendations match BeatAML's top picks?
    beataml_top_combos = (
        "Quizartinib (AC220) + Venetoclax",
        "Gilteritinib + Venetoclax",
        "Quizartinib (AC220) + Trametinib (GSK1120212)",
        "Gilteritinib + Trametinib (GSK1120212)",
    )
    tcga_top_pairs_str = [f"{p[0]} + {p[1]}" for p in top_pairs_tcga.index]
    reproduced_combos = [c for c in beataml_top_combos if c in tcga_top_pairs_str]

    summary = {
        "cfg": asdict(cfg),
        "n_tcga_patients": int(len(pf)),
        "mutation_prevalence_tcga": {
            "FLT3": round(float(pf["mut_FLT3"].mean()), 3),
            "NPM1": round(float(pf["mut_NPM1"].mean()), 3),
            "IDH1": round(float(pf["mut_IDH1"].mean()), 3),
            "IDH2": round(float(pf["mut_IDH2"].mean()), 3),
            "DNMT3A": round(float(pf["mut_DNMT3A"].mean()), 3),
        },
        "mechanism_diagnostics": diag,
        "tcga_top_combo_picks": [
            {"pair": f"{p[0]} + {p[1]}", "count": int(n)}
            for p, n in top_pairs_tcga.items()
        ],
        "reproduced_beataml_top_combos": reproduced_combos,
        "os_stats": os_stats,
    }
    (cfg.out_dir / "tcga_validation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n[tcga-val] Top combo recommendations for TCGA:")
    for p, n in top_pairs_tcga.items():
        print(f"    {p[0]:35s} + {p[1]:35s}  {n}")
    print(f"\n[tcga-val] BeatAML→TCGA reproduced combos: {reproduced_combos}")

    print(f"\n[tcga-val] OS by mech engagement:")
    for k, v in os_stats.items():
        if isinstance(v, dict):
            print(f"    {k:20s}  n={v['n']:3d}  median_os={v['median_os']:6.2f}  "
                  f"deceased={v['deceased_rate']:.1%}")
    if os_stats.get("mannwhitney_p") is not None:
        print(f"    Mann-Whitney p (driver+ vs driver-): {os_stats['mannwhitney_p']:.4f}")
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tcga-features", default="data/canonical/tcga_laml_patient_features.csv")
    ap.add_argument("--tcga-clinical", default="data/canonical/tcga_laml_clinical.csv")
    ap.add_argument("--out", default="runs/tcga_validation")
    args = ap.parse_args()
    cfg = TCGAValidationConfig(
        tcga_features_path=Path(args.tcga_features),
        tcga_clinical_path=Path(args.tcga_clinical),
        out_dir=Path(args.out),
    )
    run_tcga_validation(cfg)


if __name__ == "__main__":
    main()
