"""Path A — theoretical-viability validation experiments.

Six experiments on BeatAML 613 patients × 20 clinical drugs × 9 clones:

  E1. Clone prevalence vs AML literature
  E2. Canonical clinical regimens rank their target populations higher
  E3. Arity scaling — coverage curve for N ∈ {1, 2, 3, 4}
  E4. Benefit of triplet scales with clonal complexity (IDA prediction)
  E5. Correlation with Baseline-A single-drug sensitivity (IDA baseline)
  E6. FLT3-mut patient case studies — per-patient top-5 combos

Each experiment produces:
  - Tabular numerical result (CSV)
  - A summary entry in runs/path_a/validation_summary.json

The docs/path_a_validation.md doc is produced separately from the JSON
and the CSVs in this module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from combo_val.combo.clonal_coverage import (
    CLONE_ARCHETYPES,
    build_drug_clone_coverage,
    build_patient_clone_matrix,
    diagnose,
    rank_top_combos_per_patient,
    score_named_regimens,
)
from combo_val.combo.mechanism_prior import BEATAML_TO_MECH_ID
from combo_val.validation.head_to_head import CLINICALLY_RELEVANT_AML_DRUGS


@dataclass(frozen=True)
class PathAConfig:
    patient_features_path: Path = Path("data/canonical/beataml_patient_features.csv")
    baseline_pred_path: Path = Path(
        "runs/baseline_single_drug_mlp/predictions_all_patients_all_drugs.csv"
    )
    out_dir: Path = Path("runs/path_a")
    random_state: int = 42


# ---------------------------------------------------------------------------
# Canonical regimens for E2
# ---------------------------------------------------------------------------

# Each regimen is (name, drug-list, target-population-predicate).
# All drugs must be in CLINICALLY_RELEVANT_AML_DRUGS for scoring to work.
CANONICAL_REGIMENS = {
    # Doublets
    "Venetoclax + Azacytidine (VIALE-A)": {
        "drugs": ["Venetoclax", "Azacytidine"],
        "target_mutation": None,               # general, esp. older/unfit
    },
    "Venetoclax + Ivosidenib (IDH1-mut)": {
        "drugs": ["Venetoclax", "Ivosidenib"],
        "target_mutation": "mut_IDH1",
    },
    "Venetoclax + Enasidenib (IDH2-mut ENAVEN)": {
        "drugs": ["Venetoclax", "Enasidenib"],
        "target_mutation": "mut_IDH2",
    },
    "Venetoclax + Gilteritinib (FLT3-mut)": {
        "drugs": ["Venetoclax", "Gilteritinib"],
        "target_mutation": "mut_FLT3",
    },
    "Venetoclax + Quizartinib (FLT3-ITD)": {
        "drugs": ["Venetoclax", "Quizartinib (AC220)"],
        "target_mutation": "mut_FLT3",
    },
    # Triplets
    "AZA + Venetoclax + Gilteritinib (JCO 2024)": {
        "drugs": ["Azacytidine", "Venetoclax", "Gilteritinib"],
        "target_mutation": "mut_FLT3",
    },
    "AZA + Venetoclax + Quizartinib": {
        "drugs": ["Azacytidine", "Venetoclax", "Quizartinib (AC220)"],
        "target_mutation": "mut_FLT3",
    },
    "AZA + Venetoclax + Ivosidenib (AGILE-adjacent)": {
        "drugs": ["Azacytidine", "Venetoclax", "Ivosidenib"],
        "target_mutation": "mut_IDH1",
    },
    "AZA + Venetoclax + Enasidenib": {
        "drugs": ["Azacytidine", "Venetoclax", "Enasidenib"],
        "target_mutation": "mut_IDH2",
    },
    "AZA + Venetoclax + Midostaurin": {
        "drugs": ["Azacytidine", "Venetoclax", "Midostaurin"],
        "target_mutation": "mut_FLT3",
    },
    # Negative controls (clinically implausible)
    "NEG: Imatinib + Nilotinib (double CML)": {
        "drugs": ["Imatinib", "Nilotinib"],
        "target_mutation": None,               # no AML target
    },
    "NEG: Trametinib + Selumetinib (double MEKi)": {
        "drugs": ["Trametinib (GSK1120212)", "Selumetinib (AZD6244)"],
        "target_mutation": None,
    },
    "NEG: Crizotinib monotherapy (lung drug)": {
        "drugs": ["Crizotinib (PF-2341066)"],
        "target_mutation": None,
    },
}


# ---------------------------------------------------------------------------
# E1. Clone prevalence sanity
# ---------------------------------------------------------------------------


# AML literature prevalence rates (Papaemmanuil 2016, Bullinger 2017,
# Döhner 2022, TCGA 2013). Used as loose sanity bounds.
LIT_PREVALENCE = {
    "FLT3_clone": (0.25, 0.40),          # 25-40%
    "MENIN_HOX_clone": (0.25, 0.40),     # NPM1 + KMT2A
    "IDH1_clone": (0.04, 0.12),          # 4-12%
    "IDH2_clone": (0.06, 0.15),          # 6-15%
    "TP53_clone": (0.05, 0.15),          # 5-15%, higher in older/secondary
    "RAS_MAPK_clone": (0.15, 0.30),      # NRAS + KRAS + PTPN11
}


def E1_clone_prevalence(cfg: PathAConfig, patient_clones: pd.DataFrame) -> dict:
    results = {}
    for clone, (lo, hi) in LIT_PREVALENCE.items():
        obs = float((patient_clones[clone] > 0).mean())
        results[clone] = {
            "observed_frac": round(obs, 3),
            "literature_range": [lo, hi],
            "in_range": (lo <= obs <= hi),
        }
    # Co-occurrence check: FLT3 + NPM1 should co-occur more than by chance
    # (Papaemmanuil: ~60% of FLT3-mut patients are also NPM1-mut)
    flt3_npm1_joint = int(
        (patient_clones["FLT3_clone"] > 0)
        .astype(int).__mul__((patient_clones["MENIN_HOX_clone"] > 0).astype(int))
        .sum()
    )
    flt3_n = int((patient_clones["FLT3_clone"] > 0).sum())
    results["FLT3_NPM1_KMT2A_co_occurrence"] = {
        "joint_count": flt3_npm1_joint,
        "FLT3_n": flt3_n,
        "frac_FLT3_also_MENIN_HOX": round(flt3_npm1_joint / max(1, flt3_n), 3),
        "literature_expectation": "~60% of FLT3-mut co-occur with NPM1",
    }
    return results


# ---------------------------------------------------------------------------
# E2. Canonical regimens rank target populations higher
# ---------------------------------------------------------------------------


def E2_canonical_regimen_recognition(
    cfg: PathAConfig,
    patient_features: pd.DataFrame,
    patient_clones: pd.DataFrame,
    drug_clone_cov: pd.DataFrame,
) -> dict:
    regimens = {name: spec["drugs"] for name, spec in CANONICAL_REGIMENS.items()}
    scores = score_named_regimens(patient_clones, drug_clone_cov, regimens)

    out: dict = {}
    for name, spec in CANONICAL_REGIMENS.items():
        target_mut = spec["target_mutation"]
        regimen_scores = scores[name]
        if target_mut and target_mut in patient_features.columns:
            tgt_mask = patient_features[target_mut] == 1
            tgt_mean = float(regimen_scores[tgt_mask].mean())
            non_tgt_mean = float(regimen_scores[~tgt_mask].mean())
            # Mann-Whitney test: target patients should score higher
            u_stat, p = stats.mannwhitneyu(
                regimen_scores[tgt_mask], regimen_scores[~tgt_mask],
                alternative="greater",
            )
            out[name] = {
                "target": target_mut,
                "n_target": int(tgt_mask.sum()),
                "n_non_target": int((~tgt_mask).sum()),
                "target_mean_coverage": round(tgt_mean, 3),
                "non_target_mean_coverage": round(non_tgt_mean, 3),
                "delta": round(tgt_mean - non_tgt_mean, 3),
                "mannwhitney_p": round(float(p), 6),
            }
        else:
            out[name] = {
                "target": "general",
                "mean_coverage": round(float(regimen_scores.mean()), 3),
                "std": round(float(regimen_scores.std()), 3),
            }
    return out


# ---------------------------------------------------------------------------
# E3. Arity scaling curve
# ---------------------------------------------------------------------------


def _score_best_of_arity(
    patient_clones_arr: np.ndarray,   # (n_patients, n_clones)
    drug_cov_arr: np.ndarray,          # (n_drugs, n_clones)
    arity: int,
) -> np.ndarray:
    """For each patient, the score of the BEST combo of the given arity."""
    n_drugs = drug_cov_arr.shape[0]
    n_patients = patient_clones_arr.shape[0]
    present = (patient_clones_arr > 0).astype(np.float64)
    weights = patient_clones_arr * present
    total_w = weights.sum(axis=1)

    best = np.zeros(n_patients, dtype=np.float64)
    for combo in combinations(range(n_drugs), arity):
        cov_sub = drug_cov_arr[list(combo)]
        per_clone = 1.0 - np.prod(1.0 - np.clip(cov_sub, 0.0, 1.0), axis=0)
        num = (weights * per_clone[None, :]).sum(axis=1)
        score = np.where(total_w > 0, num / total_w, 0.0)
        best = np.maximum(best, score)
    return best


def E3_arity_scaling(
    cfg: PathAConfig,
    patient_clones: pd.DataFrame,
    drug_clone_cov: pd.DataFrame,
    arities: tuple[int, ...] = (1, 2, 3, 4),
) -> tuple[pd.DataFrame, dict]:
    pclones = patient_clones.to_numpy(dtype=np.float64)
    dcov = drug_clone_cov.to_numpy(dtype=np.float64)

    records = []
    summary = {}
    for ar in arities:
        scores = _score_best_of_arity(pclones, dcov, ar)
        for pid, s in zip(patient_clones.index, scores):
            records.append({"patient_id": pid, "arity": ar, "best_score": s})
        summary[f"arity_{ar}"] = {
            "mean_best_score": round(float(scores.mean()), 3),
            "median_best_score": round(float(np.median(scores)), 3),
            "std": round(float(scores.std()), 3),
            "q25": round(float(np.quantile(scores, 0.25)), 3),
            "q75": round(float(np.quantile(scores, 0.75)), 3),
        }
    df = pd.DataFrame(records)
    # Marginal gain per additional drug
    for i, ar in enumerate(arities):
        if i > 0:
            gain = summary[f"arity_{ar}"]["mean_best_score"] - summary[f"arity_{arities[i-1]}"]["mean_best_score"]
            summary[f"arity_{ar}"]["marginal_gain_over_{arities[i-1]}"] = round(gain, 3)
    return df, summary


# ---------------------------------------------------------------------------
# E4. Clonal complexity → triplet benefit
# ---------------------------------------------------------------------------


def _count_drivers(patient_features: pd.DataFrame) -> pd.Series:
    """Count of driver clones present (sum of mut-driven clones with weight 1.0)."""
    # A clone is "driven" if any of its presence_markers fires
    n_drivers = pd.Series(0, index=patient_features.index, dtype=int)
    for clone, spec in CLONE_ARCHETYPES.items():
        if spec["weight"] < 1.0 or "DEFAULT" in spec["presence_markers"]:
            continue
        markers = [m for m in spec["presence_markers"] if m in patient_features.columns]
        if not markers:
            continue
        present = (patient_features[markers].sum(axis=1) > 0).astype(int)
        n_drivers = n_drivers + present
    return n_drivers


def E4_clone_count_vs_triplet_benefit(
    cfg: PathAConfig,
    patient_features: pd.DataFrame,
    patient_clones: pd.DataFrame,
    drug_clone_cov: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    pclones = patient_clones.to_numpy(dtype=np.float64)
    dcov = drug_clone_cov.to_numpy(dtype=np.float64)

    best_1 = _score_best_of_arity(pclones, dcov, 1)
    best_2 = _score_best_of_arity(pclones, dcov, 2)
    best_3 = _score_best_of_arity(pclones, dcov, 3)

    n_drivers = _count_drivers(patient_features.loc[patient_clones.index])
    df = pd.DataFrame({
        "patient_id": patient_clones.index,
        "n_drivers": n_drivers.values,
        "best_single": best_1,
        "best_doublet": best_2,
        "best_triplet": best_3,
        "doublet_gain": best_2 - best_1,
        "triplet_gain_over_doublet": best_3 - best_2,
    })

    summary = {}
    for nd in sorted(df["n_drivers"].unique()):
        sub = df[df["n_drivers"] == nd]
        summary[f"n_drivers_{int(nd)}"] = {
            "n_patients": int(len(sub)),
            "mean_best_single": round(float(sub["best_single"].mean()), 3),
            "mean_best_doublet": round(float(sub["best_doublet"].mean()), 3),
            "mean_best_triplet": round(float(sub["best_triplet"].mean()), 3),
            "mean_doublet_gain": round(float(sub["doublet_gain"].mean()), 3),
            "mean_triplet_gain_over_doublet": round(float(sub["triplet_gain_over_doublet"].mean()), 3),
        }

    # Spearman: does triplet gain scale with driver count?
    rho_doublet, p_doublet = stats.spearmanr(df["n_drivers"], df["doublet_gain"])
    rho_triplet, p_triplet = stats.spearmanr(df["n_drivers"], df["triplet_gain_over_doublet"])
    summary["spearman_n_drivers_vs_doublet_gain"] = {
        "rho": round(float(rho_doublet), 3),
        "p": round(float(p_doublet), 6),
    }
    summary["spearman_n_drivers_vs_triplet_gain"] = {
        "rho": round(float(rho_triplet), 3),
        "p": round(float(p_triplet), 6),
    }
    return df, summary


# ---------------------------------------------------------------------------
# E5. Coverage correlates with Baseline-A-derived IDA score
# ---------------------------------------------------------------------------


def _baseline_ida_score(
    baseline_auc: pd.DataFrame,           # (n_patients, n_drugs) lower=better
    drug_list: list[str],
    combo: list[str],
    auc_threshold: float = 150.0,        # below = "sensitive"
) -> pd.Series:
    """IDA probability that patient is sensitive to ≥ 1 drug in the combo.

    Uses Baseline A predictions directly: each drug's sensitivity probability
    = sigmoid((threshold - auc) / scale). Bliss-independent aggregation
    gives the combo-level "at least one drug hits" probability.
    """
    sub = baseline_auc[combo]                                   # (n_patients, k)
    # Convert AUC → sensitivity probability; lower AUC = higher sensitivity
    sigmoid_scale = 30.0
    p_sens = 1.0 / (1.0 + np.exp((sub - auc_threshold) / sigmoid_scale))
    # IDA: 1 - Π(1 - p_i)
    return 1.0 - (1.0 - p_sens).prod(axis=1)


def E5_coverage_vs_ida(
    cfg: PathAConfig,
    patient_clones: pd.DataFrame,
    drug_clone_cov: pd.DataFrame,
    baseline_auc: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    drugs = drug_clone_cov.index.tolist()
    # Score a subset of key regimens both ways
    records = []
    regimens_to_test = {**{name: spec["drugs"] for name, spec in CANONICAL_REGIMENS.items()}}

    for name, drug_list in regimens_to_test.items():
        drugs_in_baseline = [d for d in drug_list if d in baseline_auc.columns]
        drugs_in_clone = [d for d in drug_list if d in drug_clone_cov.index]
        if len(drugs_in_baseline) < 1 or len(drugs_in_clone) < 1:
            continue
        # Path A coverage score
        cov_scores = score_named_regimens(
            patient_clones, drug_clone_cov, {name: drugs_in_clone}
        )[name]
        # IDA from Baseline A
        ida = _baseline_ida_score(baseline_auc, drugs, drugs_in_baseline)
        # Align indices
        common = cov_scores.index.intersection(ida.index)
        for pid in common:
            records.append({
                "regimen": name,
                "patient_id": pid,
                "path_a_coverage": float(cov_scores.loc[pid]),
                "baseline_ida": float(ida.loc[pid]),
            })
    df = pd.DataFrame(records)

    summary = {}
    for name in df["regimen"].unique():
        sub = df[df["regimen"] == name]
        if len(sub) < 20:
            continue
        rho, p = stats.spearmanr(sub["path_a_coverage"], sub["baseline_ida"])
        summary[name] = {
            "n": int(len(sub)),
            "spearman_coverage_vs_ida": round(float(rho), 3),
            "p": round(float(p), 6),
            "path_a_mean": round(float(sub["path_a_coverage"].mean()), 3),
            "ida_mean": round(float(sub["baseline_ida"].mean()), 3),
        }

    # Pooled
    rho_all, p_all = stats.spearmanr(df["path_a_coverage"], df["baseline_ida"])
    summary["pooled"] = {
        "n": int(len(df)),
        "spearman": round(float(rho_all), 3),
        "p": round(float(p_all), 6),
    }
    return df, summary


# ---------------------------------------------------------------------------
# E6. FLT3-mut patient case studies
# ---------------------------------------------------------------------------


def E6_flt3_mut_case_studies(
    cfg: PathAConfig,
    patient_features: pd.DataFrame,
    patient_clones: pd.DataFrame,
    drug_clone_cov: pd.DataFrame,
    n_cases: int = 5,
) -> tuple[pd.DataFrame, dict]:
    flt3 = patient_features[patient_features["mut_FLT3"] == 1].index
    # Pick cases with diverse co-mutation patterns
    pf = patient_features.loc[flt3]
    # Sort by co-mutation count to get variety
    pf = pf.assign(cm_count=(
        pf[["mut_NPM1", "mut_IDH1", "mut_IDH2", "mut_TP53",
            "mut_NRAS", "mut_KRAS", "mut_PTPN11"]].sum(axis=1)
    )).sort_values("cm_count")

    # Pick diverse: 1 case each of 0, 1, 2, 3+ co-mutations
    cases = []
    for cm_n in [0, 1, 2, 3]:
        bucket = pf[pf["cm_count"] == cm_n] if cm_n < 3 else pf[pf["cm_count"] >= 3]
        if len(bucket):
            cases.append(bucket.index[0])
    while len(cases) < n_cases and n_cases - len(cases) <= len(pf):
        extra = pf.index.difference(cases)[0]
        cases.append(extra)

    pc_sub = patient_clones.loc[cases]
    top3 = rank_top_combos_per_patient(pc_sub, drug_clone_cov, arity=3, top_k=5)
    top2 = rank_top_combos_per_patient(pc_sub, drug_clone_cov, arity=2, top_k=3)

    cases_df = pd.DataFrame({
        "patient_id": cases,
        "FLT3": patient_features.loc[cases, "mut_FLT3"].values,
        "NPM1": patient_features.loc[cases, "mut_NPM1"].values,
        "IDH1": patient_features.loc[cases, "mut_IDH1"].values,
        "IDH2": patient_features.loc[cases, "mut_IDH2"].values,
        "TP53": patient_features.loc[cases, "mut_TP53"].values,
        "RAS_MAPK_any": (patient_features.loc[cases, ["mut_NRAS", "mut_KRAS", "mut_PTPN11"]].sum(axis=1) > 0).astype(int).values,
    })

    # Serialize
    summary: dict = {"cases": []}
    for pid in cases:
        case_info = {
            "patient_id": int(pid),
            "clones_present": pc_sub.loc[pid][pc_sub.loc[pid] > 0].to_dict(),
            "top_3_drug_triplets": top3[top3["patient_id"] == pid].to_dict("records"),
            "top_2_drug_doublets": top2[top2["patient_id"] == pid].to_dict("records"),
        }
        summary["cases"].append(case_info)
    return cases_df, summary


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_all_experiments(cfg: PathAConfig | None = None) -> dict:
    cfg = cfg or PathAConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.random_state)

    print("[path_a] loading inputs ...")
    pf = pd.read_csv(cfg.patient_features_path).set_index("patient_id")
    baseline_pred = pd.read_csv(cfg.baseline_pred_path, index_col=0)

    drugs = list(CLINICALLY_RELEVANT_AML_DRUGS)
    patient_clones = build_patient_clone_matrix(pf)
    drug_clone_cov = build_drug_clone_coverage(drugs)

    diag = diagnose(patient_clones, drug_clone_cov)
    print(f"[path_a] {diag.n_patients} patients × {diag.n_drugs} drugs × {diag.n_clones} clones  "
          f"({diag.drugs_annotated}/{diag.n_drugs} drugs annotated)")

    patient_clones.to_csv(cfg.out_dir / "patient_clones.csv")
    drug_clone_cov.to_csv(cfg.out_dir / "drug_clone_coverage.csv")

    print("\n=== E1: Clone prevalence ===")
    e1 = E1_clone_prevalence(cfg, patient_clones)
    for k, v in e1.items():
        print(f"  {k}: {v}")

    print("\n=== E2: Canonical regimen recognition ===")
    e2 = E2_canonical_regimen_recognition(cfg, pf, patient_clones, drug_clone_cov)
    for name, stats_d in e2.items():
        marker = "✓" if stats_d.get("delta", 0) > 0.05 else ("—" if "delta" not in stats_d else "✗")
        print(f"  {marker} {name}: {stats_d}")

    print("\n=== E3: Arity scaling ===")
    e3_df, e3_sum = E3_arity_scaling(cfg, patient_clones, drug_clone_cov)
    e3_df.to_csv(cfg.out_dir / "E3_arity_scaling.csv", index=False)
    for k, v in e3_sum.items():
        print(f"  {k}: {v}")

    print("\n=== E4: Clone count → triplet benefit ===")
    e4_df, e4_sum = E4_clone_count_vs_triplet_benefit(cfg, pf, patient_clones, drug_clone_cov)
    e4_df.to_csv(cfg.out_dir / "E4_clone_count_vs_gain.csv", index=False)
    for k, v in e4_sum.items():
        print(f"  {k}: {v}")

    print("\n=== E5: Coverage vs Baseline A IDA ===")
    e5_df, e5_sum = E5_coverage_vs_ida(cfg, patient_clones, drug_clone_cov, baseline_pred)
    e5_df.to_csv(cfg.out_dir / "E5_coverage_vs_ida.csv", index=False)
    for k, v in e5_sum.items():
        print(f"  {k}: {v}")

    print("\n=== E6: FLT3-mut case studies ===")
    e6_df, e6_sum = E6_flt3_mut_case_studies(cfg, pf, patient_clones, drug_clone_cov)
    e6_df.to_csv(cfg.out_dir / "E6_flt3_cases.csv", index=False)
    for case in e6_sum["cases"]:
        pid = case["patient_id"]
        clones = list(case["clones_present"].keys())
        triplet_list = [
            f"{t['drug1']}+{t['drug2']}+{t['drug3']}@{t['score']:.2f}"
            for t in case["top_3_drug_triplets"][:3]
        ]
        print(f"  patient {pid}  clones={clones}  top-3 triplets: {triplet_list}")

    # --- Combined summary JSON ---
    summary = {
        "cfg": asdict(cfg),
        "diagnostics": {
            "n_patients": diag.n_patients,
            "n_drugs": diag.n_drugs,
            "n_clones": diag.n_clones,
            "clone_prevalence": diag.clone_prevalence,
            "drugs_annotated": diag.drugs_annotated,
            "drugs_unannotated": diag.drugs_unannotated,
        },
        "E1_clone_prevalence": e1,
        "E2_canonical_regimens": e2,
        "E3_arity_scaling": e3_sum,
        "E4_clone_count_vs_gain": e4_sum,
        "E5_coverage_vs_ida": e5_sum,
        "E6_flt3_cases": e6_sum,
    }
    out_path = cfg.out_dir / "validation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[path_a] summary → {out_path}")
    return summary


def main():
    run_all_experiments()


if __name__ == "__main__":
    main()
