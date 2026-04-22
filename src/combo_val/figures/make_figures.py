"""Generate manuscript figures from run artifacts.

Figure 1 — Head-to-head Δ distribution + FLT3 subgroup breakdown
Figure 2 — TCGA top recommendations + OS stratification
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = Path("runs")
DOCS = Path("docs")
FIG_DIR = DOCS / "figures"


def _setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def figure1_head_to_head():
    delta_df = pd.read_csv(RUNS / "head_to_head" / "clinical_drugs" / "per_patient_delta.csv")
    summary = json.load(open(RUNS / "head_to_head" / "clinical_drugs" / "summary.json"))
    # Add FLT3 mutation from patient features
    pf = pd.read_csv("data/canonical/beataml_patient_features.csv")[["patient_id", "mut_FLT3"]]
    delta_df["patient_id"] = delta_df["patient_id"].astype(int)
    pf["patient_id"] = pf["patient_id"].astype(int)
    df = delta_df.merge(pf, on="patient_id", how="left")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: overall Δ histogram
    ax = axs[0]
    deltas = df["delta"].values
    ax.hist(deltas, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.axvline(deltas.mean(), color="#C44E52", linestyle="-", linewidth=1.5, label=f"mean = {deltas.mean():.2f}")
    ax.set_xlabel("Δ = best_single_AUC − best_combo_AUC")
    ax.set_ylabel("Patients")
    ax.set_title(f"(A) Overall Δ distribution (n={len(df)})")
    ax.legend()

    # Panel B: FLT3 stratification
    ax = axs[1]
    flt3_mut = df[df["mut_FLT3"] == 1]["delta"].values
    flt3_wt  = df[df["mut_FLT3"] == 0]["delta"].values
    ax.hist(flt3_wt, bins=30, alpha=0.6, label=f"FLT3-wt (n={len(flt3_wt)})", color="#4C72B0")
    ax.hist(flt3_mut, bins=30, alpha=0.7, label=f"FLT3-mut (n={len(flt3_mut)})", color="#C44E52")
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.axvline(flt3_mut.mean(), color="#C44E52", linestyle="-", linewidth=1.2)
    ax.axvline(flt3_wt.mean(), color="#4C72B0", linestyle="-", linewidth=1.2)
    ax.set_xlabel("Δ = best_single_AUC − best_combo_AUC")
    ax.set_ylabel("Patients")
    ax.set_title(f"(B) Stratified by FLT3 status")
    ax.legend()

    fig.suptitle("Figure 1 — Head-to-head combo vs best-single-drug prediction",
                 y=1.02, fontsize=12)

    out = FIG_DIR / "figure1_head_to_head.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] wrote {out}")


def figure2_tcga_validation():
    summary = json.load(open(RUNS / "tcga_validation" / "tcga_validation_summary.json"))
    tops = summary["tcga_top_combo_picks"][:10]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: bar chart of top TCGA combos
    ax = axs[0]
    pairs = [t["pair"] for t in tops]
    counts = [t["count"] for t in tops]
    # Short-label each pair to fit axis
    short = []
    for p in pairs:
        parts = p.split(" + ")
        short.append(" + ".join(x.split()[0] for x in parts))
    y = np.arange(len(short))[::-1]
    ax.barh(y, counts[::-1], color="#4C72B0", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(short[::-1], fontsize=8.5)
    ax.set_xlabel("Patients with combo in top-3 recommendation set")
    ax.set_title(f"(A) TCGA top-3 combo recommendations (n={summary['n_tcga_patients']})")

    # Panel B: OS boxplot by driver status
    ax = axs[1]
    os_stats = summary["os_stats"]
    driver_pos = os_stats["driver_positive"]
    driver_neg = os_stats["driver_negative"]
    flt3_mut = os_stats["FLT3_mutant"]
    flt3_wt = os_stats["FLT3_wildtype"]

    data_points = [
        ("Driver+\nn=" + str(driver_pos["n"]), driver_pos["median_os"], driver_pos["deceased_rate"]),
        ("Driver-\nn=" + str(driver_neg["n"]), driver_neg["median_os"], driver_neg["deceased_rate"]),
        ("FLT3-mut\nn=" + str(flt3_mut["n"]), flt3_mut["median_os"], flt3_mut["deceased_rate"]),
        ("FLT3-wt\nn=" + str(flt3_wt["n"]), flt3_wt["median_os"], flt3_wt["deceased_rate"]),
    ]
    labels = [d[0] for d in data_points]
    medians = [d[1] for d in data_points]
    deceased = [d[2] * 100 for d in data_points]

    x = np.arange(len(labels))
    ax.bar(x, medians, color=["#C44E52", "#4C72B0", "#C44E52", "#4C72B0"],
           edgecolor="white", alpha=0.85, label="Median OS (months)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Median OS (months)")
    ax.set_title(f"(B) TCGA overall survival (p={os_stats.get('mannwhitney_p', 'NA')})")

    # Annotate deceased%
    for xi, (m, d) in enumerate(zip(medians, deceased)):
        ax.text(xi, m + 0.3, f"{d:.0f}% deceased", ha="center", fontsize=8)

    fig.suptitle("Figure 2 — TCGA-LAML independent-cohort validation",
                 y=1.02, fontsize=12)

    out = FIG_DIR / "figure2_tcga_validation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    _setup_style()
    figure1_head_to_head()
    figure2_tcga_validation()


if __name__ == "__main__":
    main()
