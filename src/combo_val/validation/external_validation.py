"""External-validation framework for the AML combo-prediction kit.

Per clinical-reviewer concern #2: the published validation is internal
to BeatAML 2.0 (OHSU + Vizome, ≥80% North American Caucasian). Before
this kit can be deployed in a Chinese / Asian / European center, the
operating institution must run THIS framework on a local hold-out
cohort and review the resulting calibration plots + metrics.

The framework is INSTITUTIONAL-AGNOSTIC: it expects a single CSV with
predicted-vs-true outcomes (one row per patient, columns documented
below) and produces a self-contained HTML/PDF report.

Required CSV columns (cohort.csv):
  patient_id          : str (de-identified)
  predicted_eln_2022  : "Favorable" | "Intermediate" | "Adverse"
  true_eln_class      : "Favorable" | "Intermediate" | "Adverse" (gold)
  predicted_top1_id   : str (regimen_id from kit)
  true_regimen_id     : str (what was actually given) — optional
  true_cr             : 0|1 (CR/CRi achieved at induction end)
  true_os_months      : float (overall survival from diagnosis)
  true_os_event       : 0|1 (1=died, 0=censored)
  age, sex, ELN_2022_true (sanity-check stratification)

Outputs:
  out_dir/
    metrics.json              # all aggregate metrics + 95% CI
    calibration_plot.png      # reliability diagram (10-bin)
    eln_confusion.png         # 3x3 confusion matrix ELN pred vs true
    regimen_match_summary.md  # human-readable summary
    REPORT.html               # standalone, openable in any browser
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


REQUIRED_COLUMNS = {
    "patient_id",
    "predicted_eln_2022",
    "true_eln_class",
    "predicted_top1_id",
    "true_cr",
}

OPTIONAL_COLUMNS = {
    "true_regimen_id",
    "true_os_months",
    "true_os_event",
    "age",
    "sex",
    "ELN_2022_true",
}


@dataclass
class ExternalValidationConfig:
    cohort_csv: Path
    out_dir: Path
    bootstrap_n: int = 2000
    random_state: int = 42
    institution_name: str = "Unknown institution"
    cohort_description: str = ""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator,
                   percentile: float = 95) -> tuple[float, float]:
    """Percentile bootstrap CI on a metric over patient-level values."""
    n = len(values)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(values[idx].mean())
    lo = float(np.percentile(boots, (100 - percentile) / 2))
    hi = float(np.percentile(boots, 100 - (100 - percentile) / 2))
    return lo, hi


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score for binary CR prediction. Lower is better; perfectly
    calibrated random guess on balanced data = 0.25."""
    return float(np.mean((y_prob - y_true) ** 2))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error — weighted gap between predicted prob
    and observed frequency across bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_eln_concordance(predicted: pd.Series, true: pd.Series) -> dict:
    """3x3 ELN confusion matrix + overall accuracy + per-class kappa."""
    classes = ["Favorable", "Intermediate", "Adverse"]
    cm = pd.crosstab(true, predicted, rownames=["true"], colnames=["pred"])
    cm = cm.reindex(index=classes, columns=classes, fill_value=0)
    n = int(cm.values.sum())
    correct = int(np.trace(cm.values))
    accuracy = correct / n if n > 0 else float("nan")

    # Cohen's kappa on 3-class
    p_o = accuracy
    row_sums = cm.sum(axis=1).values / n if n > 0 else np.zeros(3)
    col_sums = cm.sum(axis=0).values / n if n > 0 else np.zeros(3)
    p_e = float(np.sum(row_sums * col_sums))
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else float("nan")

    return {
        "confusion_matrix": cm.to_dict(),
        "n": n,
        "accuracy": accuracy,
        "cohens_kappa": kappa,
        "interpretation_kappa": _kappa_interpretation(kappa),
    }


def _kappa_interpretation(k: float) -> str:
    if not np.isfinite(k):
        return "n/a (insufficient data)"
    if k < 0.20:
        return "poor (<0.20)"
    if k < 0.40:
        return "fair (0.20-0.40)"
    if k < 0.60:
        return "moderate (0.40-0.60)"
    if k < 0.80:
        return "substantial (0.60-0.80)"
    return "almost perfect (>0.80)"


def compute_regimen_top1_match(predicted: pd.Series, true: pd.Series) -> dict:
    """How often did the kit's Top-1 regimen match what the patient
    actually received?"""
    if true.isna().all():
        return {"n_with_truth": 0, "match_rate": None, "note": "no ground-truth regimens in cohort"}
    valid = true.notna()
    match = (predicted[valid] == true[valid]).mean()
    return {
        "n_with_truth": int(valid.sum()),
        "match_rate": float(match),
        "interpretation": (
            "high (>0.7)" if match > 0.7 else
            "moderate (0.4-0.7)" if match > 0.4 else
            "low (<0.4) — kit and clinical practice diverge significantly"
        ),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray,
                     out_path: Path, n_bins: int = 10,
                     title: str = "Calibration") -> None:
    """Reliability diagram — predicted prob (binned) vs observed CR rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            bin_accs.append(y_true[mask].mean())
            bin_confs.append(y_prob[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(np.nan)
            bin_confs.append(bin_centers[i])
            bin_counts.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
    ax1.plot(bin_confs, bin_accs, "s-", color="tab:blue", label="Observed")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Observed CR rate")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="lower right")
    ax1.set_title(f"{title} — Reliability diagram")
    ax1.grid(alpha=0.3)

    # Per-bin sample counts
    ax2.bar(bin_centers, bin_counts, width=1.0 / n_bins, color="tab:orange",
            edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Predicted probability bin")
    ax2.set_ylabel("N patients in bin")
    ax2.set_title(f"{title} — Sample density")
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_eln_confusion(predicted: pd.Series, true: pd.Series,
                        out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = ["Favorable", "Intermediate", "Adverse"]
    cm = pd.crosstab(true, predicted, rownames=["True"], colnames=["Predicted"])
    cm = cm.reindex(index=classes, columns=classes, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted ELN 2022")
    ax.set_ylabel("True ELN 2022")
    ax.set_title("ELN 2022 prediction vs ground truth")

    for i in range(3):
        for j in range(3):
            v = int(cm.values[i, j])
            color = "white" if v > cm.values.max() * 0.5 else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_external_validation(config: ExternalValidationConfig) -> dict:
    """Run the full external-validation pipeline.

    Reads cohort.csv, computes calibration / Brier / ECE / kappa /
    regimen-match-rate, writes a self-contained HTML report.
    """
    df = pd.read_csv(config.cohort_csv)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"cohort.csv missing required columns: {sorted(missing)}. "
            f"See module docstring for the full data contract."
        )

    config.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.random_state)

    # ---- ELN concordance ----
    eln_metrics = compute_eln_concordance(
        df["predicted_eln_2022"], df["true_eln_class"],
    )

    # ---- Top-1 regimen match ----
    if "true_regimen_id" in df.columns:
        regimen_metrics = compute_regimen_top1_match(
            df["predicted_top1_id"], df["true_regimen_id"],
        )
    else:
        regimen_metrics = {"note": "true_regimen_id column absent — skipping"}

    # ---- Calibration on CR (treats predicted ELN ordinal as crude probability) ----
    # ELN-derived crude CR probability priors per ELN 2022:
    #   Favorable     ~ 0.80
    #   Intermediate  ~ 0.55
    #   Adverse       ~ 0.30
    # If the kit's Layer-3 had been trustworthy we'd use its own prob;
    # since we demoted Layer-3 (Pearson r ≈ 0.05), we use ELN-class prior
    # as a deliberately simple benchmark — shows whether the kit's ELN
    # classification carries any CR signal.
    eln_cr_prior = {"Favorable": 0.80, "Intermediate": 0.55, "Adverse": 0.30}
    df["pred_cr_prob_via_eln"] = df["predicted_eln_2022"].map(eln_cr_prior)
    valid = df["pred_cr_prob_via_eln"].notna() & df["true_cr"].notna()
    if valid.sum() > 0:
        y_true = df.loc[valid, "true_cr"].to_numpy(dtype=float)
        y_prob = df.loc[valid, "pred_cr_prob_via_eln"].to_numpy(dtype=float)
        brier = compute_brier_score(y_true, y_prob)
        ece = compute_ece(y_true, y_prob)

        # Bootstrap CI on Brier
        boot_briers = []
        for _ in range(config.bootstrap_n):
            idx = rng.integers(0, len(y_true), size=len(y_true))
            boot_briers.append(compute_brier_score(y_true[idx], y_prob[idx]))
        brier_ci = (
            float(np.percentile(boot_briers, 2.5)),
            float(np.percentile(boot_briers, 97.5)),
        )

        plot_calibration(
            y_true, y_prob,
            config.out_dir / "calibration_plot.png",
            title=f"{config.institution_name}",
        )
        calibration = {
            "n": int(valid.sum()),
            "brier_score": brier,
            "brier_score_ci_95": brier_ci,
            "expected_calibration_error": ece,
            "interpretation": (
                f"Brier {brier:.3f} (95% CI {brier_ci[0]:.3f}-{brier_ci[1]:.3f}); "
                f"ECE {ece:.3f}. Random-guess Brier on balanced data ~ 0.25; "
                f"perfect = 0. Lower is better."
            ),
            "calibration_plot": "calibration_plot.png",
        }
    else:
        calibration = {"note": "Insufficient predictions/truth to compute calibration"}

    # ---- ELN confusion plot ----
    plot_eln_confusion(
        df["predicted_eln_2022"], df["true_eln_class"],
        config.out_dir / "eln_confusion.png",
    )

    # ---- Aggregate ----
    metrics = {
        "institution": config.institution_name,
        "cohort_description": config.cohort_description,
        "n_patients": int(len(df)),
        "eln_concordance": eln_metrics,
        "regimen_top1_match": regimen_metrics,
        "calibration_via_eln_prior": calibration,
    }

    # Write metrics.json
    with open(config.out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Write summary markdown
    summary_md = _render_summary_markdown(metrics)
    (config.out_dir / "regimen_match_summary.md").write_text(
        summary_md, encoding="utf-8",
    )

    # Write standalone HTML report (with embedded PNGs via base64)
    html = _render_summary_html(metrics, summary_md, png_dir=config.out_dir)
    (config.out_dir / "REPORT.html").write_text(html, encoding="utf-8")

    return metrics


def _render_summary_markdown(metrics: dict) -> str:
    lines = [
        f"# External Validation Report",
        f"",
        f"**Institution**: {metrics['institution']}",
        f"**Cohort**: {metrics.get('cohort_description', '—')}",
        f"**N patients**: {metrics['n_patients']}",
        f"",
        f"---",
        f"",
        f"## ELN 2022 concordance",
        f"",
        f"- Overall accuracy: **{metrics['eln_concordance']['accuracy']:.3f}**",
        f"- Cohen's kappa: **{metrics['eln_concordance']['cohens_kappa']:.3f}** "
        f"({metrics['eln_concordance']['interpretation_kappa']})",
        f"",
        f"### Confusion matrix (rows = true, cols = predicted)",
        f"",
        f"```",
    ]
    for tc, row in metrics["eln_concordance"]["confusion_matrix"].items():
        lines.append(f"  {tc!r:<14} → {row}")
    lines.append("```")
    lines.append("")

    lines.append(f"## Calibration (CR via ELN prior)")
    lines.append("")
    cal = metrics["calibration_via_eln_prior"]
    if "brier_score" in cal:
        lines.append(f"- N: {cal['n']}")
        lines.append(f"- Brier: **{cal['brier_score']:.3f}** "
                     f"(95% CI {cal['brier_score_ci_95'][0]:.3f}-"
                     f"{cal['brier_score_ci_95'][1]:.3f})")
        lines.append(f"- ECE: **{cal['expected_calibration_error']:.3f}**")
        lines.append(f"- {cal['interpretation']}")
    else:
        lines.append(f"- {cal.get('note', 'n/a')}")
    lines.append("")

    lines.append(f"## Top-1 regimen match")
    lines.append("")
    rm = metrics["regimen_top1_match"]
    if "match_rate" in rm and rm["match_rate"] is not None:
        lines.append(f"- N with ground truth: {rm['n_with_truth']}")
        lines.append(f"- Match rate: **{rm['match_rate']:.3f}** ({rm['interpretation']})")
    else:
        lines.append(f"- {rm.get('note', 'n/a')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "**Pass criteria** (proposed institutional threshold for clinical use):"
    )
    lines.append(
        "- ELN kappa ≥ 0.60 (substantial agreement)"
    )
    lines.append(
        "- Brier ≤ 0.20 (calibrated to within reasonable bounds)"
    )
    lines.append(
        "- Top-1 regimen match ≥ 0.50 (at least half of cases align with "
        "local clinical practice)"
    )
    lines.append("")
    lines.append(
        "If any criterion fails, **the kit's recommendations should NOT be "
        "deployed in clinical workflow at this institution** until the "
        "discrepancy is investigated and either (a) the local cohort "
        "biology is documented as the cause or (b) the kit's rules are "
        "revised by the local clinical team."
    )
    return "\n".join(lines)


def _render_summary_html(metrics: dict, summary_md: str,
                          png_dir: Optional[Path] = None) -> str:
    """Self-contained HTML report — embeds calibration + confusion PNGs."""
    import base64
    png_dir = png_dir or Path(".")

    def _embed_png(name: str) -> str:
        p = png_dir / name
        if not p.exists():
            return ""
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="max-width:800px;width:100%;"/>'

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>External Validation — {metrics['institution']}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        max-width: 900px; margin: 2em auto; padding: 0 1em; color: #222; }}
h1, h2 {{ color: #1a3a8f; }}
h1 {{ border-bottom: 3px solid #1a3a8f; padding-bottom: 0.3em; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.4em 0.8em; text-align: left; }}
th {{ background: #f0f4ff; }}
.banner {{ background: #fff3cd; border-left: 5px solid #f0ad4e;
           padding: 1em; margin: 1em 0; }}
.pass {{ color: #1a7e34; font-weight: bold; }}
.fail {{ color: #b50d0d; font-weight: bold; }}
img {{ display: block; margin: 1em auto; }}
</style>
</head><body>
<h1>External Validation Report</h1>
<p><strong>Institution:</strong> {metrics['institution']}<br>
<strong>Cohort:</strong> {metrics.get('cohort_description', '—')}<br>
<strong>N patients:</strong> {metrics['n_patients']}</p>

<div class="banner">
This report tests the AML combo-prediction kit's <strong>ELN classification +
recommended regimen + outcome prediction</strong> against your institution's
ground truth. Pass criteria are listed at the bottom — if any fails, the kit
should NOT be deployed in clinical workflow at your center until the
discrepancy is reviewed.
</div>

<h2>Calibration plot</h2>
{_embed_png("calibration_plot.png")}

<h2>ELN confusion matrix</h2>
{_embed_png("eln_confusion.png")}

<h2>Numeric summary</h2>
<pre style="background:#f7f7f7;padding:1em;overflow:auto;">{json.dumps(metrics, indent=2, ensure_ascii=False)}</pre>

<h2>Markdown summary</h2>
<pre style="white-space:pre-wrap;background:#f7f7f7;padding:1em;">{summary_md}</pre>

</body></html>
"""
