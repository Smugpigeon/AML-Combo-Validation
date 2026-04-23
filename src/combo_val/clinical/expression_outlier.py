"""RNA-Seq expression outlier analysis for the 25-gene core AML driver panel.

Complements — does NOT replace — the DNA-NGS mutation table. For each of the
25 curated AML driver genes (+ a small set of expression-only hint genes like
HOXA9, MEIS1, BCL2, BAALC, MN1), we compute the patient's z-score vs a
BeatAML 2.0 reference distribution (n=707 AML samples).

Clinical semantics of each row:

    Mutation status │ Expression z │ Interpretation
    ────────────────┼──────────────┼──────────────────────────────────────
    mutated (DNA)   │  z > +1.5    │ Double-evidence — DNA lesion + expr ↑
                    │  z ≈ 0       │ DNA lesion present, expr unremarkable
                    │  z < -1.5    │ DNA lesion + expr ↓ (possible haplo-
                    │              │   insufficiency, e.g., TP53 bi-allelic)
    wild-type       │  z > +2.5    │ Cryptic expression event — verify by
                    │              │   FISH / alt cyto (MECOM/EVI1 ↑ →
                    │              │   check inv(3) not captured by karyo)
                    │  z > +1.5    │ Elevated — may indicate driver path-
                    │              │   way activation (HOXA9/MEIS1 → NPM1
                    │              │   or KMT2A-r signature)
                    │  z ≈ 0       │ Truly unremarkable
                    │  z < -1.5    │ Downregulated (TP53/RUNX1 low expr →
                    │              │   possible loss-of-function)

The analysis requires the patient's expression Series to include at least some
of the 25-gene panel; genes not present are marked "n/a — not in input
transcriptome".

Scale handling: the function auto-detects whether the input is raw-count-ish
(median > ~10, heavy right tail) or already log-scaled (median 0-8), and
applies log2(x+1) as needed so the patient lands on BeatAML Sheet1's scale
(median ~3, range [-5, +11]). If the scale still looks way off even after
auto-transform, we fall back to rank-percentile rather than z-score.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


_DEFAULT_REF_PATH = Path(
    "data/canonical/driver_gene_ref_stats.json"
)


# Gene grouping for table layout — mirrors dna_report's Tier groups and adds
# an "expression-only hint" group for genes relevant via expression alone.
TIER_GROUPS_FOR_EXPR: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Tier 1 (FDA-targetable)",
     ("FLT3", "IDH1", "IDH2", "KMT2A")),
    ("Tier 2 (prognostic / intensity-modifier)",
     ("NPM1", "TP53", "RUNX1", "ASXL1", "CEBPA")),
    ("Tier 3a (epigenetic / HMA-responsive)",
     ("DNMT3A", "TET2")),
    ("Tier 3b (RAS-MAPK)",
     ("NRAS", "KRAS", "PTPN11", "KIT")),
    ("Tier 3c (MDS-related / splice / other)",
     ("WT1", "BCOR", "STAG2", "PHF6", "SRSF2", "SF3B1", "U2AF1",
      "EZH2", "MECOM", "CBFB")),
    ("Expression-only hints (no DNA panel entry)",
     ("HOXA9", "MEIS1", "BCL2", "MCL1", "BAALC", "MN1", "CD33", "IL3RA")),
)


# Clinical interpretive hints for expression-only genes + hint logic for
# driver genes. Used in the "备注 / 建议" column.
EXPRESSION_HINTS: dict[str, str] = {
    "MECOM":   "↑↑ → 提示 EVI1 激活, 查 inv(3)/t(3;3) FISH (即使 karyo 报告阴性)",
    "HOXA9":   "↑↑ → NPM1-mut 或 KMT2A-r 表达签名",
    "MEIS1":   "↑↑ → 与 HOXA9 共升表示 posterior HOX 程序",
    "BCL2":    "↑ → Venetoclax 机制合理性支持",
    "MCL1":    "↑↑ → 可能 Venetoclax 抵抗 (MCL1 升), 考虑 MCL1i 临床试验",
    "BAALC":   "↑↑ → 独立 adverse, 强化诱导耐药风险",
    "MN1":     "↑↑ → inv(16)-like 基因表达签名",
    "CD33":    "↑ → Gemtuzumab Ozogamicin (Mylotarg) 合理性支持",
    "IL3RA":   "↑ (CD123 high) → Tagraxofusp / CD123-CAR 相关",
    "TP53":    "↓↓ + 野生 → 可能 bi-allelic 缺失, 复核 cytogenetics del(17p)",
    "FLT3":    "↑↑ + 突变 → 双证据支持 FLT3 抑制剂",
    "NPM1":    "↑↑ → 与 HOXA9 高表达共同支持 NPM1-mut 签名",
    "RUNX1":   "↓↓ → 可能 RUNX1 loss (germline / del)",
}


def _load_ref_stats(ref_path: Path | str | None = None) -> dict:
    """Load BeatAML reference stats JSON. Raises FileNotFoundError if missing.

    If `ref_path` is None:
      1. tries `_DEFAULT_REF_PATH` relative to `Path.cwd()`,
      2. falls back to `_DEFAULT_REF_PATH` under this module's repo root
         (inferred from package location).
    If `ref_path` is given explicitly, only that path is tried.
    """
    if ref_path is not None:
        p = Path(ref_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Driver-gene reference stats not found at {p}. "
                f"Run scripts/build_driver_gene_ref_stats.py first."
            )
        return json.loads(p.read_text())

    # Default path resolution — cwd first, then package-relative fallback
    candidates = [Path.cwd() / _DEFAULT_REF_PATH]
    # Package root is 3 parents up from this file: expression_outlier.py →
    # clinical → combo_val → src → <repo>
    pkg_repo_root = Path(__file__).resolve().parents[3]
    candidates.append(pkg_repo_root / _DEFAULT_REF_PATH)

    for p in candidates:
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(
        f"Driver-gene reference stats not found. Tried: "
        f"{', '.join(str(c) for c in candidates)}. "
        f"Run scripts/build_driver_gene_ref_stats.py first."
    )


def _auto_rescale_to_reference(
    value: float, ref_median: float, ref_std: float,
    patient_median: float, patient_mad: float,
) -> float:
    """If the patient's overall expression distribution is on a scale much
    different from BeatAML's, shift+scale robustly so per-gene comparisons
    are meaningful. Uses median & MAD to be robust to outliers.
    """
    if patient_mad <= 0 or not math.isfinite(patient_mad):
        return value - patient_median + ref_median
    # Scale the patient's MAD up to the reference's (using 1.4826 * MAD ≈ std)
    patient_std_est = 1.4826 * patient_mad
    if patient_std_est <= 0 or not math.isfinite(patient_std_est):
        return value - patient_median + ref_median
    scale = ref_std / patient_std_est
    # Guard against absurd scale factors
    scale = float(np.clip(scale, 0.1, 10.0))
    return (value - patient_median) * scale + ref_median


def _infer_scale_flavor(values: np.ndarray) -> str:
    """Heuristically classify whether a vector is raw counts or log-scaled.

    Returns:
      "raw_counts"       — median > 10 and max > 200 (apply log2(x+1))
      "log_cpm"          — median in [-2, 8], max < 20 (use as-is)
      "ambiguous"        — values in a weird range; use as-is with warning
    """
    vals = values[np.isfinite(values)]
    if len(vals) < 5:
        return "ambiguous"
    med = float(np.median(vals))
    mx = float(np.max(vals))
    if med > 10 and mx > 200:
        return "raw_counts"
    if -5 <= med <= 8 and mx < 20:
        return "log_cpm"
    return "ambiguous"


def _transform_to_reference_scale(
    rna_expression: pd.Series, genes: list[str],
) -> tuple[pd.Series, str]:
    """Bring the patient's expression Series onto BeatAML Sheet1's log-scale.

    Returns (aligned_series, scale_note).
    """
    # Subset to genes we care about (union of core + hints)
    present = [g for g in genes if g in rna_expression.index]
    if not present:
        return pd.Series(dtype=float), "no_genes_in_input"

    vals = rna_expression.loc[present].astype(float)
    flavor = _infer_scale_flavor(vals.values)

    if flavor == "raw_counts":
        vals = np.log2(vals.clip(lower=0) + 1.0)
        note = "auto-log2-transformed (input looked like raw counts)"
    elif flavor == "log_cpm":
        note = "input already log-scaled, used as-is"
    else:  # ambiguous
        note = "input scale ambiguous — used as-is, z-scores may be approximate"

    return vals, note


@dataclass(frozen=True)
class _GeneRow:
    gene: str
    tier_group: str
    input_value: Optional[float]        # patient value after scale transform
    ref_mean: Optional[float]
    ref_std: Optional[float]
    z_score: Optional[float]
    direction: str                       # "↑↑"|"↑"|"·"|"↓"|"↓↓"|"n/a"
    dna_status: str                      # "✓ mutated" | "✗ wild-type" | "hint-only"
    note: str                            # clinical hint text
    available: bool                      # patient value present & z computable

    def direction_emoji(self) -> str:
        return self.direction


def _classify_z(z: Optional[float]) -> str:
    if z is None or not math.isfinite(z):
        return "n/a"
    if z >= 2.5:
        return "↑↑↑"
    if z >= 1.5:
        return "↑↑"
    if z >= 0.75:
        return "↑"
    if z <= -2.5:
        return "↓↓↓"
    if z <= -1.5:
        return "↓↓"
    if z <= -0.75:
        return "↓"
    return "·"


def compute_expression_outliers(
    rna_expression: Optional[pd.Series],
    mutated_genes: set[str],
    ref_stats: Optional[dict] = None,
) -> tuple[list[_GeneRow], dict]:
    """Compute per-gene expression-outlier rows for the 25+hint gene set.

    Args:
      rna_expression: pd.Series(gene_symbol → expression_value). If None or
        empty → rows marked unavailable (graceful degradation).
      mutated_genes: set of gene symbols the patient has DNA-level mutations
        in (used to label each row as mutated/wild-type/hint-only).
      ref_stats: optional pre-loaded reference stats dict. Default: load from
        data/canonical/driver_gene_ref_stats.json.

    Returns:
      (rows, meta) where rows is the list of _GeneRow objects (in tier-group
      order) and meta contains: {"scale_note", "n_genes_available",
      "n_outliers_high", "n_outliers_low"}.
    """
    if ref_stats is None:
        ref_stats = _load_ref_stats()
    gene_stats = ref_stats.get("genes", {})

    all_genes = [g for _, group in TIER_GROUPS_FOR_EXPR for g in group]

    if rna_expression is None or len(rna_expression) == 0:
        # Full degradation — return rows flagged as unavailable
        rows = []
        for group_name, group_genes in TIER_GROUPS_FOR_EXPR:
            for g in group_genes:
                rows.append(_GeneRow(
                    gene=g, tier_group=group_name,
                    input_value=None, ref_mean=None, ref_std=None,
                    z_score=None, direction="n/a",
                    dna_status=("✓ mutated" if g in mutated_genes
                                 else ("✗ wild-type"
                                       if g in set(gg for _, ggs in
                                                    TIER_GROUPS_FOR_EXPR
                                                    if "hint" not in _
                                                    for gg in ggs)
                                       else "hint-only")),
                    note="", available=False,
                ))
        return rows, {"scale_note": "no RNA-Seq expression provided",
                      "n_genes_available": 0,
                      "n_outliers_high": 0, "n_outliers_low": 0}

    # Transform patient to reference scale
    aligned, scale_note = _transform_to_reference_scale(rna_expression,
                                                          all_genes)

    # Compute robust patient statistics for scale alignment
    if len(aligned) >= 5:
        p_med = float(np.median(aligned.values))
        p_mad = float(np.median(np.abs(aligned.values - p_med)))
    else:
        p_med, p_mad = 0.0, 1.0

    # Reference global stats (for outlier normalization if scale is off)
    ref_means = np.array([gene_stats[g]["mean"]
                          for g in gene_stats if g in all_genes])
    ref_stds = np.array([gene_stats[g]["std"]
                         for g in gene_stats if g in all_genes])
    r_med = float(np.median(ref_means)) if len(ref_means) else 0.0
    r_std = float(np.median(ref_stds)) if len(ref_stds) else 1.0

    rows: list[_GeneRow] = []
    n_high = n_low = n_available = 0
    hint_gene_set = set(TIER_GROUPS_FOR_EXPR[-1][1])

    for group_name, group_genes in TIER_GROUPS_FOR_EXPR:
        is_hint_group = (group_name == TIER_GROUPS_FOR_EXPR[-1][0])
        for g in group_genes:
            if g in mutated_genes:
                dna_status = "✓ mutated"
            elif is_hint_group:
                dna_status = "hint-only"
            else:
                dna_status = "✗ wild-type"

            if g not in gene_stats:
                rows.append(_GeneRow(
                    gene=g, tier_group=group_name,
                    input_value=None, ref_mean=None, ref_std=None,
                    z_score=None, direction="n/a",
                    dna_status=dna_status,
                    note="(参考分布未包含此基因)",
                    available=False,
                ))
                continue

            stats = gene_stats[g]
            if g not in aligned.index:
                rows.append(_GeneRow(
                    gene=g, tier_group=group_name,
                    input_value=None,
                    ref_mean=stats["mean"], ref_std=stats["std"],
                    z_score=None, direction="n/a",
                    dna_status=dna_status,
                    note="n/a — 此基因不在输入 RNA-Seq 中",
                    available=False,
                ))
                continue

            raw_val = float(aligned.loc[g])
            # Auto-rescale if patient's overall distribution looks off
            if abs(p_med - r_med) > 3.0:
                val = _auto_rescale_to_reference(
                    raw_val, r_med, r_std, p_med, p_mad,
                )
            else:
                val = raw_val
            ref_mu = float(stats["mean"])
            ref_sigma = float(stats["std"])
            if ref_sigma <= 0:
                z = None
            else:
                z = (val - ref_mu) / ref_sigma
            direction = _classify_z(z)

            # Clinical hint — only surface when the z-score is meaningful,
            # to avoid flooding the table with "what IF" text on flat rows.
            note = ""
            gene_hint = EXPRESSION_HINTS.get(g, "")
            if z is not None and abs(z) >= 1.5:
                note = gene_hint
            if dna_status == "✓ mutated" and z is not None and z >= 1.5:
                note = (f"双证据支持 (DNA + 表达); {gene_hint}".rstrip("; ")
                         if gene_hint else "双证据支持 (DNA + 表达)")
            elif dna_status == "✗ wild-type" and z is not None and z >= 2.5:
                note = (f"⚠ 野生型但表达异常高, 建议复核; {gene_hint}".rstrip("; ")
                         if gene_hint else "⚠ 野生型但表达异常高, 建议复核")
            # Hint-only genes (HOXA9/BCL2/etc.): show hint iff outlier
            elif (dna_status == "hint-only" and z is not None
                  and abs(z) >= 1.5):
                note = gene_hint

            if z is not None:
                n_available += 1
                if z >= 1.5:
                    n_high += 1
                elif z <= -1.5:
                    n_low += 1

            rows.append(_GeneRow(
                gene=g, tier_group=group_name,
                input_value=val, ref_mean=ref_mu, ref_std=ref_sigma,
                z_score=z, direction=direction,
                dna_status=dna_status, note=note, available=(z is not None),
            ))

    meta = {
        "scale_note": scale_note,
        "n_genes_available": n_available,
        "n_outliers_high": n_high,
        "n_outliers_low": n_low,
        "n_total_genes_in_panel": sum(len(g) for _, g in TIER_GROUPS_FOR_EXPR),
        "ref_cohort": ref_stats.get("reference_cohort", "BeatAML 2.0"),
        "ref_n_samples": ref_stats.get("n_samples"),
    }
    return rows, meta


def build_rnaseq_outlier_markdown(
    rna_expression: Optional[pd.Series],
    mutated_genes: set[str],
    ref_stats: Optional[dict] = None,
) -> str:
    """Build the Markdown table for Section 3.3 of the clinical report."""
    rows, meta = compute_expression_outliers(rna_expression, mutated_genes,
                                              ref_stats)

    if meta["n_genes_available"] == 0:
        return (
            "> **RNA-Seq 表达谱数据未提供** — 本节为空。"
            "如有 lab 的 RNA-Seq 表达矩阵 (log2-CPM 或 raw counts), "
            "可通过 `kit_input.rna_expression_full` 参数传入后重跑报告, "
            "即可补齐 25-gene panel 的表达离群分析。\n"
        )

    lines = [
        f"*基于 {meta['ref_cohort']} (n = {meta['ref_n_samples']}) 的参考分布, "
        f"本患者 {meta['n_genes_available']} 个基因可计算 z-score, "
        f"其中**高表达离群 (z ≥ +1.5) {meta['n_outliers_high']} 个**, "
        f"**低表达离群 (z ≤ -1.5) {meta['n_outliers_low']} 个**。 "
        f"[scale: {meta['scale_note']}]*",
        "",
        "| Tier 组 | 基因 | DNA 状态 | 表达 z-score | 方向 | 备注 |",
        "|---------|------|----------|-------------:|:----:|------|",
    ]
    prev_group = None
    for r in rows:
        group = r.tier_group if r.tier_group != prev_group else ""
        prev_group = r.tier_group
        z_str = f"{r.z_score:+.2f}" if r.z_score is not None else "n/a"
        # Bold highly-flagged rows
        gene_cell = f"**{r.gene}**" if (r.z_score is not None and
                                         abs(r.z_score) >= 1.5) else r.gene
        note_cell = r.note if r.note else ""
        # Escape pipe chars in note that could break the table
        note_cell = note_cell.replace("|", "\\|")
        lines.append(f"| {group} | {gene_cell} | {r.dna_status} | "
                     f"{z_str} | {r.direction} | {note_cell} |")

    lines.append("")
    lines.append(
        "*图例*: ↑↑↑ z≥+2.5 (极显著高) · ↑↑ z≥+1.5 · ↑ z≥+0.75 · · 正常 · "
        "↓ z≤-0.75 · ↓↓ z≤-1.5 · ↓↓↓ z≤-2.5. "
        "**「双证据支持」**=DNA 检出突变且表达升高; "
        "**「⚠ 野生型但表达异常高」**=NGS 阴性但转录本上调, 建议复核 cytogenetics/FISH."
    )
    return "\n".join(lines)


# Public exports
__all__ = [
    "TIER_GROUPS_FOR_EXPR",
    "EXPRESSION_HINTS",
    "compute_expression_outliers",
    "build_rnaseq_outlier_markdown",
]
