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
_DEFAULT_FULL_REF_PATH = Path(
    "data/canonical/full_transcriptome_ref_stats.npz"
)


# Genes we exclude from the "top-N transcriptome scan" because their
# expression variability is driven by factors uninformative for AML biology
# (sex of patient, mitochondrial housekeeping, etc.).
_TRANSCRIPTOME_SCAN_EXCLUSIONS: frozenset[str] = frozenset({
    # Sex-chromosome specific (high variance driven by patient sex)
    "XIST", "TSIX",
    "RPS4Y1", "RPS4Y2", "DDX3Y", "USP9Y", "UTY", "ZFY", "KDM5D",
    "EIF1AY", "NLGN4Y", "TMSB4Y", "TXLNG2P", "PRKY",
    # Mitochondrial (housekeeping, high-variance from library prep, no
    # AML biology interpretation)
    "MT-CO1", "MT-CO2", "MT-CO3", "MT-ND1", "MT-ND2", "MT-ND3",
    "MT-ND4", "MT-ND4L", "MT-ND5", "MT-ND6", "MT-ATP6", "MT-ATP8",
    "MT-CYB", "MT-RNR1", "MT-RNR2", "MT-TL1",
    # Hemoglobin (varies with blood contamination %)
    "HBB", "HBA1", "HBA2", "HBG1", "HBG2", "HBD", "HBE1",
})


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


# ---------------------------------------------------------------------------
# Action 1 — Phenotype signature detection
# ---------------------------------------------------------------------------


def _detect_phenotype_signatures(rows: list[_GeneRow]) -> list[str]:
    """Rule-based detection of recognizable AML transcriptional phenotypes.

    Returns a list of short natural-language signature strings ordered by
    clinical salience. Each rule uses both z-score and DNA status when
    available.
    """
    by_gene: dict[str, _GeneRow] = {r.gene: r for r in rows}

    def z(g: str) -> float:
        row = by_gene.get(g)
        if row is None or row.z_score is None:
            return 0.0
        return row.z_score

    def mut(g: str) -> bool:
        row = by_gene.get(g)
        return bool(row and row.dna_status == "✓ mutated")

    sigs: list[str] = []

    # TP53 / chr-17 deleted adverse
    if mut("TP53") and z("TP53") <= -1.5:
        sigs.append("**TP53 突变 + 表达缺失** (z={:+.2f}) — 高度提示 bi-allelic 丢失 "
                    "(del(17p) 或第二击突变), WHO 2022 multi-hit TP53 亚型".format(z("TP53")))
    elif z("TP53") <= -2.0:
        sigs.append("**TP53 表达极低** (z={:+.2f}) — 复核 del(17p) 或 bi-allelic "
                    "状态, 影响 7+3 诱导反应率".format(z("TP53")))

    # EVI1/MECOM activation (classic inv(3) signature)
    if z("MECOM") >= 2.0 and not mut("MECOM"):
        sigs.append("**MECOM/EVI1 转录激活** (z={:+.2f}) — 即使核型未报告 inv(3)/t(3;3), "
                    "强烈建议追加 **FISH 检测** 确认隐性易位".format(z("MECOM")))

    # HOX program — NPM1-mut or KMT2A-r signature
    if z("HOXA9") >= 1.5 and z("MEIS1") >= 1.0:
        ctx = ("已与 NPM1-mut 一致" if mut("NPM1")
               else "建议查 KMT2A-FISH 或 NPM1 重测")
        sigs.append("**HOXA9/MEIS1 高表达 posterior HOX 程序** (HOXA9 z={:+.2f}, "
                    "MEIS1 z={:+.2f}) — 提示 NPM1-mut 或 KMT2A-r 亚型 ({})"
                    .format(z("HOXA9"), z("MEIS1"), ctx))

    # FLT3 double-evidence
    if mut("FLT3") and z("FLT3") >= 1.5:
        sigs.append("**FLT3 双证据支持** (DNA 突变 + 表达 z={:+.2f}) — "
                    "FLT3i (Midostaurin/Gilteritinib/Quizartinib) 机制合理性"
                    "强, 疗效预期良好".format(z("FLT3")))

    # BCL2 / Venetoclax support
    if z("BCL2") >= 1.5:
        if z("MCL1") >= 1.5:
            sigs.append("**BCL2 与 MCL1 同时升高** (BCL2 z={:+.2f}, MCL1 z={:+.2f}) — "
                        "Venetoclax 可能面临 **MCL1-mediated 抵抗**, 考虑"
                        "三联 (Ven+Aza+Flotetuzumab) 或 MCL1-inhibitor 临床试验"
                        .format(z("BCL2"), z("MCL1")))
        else:
            sigs.append("**BCL2 过表达** (z={:+.2f}) — Venetoclax 机制合理性强"
                        .format(z("BCL2")))
    elif z("MCL1") >= 1.5:
        sigs.append("**MCL1 单独升高** (z={:+.2f}) — Venetoclax 抵抗风险, "
                    "考虑 MCL1-inhibitor 临床试验".format(z("MCL1")))

    # Adverse signature: BAALC + MN1
    if z("BAALC") >= 1.5:
        sigs.append("**BAALC 高表达** (z={:+.2f}) — 独立 adverse 预后因子, "
                    "强化诱导耐药风险升高".format(z("BAALC")))

    if z("MN1") >= 1.5:
        sigs.append("**MN1 高表达** (z={:+.2f}) — inv(16)-like 基因表达签名"
                    .format(z("MN1")))

    # CD33 for Mylotarg
    if z("CD33") >= 1.5:
        sigs.append("**CD33 高表达** (z={:+.2f}) — Gemtuzumab Ozogamicin "
                    "(Mylotarg) 适应症支持 (对 CD33+ blasts 有效)".format(z("CD33")))

    # IL3RA (CD123) for Tagraxofusp
    if z("IL3RA") >= 1.5:
        sigs.append("**IL3RA (CD123) 高表达** (z={:+.2f}) — Tagraxofusp 或 "
                    "CD123-CAR-T 相关 (BPDCN 除外, AML 中研究阶段)".format(z("IL3RA")))

    # RUNX1 loss hint
    if z("RUNX1") <= -1.5 and not mut("RUNX1"):
        sigs.append("**RUNX1 表达低下** (z={:+.2f}) — 可能 RUNX1 缺失 / "
                    "germline RUNX1 综合症 (FPD), 建议 germline testing"
                    .format(z("RUNX1")))

    return sigs


def _build_highlights_paragraph(
    rows: list[_GeneRow], meta: dict,
) -> str:
    """Natural-language summary of transcriptional phenotype for report header."""
    if meta.get("n_genes_available", 0) == 0:
        return ""
    sigs = _detect_phenotype_signatures(rows)
    n_avail = meta.get("n_genes_available", 0)
    n_high = meta.get("n_outliers_high", 0)
    n_low = meta.get("n_outliers_low", 0)

    if not sigs:
        return (f"**RNA-Seq 高亮**: 本患者 25+8 基因 panel "
                f"(n={n_avail} 可检) 中未发现显著转录离群 "
                f"(所有 |z| < 1.5)。提示转录组呈典型 AML 背景, "
                f"无需追加 FISH / 表达驱动靶点。\n")

    head = (f"**RNA-Seq 高亮**: {n_high} 个基因高表达离群 (z≥+1.5), "
            f"{n_low} 个低表达离群 (z≤-1.5), 归纳出以下转录表型签名:\n\n")
    body = "\n".join(f"- {s}" for s in sigs)
    return head + body + "\n"


# ---------------------------------------------------------------------------
# Action 3 — Top-N full-transcriptome outlier scan
# ---------------------------------------------------------------------------


def _load_full_transcriptome_stats(
    path: Path | str | None = None,
) -> Optional[dict]:
    """Load full-transcriptome stats .npz. Returns None if file missing
    (graceful degradation — Action 3 is optional)."""
    if path is not None:
        p = Path(path)
    else:
        candidates = [Path.cwd() / _DEFAULT_FULL_REF_PATH]
        pkg_repo_root = Path(__file__).resolve().parents[3]
        candidates.append(pkg_repo_root / _DEFAULT_FULL_REF_PATH)
        p = next((c for c in candidates if c.exists()), None)
        if p is None:
            return None
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return {
        "genes": data["genes"],
        "means": data["means"],
        "stds": data["stds"],
        "medians": data["medians"],
        "meta": meta,
    }


def find_transcriptome_outliers(
    rna_expression: Optional[pd.Series],
    exclude_genes: set[str],
    top_n: int = 10,
    min_abs_z: float = 3.0,
    full_ref: Optional[dict] = None,
) -> tuple[list[dict], dict]:
    """Scan the full transcriptome for extreme outliers NOT in the curated
    25+8 gene list.

    Args:
      rna_expression: patient expression Series (full transcriptome).
      exclude_genes: set of genes already covered by the curated table (so
        they don't appear again here).
      top_n: max number of extra outliers to return.
      min_abs_z: minimum |z| to count as an outlier (default 3σ, very
        conservative to avoid noise).
      full_ref: optional pre-loaded full-transcriptome stats.

    Returns:
      (rows, meta) where rows is [{gene, z_score, direction, ref_mean,
      ref_std, patient_value}, ...] sorted by |z| descending.
    """
    if rna_expression is None or len(rna_expression) == 0:
        return [], {"available": False,
                    "reason": "no RNA-Seq expression provided"}

    if full_ref is None:
        full_ref = _load_full_transcriptome_stats()
    if full_ref is None:
        return [], {"available": False,
                    "reason": "full-transcriptome stats file not found"}

    # Align patient to reference scale (re-use helper)
    # Use the full gene list of the reference for scale detection
    ref_genes = full_ref["genes"]
    ref_mean_arr = full_ref["means"]
    ref_std_arr = full_ref["stds"]

    # Build patient-gene → value on ref scale
    patient_vals: dict[str, float] = {}
    # Scale detection on patient's overall distribution
    patient_arr = rna_expression.astype(float).values
    flavor = _infer_scale_flavor(patient_arr)
    if flavor == "raw_counts":
        patient_vals_clean = np.log2(rna_expression.astype(float).clip(lower=0) + 1.0)
        scale_note = "auto-log2-transformed"
    else:
        patient_vals_clean = rna_expression.astype(float)
        scale_note = f"used as-is ({flavor})"

    # Compute z-scores for genes present in BOTH patient and reference
    gene_to_ref_idx = {g: i for i, g in enumerate(ref_genes)}
    outlier_candidates: list[dict] = []

    exclude_upper = {g.upper() for g in exclude_genes} | _TRANSCRIPTOME_SCAN_EXCLUSIONS

    for g in patient_vals_clean.index:
        g_up = g.upper()
        if g_up in exclude_upper:
            continue
        idx = gene_to_ref_idx.get(g)
        if idx is None:
            continue
        ref_mu = float(ref_mean_arr[idx])
        ref_sigma = float(ref_std_arr[idx])
        if ref_sigma <= 0:
            continue
        val = float(patient_vals_clean.loc[g])
        z = (val - ref_mu) / ref_sigma
        if abs(z) >= min_abs_z:
            outlier_candidates.append({
                "gene": g, "z_score": z, "direction": _classify_z(z),
                "patient_value": val, "ref_mean": ref_mu, "ref_std": ref_sigma,
            })

    # Sort by |z| descending, take top-N
    outlier_candidates.sort(key=lambda r: -abs(r["z_score"]))
    rows = outlier_candidates[:top_n]

    meta = {
        "available": True,
        "scale_note": scale_note,
        "n_candidates_total": len(outlier_candidates),
        "n_returned": len(rows),
        "min_abs_z_threshold": min_abs_z,
        "excluded_categories": ("sex-specific, mitochondrial, hemoglobin, "
                                 "+ curated 25+8 gene panel"),
    }
    return rows, meta


def build_transcriptome_scan_markdown(
    rna_expression: Optional[pd.Series],
    exclude_genes: set[str],
    top_n: int = 10,
    min_abs_z: float = 3.0,
) -> str:
    """Markdown block for the Action-3 top-N transcriptome scan."""
    rows, meta = find_transcriptome_outliers(
        rna_expression, exclude_genes=exclude_genes, top_n=top_n,
        min_abs_z=min_abs_z,
    )
    if not meta.get("available"):
        return (f"*全转录组补充扫描: 未启用 "
                f"({meta.get('reason', 'unknown')})。* \n")

    if not rows:
        return (f"*全转录组扫描: **未发现核心 panel 之外的极端离群** "
                f"(|z| ≥ {min_abs_z}σ). 排除了 sex / mitochondrial / "
                f"hemoglobin 等非临床变异基因。*\n")

    lines = [
        f"*本扫描从 BeatAML 2.0 全转录组 (~{meta.get('n_candidates_total', '?')} "
        f"个候选基因中)挑选出 |z| ≥ {meta.get('min_abs_z_threshold', min_abs_z)}σ "
        f"的 top-{meta.get('n_returned', 0)} 离群基因 (已排除: "
        f"{meta['excluded_categories']}). 这些基因**不在临床策展的 25+8 "
        f"panel 里**, 可能提示尚未被常规检测捕获的生物学信号, 需要临床人员"
        f"结合基因功能自行解读。*",
        "",
        "| # | 基因 | z-score | 方向 | 患者值 | BeatAML 均值 ± std |",
        "|---|------|--------:|:----:|-------:|---------------------|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | **{r['gene']}** | {r['z_score']:+.2f} | "
            f"{r['direction']} | {r['patient_value']:.2f} | "
            f"{r['ref_mean']:+.2f} ± {r['ref_std']:.2f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action 2 — Split table: outliers vs collapsed-normal
# ---------------------------------------------------------------------------


def build_rnaseq_outlier_markdown_v2(
    rna_expression: Optional[pd.Series],
    mutated_genes: set[str],
    ref_stats: Optional[dict] = None,
    outlier_threshold: float = 0.75,
) -> str:
    """Build the new (v2) Markdown for Section 3.3 — highlights paragraph
    + outliers-first table + collapsed normal + optional transcriptome scan.

    Sections produced:
      A. Highlights paragraph (phenotype signatures)
      B. Outlier table — only rows with |z| ≥ outlier_threshold OR
         mutated genes OR hint-only genes with z flagged
      C. Normal-range collapsed line listing remaining genes
      D. Full-transcriptome top-N scan (if stats file available)

    `outlier_threshold=0.75` catches mild outliers (≥0.75σ) so we don't lose
    subtle findings; strong outliers (≥1.5σ) are bolded.
    """
    rows, meta = compute_expression_outliers(
        rna_expression, mutated_genes, ref_stats,
    )

    if meta["n_genes_available"] == 0 and not any(r.available for r in rows):
        return (
            "> **RNA-Seq 表达谱数据未提供** —— 本节为空。"
            "如 lab 出的 RNA-Seq 表达矩阵可用, 请通过 "
            "`KitInput.rna_expression_full` 传入后重跑报告。\n"
        )

    # --- A. Highlights paragraph ---
    parts = [_build_highlights_paragraph(rows, meta)]

    # --- B. Outlier sub-table ---
    outlier_rows = [r for r in rows
                     if r.available and (
                         abs(r.z_score) >= outlier_threshold or
                         r.dna_status == "✓ mutated"
                     )]

    parts.append(f"#### A. 核心 panel 离群基因 + 全部 DNA 突变 "
                 f"({len(outlier_rows)} 行)")
    parts.append("")
    if not outlier_rows:
        parts.append("*（无）*\n")
    else:
        parts.append(
            "| Tier 组 | 基因 | DNA 状态 | z-score | 方向 | 临床提示 |"
        )
        parts.append(
            "|---------|------|----------|--------:|:----:|----------|"
        )
        prev_group = None
        for r in outlier_rows:
            group = r.tier_group if r.tier_group != prev_group else ""
            prev_group = r.tier_group
            gene = (f"**{r.gene}**" if abs(r.z_score) >= 1.5 else r.gene)
            z = f"{r.z_score:+.2f}"
            note = (r.note or "").replace("|", "\\|")
            parts.append(f"| {group} | {gene} | {r.dna_status} | "
                          f"{z} | {r.direction} | {note} |")
        parts.append("")

    # --- C. Collapsed-normal paragraph ---
    normal_rows = [r for r in rows
                    if r.available and
                    abs(r.z_score) < outlier_threshold and
                    r.dna_status != "✓ mutated"]
    unavailable_rows = [r for r in rows if not r.available]

    parts.append(f"#### B. 核心 panel 正常范围基因 ({len(normal_rows)} 个)")
    parts.append("")
    if normal_rows:
        gene_list = ", ".join(r.gene for r in normal_rows)
        parts.append(f"以下基因表达在 BeatAML 正常范围内 "
                     f"(|z| < {outlier_threshold}σ), DNA 状态: "
                     f"全部野生型 / hint-only: **{gene_list}**\n")

    if unavailable_rows:
        missing_list = ", ".join(r.gene for r in unavailable_rows)
        parts.append(f"*未在输入 RNA-Seq 中覆盖的基因 "
                     f"({len(unavailable_rows)}): {missing_list}*\n")

    # --- D. Full-transcriptome top-N scan ---
    excluded = {r.gene for r in rows}
    transcriptome_scan_md = build_transcriptome_scan_markdown(
        rna_expression, exclude_genes=excluded, top_n=10, min_abs_z=3.0,
    )
    parts.append(f"#### C. 全转录组扩展扫描 (core panel 之外)")
    parts.append("")
    parts.append(transcriptome_scan_md)

    # Legend footer
    parts.append("")
    parts.append(
        f"*图例*: ↑↑↑ z≥+2.5 · ↑↑ z≥+1.5 · ↑ z≥+0.75 · · 正常 · "
        f"↓ z≤-0.75 · ↓↓ z≤-1.5 · ↓↓↓ z≤-2.5. "
        f"参考分布: {meta.get('ref_cohort', 'BeatAML 2.0')} "
        f"(n={meta.get('ref_n_samples', '?')}). "
        f"[scale: {meta.get('scale_note', '')}]"
    )
    return "\n".join(parts)


# Public exports
__all__ = [
    "TIER_GROUPS_FOR_EXPR",
    "EXPRESSION_HINTS",
    "compute_expression_outliers",
    "build_rnaseq_outlier_markdown",
    "build_rnaseq_outlier_markdown_v2",
    "find_transcriptome_outliers",
    "build_transcriptome_scan_markdown",
    "_detect_phenotype_signatures",
    "_build_highlights_paragraph",
]
