"""Clinical-grade per-patient report generator — Markdown + PDF.

Output is a narrative-first clinical report modeled on Foundation Medicine /
Caris molecular tumor-board report conventions, intended for MDT review
without requiring the clinician to learn the kit's internal data structures.

Sections:
  1. Executive summary (3-4 sentences the MDT reads aloud)
  2. Patient demographics and specimen QC
  3. Molecular profile narrative (mutations + fusions + cytogenetics in prose)
  4. ELN 2017 risk stratification with rationale
  5. Treatment recommendations (top 3, paragraph each with trial PMID + CR/OS)
  6. Model-based combination prediction (optional)
  7. Clonal biology rationale
  8. Confidence + limitations + reviewer checklist
  9. Methodology + key references

PDF rendering via pandoc (already installed on user's machine).
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

from combo_val.clinical.dna_report import CORE_DRIVER_GENES
from combo_val.clinical.expression_outlier import (
    build_rnaseq_outlier_markdown,
    build_rnaseq_outlier_markdown_v2,
)
from combo_val.clinical.kit_schema import KitInput, KitOutput, MutationCall


def _weasyprint_env() -> dict[str, str]:
    """Return an env with DYLD_FALLBACK_LIBRARY_PATH patched for Homebrew dylibs.

    On macOS (especially Apple Silicon), WeasyPrint needs pango/cairo libraries
    that Homebrew installs under /opt/homebrew/lib, but Anaconda's Python
    doesn't search there by default. Adding this path lets weasyprint find
    libpango/libcairo without root privileges.
    """
    env = os.environ.copy()
    homebrew_lib = "/opt/homebrew/lib"
    if Path(homebrew_lib).exists():
        existing = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = (
            f"{homebrew_lib}:{existing}" if existing else homebrew_lib
        )
    return env


# ---------------------------------------------------------------------------
# Prose generators (narrative building blocks)
# ---------------------------------------------------------------------------


_GENE_PROSE: dict[str, str] = {
    "FLT3": "**FLT3** 是 AML 最常见的可靶向驱动基因（发生率约 25-30%）。FLT3-ITD (内部串联重复) 与 FLT3-TKD (tyrosine kinase domain point mutation) 临床意义不同：ITD 使 FLT3 受体持续激活，带来更差预后；TKD 相对 ITD 预后更好。FDA 批准的 FLT3 抑制剂包括 Midostaurin (多激酶，一线与 7+3 联用)、Quizartinib (ITD 专选，与 7+3 联用)、Gilteritinib (ITD+TKD 均活性，R/R 单药或联合 Ven)。",
    "NPM1": "**NPM1** 突变出现在约 30% AML 患者，是 WHO 2022 的独立 AML 亚型。Isolated NPM1 或与 FLT3-ITD low-AR 合并时预后良好；与 FLT3-ITD high-AR 合并时需 ELN 2017 修正归为 Intermediate。Menin 抑制剂 (Revumenib, FDA 2024 KMT2A-r 批准，NPM1 indication 在研究阶段) 对 NPM1-mut AML 的 HOXA/MEIS1 表达程序有靶向性。",
    "IDH1": "**IDH1** R132 neomorphic 突变产生 2-hydroxyglutarate (2-HG)，阻断分化。**Ivosidenib** (FDA 2018 R/R 批准, AGILE 2022 一线批准联合 Aza) 是特异性 IDH1 抑制剂；**Olutasidenib** 2022 年 FDA 批准用于 R/R。",
    "IDH2": "**IDH2** R140/R172 突变机制类似 IDH1。**Enasidenib** (FDA 2017 R/R 批准) 是特异性 IDH2 抑制剂。Enasidenib + Venetoclax (ENAVEN 试验) 对 IDH2-mut 有强协同。",
    "TP53": "**TP53** 突变是 AML 最不良预后因素之一。传统 7+3 诱导对 TP53-mut 患者 CR 率低 (~20-30%)，中位 OS 约 4-6 个月。WHO 2022 将 multi-hit TP53 (双等位命中或伴 complex karyotype) 定义为独立 adverse 亚型。治疗方面，传统方案效果差，应考虑**临床试验入组** + **早期 allo-SCT 桥接**。",
    "RUNX1": "**RUNX1** 突变按 ELN 2017 归为 Adverse (尽管在 ELN 2022 有例外，de novo 情境下与 CBF 融合共存时可能 Intermediate)。无特异性靶向药。",
    "ASXL1": "**ASXL1** 突变常见于继发性 AML (MDS/MPN 转化)，按 ELN 2017 归为 Adverse。常与 SRSF2、TET2、DNMT3A 共突变形成 CHIP-AML 模式。",
    "CEBPA": "**CEBPA** 双等位突变 (biallelic) 按 ELN 2017 归为 Favorable；单等位 (monoallelic) 无此优势。WHO 2022 要求 bZIP domain 特异性检测以精确分类。",
    "DNMT3A": "**DNMT3A** 突变是 AML 最常见背景事件 (~25%)，多为 R882H 热点。独立预后影响中等偏负。HMA (Azacitidine, Decitabine) 对 DNMT3A-mut 克隆有证据支持的协同。",
    "TET2": "**TET2** 突变也是 epigenetic 背景事件，与 DNMT3A 功能相关。HMA 响应预测弱阳性。",
    "KIT": "**c-KIT** 突变在 core-binding factor AML (t(8;21) 或 inv(16)) 中出现时恶化预后。Dasatinib 有研究阶段应用。",
    "NRAS": "**NRAS** 激活 RAS-MAPK 通路，与 AML 治疗抵抗相关。MEK 抑制剂 (Trametinib) 有研究阶段数据。",
    "KRAS": "**KRAS** 激活 RAS-MAPK 通路，AML 中较 NRAS 少见。",
    "PTPN11": "**PTPN11** 激活 RAS-MAPK 通路，某些队列中 adverse。",
    "KMT2A": "**KMT2A** (MLL1) 基因突变罕见。真正临床重要的是 **KMT2A-r** (KMT2A 融合)，menin 抑制剂 Revumenib 2024 FDA 批准。",
    "MECOM": "**MECOM** 常通过 inv(3)/t(3;3) 融合激活，ELN 2017 uniformly adverse。无靶向药。",
    "CBFB": "**CBFB** 通过 inv(16) 或 t(16;16) 与 MYH11 融合，core-binding factor AML，ELN 2017 Favorable。7+3 + GO 是标准方案。",
}


_FUSION_PROSE: dict[str, str] = {
    "PML-RARA": "**PML-RARA** 融合定义急性早幼粒细胞白血病 (APL)，是 AML 的特殊亚型。**ATRA (维甲酸) + ATO (三氧化二砷) 是 APL 的一线方案**，可治愈率 > 90%。**7+3 诱导对 APL 不适当**，因为存在分化综合症风险且不如 ATRA+ATO 有效。WBC ≥ 10 × 10^9/L 的 high-risk APL 可加 GO 或 Idarubicin。",
    "KMT2A-r": "**KMT2A (11q23) 重排** 通过多种融合伴侣 (MLLT3、AFDN、ENL、ELL 等) 驱动 AML。按 ELN 2017，t(9;11) KMT2A-MLLT3 为 Intermediate，其他 KMT2A 融合为 Adverse。**Revumenib** (FDA 2024 批准) 是 menin 抑制剂，针对 KMT2A-r AML 的 HOXA/MEIS1 程序。",
    "CBFB-MYH11": "**inv(16) 或 t(16;16)** 产生 CBFB-MYH11 融合，属 core-binding factor AML (CBF-AML)。ELN 2017 Favorable，5 年 OS ~60-70% 用标准 7+3 + GO。",
    "RUNX1-RUNX1T1": "**t(8;21)** 产生 RUNX1-RUNX1T1 融合，属 core-binding factor AML。ELN 2017 Favorable，标准治疗 7+3 + GO，高治愈率。KIT 共突变恶化预后。",
}


def _executive_summary(kit: KitInput, kit_out: KitOutput) -> str:
    """3-4 sentence clinical summary at the top of the report."""
    age_str = f"{kit.age:.0f} 岁" if kit.age else "年龄未知"
    sex_cn = {"male": "男性", "female": "女性"}.get((kit.sex or "").lower(), "")
    eln = kit_out.predicted_eln2017
    fitness = "适合强化诱导" if kit_out.fitness_flag == "fit_for_intensive" else "不适合强化诱导"

    # Key drivers
    driver_flags = kit_out.driver_flags
    active_drivers = [k.replace("_", "-") for k, v in driver_flags.items() if v]
    driver_str = "、".join(active_drivers) if active_drivers else "未检出核心驱动突变"

    # Top regimen
    top_regimen = (kit_out.top_regimens or [{}])[0]
    top_reg_name = top_regimen.get("name", "方案待定")
    top_reg_cr = top_regimen.get("published_cr_cri_rate")
    cr_str = f"（CR/CRi {100 * top_reg_cr:.0f}%）" if top_reg_cr else ""

    return (
        f"本患者为 **{age_str} {sex_cn}**，分子分型提示 **{driver_str}**，"
        f"综合 ELN 2017 风险分层为 **{eln}**，临床评估为 **{fitness}**。"
        f"基于分子特征与已发表临床试验证据，**一线方案首选 {top_reg_name}{cr_str}**。"
        f"详见下文各节。"
    )


def _detected_mutations_table(driver_mutations: list[dict]) -> str:
    """Markdown table of all detected drivers with Tier, VAF, targetable drugs.

    `driver_mutations` comes from `kit_out.dna_summary["driver_mutations"]` —
    same source the DNA-profile PNG is built from, so the table always matches
    the figure.
    """
    if not driver_mutations:
        return ("> **未检出 25-gene 核心 panel 中的驱动基因突变**。"
                "如 lab 报告包含其他 (panel 外) 基因变异，请人工结合原始报告解读。\n")

    lines = [
        "| # | 基因 | 变异类型 | VAF | Tier | 可靶向药物 | ELN 意义 |",
        "|---|------|----------|-----|------|------------|----------|",
    ]
    for i, m in enumerate(driver_mutations, 1):
        gene = m.get("gene", "—")
        var = m.get("variant_type") or "—"
        if m.get("allelic_ratio") is not None:
            var = f"{var} (AR={m['allelic_ratio']:.2f})"
        if m.get("is_biallelic"):
            var = f"{var} · biallelic"
        vaf = m.get("vaf")
        vaf_str = f"{vaf:.2f}" if isinstance(vaf, (int, float)) else "—"
        tier = m.get("tier", "—")
        tier_str = f"T{tier}" if isinstance(tier, int) else str(tier)
        drugs = m.get("targetable_by") or []
        drugs_str = ", ".join(drugs[:3]) + (f" (+{len(drugs) - 3})" if len(drugs) > 3 else "")
        drugs_str = drugs_str or "无 FDA 批准靶向药"
        eln = m.get("eln_implication") or "—"
        # Cell-wrap for long text
        eln_short = eln if len(eln) <= 38 else eln[:36] + "…"
        lines.append(f"| {i} | **{gene}** | {var} | {vaf_str} | {tier_str} | "
                     f"{drugs_str} | {eln_short} |")
    return "\n".join(lines)


def _panel_coverage_table(driver_mutations: list[dict]) -> str:
    """A 25-gene core-panel checklist: which were detected, which were not.

    Groups genes by Tier so the clinician immediately sees whether any
    actionable (Tier 1) gene went untested or is wild-type.
    """
    detected_genes = {m.get("gene", "").upper() for m in driver_mutations}

    tier_groups = {
        "Tier 1 (FDA-targetable)": ["FLT3", "IDH1", "IDH2", "KMT2A"],
        "Tier 2 (prognostic / intensity-modifier)":
            ["NPM1", "TP53", "RUNX1", "ASXL1", "CEBPA"],
        "Tier 3a (epigenetic / HMA-responsive)":
            ["DNMT3A", "TET2"],
        "Tier 3b (RAS-MAPK)":
            ["NRAS", "KRAS", "PTPN11", "KIT"],
        "Tier 3c (MDS-related / splice / other)":
            ["WT1", "BCOR", "STAG2", "PHF6", "SRSF2", "SF3B1", "U2AF1",
             "EZH2", "MECOM", "CBFB"],
    }

    lines = [
        "| Tier 组 | 基因 (✓ = 检出 · ✗ = 野生型) |",
        "|---------|------------------------------|",
    ]
    for group, genes in tier_groups.items():
        gene_cells = []
        for g in genes:
            mark = "**✓**" if g in detected_genes else "✗"
            gene_cells.append(f"{g} {mark}")
        lines.append(f"| *{group}* | {' · '.join(gene_cells)} |")
    total_panel = sum(len(g) for g in tier_groups.values())
    lines.append("")
    lines.append(f"_25-gene 核心 panel 总覆盖: {len(detected_genes)}/{total_panel} "
                 f"检出突变 (野生型 = 正常序列, 不代表无遗传改变 — 仍可能有 panel 外变异)_")
    return "\n".join(lines)


def _rnaseq_outlier_section(kit: KitInput, kit_out: KitOutput) -> str:
    """Render Section 3.3 — RNA-Seq expression outlier analysis (v2 layout).

    Preferred path: use precomputed rna_outlier dict from predict_for_patient,
    which already includes phenotype signatures and transcriptome scan.
    Fallback: compute live from kit.rna_expression_full for test fixtures.

    The v2 layout has four subsections:
      - Highlights paragraph (transcriptional phenotype summary)
      - A. Outliers + DNA-mutated rows (full table, only the interesting rows)
      - B. Normal-range genes (one line collapsed)
      - C. Full-transcriptome top-N scan (optional, shows extra outliers
            outside the curated 25+8 panel)
    """
    rna_outlier = getattr(kit_out, "rna_outlier", None) or {}
    rows = rna_outlier.get("rows", []) if rna_outlier else []
    meta = rna_outlier.get("meta", {}) if rna_outlier else {}
    phenotype_sigs = rna_outlier.get("phenotype_signatures", [])
    scan_rows = rna_outlier.get("transcriptome_scan_rows", [])
    scan_meta = rna_outlier.get("transcriptome_scan_meta", {})

    if rows:
        return _render_rna_outlier_table_v2(rows, meta, phenotype_sigs,
                                              scan_rows, scan_meta)

    # Fallback: compute live (used in tests that build KitOutput manually)
    rna_expr = getattr(kit, "rna_expression_full", None)
    if rna_expr is None:
        return ("> **RNA-Seq 表达谱数据未提供** —— 本节为空。"
                "要启用本分析，请在 `KitInput.rna_expression_full` 提供全转录组 "
                "(log2-CPM 或 raw counts 均可, kit 会自动判断).\n")
    mutated_genes = {m.gene.upper() for m in (kit.mutations or [])}
    try:
        return build_rnaseq_outlier_markdown_v2(rna_expr, mutated_genes)
    except FileNotFoundError:
        return ("> 参考分布统计文件缺失 —— 请运行 "
                "`python scripts/build_driver_gene_ref_stats.py` 生成。\n")


def _render_rna_outlier_table_v2(
    rows: list[dict], meta: dict,
    phenotype_sigs: list[str],
    scan_rows: list[dict], scan_meta: dict,
    outlier_threshold: float = 0.75,
) -> str:
    """v2 renderer from cached dict rows: highlights + outliers + collapsed
    + transcriptome scan. Uses the cached data from KitOutput.rna_outlier."""
    if meta.get("n_genes_available", 0) == 0 and not any(
        r.get("available") for r in rows
    ):
        return (
            "> **RNA-Seq 表达谱数据未提供** —— 本节为空。"
            "如 lab 出的 RNA-Seq 表达矩阵可用, 请通过 "
            "`KitInput.rna_expression_full` 传入后重跑报告。\n"
        )

    parts: list[str] = []

    # --- Highlights paragraph ---
    n_avail = meta.get("n_genes_available", 0)
    n_high = meta.get("n_outliers_high", 0)
    n_low = meta.get("n_outliers_low", 0)
    if phenotype_sigs:
        parts.append(f"**RNA-Seq 高亮**: {n_high} 个基因高表达离群 (z≥+1.5), "
                      f"{n_low} 个低表达离群 (z≤-1.5), 归纳出以下转录表型签名:\n")
        for s in phenotype_sigs:
            parts.append(f"- {s}")
        parts.append("")
    else:
        parts.append(f"**RNA-Seq 高亮**: 本患者 25+8 基因 panel "
                      f"(n={n_avail} 可检) 中未发现显著转录离群 (所有 |z| < 1.5)。"
                      f"提示转录组呈典型 AML 背景, 无需追加表达驱动靶点。\n")

    # --- A. Outliers + mutated ---
    outlier_rows = [r for r in rows if r.get("available") and (
        (r.get("z_score") is not None and
         abs(r.get("z_score", 0)) >= outlier_threshold) or
        r.get("dna_status") == "✓ mutated"
    )]
    parts.append(f"#### A. 核心 panel 离群基因 + 全部 DNA 突变 "
                 f"({len(outlier_rows)} 行)")
    parts.append("")
    if not outlier_rows:
        parts.append("*（无）*\n")
    else:
        parts.append("| Tier 组 | 基因 | DNA 状态 | z-score | 方向 | 临床提示 |")
        parts.append("|---------|------|----------|--------:|:----:|----------|")
        prev_group = None
        for r in outlier_rows:
            group = r.get("tier_group", "")
            group_cell = group if group != prev_group else ""
            prev_group = group
            z = r.get("z_score")
            z_str = f"{z:+.2f}" if isinstance(z, (int, float)) else "n/a"
            gene = r.get("gene", "")
            gene_cell = (f"**{gene}**" if isinstance(z, (int, float))
                         and abs(z) >= 1.5 else gene)
            note = (r.get("note") or "").replace("|", "\\|")
            parts.append(f"| {group_cell} | {gene_cell} | "
                         f"{r.get('dna_status', '')} | {z_str} | "
                         f"{r.get('direction', 'n/a')} | {note} |")
        parts.append("")

    # --- B. Normal-range collapsed ---
    normal_rows = [r for r in rows if r.get("available") and
                   r.get("z_score") is not None and
                   abs(r.get("z_score", 0)) < outlier_threshold and
                   r.get("dna_status") != "✓ mutated"]
    unavailable_rows = [r for r in rows if not r.get("available")]

    parts.append(f"#### B. 核心 panel 正常范围基因 ({len(normal_rows)} 个)")
    parts.append("")
    if normal_rows:
        gene_list = ", ".join(r.get("gene", "") for r in normal_rows)
        parts.append(f"以下基因表达在 BeatAML 正常范围内 "
                     f"(|z| < {outlier_threshold}σ), DNA 状态: "
                     f"野生型 / hint-only: **{gene_list}**\n")

    if unavailable_rows:
        missing_list = ", ".join(r.get("gene", "") for r in unavailable_rows)
        parts.append(f"*未在输入 RNA-Seq 中覆盖的基因 "
                     f"({len(unavailable_rows)}): {missing_list}*\n")

    # --- C. Full-transcriptome scan ---
    parts.append(f"#### C. 全转录组扩展扫描 (core panel 之外)")
    parts.append("")
    if not scan_meta.get("available"):
        parts.append(f"*未启用 ({scan_meta.get('reason', '不可用')}).*\n")
    elif not scan_rows:
        parts.append(f"*未发现核心 panel 之外的极端离群 "
                     f"(|z| ≥ {scan_meta.get('min_abs_z_threshold', 3.0)}σ). "
                     f"已排除 sex / mitochondrial / hemoglobin 等非临床变异基因。*\n")
    else:
        parts.append(
            f"*本扫描从 BeatAML 2.0 全转录组 (~{scan_meta.get('n_candidates_total', '?')} "
            f"个候选基因)挑选出 |z| ≥ {scan_meta.get('min_abs_z_threshold', 3.0)}σ "
            f"的 top-{scan_meta.get('n_returned', 0)} 离群基因。这些基因**不在临床"
            f"策展的 25+8 panel 里**, 可能提示尚未被常规检测捕获的生物学信号, "
            f"需要临床人员结合基因功能自行解读。*")
        parts.append("")
        parts.append("| # | 基因 | z-score | 方向 | 患者值 | BeatAML 均值 ± std |")
        parts.append("|---|------|--------:|:----:|-------:|---------------------|")
        for i, r in enumerate(scan_rows, 1):
            parts.append(
                f"| {i} | **{r.get('gene', '')}** | "
                f"{r.get('z_score', 0):+.2f} | {r.get('direction', '')} | "
                f"{r.get('patient_value', 0):.2f} | "
                f"{r.get('ref_mean', 0):+.2f} ± {r.get('ref_std', 0):.2f} |"
            )
        parts.append("")

    # --- Legend ---
    parts.append(
        f"*图例*: ↑↑↑ z≥+2.5 · ↑↑ z≥+1.5 · ↑ z≥+0.75 · · 正常 · "
        f"↓ z≤-0.75 · ↓↓ z≤-1.5 · ↓↓↓ z≤-2.5. "
        f"参考分布: {meta.get('ref_cohort', 'BeatAML 2.0')} "
        f"(n={meta.get('ref_n_samples', '?')}). "
        f"[scale: {meta.get('scale_note', '')}]"
    )
    return "\n".join(parts)


def _hotspot_codon_section(mutations: list[MutationCall]) -> str:
    """Per issues #9 + #13 — surface hotspot-codon-specific clinical
    interpretation (DNMT3A R882, IDH1 R132, IDH2 R140/R172, FLT3-TKD
    D835/I836/F691L, NPM1 exon 12, KIT D816)."""
    from combo_val.clinical.hotspot_codons import hotspot_summary_for_report

    summary = hotspot_summary_for_report(mutations or [])
    if not summary:
        return ""  # no hotspot info to add — silent

    lines = ["**Hotspot codon analysis** (per issues #9 + #13):", ""]
    for s in summary:
        gene = s["gene"]
        codon = s.get("codon") or "?"
        label = s.get("hotspot_label") or "non-hotspot"
        conf = s.get("confidence", "low")
        marker = "⭐" if s["is_hotspot"] else "—"
        lines.append(f"- {marker} **{gene} {codon}** ({label}, confidence: {conf})")
        lines.append(f"  - {s['interpretation']}")
        lines.append("")
    return "\n".join(lines)


def _mutation_narrative(mutations: list[MutationCall]) -> str:
    """Build a prose paragraph describing each driver mutation's clinical meaning."""
    if not mutations:
        return "NGS 突变 panel 未检出核心驱动基因 (25-gene panel)。不排除其他 panel 外基因突变或样本敏感度限制，建议回顾 lab 原始 variant list。"

    lines = []
    for m in mutations:
        gene = m.gene.upper()
        vaf_str = f"VAF {m.vaf:.2f}" if m.vaf is not None else "VAF 未报告"
        variant = m.variant_type or "未分类"

        specific = ""
        if gene == "FLT3":
            if m.is_ITD:
                ar = m.allelic_ratio or 0
                ar_flag = ("**高负荷 ITD (AR ≥ 0.5)**" if ar >= 0.5
                           else "低负荷 ITD (AR < 0.5)")
                specific = f" 检测到 **FLT3-ITD** (AR={ar:.2f})，属于 {ar_flag}。"
            elif m.is_TKD:
                specific = " 检测到 **FLT3-TKD** 点突变 (通常 D835Y/F691L 等)。"
        elif gene == "CEBPA" and m.is_biallelic:
            specific = " 为 **biallelic** (双等位突变)，按 ELN 2017 归 Favorable。"

        prose = _GENE_PROSE.get(gene, "")
        lines.append(f"- **{gene}** ({variant}, {vaf_str}).{specific}\n  {prose}")
    return "\n".join(lines)


def _fusion_narrative(fusions: list[str]) -> str:
    if not fusions:
        return "细胞遗传学未报告融合基因。核型分析详见第 2.3 节。"
    lines = []
    for f in fusions:
        prose = ""
        for known, desc in _FUSION_PROSE.items():
            if known.upper() in f.upper():
                prose = desc
                break
        lines.append(f"- **{f}**\n  {prose or '此融合临床意义需文献核查。'}")
    return "\n".join(lines)


def _cytogenetic_narrative(cytogenetics: list[dict], karyotype_text: str | None) -> str:
    if not karyotype_text:
        return "无核型报告。建议血液科 / 细胞遗传学补充分析。"
    lines = [f"**原始核型**: `{karyotype_text}`\n"]
    abnormal = [c for c in cytogenetics
                if c.get("present") and "Normal" not in c.get("finding", "")]
    normal_flag = next((c for c in cytogenetics
                         if "Normal" in c.get("finding", "")), None)

    if not abnormal:
        if normal_flag and normal_flag.get("present"):
            lines.append("核型分析示**正常核型 (46,XX or 46,XY)**，未检出 ELN 2017 高危异常。")
        else:
            lines.append("核型解析未检出显著异常。")
    else:
        lines.append("**检出的 cytogenetic 异常：**\n")
        for c in abnormal:
            lines.append(f"- **{c['finding']}** — {c.get('interpretation', '')}")
        lines.append("")
    return "\n".join(lines)


def _eln_dual_section(kit: KitInput, kit_out: KitOutput) -> str:
    """Per issue #4 — render BOTH ELN 2017 and ELN 2022 with 2022 as primary
    clinical guidance. Adds a transition note when the two systems disagree
    so clinicians can see what changed and why."""
    from combo_val.clinical.eln_2022 import compare_eln_versions

    eln_2017_dict = kit_out.eln_2017 or {}
    eln_2022_dict = kit_out.eln_2022 or {}
    cat_2017 = eln_2017_dict.get("category") or kit_out.predicted_eln2017
    cat_2022 = eln_2022_dict.get("category") or cat_2017
    rationale_2017 = eln_2017_dict.get("rationale", [])
    rationale_2022 = eln_2022_dict.get("rationale", [])

    # Color emoji per category (same convention as Section 3.8)
    def _emoji(c: str) -> str:
        return {"Favorable": "🟢", "Intermediate": "🟡", "Adverse": "🔴"}.get(c, "⚪")

    lines = [
        "**ELN 2022 (Döhner Blood 2022, PMID 35797463) — primary, current standard**",
        f"{_emoji(cat_2022)} 分层: **{cat_2022}**",
        "",
        "依据：",
    ]
    for r in rationale_2022:
        lines.append(f"- {r}")
    lines.append("")
    lines.append(
        f"**ELN 2017 (Döhner Blood 2017, PMID 27895058) — historical, "
        f"matches BeatAML training labels**"
    )
    lines.append(f"{_emoji(cat_2017)} 分层: **{cat_2017}**")
    lines.append("")
    lines.append("依据：")
    for r in rationale_2017:
        lines.append(f"- {r}")
    lines.append("")

    # If the two versions diverge — explain why
    transition_note = compare_eln_versions(cat_2017, cat_2022)
    if transition_note:
        lines.append(
            f"> ⚠ **ELN 2017 vs ELN 2022 不一致** — "
            f"2017 = {cat_2017}, 2022 = {cat_2022}\n"
        )
        lines.append(f"> {transition_note}")
        lines.append("")
        lines.append(
            "*临床建议遵循 ELN 2022 (current standard)。ELN 2017 仅供与历史"
            "数据 (BeatAML training cohort) 对照。*"
        )
        lines.append("")

    # Append the existing prose-style narrative for clinical context
    lines.append(_eln_rationale_prose(kit, kit_out))
    return "\n".join(lines)


def _eln_rationale_prose(kit: KitInput, kit_out: KitOutput) -> str:
    """Narrative explanation of WHY the patient got their ELN category.
    Still public (used by tests) and called by _eln_dual_section."""
    eln = kit_out.predicted_eln2017
    flags = kit_out.driver_flags
    drivers_active = [k for k, v in flags.items() if v]
    fusions_str = " ".join(kit.fusions or []).upper()

    if eln == "Favorable":
        # APL is a distinct ELN Favorable subtype — NOT CBF-AML
        if "PML-RARA" in fusions_str or "PML_RARA" in fusions_str:
            return ("ELN 2017 分层为 **Favorable**，依据: **PML-RARA 融合 (APL)**。"
                    "**急性早幼粒细胞白血病是独立亚型，治疗方案与其他 AML 完全不同**: "
                    "首选 ATRA+ATO (low-risk) 或 ATRA+ATO+Idarubicin (high-risk)，"
                    "可治愈率 > 90%。**不适用 7+3 诱导**。")
        # CBF-AML (true core-binding factor)
        if any(f in fusions_str for f in
               ("RUNX1-RUNX1T1", "RUNX1_RUNX1T1", "AML1-ETO",
                "CBFB-MYH11", "CBFB_MYH11", "INV(16)")):
            return ("ELN 2017 分层为 **Favorable**，依据: **core-binding factor 融合** "
                    "(t(8;21) RUNX1-RUNX1T1 或 inv(16) CBFB-MYH11)。"
                    "标准 7+3 + GO (Mylotarg) 有效，5 年 OS 60-70%。")
        if flags.get("NPM1") and not flags.get("FLT3_ITD"):
            return ("ELN 2017 分层为 **Favorable**，依据: NPM1 突变存在且无 FLT3-ITD。"
                    "该亚型预后好，5-year OS 可达 60-70%，强化治疗反应率高。")
        if "CEBPA_biallelic" in drivers_active:
            return ("ELN 2017 分层为 **Favorable**，依据: CEBPA biallelic 双等位突变。"
                    "此为 WHO 2022 独立 AML 亚型，预后好。")
        return ("ELN 2017 分层为 **Favorable**，依据综合 favorable 特征。"
                "标准强化诱导方案有效。")

    if eln == "Adverse":
        parts = ["ELN 2017 分层为 **Adverse**，依据："]
        if flags.get("TP53"):
            parts.append("TP53 突变 (ELN 2017 明确 adverse)")
        if flags.get("RUNX1"):
            parts.append("RUNX1 突变 (ELN 2017 adverse)")
        if flags.get("FLT3_ITD") and not flags.get("NPM1"):
            parts.append("FLT3-ITD 高负荷 (无 NPM1 修正)")
        return (", ".join(parts) +
                "。此组预后差,传统 7+3 效果有限,**强烈建议临床试验入组 + CR1 阶段"
                "尽早 allo-SCT 桥接 — 立即启动 HLA 配型 + 供者搜索**(详见 §3.9)。")

    if eln == "Intermediate":
        if flags.get("NPM1") and flags.get("FLT3_ITD"):
            return ("ELN 2017 分层为 **Intermediate**,依据: NPM1 突变 + FLT3-ITD 高负荷组合 "
                    "(按 ELN 2017 修正规则,高 AR FLT3-ITD 原为 adverse,但 NPM1 "
                    "co-mutation 将整体归为 Intermediate)。"
                    "**建议路径**:(1) 标准 7+3 + FLT3 抑制剂(Mido per RATIFY 或 Quiz "
                    "per QUANTUM-First)强化诱导;(2) **CR1 阶段强烈推荐 allo-SCT** —— "
                    "FLT3-ITD 高 AR 即使有 NPM1 共突变,allo-SCT 仍显著降低复发,"
                    "**立即启动 HLA 配型 + 供者搜索**(详见 §3.9)。")
        return ("ELN 2017 分层为 **Intermediate**。无明确 favorable 或 adverse 特征。"
                "**CR1 allo-SCT 通常推荐**(尤其有不良共突变如 RUNX1/ASXL1/TP53),"
                "需结合体能状态、年龄、共病决定;启动 HLA 配型作为备选(详见 §3.9)。")

    return f"ELN 2017 分层: {eln}。"


def _who_icc_section(kit: KitInput) -> str:
    """Per issue #12 — WHO 2022 + ICC 2022 entity classification.
    Both systems revised in 2022 (WHO: Khoury Leukemia 2022 PMID 35732831,
    ICC: Arber Blood 2022 PMID 35797568). They mostly agree but differ on
    blast thresholds and TP53 handling."""
    from combo_val.clinical.who_icc_2022 import classify_who_icc_2022
    r = classify_who_icc_2022(
        karyotype_text=kit.karyotype_text,
        mutations=kit.mutations or [],
        fusions=kit.fusions or [],
        blast_pct=kit.blast_pct_bm,
        prior_mds=kit.prior_mds,
    )
    lines = [
        "**WHO 2022** (Khoury et al. *Leukemia* 2022, PMID 35732831):",
        f"- 实体: **{r.who_2022}**",
        "",
        "**ICC 2022** (Arber et al. *Blood* 2022, PMID 35797568):",
        f"- 实体: **{r.icc_2022}**",
        "",
    ]
    if not r.concordant:
        lines.extend([
            f"> ⚠ **WHO 2022 vs ICC 2022 分歧** — 两个分类系统对本患者的"
            f"诊断分类不一致:",
            f">",
            f"> - WHO 2022: `{r.who_2022}`",
            f"> - ICC 2022: `{r.icc_2022}`",
            f">",
            f"> 常见差异原因:",
            f"> 1. 母细胞百分比阈值不同 (WHO 2022 不再要求 ≥ 20% blast for "
            f"defining genetic abnormality; ICC 2022 保留 10%/20% 阈值)",
            f"> 2. TP53 (ICC 2022 单独实体, WHO 2022 归 AML-MR)",
            f">",
            f"> **临床建议**: 依据您所在区域 / 中心的报告标准选择。两系统在"
            f"治疗推荐上多数情况下一致, 主要差异在于诊断登记 (registry coding)。",
            "",
        ])
    if r.rationale:
        lines.append("**分类依据**:")
        for s in r.rationale:
            lines.append(f"- {s}")
    return "\n".join(lines)


def _allo_sct_section(kit: KitInput, kit_out: KitOutput) -> str:
    """Per-issue-#2: replace 'consider allo-SCT' with explicit recommendation
    + readiness checklist when ELN ≥ Intermediate AND patient is fit.

    Recommendation strength tiers:
      - **Strongly recommended** (大写,加粗) — Adverse OR (Intermediate
        with FLT3-ITD high AR / TP53 / RUNX1 / ASXL1 / cytogenetic adverse)
      - **Recommended** — Intermediate without high-risk co-mutations
      - **Discuss case-by-case** — Favorable (NPM1 isolated, CEBPA
        biallelic, CBF-AML in CR1 with MRD negative)
      - **Not indicated** — APL (PML-RARA) in CR
    """
    eln = kit_out.predicted_eln2017
    flags = kit_out.driver_flags
    fit = kit_out.fitness_flag == "fit_for_intensive"
    fusions_str = " ".join((kit.fusions or [])).upper()

    # APL: SCT not standard for low-risk APL in CR
    if "PML-RARA" in fusions_str or "PML_RARA" in fusions_str:
        return (
            "**APL (PML-RARA)** — Allo-SCT 通常**不作为 CR1 阶段标准推荐**。"
            "ATRA + ATO 治疗后达到分子学缓解的 APL 患者长期生存极好(>90%);"
            "仅在分子学复发或难治时再评估 SCT。"
        )

    # Decide recommendation strength
    if not fit:
        strength = "**Allo-SCT 须结合体能 + 共病评估**"
        rationale = (
            "患者当前体能/共病评估为 **unfit for intensive induction**。"
            "传统 myeloablative SCT 风险高;可考虑**减低强度预处理(RIC)** "
            "SCT,需 transplant 团队评估 HCT-CI 评分。"
        )
    elif eln == "Adverse":
        strength = "🔴 **CR1 阶段强烈推荐 Allo-SCT(Strongly recommended)**"
        rationale = (
            "ELN Adverse 风险组(TP53 / RUNX1 / ASXL1 / 复杂核型 / "
            "FLT3-ITD 无 NPM1 等),传统化疗治愈率 < 20%,"
            "**SCT 是唯一可能治愈的途径**。**今日**启动 HLA + 供者搜索。"
        )
    elif eln == "Intermediate":
        # FLT3-ITD high AR + NPM1 still benefits from SCT
        if flags.get("FLT3_ITD"):
            strength = "🔴 **CR1 阶段强烈推荐 Allo-SCT(Strongly recommended)**"
            rationale = (
                "FLT3-ITD 高 AR 即使有 NPM1 共突变,SCT 仍显著降低复发率"
                "(SORMAIN, GIMEMA AML0310 等多项研究)。**今日**启动 HLA + "
                "供者搜索,目标 CR1 后 3 个月内移植。"
            )
        elif flags.get("RUNX1") or flags.get("ASXL1") or flags.get("TP53"):
            strength = "🔴 **CR1 阶段强烈推荐 Allo-SCT**"
            rationale = (
                "Intermediate 风险伴不良共突变(RUNX1/ASXL1/TP53),复发风险 "
                "实际偏高,移植获益明确。"
            )
        else:
            strength = "🟡 **CR1 阶段推荐 Allo-SCT**(Recommended)"
            rationale = (
                "Intermediate 无明显高危共突变。需结合 MRD 状态、年龄、"
                "供者可及性、患者意愿决定。"
            )
    elif eln == "Favorable":
        if flags.get("NPM1") and not flags.get("FLT3_ITD"):
            strength = "🟢 **CR1 不常规推荐 Allo-SCT**(NPM1-only Favorable)"
            rationale = (
                "Isolated NPM1-mut 是 ELN Favorable,5 年 OS 60-70%。"
                "若 CR1 + MRD 阴性,**不推荐**前置 SCT;若 MRD 持续阳性"
                "或早期复发,再考虑 SCT。"
            )
        else:
            strength = "🟢 **CR1 不常规推荐 Allo-SCT**(Favorable)"
            rationale = (
                "CBF-AML / CEBPA biallelic 等 Favorable 亚型,化疗"
                "+ HiDAC 巩固足够,SCT 留作复发挽救。"
            )
    else:
        strength = "**Allo-SCT 推荐待评估**"
        rationale = "ELN 分层信息不足,需 MDT 综合判断。"

    # Build the readiness checklist (only render full checklist when actually
    # recommended — avoid clutter for Favorable / APL)
    show_checklist = (
        eln in ("Adverse", "Intermediate")
        and (kit_out.predicted_eln2017 != "Favorable")
        and "PML-RARA" not in fusions_str
    )

    body = [strength, "", rationale]
    if show_checklist:
        body.extend([
            "",
            "**SCT 准备 checklist(MDT 团队按时点核对)**:",
            "",
            "**今日 / D0**:",
            "- [ ] HLA 高分辨配型 — 患者 + 一级亲属(同胞/父母/子女)",
            "- [ ] 转诊移植中心(CR1 起 30 天内确诊转诊)",
            "- [ ] 计算 **HCT-CI 评分**(共病指数,> 3 提示 RIC)",
            "- [ ] 评估 ECOG / KPS 体能状态",
            "- [ ] 启动**供者搜索**(BMDW、骨髓库 / NMDP)",
            "",
            "**诱导期间**:",
            "- [ ] 心脏 baseline:ECG、ECHO(LVEF)",
            "- [ ] 肺功能:DLCO 校正、FEV1",
            "- [ ] 肝肾功能 baseline、HBV / HCV / HIV / CMV / EBV 血清学",
            "- [ ] 牙科评估、感染灶清除",
            "- [ ] 生育力咨询(育龄患者)",
            "- [ ] 心理 + 社工支持评估",
            "",
            "**CR1 巩固阶段**:",
            "- [ ] 确认 MRD 状态(NPM1 RT-qPCR / 流式)",
            "- [ ] 选择供者:Matched Sibling > MUD > Haploidentical > Cord",
            "- [ ] 预处理方案(MAC vs RIC,基于 HCT-CI + 年龄 + 供者类型)",
            "- [ ] 入院 SCT 时间窗:CR1 后 ≤ 3 个月内为佳",
            "",
            "_⚠ 不做 SCT 的复发风险:_ FLT3-ITD high AR + NPM1 患者 5 年 RFS"
            "化疗组 ~30-40%,SCT 组 ~50-65%(GIMEMA AML0310, SORMAIN 等)。",
        ])

    return "\n".join(body)


def _baseline_workup_section(kit: KitInput, kit_out: KitOutput) -> str:
    """Per issue #6 — pre-induction workup checklist with conditional rules.

    Rendered as Markdown task-list (`- [ ]`) so MD/HTML viewers show
    checkboxes. PDF export keeps the same structure as plain text bullets.
    """
    lines = [
        "**Pre-induction workup**: complete BEFORE starting treatment. "
        "组织起来按时点,避免临床决策延误。",
        "",
        "**Universal — 所有 AML 患者**:",
        "- [ ] CBC + diff + reticulocyte count",
        "- [ ] CMP (Na/K/Cl/HCO₃, BUN/Cr, Ca/P/Mg, AST/ALT, total bilirubin, "
        "albumin, total protein)",
        "- [ ] **Coagulation panel** — PT/INR, aPTT, fibrinogen, D-dimer "
        "(critical: APL screen + DIC baseline)",
        "- [ ] **ECG** (baseline rhythm + QTc; pre-anthracycline + QT-prolonging "
        "agents reference)",
        "- [ ] **HBV / HCV / HIV** — surface antigen + viral load if positive "
        "(reactivation risk on chemo / biologics; hepatitis B requires "
        "entecavir or tenofovir prophylaxis)",
        "- [ ] **Pregnancy test (β-hCG)** — premenopausal female",
        "- [ ] **Type and screen / cross-match** — pretransfusion baseline",
        "- [ ] **CYP3A4 medication review** — Ven, Quiz, Mido, azoles, "
        "rifampin, calcium channel blockers; document concomitant meds",
        "",
    ]

    age = kit.age
    fitness = kit_out.fitness_flag
    flags = kit_out.driver_flags or {}
    cat_2022 = (kit_out.eln_2022 or {}).get("category") or kit_out.predicted_eln2017
    fusions_str = " ".join(kit.fusions or []).upper()
    is_apl = "PML-RARA" in fusions_str or "PML_RARA" in fusions_str

    # Conditional: anthracycline-eligible (fit + not APL using ATRA+ATO low risk)
    anthracycline_likely = (
        fitness == "fit_for_intensive" and not is_apl
        and (age is None or age <= 75)
    )
    if anthracycline_likely:
        lines.extend([
            "**强化诱导 (7+3 / CPX-351 等含蒽环类)**:",
            "- [ ] **ECHO with EF** (anthracycline cardiotoxicity baseline; "
            "stop if LVEF < 50% — switch to non-anthracycline regimen)",
            "- [ ] Cardiac troponin baseline (high-risk: prior chest XRT, "
            "diabetes, hypertension)",
            "- [ ] Lipid panel + HbA1c (fitness optimization)",
            "",
        ])

    # Conditional: FLT3i (any regimen recommends Mido / Quiz / Gilt)
    flt3i_likely = flags.get("FLT3_ITD") or flags.get("FLT3_TKD")
    if flt3i_likely:
        lines.extend([
            "**FLT3 抑制剂 (Mido/Quiz/Gilt)**:",
            "- [ ] **ECG with QTc** (Quizartinib FDA black-box: cardiac "
            "arrest with QTc > 500ms)",
            "- [ ] **Electrolytes** — K ≥ 4.0 mmol/L, Mg ≥ 2.0 mg/dL "
            "(target before + during therapy; QT-prolongation prophylaxis)",
            "- [ ] **Avoid concurrent QT-prolonging agents** — review and "
            "stop ondansetron, fluconazole, levofloxacin, quetiapine, etc.",
            "",
        ])

    # Conditional: Venetoclax (any regimen recommends Ven)
    ven_likely = (
        kit_out.top_regimens and any(
            "Venetoclax" in (r.get("drugs") or [])
            for r in (kit_out.top_regimens or [])[:3]
        )
    )
    if ven_likely:
        lines.extend([
            "**Venetoclax (Ven+Aza, Ven+LDAC, Ven+Dec, etc.)**:",
            "- [ ] **TLS labs** — uric acid, K, P, Ca, Cr, LDH (重复 q6h × 24h "
            "post first dose; aggressive ramp-up prophylaxis if WBC > 25 or "
            "high disease burden)",
            "- [ ] **Allopurinol 300mg PO** (start 24–48h pre-Ven) — "
            "or **rasburicase** if UA > 7.5 mg/dL or rapid TLS risk",
            "- [ ] **CYP3A4 strong inhibitor adjustment** — posaconazole "
            "→ Ven dose reduce 75%; voriconazole → 50%; "
            "fluconazole → 50%. **Never** with strong CYP3A inducers.",
            "",
        ])

    # Conditional: hyperleukocytic OR monocytic OR M4/M5 (we don't have FAB,
    # but high WBC is the proxy) → LP + IT MTX prophylaxis
    high_wbc = kit.wbc is not None and float(kit.wbc) > 50
    monocytic_signal = (
        # Heuristic: NPM1, KMT2A-rearranged, or MLL fusions associate with M4/M5
        flags.get("NPM1") or "KMT2A" in fusions_str or "MLLT" in fusions_str
    )
    if high_wbc or monocytic_signal:
        lines.extend([
            "**CNS leukemia prophylaxis (高 WBC / monocytic / KMT2A-r)**:",
            "- [ ] **Lumbar puncture** with cytology + flow cytometry "
            "(after platelets ≥ 50 ×10⁹/L, INR ≤ 1.5; transfuse if needed)",
            "- [ ] **Intrathecal methotrexate 12–15 mg** prophylaxis "
            "× 4–6 doses (or as per institutional protocol); "
            "hold if CNS+ disease (treatment dosing instead)",
            "- [ ] Brain MRI if focal neurologic findings",
            "",
        ])

    # Conditional: ELN ≥ Intermediate AND fit → urgent HLA + transplant referral
    if cat_2022 in ("Intermediate", "Adverse") and fitness == "fit_for_intensive":
        lines.extend([
            "**HLA + transplant prep (ELN ≥ Intermediate, fit)**:",
            "- [ ] **HLA high-resolution typing** — patient + first-degree "
            "relatives (siblings priority, then parents/children)",
            "- [ ] **Refer to transplant center** within 30 days of CR1 "
            "(MDT discussion, donor search, conditioning regimen choice)",
            "- [ ] **HCT-CI (Hematopoietic Cell Transplant Comorbidity "
            "Index)** scoring — guides MAC vs RIC decision",
            "- [ ] **Unrelated donor search** initiated (NMDP / BMDW) — "
            "median time-to-MUD ~3 months",
            "",
        ])

    # Always: fertility + psychosocial
    if age is not None and age <= 50:
        lines.extend([
            "**生育力 + 心理支持 (≤ 50 岁)**:",
            "- [ ] Fertility consult — sperm banking (♂) or oocyte/embryo "
            "cryopreservation (♀) BEFORE chemotherapy (chemo-induced "
            "azoospermia/POI common after anthracycline)",
            "- [ ] Social work / patient navigator referral",
            "",
        ])

    return "\n".join(lines)


def _mrd_monitoring_section(kit: KitInput, kit_out: KitOutput) -> str:
    """Per issue #7 — MRD monitoring plan based on patient genomic profile.

    References ELN 2021 MRD consensus (Heuser et al., Blood 2021;
    PMID 33591443). Conditional rules:
      - NPM1mut → NPM1 RT-qPCR
      - FLT3-ITD → NGS-MRD (ClonoSEQ or in-house FLT3-ITD assay)
      - CBF-AML (RUNX1-RUNX1T1 / CBFB-MYH11) → fusion-transcript RT-qPCR
      - Other → multiparametric flow cytometry (MFC, sensitivity 1e-4)
    """
    flags = kit_out.driver_flags or {}
    fusions_str = " ".join(kit.fusions or []).upper()

    has_npm1 = bool(flags.get("NPM1"))
    has_flt3_itd = bool(flags.get("FLT3_ITD"))
    # Detect CBF from karyotype OR fusion list
    from combo_val.clinical.karyotype_parser import parse_karyotype
    k = parse_karyotype(kit.karyotype_text)
    has_cbf = (
        k.t_8_21 or k.inv_16
        or any(
            f in fusions_str
            for f in ("RUNX1-RUNX1T1", "RUNX1_RUNX1T1",
                      "CBFB-MYH11", "CBFB_MYH11", "AML1-ETO")
        )
    )
    has_apl = "PML-RARA" in fusions_str or "PML_RARA" in fusions_str or k.t_15_17

    lines = [
        "**MRD monitoring plan** — measurable residual disease drives "
        "post-CR1 decisions (consolidation choice, allo-SCT timing, "
        "preemptive intervention). 参考 ELN 2021 MRD consensus "
        "(Heuser et al., Blood 2021, PMID 33591443).",
        "",
        "**Standard timepoints (all patients)**:",
        "- 诱导后 (end of induction, ~D28-D35)",
        "- 巩固 1 后 (post-consolidation 1)",
        "- 巩固 2 后 (post-consolidation 2)",
        "- 巩固 3 后 / 移植前 (post-consolidation 3 / pre-SCT)",
        "- 巩固后 q3 mo × 2 yr (post-treatment surveillance)",
        "",
    ]

    # Choose primary modality based on patient profile (priority: APL > CBF > NPM1 > FLT3-ITD > MFC fallback)
    if has_apl:
        lines.extend([
            "**主要 modality: PML-RARA RT-qPCR** (APL — distinct from other AML)",
            "- 灵敏度 1e-4",
            "- ATRA+ATO 后 PML-RARA 阴性 = 分子 CR (ELN 2021 APL: 巩固后 BCR ≥ 4 logs reduction)",
            "- 阳性 → 早期 ATO/MTX 强化 (preemptive); confirmed molecular relapse → ATO + GO + ATRA",
            "",
        ])
    elif has_cbf:
        lines.extend([
            "**主要 modality: 融合转录本 RT-qPCR**",
            "- t(8;21) → **RUNX1-RUNX1T1** 转录本 (灵敏度 1e-4 to 1e-5)",
            "- inv(16)/t(16;16) → **CBFB-MYH11** 转录本 (Type A 最常见)",
            "- 目标: 巩固后 ≥ 3-log reduction = MRD-negative; 持续阳性或 > 0.1% "
            "→ 高复发风险, 考虑 allo-SCT",
            "- ELN 2021: CBF-AML 巩固后仍可检出低水平 MRD 但保持稳定 → 不一定需要"
            "干预; 上升趋势 (rising MRD) 才是 actionable",
            "",
        ])
    elif has_npm1:
        lines.extend([
            "**主要 modality: NPM1 RT-qPCR** (ELN 2021 SOC for NPM1mut AML)",
            "- 灵敏度 1e-5 to 1e-6 (gold standard)",
            "- Express as **NPM1mut/ABL ratio** (NCRI/AMLSG harmonized)",
            "- **MRD-negative** = below detection limit OR < 2-log reduction maintained",
            "- **MRD-positive at end of induction** → 巩固后 SCT；**MRD-positive "
            "at end of consolidation** → 立即 SCT (Ivey et al. NEJM 2016 — "
            "NPM1 MRD post-cons 是最强 RFS 预测因子)",
            "- **Rising MRD (≥ 1-log increase)** during follow-up → preemptive "
            "salvage (Ven+Aza or trial); 等到 frank relapse 治愈率显著降低",
            "",
        ])
    else:
        lines.extend([
            "**主要 modality: 多参数流式细胞术 (MFC)** — non-NPM1 / non-CBF / "
            "non-APL profile",
            "- 灵敏度 1e-3 to 1e-4 (operator-dependent)",
            "- 使用 difference-from-normal (DfN) approach 检测 leukemia-"
            "associated immunophenotype (LAIP)",
            "- ELN 2021: MFC MRD ≥ 0.1% 视为 MRD-positive; < 0.1% = "
            "MRD-negative (但 < 0.01% 更可靠)",
            "- 局限: 表型漂移 (treatment-induced phenotype change) 可能导致"
            "假阴性; NGS-MRD (e.g., ClonoSEQ) 是更鲁棒的备选",
            "",
        ])

    # Additional FLT3-ITD-specific monitoring (independent of primary modality)
    if has_flt3_itd:
        lines.extend([
            "**FLT3-ITD 附加监测 (除主要 modality 外)**:",
            "- **NGS-MRD for FLT3-ITD** (e.g., ClonoSEQ, in-house ITD-NGS) "
            "灵敏度 1e-4 to 1e-5",
            "- 时间点同上; FLT3-ITD MRD 阳性是 ADMIRAL/QUANTUM-First post-SCT "
            "维持治疗 (Gilteritinib / Quizartinib) 的入组依据",
            "- ELN 2021 FLT3-ITD MRD 共识仍在演化; 当前实践: NGS-MRD positive "
            "post-SCT → 启动 FLT3i 维持 (per RATIFY/ADMIRAL protocols)",
            "",
        ])

    lines.extend([
        "**Reporting standards**:",
        "- 所有 MRD 报告必须包含: 检测方法、灵敏度、对照基因 (e.g., ABL)、"
        "样本来源 (BM > PB)、报告日期、检测实验室认证 (CAP/CLIA)",
        "- ELN 2021 推荐使用 standardized reporting (NCRI/AMLSG schema)",
        "",
        "*References*: Heuser M et al. ELN 2021 MRD consensus. Blood 2021; "
        "138(26):2753-2767. PMID 33591443. "
        "Ivey A et al. NPM1 MRD predicts relapse. NEJM 2016; 374(5):422-433.",
    ])

    return "\n".join(lines)


def _regimen_section(kit_out: KitOutput) -> str:
    """Render top 3 regimens as narrative paragraphs."""
    regimens = kit_out.top_regimens or []
    if not regimens:
        return "*未返回匹配方案。可能原因：患者驱动基因组合超出数据库覆盖范围；建议按 ELN 风险 + 体能状态由临床医生选择。*"

    lines = []
    for i, r in enumerate(regimens[:3], 1):
        name = r.get("name", f"Regimen {i}")
        drugs = " + ".join(r.get("drugs", []))
        trial = r.get("trial_name", "")
        phase = r.get("trial_phase", "")
        pmid = r.get("pmid")
        cr = r.get("published_cr_cri_rate")
        os_m = r.get("published_median_os_months")

        cr_str = f"**CR/CRi = {100 * cr:.0f}%**" if cr else ""
        os_str = f"，中位 OS = {os_m:.1f} 月" if os_m else ""
        pmid_link = f" (PMID [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/))" if pmid else ""

        match_reason = r.get("biomarker_matches") or []
        match_str = ("患者匹配该方案的依据: " + "、".join(match_reason) + "。"
                      if match_reason else "")

        cautions = r.get("cautions") or []
        caution_str = ("\n\n  **用药警告**:\n" +
                        "\n".join(f"  - {c}" for c in cautions)
                        if cautions else "")

        rank_prefix = {1: "### 4.1 首选方案", 2: "### 4.2 次选方案",
                       3: "### 4.3 备选方案"}.get(i, f"### 4.{i} 方案")
        lines.append(
            f"{rank_prefix}: {name}\n\n"
            f"**组成**: {drugs}\n\n"
            f"**证据**: {phase} {trial}{pmid_link}. 原始试验报告 {cr_str}{os_str}。\n\n"
            f"{match_str}{caution_str}"
        )
    return "\n\n".join(lines)


def _combo_prediction_narrative(kit_out: KitOutput) -> str:
    """Layer 3 MLP prediction summary."""
    combos = kit_out.top_combinations or []
    if not combos:
        return "*模型未返回组合预测。*"

    # Per issue #3 — Layer-3 OOD suppression marker takes precedence.
    # When the upstream kit detects that the RNA-Seq sample is far from
    # BeatAML training distribution, top_combos is replaced with a single
    # suppression sentinel: drug1/drug2/auc fields are NOT populated. We
    # render a yellow warning banner instead of crashing on missing keys.
    top = combos[0]
    if top.get("suppressed"):
        backbone = top.get("layer3_backbone", "unknown")
        severity = top.get("ood_severity", "unknown")
        m_raw = top.get("mahalanobis_raw")
        m_qn = top.get("mahalanobis_qn")
        reason = top.get("suppress_reason") or ""
        m_raw_str = f"{m_raw:.1f}" if isinstance(m_raw, (int, float)) else "n/a"
        m_qn_str = f"{m_qn:.1f}" if isinstance(m_qn, (int, float)) else "n/a"
        return (
            f"> ⚠ **Layer-3 组合预测已禁用 (per issue #3 — RNA-Seq OOD)**\n\n"
            f"上游 QC 检测到本样本 RNA-Seq 与 BeatAML 2.0 训练分布偏离过远，"
            f"MLP 输出在数值上看似合理，实际为外推幻觉，不可作为临床决策依据。\n\n"
            f"- **OOD 等级**: `{severity}`\n"
            f"- **Mahalanobis 距离 (raw)**: {m_raw_str}\n"
            f"- **Mahalanobis 距离 (post-QN)**: {m_qn_str}\n"
            f"- **Backbone**: `{backbone}`\n\n"
            f"*详情*: {reason}\n\n"
            f"**替代依据**: 请优先参考第 3 节 (循证一线方案) 与第 4 节 "
            f"(克隆覆盖三联体)；这两层不依赖 RNA-Seq 表达，对 OOD 样本仍稳健。"
        )

    backbone = top.get("layer3_backbone", "unknown")
    lines = [
        f"基于 `{backbone}` backbone 在 BeatAML 2.0 (613 患者 × 165 药) 数据集上训练，"
        f"对本患者预测 AUC 最低 (理论细胞杀伤最强) 的 top-3 组合：\n",
    ]
    for c in combos[:3]:
        pair = f"{c['drug1']} + {c['drug2']}"
        auc = c.get("predicted_combo_auc", "?")
        mech = c.get("mech_score", 0)
        cov = c.get("clonal_coverage_score")
        cov_str = f"，克隆覆盖 {cov:.2f}" if cov is not None else ""
        lines.append(
            f"- **{pair}** — 预测 AUC {auc:.1f} (机制先验 {mech:+.2f}{cov_str})"
        )
    lines.append(
        "\n*注意*: 组合 AUC 预测基于 ex-vivo (体外) 数据，不等同于临床 CR 预测。"
        "建议对照第三节的临床试验证据综合判断。"
    )
    return "\n".join(lines)


def _clonal_coverage_narrative(kit_out: KitOutput) -> str:
    cc = kit_out.clonal_coverage or {}
    if not cc:
        return "*克隆分析未可用。*"
    clones = cc.get("patient_clones", {})
    if not clones:
        return "患者突变 panel 未识别显著 clonal archetype。"
    clone_list = "、".join(f"{k} (权重 {v})" for k, v in clones.items())
    dominant = cc.get("dominant_clones", [])
    dom_str = f" 主要克隆: {'、'.join(dominant)}。" if dominant else ""

    top_triplets = cc.get("top_triplets_by_coverage", [])
    triplet_lines = []
    for t in top_triplets[:3]:
        drugs = " + ".join(t.get("drugs", []))
        cov = t.get("coverage_score", 0)
        triplet_lines.append(f"- {drugs} — 覆盖 {cov:.2f}")

    narrative = (
        f"本患者分子分型在 Palmer-Sorger Independent Drug Action (IDA) 框架下分解为 "
        f"{cc.get('n_clones_present', 0)} 个活跃克隆原型: {clone_list}。{dom_str}\n\n"
        f"按 Bliss-IDA 理论，三药联合能覆盖更多克隆:\n"
    )
    return narrative + "\n".join(triplet_lines)


def _confidence_narrative(kit_out: KitOutput) -> str:
    """Sample QC + confidence caveats in prose."""
    dna = kit_out.dna_summary or {}
    qc = dna.get("sample_qc", {})
    notes = kit_out.confidence_notes or []

    lines = []
    n_mut = qc.get("n_mutations_called", 0)
    n_above = qc.get("n_above_vaf_0_20", 0)
    karyotype_ok = qc.get("karyotype_parsed", False)
    fusions_n = qc.get("fusions_reported", 0)

    lines.append(f"**样本质控**:\n")
    lines.append(f"- NGS 突变检出: {n_mut} 个，其中 VAF ≥ 0.20 的有 {n_above} 个")
    lines.append(f"- 核型解析: {'成功' if karyotype_ok else '未成功或未提供'}")
    lines.append(f"- 融合报告: {fusions_n} 个\n")

    if notes:
        lines.append("**模型置信度说明**:\n")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines)


def _drug_interaction_warnings(kit_out: KitOutput) -> list[str]:
    """Per issue #8 — generate drug-interaction warnings (CYP3A4, QT, etc.)
    based on the patient's top recommended regimens.

    Returns a list of formatted warning bullets ready to merge into the
    cautions section.
    """
    regimens = kit_out.top_regimens or []
    if not regimens:
        return []

    # Collect distinct drugs across the top-3 regimens
    top3_drugs: set[str] = set()
    for r in regimens[:3]:
        top3_drugs.update(d for d in (r.get("drugs") or []))

    warnings: list[str] = []

    # ---- Venetoclax + CYP3A4 inhibitor (azole prophylaxis is SOC) ----
    if "Venetoclax" in top3_drugs:
        warnings.append(
            "💊 **Venetoclax + 强效 CYP3A4 抑制剂 (常见: 唑类抗真菌)**: "
            "Ven + posaconazole → AUC ↑ 6-7×, **必须减量**. "
            "**Posaconazole / itraconazole / voriconazole** (强效抑制) → "
            "Ven 维持剂量降至 **100 mg/day**, ramp-up 也按比例 (10 → 20 → 50 → 100 mg). "
            "**Fluconazole** (中效) → Ven 减 50%. "
            "**与强效 CYP3A4 诱导剂 (rifampin, phenytoin, carbamazepine, St. John's wort) "
            "同用** → Ven 暴露下降 ≥ 70%, **应避免同用**; 必须时改用其他方案。"
        )

    # ---- Quizartinib QT/CYP3A4 interactions (FDA black-box) ----
    if "Quizartinib (AC220)" in top3_drugs or "Quizartinib" in top3_drugs:
        warnings.append(
            "⚠ **Quizartinib + 强效 CYP3A4 抑制剂 (FDA black-box: cardiac arrest)**: "
            "Quiz 是 CYP3A4 底物 + QT 延长药物. 强效 CYP3A4 抑制剂 (azoles) → "
            "Quiz 暴露 ↑ 90%, **维持期减量至 30 mg/day** (诱导 35.4 mg/day, "
            "巩固 53 mg/day). **避免**与其他 QT 延长药物同用 (ondansetron, "
            "fluoroquinolones, methadone, antipsychotics, tricyclics). "
            "**基线 QTc > 450 ms 或药物相互作用无法管理 → 改用 Gilteritinib** "
            "(QT 风险更低)."
        )

    # ---- Midostaurin CYP3A4 caveat (less stringent than Quiz) ----
    if "Midostaurin" in top3_drugs:
        warnings.append(
            "💊 **Midostaurin + CYP3A4 调节剂**: Mido 是 CYP3A4 底物且自我诱导 "
            "(steady-state AUC ↓ ~75% over 28 days). 强效抑制剂 (posaconazole) → "
            "Cmax ↑ 1.6×, 临床通常无需调整 (RATIFY 允许同用). 强效诱导剂 "
            "(rifampin) → Cmax ↓ ~94%, **避免同用**. 与 Ven 同方案时 (e.g., "
            "假设性 7+3 + Mido + Ven 联合) 注意累加 CYP3A4 负担."
        )

    # ---- Gilteritinib CYP3A4 (less drastic) ----
    if "Gilteritinib" in top3_drugs:
        warnings.append(
            "💊 **Gilteritinib + 强效 CYP3A4 抑制剂**: Gilt 是 CYP3A4 + P-gp 底物. "
            "Itraconazole → Gilt AUC ↑ 2.2×; 监测毒性 (主要: 骨髓抑制、QT). "
            "强效诱导剂 (rifampin) → Gilt AUC ↓ 70%, **避免同用**. "
            "Gilt 的 QT 风险显著低于 Quiz; 仍建议基线 + 治疗期 ECG 监测."
        )

    # ---- Ivosidenib differentiation syndrome + CYP3A4 ----
    if "Ivosidenib" in top3_drugs:
        warnings.append(
            "⚠ **Ivosidenib differentiation syndrome (~25%) + CYP3A4 互作**: "
            "Ivo 是 CYP3A4 底物 + 诱导剂. 强效抑制剂 (posaconazole) → Ivo AUC ↑ 2×; "
            "**避免**与强效诱导剂同用. **DS 警示**: 治疗早期 (D1-D60) 不明原因发热、"
            "肺浸润、皮疹、外周水肿 → 立即 dexamethasone 10 mg IV q12h + 中断 Ivo "
            "至缓解. 同时关注 QT 延长 (基线 + 周 1, 2, 3, 4, 然后月度 ECG)."
        )

    # ---- Enasidenib similar warning (often co-treated with azoles) ----
    if "Enasidenib" in top3_drugs:
        warnings.append(
            "⚠ **Enasidenib differentiation syndrome (~15%) + CYP3A4 互作**: "
            "Ena 是多 CYP 通路底物 (CYP3A4 + 1A2 + 2C19). 强效抑制剂 (azole) → "
            "Ena AUC ↑ 1.5×. **DS 警示** 同 Ivo (dexamethasone + 暂停). "
            "Ena 的 QT 风险较低但仍需基线 ECG."
        )

    # ---- Anthracycline + cardiotoxic medication interactions ----
    if "Daunorubicin" in top3_drugs or "Idarubicin" in top3_drugs:
        warnings.append(
            "💉 **蒽环类心毒性 + 联合心毒性药物**: 累积剂量限制 - "
            "Daunorubicin ≤ 550 mg/m² lifetime, Idarubicin ≤ 90 mg/m². "
            "**避免**同用其他心毒性药物 (trastuzumab, 高剂量 cyclophosphamide). "
            "基线 ECHO (LVEF ≥ 50%) + 累积剂量后或剂量调整后复查. "
            "Dexrazoxane 可作为心保护剂 (Idarubicin 高累积剂量时考虑)."
        )

    return warnings


def _cautions_section(kit_out: KitOutput) -> str:
    cautions = list(kit_out.cautions or [])
    # Issue #8 — append regimen-specific drug-interaction warnings.
    interaction_warnings = _drug_interaction_warnings(kit_out)
    cautions.extend(interaction_warnings)
    if not cautions:
        return "*未检测到特殊用药禁忌。*"
    return "\n".join(f"- {c}" for c in cautions)


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------


def build_clinical_report_markdown(
    kit: KitInput,
    kit_out: KitOutput,
    dna_figure_rel_path: str | None = "dna_profile.png",
    audit_mode: bool = False,
) -> str:
    """Build a complete clinical-grade Markdown report for one patient.

    Args:
      kit, kit_out: patient input + kit output.
      dna_figure_rel_path: relative path (from the report file's directory)
        to the DNA-level summary PNG. If the file doesn't exist at render
        time, the image link simply renders as a broken image — we also
        include a short paragraph explaining what the figure depicts.
        Set to None to omit the figure entirely.
      audit_mode: kept for backwards-compat. As of v0.5 (Prospective
        Validation Phase) Layer-3 is visible BY DEFAULT but framed as
        a research output requiring institutional consent + prospective
        outcome capture. The earlier v0.4 default-hide is no longer the
        right framing — see docs/PROSPECTIVE_VALIDATION_PROTOCOL.md.
        Setting audit_mode=True is now an alias for "research engineering
        view" with extra backbone-comparison details.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    dna = kit_out.dna_summary or {}

    sections = [
        # ---- Header ----
        f"# AML 精准用药评估报告",
        f"**Patient ID**: `{kit.patient_id}`  ",
        f"**报告时间**: {timestamp}  ",
        f"**Kit 版本**: v0.5 — Prospective Validation Phase  ",
        f"**报告性质**: 临床决策辅助 + Layer-3 前瞻性验证研究 (research)  ",
        f"**View mode**: {'engineering audit' if audit_mode else 'standard prospective'}",
        "",
        # v0.5 prospective-phase banner — replaces the v0.4 demote.
        "> 🔬 **v0.5 Prospective Validation Phase**",
        ">",
        "> 本报告的 **第七节 (Layer-3 ML 组合预测)** 是研究输出, 不是临床指导。",
        "> Layer-3 在 BeatAML 2.0 内部 hold-out 上 vs 临床 CR 率 Pearson r ≈ 0.05 — ",
        "> 我们正在**前瞻性收集中国队列**验证它在真实临床场景下的相关性。",
        ">",
        "> 患者参加验证队列**完全自愿**, 不影响标准治疗(以本中心 MDT 决定为准)。",
        "> 验证完成 (target N=200, 12-month follow-up) 后再决定 Layer-3 是否进入临床流程。",
        "> 详见 `docs/PROSPECTIVE_VALIDATION_PROTOCOL.md` + IRB-approved consent。",
        "",
        # Population-validation banner — model has not been validated on
        # Chinese/Asian AML cohorts, training distribution is ~80% NA Caucasian.
        "> 🚩 **未在中国 / 亚洲 AML 队列上验证 (NOT VALIDATED ON CHINESE / ASIAN AML COHORTS)**",
        ">",
        "> 模型训练数据 BeatAML 2.0 (n=613) 主要来自 OHSU + Vizome (北美高加索人群 ≥80%)。",
        "> 中国 AML 在 t(8;21) 比例 (~15-20% vs 西方 ~7%)、APL 比例、NPM1 突变频率",
        "> 上与西方队列存在系统性差异。**本工具的 calibration 在亚裔队列上未知**，",
        "> 临床医师必须将所有推荐结果与本中心既往真实世界经验对照,不可直接采纳。",
        "",
        "---",
        "",
        # ---- Section 1: Executive Summary ----
        "## 一、临床快报 (Executive Summary)",
        "",
        _executive_summary(kit, kit_out),
        "",
        "---",
        "",
        # ---- Section 2: Patient + Specimen ----
        "## 二、患者基本信息与样本",
        "",
        "### 2.1 基本信息",
        "",
        f"- **患者 ID**: {kit.patient_id}",
        f"- **年龄**: {kit.age or '未知'} 岁",
        f"- **性别**: {'男性' if (kit.sex or '').lower() == 'male' else '女性' if (kit.sex or '').lower() == 'female' else '未知'}",
        f"- **体能状态**: {kit_out.fitness_flag}",
        f"- **疾病阶段**: {'复发/难治' if kit.is_relapse else '新诊断'}{' (继发于 MDS)' if kit.prior_mds else ''}",
        "",
        "### 2.2 化验室指标",
        "",
    ]

    # Labs
    labs = []
    if kit.wbc is not None:
        labs.append(f"- **WBC**: {kit.wbc:.1f} × 10⁹/L")
    if kit.platelet is not None:
        labs.append(f"- **血小板**: {kit.platelet:.0f} × 10⁹/L")
    if kit.hemoglobin is not None:
        labs.append(f"- **血红蛋白**: {kit.hemoglobin:.1f} g/dL")
    if kit.ldh is not None:
        labs.append(f"- **LDH**: {kit.ldh:.0f} U/L")
    if kit.blast_pct_bm is not None:
        labs.append(f"- **骨髓原始细胞比例**: {kit.blast_pct_bm:.0f}%")
    if kit.blast_pct_pb is not None:
        labs.append(f"- **外周原始细胞比例**: {kit.blast_pct_pb:.0f}%")
    sections.extend(labs if labs else ["_化验数据未提供_"])
    sections.extend(["", "---", ""])

    # ---- Section 3: Molecular Profile ----
    figure_block: list[str] = []
    if dna_figure_rel_path:
        figure_block = [
            f"![DNA-level profile: 核心基因 / 突变 / 融合 / 核型 / ELN 分层 / 可靶向性 总览]({dna_figure_rel_path})",
            "",
            "*图 3.0*: 上图为 DNA 级别分子综合图。左上显示检出的核心驱动突变（按 "
            "Tier 1-3 着色，Tier 1=FDA 批准靶向药）；右上为融合基因与核型异常汇总；"
            "左下为 25-gene 核心 panel 中未检出基因的背景参照；右下为 ELN 2017 "
            "风险分层依据高亮。**报告正文的叙事均以此图数据为基础**。",
            "",
        ]
    driver_muts_summary = dna.get("driver_mutations", []) or []
    sections.extend([
        "## 三、分子特征 (Molecular Profile)",
        "",
        *figure_block,
        "### 3.1 检出的核心驱动突变（速查表）",
        "",
        _detected_mutations_table(driver_muts_summary),
        "",
        "### 3.2 25-gene 核心 panel 覆盖情况",
        "",
        _panel_coverage_table(driver_muts_summary),
        "",
        "### 3.3 RNA-Seq 表达离群分析 (25-gene panel + 表达提示基因)",
        "",
        _rnaseq_outlier_section(kit, kit_out),
        "",
        "### 3.4 核心驱动突变 — 临床解读",
        "",
        _mutation_narrative(kit.mutations or []),
        "",
        _hotspot_codon_section(kit.mutations or []),
        "",
        "### 3.5 融合基因",
        "",
        _fusion_narrative(kit.fusions or []),
        "",
        "### 3.6 细胞遗传学",
        "",
        _cytogenetic_narrative(
            dna.get("cytogenetics", []),
            kit.karyotype_text,
        ),
        "",
        "### 3.7 ELN 风险分层 (2022 优先, 2017 对照)",
        "",
        _eln_dual_section(kit, kit_out),
        "",
        "### 3.8 WHO 2022 + ICC 2022 分类",
        "",
        _who_icc_section(kit),
        "",
        "### 3.9 Allo-SCT 推荐 + 准备 checklist",
        "",
        _allo_sct_section(kit, kit_out),
        "",
        "---",
        "",
    ])

    # ---- Section 4: Pre-induction workup checklist (issue #6) ----
    sections.extend([
        "## 四、Pre-induction Workup Checklist",
        "",
        "在启动治疗前必须完成的基线评估。条件性 (conditional) 项目根据本患者的"
        "基因型 + ELN 分层 + 体能状态自动激活。",
        "",
        _baseline_workup_section(kit, kit_out),
        "",
        "---",
        "",
    ])

    # ---- Section 5: Treatment Recommendations ----
    sections.extend([
        "## 五、治疗方案推荐",
        "",
        "以下方案按综合证据强度排序 (临床试验阶段、患者生物标志物匹配度、适应症严格度)。**每个方案的选择责任最终在主治医师**，本报告为辅助信息。",
        "",
        _regimen_section(kit_out),
        "",
        "---",
        "",
    ])

    # ---- Section 6: MRD monitoring plan (issue #7) ----
    sections.extend([
        "## 六、MRD 监测计划 (per ELN 2021)",
        "",
        _mrd_monitoring_section(kit, kit_out),
        "",
        "---",
        "",
    ])

    # ---- Section 7: Layer-3 ML prediction (Prospective Validation Phase) ----
    # v0.5 framing: Layer-3 visible by default, but explicitly framed as
    # research output requiring prospective outcome capture.
    # Internal Route B Pearson r ≈ 0.05 vs clinical CR — kit needs real
    # prospective N=200 cohort with 12-month follow-up to determine
    # whether Layer-3 enters clinical workflow. See
    # docs/PROSPECTIVE_VALIDATION_PROTOCOL.md.
    sections.extend([
        "## 七、Layer-3 ML 组合预测 (Prospective Validation — RESEARCH ONLY)",
        "",
        "> 🔬 **本节是 v0.5 前瞻性验证研究内容, 不是临床决策依据**",
        ">",
        "> Layer-3 MLP 在 BeatAML 2.0 内部 hold-out 上 vs 临床 CR 率 ",
        "> **Pearson r ≈ 0.05**(统计上无显著相关)。",
        "> 我们正在前瞻性收集中国 AML 队列(目标 N=200, 12-month follow-up)",
        "> 来验证此预测在真实临床场景下的实际相关性。",
        ">",
        "> **若本中心已加入验证队列**: kit 的 Top-1 预测会被 immutable-locked 并",
        "> 与患者实际治疗 + 6/12 月 outcome 配对存档 (per IRB-approved protocol)。",
        "> **若本中心未加入验证队列**: 本节仅供学术参考, 治疗决策请以 ",
        "> 第五节(循证方案 Layer 1)+ §3.7-3.9(ELN/WHO/ICC + Allo-SCT)为准。",
        "",
        "### 7.1 组合 AUC 预测 (Layer 3 — MLP)",
        "",
        _combo_prediction_narrative(kit_out),
        "",
        "### 7.2 克隆生物学 rationale (Layer 2 — Path A IDA coverage)",
        "",
        _clonal_coverage_narrative(kit_out),
        "",
    ])
    if audit_mode:
        sections.extend([
            "### 7.3 Engineering audit (multi-backbone comparison)",
            "",
            "> Engineering view — backbone label + Mahalanobis QC + checkpoint hash. ",
            "> This pane is for kit developers comparing MLP / Set-Transformer / "
            "Synergy-head outputs head-to-head. Not for clinicians.",
            "",
            f"- Backbone: `{(kit_out.top_combinations or [{}])[0].get('layer3_backbone', 'unknown')}`",
            f"- Top-1 combo: `{(kit_out.top_combinations or [{}])[0].get('drug1', '?')}` + "
            f"`{(kit_out.top_combinations or [{}])[0].get('drug2', '?')}`",
            "",
        ])
    sections.extend(["---", ""])

    # ---- Section 8: Cautions ----
    sections.extend([
        "## 八、用药警告与注意事项",
        "",
        _cautions_section(kit_out),
        "",
        "---",
        "",
    ])

    # ---- Section 9: QC & Limitations ----
    sections.extend([
        "## 九、质量控制与局限 (Confidence & Limitations)",
        "",
        _confidence_narrative(kit_out),
        "",
        "### 9.1 已知限制",
        "",
        "- **Panel 覆盖**: 25 个核心驱动基因 panel，未覆盖 comprehensive gene list。"
        "  如 lab 报告含其他基因，请人工结合原始报告解读。",
        "- **ELN 版本**: 本报告同时输出 ELN 2017 (训练 label 用) 与 ELN 2022 "
        "  (current standard, Döhner Blood 2022); §3.7 显示双版本结果, "
        "  不一致时给出 transition 解释。",
        "- **核型解析**: 正则启发式，覆盖约 85% 常见 ISCN 模式。"
        "  复杂/罕见核型（如 i(17q), idic(X), chromothripsis）需 cytogeneticist 人工审阅。",
        "- **组合预测**: 基于 613 BeatAML 患者的 ex-vivo drug sensitivity (AUC)，"
        "  **未经前瞻性临床验证**。AUC 预测与临床 CR 率相关性已在本 kit Route B 测试"
        "  中验证无显著相关 (Pearson ≈ 0.05)，建议以第三节试验证据为主。",
        "",
        "### 9.2 审核建议 (Checklist)",
        "",
        "请 MDT 团队核查以下项目：",
        "",
        "- [ ] ELN 2022 风险分层与院内 cytogenetics 报告一致",
        "- [ ] Pre-induction workup (§4) 全部完成",
        "- [ ] MRD 监测计划 (§6) 已纳入治疗决策时间表",
        "- [ ] 推荐方案在本地药物可及性 + 保险范围内",
        "- [ ] 患者知情同意 + 适合接受推荐强度治疗",
        "- [ ] 特殊用药警告 (TLS、QT 延长、心肝肾功能、leukostasis) 已核查",
        "- [ ] FLT3-ITD allelic ratio 与 lab 报告数值一致 (不同 lab denominator 定义可能略异)",
        "- [ ] TP53 如存在，是否已进行 allelic state (mono vs multi-hit) 判断",
        "",
        "---",
        "",
    ])

    # ---- Section 10: Methodology ----
    sections.extend([
        "## 十、方法学背景",
        "",
        "本报告由 **AML Combo-Prediction Kit v0.2** 自动生成。该工具集成三层推荐：",
        "",
        "1. **Layer 1 — Evidence-based retrieval**: "
        "20 个已发表 AML 临床试验方案数据库 (覆盖 FDA 批准 + Phase 2/3 阶段)，"
        "按患者 biomarker 驱动 + 分期 + 体能状态做严格匹配。",
        "2. **Layer 2 — Biology (Clonal Coverage × Bliss-IDA)**: "
        "基于 Palmer-Sorger Independent Drug Action 框架，"
        "将患者分解为克隆原型，对组合计算 Bliss 独立性覆盖率。",
        "3. **Layer 3 — Prediction (MLP + Mechanism Prior)**: "
        "多任务 MLP 在 BeatAML 2.0 (613 患者 × 165 药 × 55K ex-vivo 测量) 上训练，"
        "叠加 39-轴手工机制先验 (target / cell-state / regimen-role / toxicity)。",
        "",
        "**数据来源**:",
        "- BeatAML 2.0 (Tyner et al., 2018): 613 AML 患者的 RNA-Seq + NGS + ex-vivo 药敏",
        "- DrugComb v1.5: 186 AML 细胞系组合协同数据 (ALMANAC-HL60)",
        "- TCGA-LAML: 173 独立队列验证",
        "",
        "---",
        "",
    ])

    # ---- Section 11: References ----
    sections.extend([
        "## 十一、关键参考文献",
        "",
        "### 11.1 指南",
        "",
        "- Döhner H et al. **ELN 2017**. *Blood* 2017, [PMID 27895058](https://pubmed.ncbi.nlm.nih.gov/27895058)",
        "- Döhner H et al. **ELN 2022**. *Blood* 2022, [PMID 35797463](https://pubmed.ncbi.nlm.nih.gov/35797463)",
        "- Heuser M et al. **ELN 2021 MRD consensus**. *Blood* 2021, [PMID 33591443](https://pubmed.ncbi.nlm.nih.gov/33591443)",
        "- Khoury JD et al. **WHO 2022 Hematolymphoid Classification**. *Leukemia* 2022, [PMID 35732831](https://pubmed.ncbi.nlm.nih.gov/35732831)",
        "- NCCN Clinical Practice Guidelines in Oncology: AML v2.2024",
        "",
        "### 11.2 关键临床试验",
        "",
        "- Stone RM et al. **RATIFY** (Mid + 7+3). *NEJM* 2017, [PMID 28644114](https://pubmed.ncbi.nlm.nih.gov/28644114)",
        "- DiNardo CD et al. **VIALE-A** (Ven + Aza). *NEJM* 2020, [PMID 32813947](https://pubmed.ncbi.nlm.nih.gov/32813947)",
        "- Perl AE et al. **ADMIRAL** (Gilteritinib mono R/R). *NEJM* 2019, [PMID 31665578](https://pubmed.ncbi.nlm.nih.gov/31665578)",
        "- Montesinos P et al. **AGILE** (Aza + Ivo IDH1-mut). *NEJM* 2022, [PMID 35443106](https://pubmed.ncbi.nlm.nih.gov/35443106)",
        "- Short NJ, Daver N et al. **Aza + Ven + Gilt triplet**. *JCO* 2024, [PMID 38277619](https://pubmed.ncbi.nlm.nih.gov/38277619)",
        "- Erba HP et al. **QUANTUM-First** (Quiz + 7+3). *Lancet* 2023, [PMID 37116523](https://pubmed.ncbi.nlm.nih.gov/37116523)",
        "- Ivey A et al. **NPM1 MRD predicts relapse**. *NEJM* 2016, [PMID 26789727](https://pubmed.ncbi.nlm.nih.gov/26789727)",
        "",
        "### 11.3 方法学",
        "",
        "- Palmer AC, Sorger PK. **Independent Drug Action**. *Cancer Discov* 2022, [PMID 34983746](https://pubmed.ncbi.nlm.nih.gov/34983746)",
        "- Julkunen H et al. **comboFM: Multi-way drug combination prediction**. *Nat Commun* 2020, [PMID 33262326](https://pubmed.ncbi.nlm.nih.gov/33262326)",
        "- Li MM et al. **AMP/ASCO/CAP Standards for Somatic Variant Interpretation**. *J Mol Diagn* 2017, [PMID 27993330](https://pubmed.ncbi.nlm.nih.gov/27993330)",
        "",
        "---",
        "",
        "## 免责声明",
        "",
        "本报告由 AML Combo-Prediction Kit v0.2 自动生成。所有推荐为**研究辅助性质**，",
        "不构成诊断或医疗建议。最终治疗决策必须由具备执业资质的血液肿瘤专科医师",
        "结合全面临床评估做出。报告中引用的文献与 FDA 标签信息可能随时间更新，",
        "请以最新版本为准。",
        "",
        "---",
        "",
        f"*Generated by AML Combo-Prediction Kit v0.2 on {timestamp}*  ",
        f"*For research use only · Not for clinical diagnosis*  ",
        f"*Full reading guide: `docs/clinical_reader_guide.md`*",
    ])

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------


_HTML_CSS = """
body {
  font-family: -apple-system, "Helvetica Neue", "PingFang SC",
               "Microsoft YaHei", sans-serif;
  max-width: 860px;
  margin: 2em auto;
  padding: 0 2em;
  line-height: 1.65;
  color: #222;
}
h1 { border-bottom: 3px solid #1a5490; padding-bottom: 0.3em; color: #1a5490; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.2em; margin-top: 2em;
     color: #1a5490; }
h3 { color: #333; margin-top: 1.5em; }
h4 { color: #555; }
hr { border: 0; border-top: 1px solid #ddd; margin: 1.5em 0; }
code { background: #f4f4f4; padding: 0.1em 0.3em; border-radius: 3px;
       font-size: 90%; }
pre { background: #f4f4f4; padding: 1em; border-radius: 5px; overflow-x: auto; }
blockquote { border-left: 4px solid #1a5490; padding-left: 1em;
             color: #555; margin-left: 0; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #ccc; padding: 0.4em 0.8em; }
th { background: #f0f0f0; }
a { color: #1a5490; text-decoration: none; }
a:hover { text-decoration: underline; }
ul, ol { padding-left: 1.5em; }
li { margin: 0.2em 0; }
.footnote { font-size: 90%; color: #666; }
img { max-width: 100%; height: auto; display: block;
      margin: 1em auto; page-break-inside: avoid; }
figure { margin: 1em 0; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 90%; color: #555; margin-top: 0.3em; }
@media print {
  body { max-width: none; margin: 0; }
  h1, h2 { page-break-after: avoid; }
  img { max-width: 100%; max-height: 90vh; }
  @page { size: letter; margin: 1.5cm; }
}
"""


def render_markdown_to_html(
    md_path: Path | str,
    html_path: Path | str,
    pandoc_bin: str = "pandoc",
) -> str:
    """Convert Markdown → standalone HTML via pandoc (always available).

    HTML is a universal fallback: any browser opens it, and the user can
    hit Cmd-P / Ctrl-P → Save as PDF from the browser.
    """
    md_path = Path(md_path)
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSS to a sibling file to inline into the HTML
    css_path = html_path.parent / "_report_style.css"
    css_path.write_text(_HTML_CSS, encoding="utf-8")

    result = subprocess.run(
        [pandoc_bin, str(md_path), "-o", str(html_path),
         "--standalone",
         "--metadata", "title=AML 精准用药评估报告",
         "--css", css_path.name,
         "--toc", "--toc-depth=2"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pandoc HTML render failed: {result.stderr}")
    return str(html_path)


_CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
]


def _find_chrome_binary() -> str | None:
    """Locate a headless-capable Chromium-family browser binary, or None."""
    for cand in _CHROME_CANDIDATES:
        if Path(cand).exists():
            return cand
    # PATH-based lookup
    for name in ("google-chrome", "chromium", "chromium-browser"):
        probe = subprocess.run(["which", name], capture_output=True, text=True)
        if probe.returncode == 0 and probe.stdout.strip():
            return probe.stdout.strip()
    return None


def _render_via_chrome(html_path: Path, pdf_path: Path) -> str:
    """Use headless Chrome/Chromium to print HTML → PDF. Most reliable for CJK."""
    chrome = _find_chrome_binary()
    if not chrome:
        raise FileNotFoundError("No Chrome/Chromium binary found on this system")
    result = subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         "--no-sandbox", f"--print-to-pdf={pdf_path}",
         f"file://{html_path.resolve()}"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"chrome print-to-pdf failed: {result.stderr[:300]}")
    if not pdf_path.exists():
        raise RuntimeError("chrome print-to-pdf produced no output file")
    return str(pdf_path)


def render_markdown_to_pdf(
    md_path: Path | str,
    pdf_path: Path | str,
    pandoc_bin: str = "pandoc",
) -> str:
    """Convert Markdown → PDF via the first available engine.

    Engine chain (best for rich HTML + CJK first):
      1. Headless Chrome/Chromium   — renders the styled HTML, most reliable
      2. xelatex / pdflatex         — classic pandoc LaTeX, needs MacTeX installed
      3. wkhtmltopdf                — standalone Qt-WebKit binary
      4. weasyprint                 — Python native (needs pango dylibs on PATH)

    Raises RuntimeError with actionable install hint if all fail.
    """
    md_path = Path(md_path)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    tried: list[str] = []
    last_err = None

    # --- Preferred: headless Chrome via the pre-rendered HTML ---
    html_path = md_path.with_suffix(".html")
    if not html_path.exists():
        try:
            render_markdown_to_html(md_path, html_path, pandoc_bin=pandoc_bin)
        except Exception as e:
            last_err = f"HTML prep for chrome failed: {e}"
    if html_path.exists():
        try:
            return _render_via_chrome(html_path, pdf_path)
        except Exception as e:
            tried.append("chrome")
            last_err = str(e)

    # --- pandoc-based engines ---
    engines = ["xelatex", "pdflatex", "wkhtmltopdf", "weasyprint"]
    for engine in engines:
        probe = subprocess.run(["which", engine], capture_output=True, text=True)
        if probe.returncode != 0:
            continue
        tried.append(engine)
        try:
            args = [pandoc_bin, str(md_path), "-o", str(pdf_path),
                    f"--pdf-engine={engine}", "--standalone"]
            if engine in ("xelatex", "pdflatex"):
                args += ["--variable", "geometry:margin=2cm",
                         "--variable", "fontsize=11pt",
                         "--variable", "mainfont=Helvetica",
                         "--variable", "CJKmainfont=PingFang SC",
                         "--variable", "colorlinks=true",
                         "--variable", "linkcolor=blue"]
            env = _weasyprint_env() if engine == "weasyprint" else None
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=180, env=env,
            )
            if result.returncode == 0:
                return str(pdf_path)
            last_err = result.stderr[:300]
        except subprocess.TimeoutExpired:
            last_err = f"{engine}: timed out after 180s"
            continue

    raise RuntimeError(
        f"No PDF engine available. Tried: {tried or ['<none>']}. "
        f"Last error: {last_err}\n"
        f"Options: install Google Chrome, or run "
        f"`brew install --cask mactex` (xelatex), "
        f"`brew install wkhtmltopdf`, or `pip install weasyprint`. "
        f"The HTML file at {html_path} is already fully styled and can be "
        f"opened in any browser, then printed to PDF (Cmd-P → Save as PDF)."
    )


def export_clinical_report(
    kit: KitInput,
    kit_out: KitOutput,
    out_dir: Path | str,
    also_render_pdf: bool = True,
    also_render_html: bool = True,
    audit_mode: bool = False,
) -> dict[str, str]:
    """One-shot: generate Markdown + HTML + (optional) PDF + return paths.

    Args:
      audit_mode: pass-through to build_clinical_report_markdown — when
        True, Layer-3 ML predictions appear in the report (engineering
        audit only). Default False — Layer-3 collapsed per clinical
        reviewer concern (Pearson r ≈ 0.05 vs CR creates anchoring).

    Returns:
      {"markdown": ..., "html": ... or None, "pdf": ... or None,
       "pdf_error": ... (only on failure)}

    HTML is the reliable fallback: always produced, opens in any browser,
    user can Cmd-P → "Save as PDF" if the native PDF engine is unavailable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = build_clinical_report_markdown(kit, kit_out, audit_mode=audit_mode)
    md_path = out_dir / "clinical_report.md"
    md_path.write_text(md, encoding="utf-8")

    paths: dict[str, str] = {"markdown": str(md_path)}

    if also_render_html:
        html_path = out_dir / "clinical_report.html"
        try:
            render_markdown_to_html(md_path, html_path)
            paths["html"] = str(html_path)
        except Exception as e:
            paths["html"] = ""
            paths["html_error"] = str(e)

    if also_render_pdf:
        pdf_path = out_dir / "clinical_report.pdf"
        try:
            render_markdown_to_pdf(md_path, pdf_path)
            paths["pdf"] = str(pdf_path)
        except Exception as e:
            # Don't fail the whole pipeline if PDF rendering fails;
            # the Markdown + HTML are still useful.
            paths["pdf"] = ""
            paths["pdf_error"] = str(e)

    return paths
