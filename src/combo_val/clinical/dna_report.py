"""Per-patient DNA-level profile report — core genes, core mutations,
and their clinical implications.

Consolidates the patient's mutation panel + FLT3 detail + CEBPA biallelic
flag + fusions + karyotype flags into a structured report the clinical team
can cross-reference against NGS lab output. Tables-first, not prose.

Knowledge base below is curated from:
  - ELN 2017 risk stratification (Döhner et al. Blood 2017)
  - WHO 2022 / ICC 2022 AML classification
  - NCCN Guidelines for AML v2.2024
  - FDA drug labels for all targeted AML therapies

Output: a dict structured for easy JSON serialization + a string pretty-
printer. Exposed via KitOutput.dna_summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from combo_val.clinical.kit_schema import KitInput, MutationCall


# ---------------------------------------------------------------------------
# Knowledge base: per-gene clinical interpretation
# ---------------------------------------------------------------------------

# Core 25-gene AML driver panel (matches patient_features mut_* columns)
# Ordered by clinical actionability tier.
CORE_DRIVER_GENES: tuple[str, ...] = (
    # Tier 1 — FDA-approved targeted therapy exists
    "FLT3", "IDH1", "IDH2", "KMT2A",
    # Tier 2 — strong prognostic / informs therapy intensity
    "NPM1", "TP53", "RUNX1", "ASXL1", "CEBPA",
    # Tier 3 — background / co-mutation / subtype
    "DNMT3A", "TET2", "KIT",
    "NRAS", "KRAS", "PTPN11",  # RAS-MAPK
    "WT1", "BCOR",
    "STAG2", "PHF6",
    "SRSF2", "SF3B1", "U2AF1", "EZH2",  # splice/MDS-related
    "MECOM", "CBFB",  # fusion partners
)

# Gene-level clinical annotation. Empty {} for genes with no specific
# targeted therapy; caller should still list them in the report.
GENE_IMPLICATIONS: dict[str, dict] = {
    "FLT3": {
        "tier": 1,
        "targetable_by": ["Midostaurin", "Quizartinib", "Gilteritinib", "Sorafenib", "Crenolanib"],
        "eln_adverse_if": "ITD positive with high allelic ratio (≥0.5) AND no NPM1 mutation",
        "eln_favorable_if": "ITD negative OR ITD with low AR AND NPM1 mutation present",
        "notes": "ITD ≠ TKD — verify variant type. Allelic ratio ≥0.5 triggers RATIFY / QUANTUM-First intensive triplet eligibility.",
    },
    "IDH1": {
        "tier": 1,
        "targetable_by": ["Ivosidenib", "Olutasidenib"],
        "fda_combo": "Azacytidine + Ivosidenib (AGILE trial, newly dx 2022)",
        "notes": "R132 hotspot. Ivosidenib monotherapy for R/R; Aza+IVO for newly dx unfit.",
    },
    "IDH2": {
        "tier": 1,
        "targetable_by": ["Enasidenib"],
        "notes": "R140 / R172 hotspots. Enasidenib FDA 2017 for R/R.",
    },
    "KMT2A": {
        "tier": 1,
        "targetable_by": ["Revumenib", "Ziftomenib"],
        "notes": "KMT2A-rearranged (fusion) → menin inhibitor eligible. MLL point mutations rare.",
    },
    "NPM1": {
        "tier": 2,
        "targetable_by": ["Revumenib (research)"],
        "eln_favorable_if": "Isolated or with FLT3-ITD low allelic ratio",
        "notes": "Most common mutation in AML. Favorable as monotherapy-response modifier.",
    },
    "TP53": {
        "tier": 2,
        "targetable_by": [],
        "eln_adverse": True,
        "notes": "ADVERSE. Conventional induction poorly effective. Trial enrollment + SCT priority.",
    },
    "RUNX1": {
        "tier": 2,
        "targetable_by": [],
        "eln_adverse": True,
        "notes": "ADVERSE by ELN 2017 (partial — context-dependent in ICC 2022).",
    },
    "ASXL1": {
        "tier": 2,
        "targetable_by": [],
        "eln_adverse": True,
        "notes": "ADVERSE. Often co-mutates with SRSF2 in MDS-AML.",
    },
    "CEBPA": {
        "tier": 2,
        "targetable_by": [],
        "eln_favorable_if": "BIALLELIC (both alleles mutated); monoallelic is NOT favorable",
        "notes": "Distinguish biallelic vs monoallelic — matters for risk stratification.",
    },
    "DNMT3A": {
        "tier": 3,
        "notes": "Common background in AML; moderate prognostic signal. HMA (aza/dec) synergy reported.",
    },
    "TET2": {
        "tier": 3,
        "notes": "Epigenetic. HMA response predictor (weak).",
    },
    "KIT": {
        "tier": 3,
        "notes": "c-KIT. When comut with CBFB/RUNX1-RUNX1T1, worsens prognosis. Dasatinib research use.",
    },
    "NRAS": {"tier": 3, "notes": "RAS-MAPK pathway activation. MEKi (Trametinib) research."},
    "KRAS": {"tier": 3, "notes": "RAS-MAPK pathway activation. Less common than NRAS in AML."},
    "PTPN11": {"tier": 3, "notes": "Activates RAS-MAPK. Adverse in some cohorts."},
    "WT1": {"tier": 3, "notes": "Less common; some evidence of MRD tracking utility."},
    "BCOR": {"tier": 3, "notes": "Adverse. Often co-mutates with TET2 or DNMT3A."},
    "STAG2": {"tier": 3, "notes": "Cohesin complex. MDS-AML context."},
    "PHF6": {"tier": 3, "notes": "X-linked; more common in male AML."},
    "SRSF2": {"tier": 3, "notes": "Splicing. MDS/MDS-AML adverse feature."},
    "SF3B1": {"tier": 3, "notes": "Splicing. Usually MDS context. Adverse if de novo AML."},
    "U2AF1": {"tier": 3, "notes": "Splicing. Adverse in AML."},
    "EZH2": {"tier": 3, "notes": "Polycomb. Adverse; tazemetostat research."},
    "MECOM": {"tier": 3, "notes": "Usually via inv(3)/t(3;3). ADVERSE; uniformly poor prognosis."},
    "CBFB": {"tier": 3, "notes": "Usually via inv(16) fusion with MYH11. FAVORABLE — CBF-AML."},
}


FUSION_IMPLICATIONS: dict[str, dict] = {
    "PML-RARA": {
        "eln": "Favorable (special — APL category)",
        "regimen": "ATRA + ATO mandatory (first-line APML4 / LPA regimen)",
        "notes": "**7+3 is INAPPROPRIATE** for PML-RARA. ATRA + ATO is curative in >90%.",
    },
    "KMT2A-r": {
        "eln": "Adverse (historically) / context in ICC 2022",
        "regimen": "Menin inhibitor (Revumenib) candidate",
        "notes": "KMT2A-MLLT3 t(9;11) is Intermediate; other KMT2A fusions are Adverse.",
    },
    "CBFB-MYH11": {
        "eln": "Favorable (core-binding factor AML)",
        "regimen": "Standard 7+3 with Gemtuzumab ozogamicin (GO) add-on",
        "notes": "inv(16) / t(16;16). Excellent prognosis with chemo+GO.",
    },
    "RUNX1-RUNX1T1": {
        "eln": "Favorable (core-binding factor AML)",
        "regimen": "Standard 7+3 with GO add-on",
        "notes": "t(8;21). Excellent prognosis if treated intensively.",
    },
}


KARYO_IMPLICATIONS: dict[str, dict] = {
    "karyo_complex": {
        "eln": "Adverse",
        "notes": "Complex karyotype (≥3 abnormalities). Poor prognosis regardless of other features.",
    },
    "karyo_monosomy_5_or_7": {
        "eln": "Adverse",
        "notes": "Monosomy 5/7 or del(5q)/del(7q). MDS-related. Vyxeos (CPX-351) eligible.",
    },
    "karyo_del_17p": {
        "eln": "Adverse (usually with TP53)",
        "notes": "17p deletion affects TP53 locus. Conventional induction poorly effective.",
    },
}


# ---------------------------------------------------------------------------
# Building the DNA summary
# ---------------------------------------------------------------------------


def _mutation_detail(mut: MutationCall) -> dict:
    """Per-mutation structured detail with clinical interpretation."""
    gene = mut.gene.upper()
    info = GENE_IMPLICATIONS.get(gene, {})

    # FLT3-specific handling
    flt3_extra = {}
    if gene == "FLT3":
        if mut.is_ITD:
            flt3_extra["variant_type"] = "ITD"
            if mut.allelic_ratio is not None:
                flt3_extra["allelic_ratio"] = round(mut.allelic_ratio, 3)
                flt3_extra["ar_interpretation"] = (
                    "HIGH (≥0.5) — adverse modifier unless NPM1-mut"
                    if mut.allelic_ratio >= 0.5
                    else "LOW (<0.5) — less adverse impact"
                )
        elif mut.is_TKD:
            flt3_extra["variant_type"] = "TKD"
        else:
            flt3_extra["variant_type"] = mut.variant_type or "unspecified"

    cebpa_extra = {}
    if gene == "CEBPA":
        cebpa_extra["biallelic"] = bool(mut.is_biallelic)
        cebpa_extra["allelic_interpretation"] = (
            "BIALLELIC → favorable ELN risk"
            if mut.is_biallelic
            else "monoallelic → NOT automatically favorable (verify second hit)"
        )

    return {
        "gene": gene,
        "variant_type": mut.variant_type,
        "vaf": round(mut.vaf, 3) if mut.vaf is not None else None,
        **flt3_extra,
        **cebpa_extra,
        "tier": info.get("tier"),
        "targetable_by": info.get("targetable_by", []),
        "eln_implication": info.get("eln_adverse_if") or info.get("eln_favorable_if"),
        "is_adverse_driver": info.get("eln_adverse") is True,
        "notes": info.get("notes", ""),
    }


def _targetability_map(patient_mutations: list[MutationCall],
                      fusions: list[str]) -> dict[str, list[str]]:
    """Which drug classes are indicated for this patient's genetic profile?"""
    targetable: dict[str, list[str]] = {}
    fus_up = {f.upper() for f in (fusions or [])}

    for mut in patient_mutations:
        g = mut.gene.upper()
        info = GENE_IMPLICATIONS.get(g, {})
        drugs = info.get("targetable_by", [])
        if drugs:
            targetable.setdefault(g, []).extend(drugs)

    # Fusion-driven additions
    if any("PML-RARA" in f for f in fus_up):
        targetable["PML-RARA"] = ["ATRA", "ATO (arsenic trioxide)", "GO (optional)"]
    if any("CBFB-MYH11" in f for f in fus_up):
        targetable["CBFB-MYH11"] = ["7+3 with GO add-on", "Gemtuzumab ozogamicin"]
    if any("RUNX1-RUNX1T1" in f for f in fus_up):
        targetable["RUNX1-RUNX1T1"] = ["7+3 with GO add-on", "Gemtuzumab ozogamicin"]

    # Universal BCL2 dependency — venetoclax indicated in unfit or post-HMA
    targetable["universal"] = ["Venetoclax (combination)"]

    return targetable


def build_dna_summary(kit: KitInput, computed_eln: str) -> dict:
    """Build the per-patient DNA-level report dict.

    Returns a dict with sections:
      - driver_mutations: list of per-mutation detail rows
      - fusion_analysis: list of fusion interpretations
      - cytogenetics: list of karyo flag interpretations
      - eln_risk: computed + rationale
      - targetability: drug class lookup per finding
      - sample_qc: coverage / VAF summary
    """
    mutations = kit.mutations or []
    fusions = kit.fusions or []

    # Organize mutations into tiers
    mut_details = [_mutation_detail(m) for m in mutations]
    mut_details.sort(key=lambda r: (r.get("tier") or 99, r["gene"]))

    # Fusion analysis
    fusion_rows = []
    for f in fusions:
        f_up = f.upper()
        info = None
        for known_key, known_info in FUSION_IMPLICATIONS.items():
            if known_key.upper() in f_up:
                info = known_info
                break
        fusion_rows.append({
            "fusion": f,
            "eln_implication": info["eln"] if info else "unknown",
            "regimen_note": info["regimen"] if info else "",
            "notes": info["notes"] if info else "Uncommon fusion — check literature.",
        })

    # Cytogenetics — derived from karyotype_text via karyo parser
    from combo_val.clinical.karyotype_parser import parse_karyotype
    k = parse_karyotype(kit.karyotype_text)
    k_flags = k.as_dict()
    cytogenetics_rows = [
        {
            "finding": "Complex karyotype",
            "present": bool(k_flags["karyo_complex"]),
            "interpretation": KARYO_IMPLICATIONS["karyo_complex"]["notes"]
                if k_flags["karyo_complex"] else "Not present",
        },
        {
            "finding": "Monosomy 5/7 or del(5q)/del(7q)",
            "present": bool(k_flags["karyo_monosomy_5_or_7"]),
            "interpretation": KARYO_IMPLICATIONS["karyo_monosomy_5_or_7"]["notes"]
                if k_flags["karyo_monosomy_5_or_7"] else "Not present",
        },
        {
            "finding": "del(17p) / TP53 locus",
            "present": bool(k_flags["karyo_del_17p"]),
            "interpretation": KARYO_IMPLICATIONS["karyo_del_17p"]["notes"]
                if k_flags["karyo_del_17p"] else "Not present",
        },
        {
            "finding": "Normal karyotype",
            "present": bool(k_flags["karyo_normal"]),
            "interpretation": "46,XX or 46,XY, no abnormalities"
                if k_flags["karyo_normal"] else "",
        },
    ]

    # Targetability map (which drug classes are clinically indicated?)
    targetability = _targetability_map(mutations, fusions)

    # Sample QC summary
    n_called = len(mutations)
    n_with_vaf = sum(1 for m in mutations if m.vaf is not None)
    vaf_above_threshold = sum(1 for m in mutations if m.vaf is not None and m.vaf >= 0.20)
    sample_qc = {
        "n_mutations_called": n_called,
        "n_with_vaf_annotation": n_with_vaf,
        "n_above_vaf_0_20": vaf_above_threshold,
        "karyotype_parsed": bool(kit.karyotype_text and kit.karyotype_text.strip()),
        "fusions_reported": len(fusions),
    }

    return {
        "driver_mutations": mut_details,
        "fusion_analysis": fusion_rows,
        "cytogenetics": cytogenetics_rows,
        "eln_risk": {
            "category": computed_eln,
            "source": "computed via eln_computer.compute_eln2017 (ELN 2017 rules)",
        },
        "targetability": targetability,
        "sample_qc": sample_qc,
    }


def export_dna_summary_csv(summary: dict, patient_id: str, out_dir: Path | str) -> dict:
    """Write per-section CSV tables + a unified JSON file.

    out_dir/patient_<id>/
      ├── driver_mutations.csv       ← the core "table" clinicians care about
      ├── fusion_analysis.csv
      ├── cytogenetics.csv
      ├── targetability.csv
      ├── sample_qc.csv
      └── dna_summary.json            ← full structured dict
    """
    import json
    import pandas as pd
    out = Path(out_dir) / f"patient_{patient_id}"
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # Driver mutations table — the main "DNA-level predictions 表"
    if summary.get("driver_mutations"):
        mdf = pd.DataFrame(summary["driver_mutations"])
        # Order columns for human readability
        priority = ["gene", "variant_type", "vaf", "tier", "allelic_ratio",
                    "ar_interpretation", "biallelic", "allelic_interpretation",
                    "targetable_by", "eln_implication", "is_adverse_driver", "notes"]
        cols = [c for c in priority if c in mdf.columns] + \
               [c for c in mdf.columns if c not in priority]
        mdf = mdf[cols]
        # Serialize list columns as semicolon-joined strings for Excel readability
        for c in mdf.columns:
            if mdf[c].apply(lambda v: isinstance(v, list)).any():
                mdf[c] = mdf[c].apply(
                    lambda v: "; ".join(map(str, v)) if isinstance(v, list) else v
                )
        p = out / "driver_mutations.csv"
        mdf.to_csv(p, index=False)
        paths["driver_mutations"] = str(p)

    # Fusions
    if summary.get("fusion_analysis"):
        p = out / "fusion_analysis.csv"
        pd.DataFrame(summary["fusion_analysis"]).to_csv(p, index=False)
        paths["fusion_analysis"] = str(p)

    # Cytogenetics
    if summary.get("cytogenetics"):
        p = out / "cytogenetics.csv"
        pd.DataFrame(summary["cytogenetics"]).to_csv(p, index=False)
        paths["cytogenetics"] = str(p)

    # Targetability — pivot dict → long-form table
    targ = summary.get("targetability", {})
    if targ:
        rows = []
        for finding, drugs in targ.items():
            for drug in drugs:
                rows.append({"finding": finding, "indicated_drug_class": drug})
        p = out / "targetability.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        paths["targetability"] = str(p)

    # Sample QC
    qc = summary.get("sample_qc", {})
    if qc:
        p = out / "sample_qc.csv"
        pd.DataFrame([qc]).to_csv(p, index=False)
        paths["sample_qc"] = str(p)

    # Full JSON
    p = out / "dna_summary.json"
    # Sanitize non-serializable values
    def _safe(x):
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if isinstance(x, (list, tuple)):
            return [_safe(v) for v in x]
        if isinstance(x, dict):
            return {k: _safe(v) for k, v in x.items()}
        return str(x)
    p.write_text(json.dumps(_safe(summary), indent=2, ensure_ascii=False),
                 encoding="utf-8")
    paths["dna_summary_json"] = str(p)
    paths["_output_dir"] = str(out)
    return paths


def render_dna_summary_figure(
    summary: dict,
    patient_id: str,
    out_path: Path | str,
    dpi: int = 150,
) -> str:
    """Render a publication-quality figure of the DNA profile.

    Creates a 2x2 grid: driver mutations table, fusion+cytogenetics,
    targetability summary, ELN+QC summary. Saves PNG at out_path.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Patient {patient_id} — DNA-Level Profile\n"
        f"ELN 2017 Risk: {summary.get('eln_risk', {}).get('category', '?')}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.15)

    # ---------- Panel A: Driver mutations table (top-left, spans wide) ----------
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.axis("off")
    ax_a.set_title("A. Driver mutations (25-gene core panel)", loc="left",
                    fontweight="bold", fontsize=11)
    muts = summary.get("driver_mutations", [])
    if muts:
        rows = []
        for m in muts:
            gene = m["gene"]
            variant = m.get("variant_type") or ""
            if gene == "FLT3" and m.get("allelic_ratio") is not None:
                variant = f"{m.get('variant_type', 'ITD')} AR={m['allelic_ratio']:.2f}"
            elif gene == "CEBPA" and m.get("biallelic"):
                variant = "biallelic"
            vaf = f"{m['vaf']:.2f}" if m.get("vaf") is not None else "—"
            tier = f"T{m['tier']}" if m.get("tier") else "—"
            targetable = ", ".join(m.get("targetable_by", [])[:3]) or "—"
            rows.append([gene, variant, vaf, tier, targetable])
        table = ax_a.table(
            cellText=rows,
            colLabels=["Gene", "Variant", "VAF", "Tier", "Targetable (top 3)"],
            cellLoc="left", colLoc="left",
            loc="upper left",
            colWidths=[0.10, 0.20, 0.08, 0.06, 0.56],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.6)
        # Color header
        for j in range(5):
            table[(0, j)].set_facecolor("#4A6FA5")
            table[(0, j)].get_text().set_color("white")
            table[(0, j)].get_text().set_fontweight("bold")
        # Color tier-1 rows
        for i, m in enumerate(muts, start=1):
            if m.get("tier") == 1:
                for j in range(5):
                    table[(i, j)].set_facecolor("#E8F4EA")
            elif m.get("is_adverse_driver"):
                for j in range(5):
                    table[(i, j)].set_facecolor("#FDECEA")
    else:
        ax_a.text(0.5, 0.5, "No driver mutations called",
                  ha="center", va="center", fontsize=11, style="italic")

    # ---------- Panel B: Fusion + Cytogenetics (bottom-left) ----------
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.axis("off")
    ax_b.set_title("B. Fusions + cytogenetics", loc="left",
                    fontweight="bold", fontsize=11)
    y = 0.95
    fusions = summary.get("fusion_analysis", [])
    if fusions:
        ax_b.text(0.0, y, "Fusions:", fontweight="bold", fontsize=10,
                   transform=ax_b.transAxes)
        y -= 0.08
        for f in fusions:
            ax_b.text(0.04, y, f"• {f['fusion']} → {f.get('eln_implication', '?')}",
                       fontsize=9, transform=ax_b.transAxes)
            y -= 0.06
    else:
        ax_b.text(0.0, y, "Fusions: none detected", fontsize=10,
                   color="gray", transform=ax_b.transAxes)
        y -= 0.08
    y -= 0.04
    ax_b.text(0.0, y, "Cytogenetics:", fontweight="bold", fontsize=10,
               transform=ax_b.transAxes)
    y -= 0.08
    for c in summary.get("cytogenetics", []):
        mark = "✓" if c["present"] else "✗"
        color = "#B71C1C" if c["present"] and "Normal" not in c["finding"] else \
                "#2E7D32" if c["present"] else "gray"
        ax_b.text(0.04, y, f"[{mark}] {c['finding']}", fontsize=9,
                   color=color, transform=ax_b.transAxes)
        y -= 0.06

    # ---------- Panel C: Targetability (bottom-right) ----------
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.axis("off")
    ax_c.set_title("C. Targetable drug classes", loc="left",
                    fontweight="bold", fontsize=11)
    y = 0.95
    for finding, drugs in summary.get("targetability", {}).items():
        ax_c.text(0.0, y, f"{finding}:", fontweight="bold", fontsize=9,
                   transform=ax_c.transAxes)
        y -= 0.06
        for d in drugs[:3]:
            ax_c.text(0.05, y, f"→ {d}", fontsize=9, transform=ax_c.transAxes)
            y -= 0.05
        if len(drugs) > 3:
            ax_c.text(0.05, y, f"  (+{len(drugs) - 3} more)",
                       fontsize=8, style="italic", color="gray",
                       transform=ax_c.transAxes)
            y -= 0.05
        y -= 0.03

    # Sample QC footer
    qc = summary.get("sample_qc", {})
    fig.text(0.5, 0.02,
              f"Sample QC: {qc.get('n_mutations_called', 0)} mutations called · "
              f"{qc.get('n_above_vaf_0_20', 0)} above VAF 0.20 · "
              f"karyotype parsed: {qc.get('karyotype_parsed', False)} · "
              f"fusions: {qc.get('fusions_reported', 0)}",
              ha="center", fontsize=8, color="gray")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def pretty_print_dna_summary(summary: dict) -> str:
    """Clinician-facing table format."""
    lines = ["║", "║ DNA-LEVEL PROFILE (core genes, mutations, targetability)"]

    # Driver mutations table
    muts = summary.get("driver_mutations", [])
    if muts:
        lines.append("║ ┌─ DRIVER MUTATIONS (25-gene core panel) ─────────────────────────────────┐")
        lines.append(f"║ │ {'Gene':<8s} {'Variant':<16s} {'VAF':>6s}  {'Tier':>4s}  {'Targetable':<30s} │")
        lines.append("║ ├" + "─" * 76 + "┤")
        for m in muts:
            gene = m["gene"]
            variant = m.get("variant_type") or ""
            if gene == "FLT3" and m.get("allelic_ratio") is not None:
                variant = f"{m.get('variant_type', 'ITD')} AR={m['allelic_ratio']:.2f}"
            if gene == "CEBPA" and m.get("biallelic"):
                variant = "biallelic"
            vaf = f"{m['vaf']:.2f}" if m.get("vaf") is not None else "—"
            tier = f"T{m['tier']}" if m.get("tier") else "—"
            targetable = ", ".join(m.get("targetable_by", [])[:2]) or "—"
            if len(targetable) > 30:
                targetable = targetable[:27] + "..."
            lines.append(f"║ │ {gene:<8s} {variant:<16s} {vaf:>6s}  {tier:>4s}  {targetable:<30s} │")
        lines.append("║ └" + "─" * 76 + "┘")
    else:
        lines.append("║   No driver mutations called.")

    # Fusions
    fus = summary.get("fusion_analysis", [])
    if fus:
        lines.append("║")
        lines.append("║ FUSION ANALYSIS")
        for f in fus:
            lines.append(f"║   • {f['fusion']:<20s} ELN: {f['eln_implication']}")
            if f.get("regimen_note"):
                lines.append(f"║     Regimen note: {f['regimen_note']}")
    else:
        lines.append("║ FUSION ANALYSIS: no fusions reported")

    # Cytogenetics
    lines.append("║")
    lines.append("║ CYTOGENETIC FLAGS")
    for c in summary.get("cytogenetics", []):
        mark = "✓" if c["present"] else "✗"
        lines.append(f"║   [{mark}] {c['finding']:<36s}")

    # Targetability summary
    lines.append("║")
    lines.append("║ TARGETABILITY (which drug classes are indicated)")
    for finding, drugs in summary.get("targetability", {}).items():
        top_drugs = ", ".join(drugs[:3])
        if len(drugs) > 3:
            top_drugs += "..."
        lines.append(f"║   {finding:<18s} → {top_drugs}")

    # ELN
    eln = summary.get("eln_risk", {})
    lines.append("║")
    lines.append(f"║ ELN 2017 RISK:  {eln.get('category', '?')}")

    # Sample QC
    qc = summary.get("sample_qc", {})
    if qc:
        lines.append("║")
        lines.append("║ SAMPLE QC (DNA level)")
        lines.append(f"║   Mutations called:       {qc.get('n_mutations_called', 0)}")
        lines.append(f"║   With VAF ≥ 0.20:        {qc.get('n_above_vaf_0_20', 0)} / {qc.get('n_mutations_called', 0)}")
        lines.append(f"║   Karyotype parsed:       {qc.get('karyotype_parsed', False)}")
        lines.append(f"║   Fusions reported:       {qc.get('fusions_reported', 0)}")

    return "\n".join(lines)
