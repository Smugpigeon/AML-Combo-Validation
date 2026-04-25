"""ELN 2022 risk stratification (Döhner et al., Blood 2022, PMID 35797463).

Key differences vs ELN 2017 (issue #4):

1. **FLT3-ITD allelic ratio is no longer used** for stratification.
   - NPM1mut + FLT3-ITD (any AR) → Intermediate
     (was conditional on AR < 0.5 → Favorable in 2017)
   - NPM1wt + FLT3-ITD (any AR) → Intermediate
     (was Adverse for high AR in 2017)

2. **MDS-related (myelodysplasia-related) gene mutations** → Adverse:
   ASXL1, BCOR, EZH2, RUNX1, SF3B1, SRSF2, STAG2, U2AF1, ZRSR2.
   (RUNX1 + ASXL1 were already Adverse in 2017; the spliceosome / chromatin
   genes are new.)

3. **TP53 multi-hit** → independent Adverse subtype.
   Defined as: ≥2 distinct TP53 mutations OR a single TP53 mutation with
   VAF ≥ 0.5 OR a TP53 mutation with concurrent del(17p)/loss of TP53.
   Single hit TP53 (low VAF, no del17p) is still Adverse but tracked
   separately for prognosis.

4. **CEBPA bZIP in-frame mutation** (single allele) → Favorable.
   In ELN 2017 this required biallelic CEBPA. ELN 2022 found that the
   bZIP in-frame mutation alone (even monoallelic) confers the same
   favorable prognosis.

5. **MDS-related cytogenetics** → Adverse:
   complex karyotype, monosomy/del 5/7, del(11q), del(12p), monosomy 13,
   del(17p), idic(X)(q13). Most of these were already Adverse in 2017.

6. **t(9;11) KMT2A-MLLT3 reclassified to Intermediate** (already in 2017)
   but the rule wording is cleaner.

7. **AML with myelodysplasia-related changes** (clinical history of MDS)
   defaults to Adverse.

References:
  - Döhner et al. Blood 2022; 140(12):1345-1377. PMID 35797463.
  - https://ashpublications.org/blood/article/140/12/1345/485890/

Module shape mirrors `eln_computer.py` so they can be wired in parallel
(report shows BOTH 2017 + 2022 with 2022 as primary, per issue #4).
"""

from __future__ import annotations

from dataclasses import dataclass

from combo_val.clinical.karyotype_parser import parse_karyotype


# Genes whose mutation alone (in the absence of overriding favorable
# features) qualifies the patient as ELN 2022 Adverse via the
# "myelodysplasia-related (MDS-related) gene mutations" criterion.
ELN2022_MDS_RELATED_GENES: frozenset[str] = frozenset({
    "ASXL1", "BCOR", "EZH2", "RUNX1", "SF3B1", "SRSF2",
    "STAG2", "U2AF1", "ZRSR2",
})


@dataclass(frozen=True)
class ELN2022Result:
    category: str       # "Favorable" | "Intermediate" | "Adverse"
    ordinal: float      # 0.0 / 1.0 / 2.0
    rationale: list[str]


def _has_mutation(mutations: list, gene: str) -> bool:
    return any(m.gene.upper() == gene.upper() for m in mutations)


def _flt3_itd(mutations: list) -> bool:
    """ELN 2022: presence of FLT3-ITD; allelic ratio NOT used."""
    return any(
        m.gene.upper() == "FLT3" and getattr(m, "is_ITD", False)
        for m in mutations
    )


def _cebpa_bzip(mutations: list) -> bool:
    """ELN 2022: any in-frame bZIP CEBPA mutation (mono or biallelic)."""
    for m in mutations:
        if m.gene.upper() != "CEBPA":
            continue
        if getattr(m, "is_bzip", False):
            return True
        # Backwards-compat: a biallelic CEBPA flagged in older inputs
        # almost certainly captures the bZIP case (most ELN 2017 biallelic
        # CEBPA cases involve a bZIP variant). Accept it as bZIP-equivalent
        # so 2022 doesn't *under*-classify legacy inputs.
        if getattr(m, "is_biallelic", False):
            return True
    return False


def _tp53_multi_hit(mutations: list, del_17p: bool) -> bool:
    """ELN 2022 TP53 multi-hit definition:
       ≥2 distinct TP53 mutations OR
       a TP53 mutation with VAF ≥ 0.5 OR
       a TP53 mutation with concurrent del(17p)."""
    tp53_calls = [m for m in mutations if m.gene.upper() == "TP53"]
    if not tp53_calls:
        return False
    # Explicit annotation wins
    if any(getattr(m, "is_multi_hit", False) for m in tp53_calls):
        return True
    # ≥2 calls
    if len(tp53_calls) >= 2:
        return True
    # Single call + del17p
    if del_17p:
        return True
    # Single call with high VAF (>= 0.5) — biallelic by VAF
    for m in tp53_calls:
        vaf = getattr(m, "vaf", None)
        if vaf is not None:
            try:
                if float(vaf) >= 0.5:
                    return True
            except (TypeError, ValueError):
                pass
    return False


def _mds_related_gene_present(mutations: list) -> str | None:
    """Return the first MDS-related gene name found in the panel, else None."""
    for m in mutations:
        gene = m.gene.upper()
        if gene in ELN2022_MDS_RELATED_GENES:
            return gene
    return None


def compute_eln2022(
    karyotype_text: str | None,
    mutations: list,
    fusions: list[str] | None = None,
    prior_mds: bool | None = None,
) -> ELN2022Result:
    """Compute ELN 2022 risk category.

    Order of evaluation matters: Adverse criteria checked first (they
    dominate), then Favorable, with Intermediate as the residual default.
    """
    k = parse_karyotype(karyotype_text)
    fusions_up = [f.upper() for f in (fusions or [])]
    rationale: list[str] = []

    # ------- Detect overriding cytogenetics / fusions -------
    has_cbf_t821 = k.t_8_21 or any("RUNX1-RUNX1T1" in f for f in fusions_up)
    has_cbf_inv16 = k.inv_16 or any("CBFB-MYH11" in f for f in fusions_up)
    has_apl = k.t_15_17 or any("PML-RARA" in f for f in fusions_up)
    has_kmt2a_non_t911 = any(
        "KMT2A" in f and "MLLT3" not in f and "9-11" not in f
        for f in fusions_up
    )
    has_t_9_11 = k.t_9_11

    flt3_itd = _flt3_itd(mutations)
    npm1 = _has_mutation(mutations, "NPM1")

    # ============== ADVERSE (checked first) ===================================
    # TP53 multi-hit — independent Adverse subtype, highest priority
    if _tp53_multi_hit(mutations, del_17p=k.del_17p):
        rationale.append("TP53 multi-hit (ELN 2022: independent Adverse subtype)")
        return ELN2022Result("Adverse", 2.0, rationale)
    # Single-hit TP53 still Adverse
    if _has_mutation(mutations, "TP53") or k.del_17p:
        rationale.append("TP53 mutation or del(17p) (ELN 2022 Adverse)")
        return ELN2022Result("Adverse", 2.0, rationale)
    # Complex karyotype
    if k.complex:
        rationale.append("Complex karyotype, ≥3 abnormalities (ELN 2022 Adverse)")
        return ELN2022Result("Adverse", 2.0, rationale)
    # Monosomy / del 5/7
    if k.monosomy_5_or_7 or k.del_5q or k.del_7q:
        rationale.append("Monosomy/del 5 or 7 (ELN 2022 Adverse)")
        return ELN2022Result("Adverse", 2.0, rationale)
    # KMT2A-rearranged (excluding t(9;11))
    if has_kmt2a_non_t911:
        rationale.append("KMT2A-rearranged non-t(9;11) (ELN 2022 Adverse)")
        return ELN2022Result("Adverse", 2.0, rationale)
    # MDS-related gene mutations — NEW in 2022 (vs 2017)
    mds_gene = _mds_related_gene_present(mutations)
    if mds_gene is not None:
        rationale.append(
            f"MDS-related gene mutation: {mds_gene} "
            f"(ELN 2022 Adverse — new criterion vs 2017 for "
            f"BCOR/EZH2/SF3B1/SRSF2/STAG2/U2AF1/ZRSR2)"
        )
        return ELN2022Result("Adverse", 2.0, rationale)
    # Prior MDS history — AML with myelodysplasia-related changes
    if prior_mds is True:
        rationale.append("Prior MDS history → AML-MR (ELN 2022 Adverse)")
        return ELN2022Result("Adverse", 2.0, rationale)

    # ============== FAVORABLE ================================================
    if has_cbf_t821:
        rationale.append("t(8;21) RUNX1-RUNX1T1 (core-binding factor)")
        return ELN2022Result("Favorable", 0.0, rationale)
    if has_cbf_inv16:
        rationale.append("inv(16)/t(16;16) CBFB-MYH11 (core-binding factor)")
        return ELN2022Result("Favorable", 0.0, rationale)
    if has_apl:
        # APL is a distinct entity; clinically Favorable with ATRA+ATO.
        rationale.append("t(15;17) PML-RARA (APL)")
        return ELN2022Result("Favorable", 0.0, rationale)
    # CEBPA bZIP — ELN 2022 relaxed from biallelic-only to bZIP-monoallelic OK
    if _cebpa_bzip(mutations):
        rationale.append(
            "CEBPA bZIP in-frame mutation "
            "(ELN 2022 — relaxed from 2017's biallelic-only)"
        )
        return ELN2022Result("Favorable", 0.0, rationale)
    # NPM1 mutation WITHOUT FLT3-ITD — Favorable in both 2017 and 2022
    if npm1 and not flt3_itd:
        rationale.append("NPM1-mut without FLT3-ITD (ELN 2022 Favorable)")
        return ELN2022Result("Favorable", 0.0, rationale)

    # ============== INTERMEDIATE (residual + explicit cases) ================
    # NPM1 + FLT3-ITD (any AR) — ELN 2022 changed: Intermediate regardless of AR
    if npm1 and flt3_itd:
        rationale.append(
            "NPM1-mut + FLT3-ITD (ELN 2022 Intermediate — AR no longer used)"
        )
        return ELN2022Result("Intermediate", 1.0, rationale)
    # FLT3-ITD without NPM1 — ELN 2022 Intermediate (AR-agnostic; was Adverse
    # for high AR in 2017)
    if flt3_itd and not npm1:
        rationale.append(
            "FLT3-ITD without NPM1 (ELN 2022 Intermediate — "
            "downgraded from 2017 Adverse for high AR)"
        )
        return ELN2022Result("Intermediate", 1.0, rationale)
    # t(9;11) KMT2A-MLLT3 — Intermediate
    if has_t_9_11:
        rationale.append("t(9;11) KMT2A-MLLT3 (ELN 2022 Intermediate)")
        return ELN2022Result("Intermediate", 1.0, rationale)

    rationale.append(
        "No favorable/adverse rules fired — Intermediate by default"
    )
    return ELN2022Result("Intermediate", 1.0, rationale)


def compare_eln_versions(eln_2017: str, eln_2022: str) -> str | None:
    """If the two versions differ, return a short human-readable note
    explaining the most likely cause. Otherwise return None.

    Used by the report to flag patients whose risk class changed between
    the two systems."""
    if eln_2017 == eln_2022:
        return None
    transitions = {
        ("Favorable", "Intermediate"): (
            "ELN 2022 升级到 Intermediate — 最常见原因: NPM1+FLT3-ITD low AR "
            "在 ELN 2017 为 Favorable, 在 ELN 2022 (AR 不再使用) 归 Intermediate。"
        ),
        ("Favorable", "Adverse"): (
            "ELN 2022 升级到 Adverse — 检测到新的 MDS-related 基因 "
            "(BCOR/EZH2/SF3B1/SRSF2/STAG2/U2AF1) 或 TP53 multi-hit, "
            "ELN 2017 未涵盖这些。"
        ),
        ("Intermediate", "Adverse"): (
            "ELN 2022 升级到 Adverse — 最常见原因: MDS-related 基因 (SF3B1/"
            "SRSF2/STAG2/U2AF1/BCOR/EZH2) 在 ELN 2017 不归 Adverse, "
            "在 ELN 2022 归 Adverse。"
        ),
        ("Intermediate", "Favorable"): (
            "ELN 2022 降级到 Favorable — 最常见原因: CEBPA bZIP 单等位突变 "
            "在 ELN 2017 不算 Favorable (要求 biallelic), 在 ELN 2022 算。"
        ),
        ("Adverse", "Intermediate"): (
            "ELN 2022 降级到 Intermediate — 最常见原因: FLT3-ITD 高 AR "
            "(without NPM1) 在 ELN 2017 为 Adverse, ELN 2022 不再使用 AR, "
            "归 Intermediate。"
        ),
        ("Adverse", "Favorable"): (
            "ELN 2022 降级到 Favorable — 罕见: 通常因 CEBPA bZIP 重新评估或 "
            "NPM1 状态修订。建议核对原始测序数据。"
        ),
    }
    return transitions.get(
        (eln_2017, eln_2022),
        f"ELN 2017 ({eln_2017}) 与 ELN 2022 ({eln_2022}) 不一致，"
        f"建议人工核对突变 / 核型详细信息。",
    )
