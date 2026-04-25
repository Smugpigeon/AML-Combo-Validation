"""Per issues #9 + #13 — AML driver gene hotspot codon classifier.

Many AML driver genes have **canonical hotspot codons** with independent
prognostic / therapeutic significance distinct from non-hotspot mutations
in the same gene:

  - DNMT3A R882 (H > C > P) — independent adverse, MRD-persistence common
    (Ley TJ et al., NEJM 2010 PMID 21067377; Patel JP NEJM 2012 PMID 22417203)
  - IDH1 R132 (H > C > G > L > S) — primary driver; FDA-approved Ivosidenib targets it
  - IDH2 R140Q vs R172K — both targeted by Enasidenib but R140Q more common
    and has slightly different cooperating mutation pattern
  - FLT3-TKD D835 / I836 / F691L — D835Y is most common; F691L is the
    Gilteritinib-resistance gatekeeper mutation (clinical implication for
    salvage choice)
  - CEBPA bZIP — ELN 2022 Favorable for SINGLE allele in-frame bZIP

`classify_hotspot()` takes a MutationCall and returns a structured
description of whether the mutation hits a known hotspot, plus a
clinician-facing interpretation string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class HotspotResult:
    """Summary of hotspot classification for one mutation call."""
    is_hotspot: bool
    hotspot_label: str | None         # e.g. "DNMT3A R882"
    interpretation: str                # human-readable clinical commentary
    confidence: str                    # "high" | "medium" | "low"


# Per-gene hotspot codon patterns (regex on protein_codon string)
_HOTSPOT_PATTERNS = {
    # DNMT3A R882 — strict hotspot, ~60% of DNMT3A mutations in AML
    "DNMT3A": [
        (re.compile(r"R882[HCPYS]"),
         "DNMT3A R882",
         "**R882 hotspot** — canonical AML hotspot (~60% of DNMT3A "
         "mutations). Independent adverse prognosis: MRD-persistence "
         "common, slightly worse OS even when controlling for ELN risk. "
         "R882H most common (~50%), R882C second. Patel NEJM 2012 "
         "(PMID 22417203). Cooperates with NPM1 + FLT3-ITD (classic triplet).",
         "high"),
        (re.compile(r"R635[WGEK]|R736[CHL]"),
         "DNMT3A non-R882 missense",
         "Non-R882 DNMT3A missense — less defined prognostic impact than "
         "R882 hotspot; some series suggest similar adverse effect, others "
         "are neutral. **Verify with codon-specific literature**.",
         "medium"),
    ],
    # IDH1 R132 — primary AML hotspot; FDA-approved Ivosidenib targets it
    "IDH1": [
        (re.compile(r"R132[HCGLSY]"),
         "IDH1 R132",
         "**R132 hotspot** — produces 2-HG oncometabolite (cellular "
         "differentiation block). **Ivosidenib (Tibsovo) FDA-approved** "
         "(IDH1 R132 specific). R132H most common (~80%), then R132C. "
         "Differentiation-syndrome risk on Ivo (~25%). AGILE trial "
         "(Aza+Ivo) is FDA SOC for IDH1-mut newly-Dx unfit (PMID 35443106).",
         "high"),
    ],
    # IDH2 R140 / R172 — Enasidenib targets both
    "IDH2": [
        (re.compile(r"R140[QWLR]"),
         "IDH2 R140",
         "**R140 hotspot** (more common, ~75%). Cooperates with NPM1+/-FLT3. "
         "**Enasidenib (Idhifa) FDA-approved** for R/R IDH2-mut. Newly-Dx "
         "Aza+Ena+Ven triplet is Phase 2 experimental. DS risk ~15%.",
         "high"),
        (re.compile(r"R172[KGS]"),
         "IDH2 R172",
         "**R172 hotspot** (less common, ~25%). Slightly distinct "
         "epigenetic phenotype vs R140 (more global hypermethylation). "
         "Same Enasidenib indication. R172 cohorts have somewhat lower "
         "complete remission rate than R140 in IDHENTIFY.",
         "high"),
    ],
    # FLT3-TKD — D835Y most common; F691L is gilt-resistance hotspot
    "FLT3": [
        (re.compile(r"D835[YFVHN]"),
         "FLT3-TKD D835",
         "**D835 hotspot** — canonical FLT3 TKD mutation (activation loop). "
         "Different from FLT3-ITD: D835 has slightly better prognosis than "
         "ITD and is **more sensitive to Type-1 FLT3i (Gilteritinib, "
         "Crenolanib)** than Type-2 (Quizartinib, Sorafenib). Choose "
         "Gilteritinib over Quiz when D835 isolated.",
         "high"),
        (re.compile(r"I836"),
         "FLT3-TKD I836",
         "**I836** — adjacent activation-loop hotspot. Same Type-1 FLT3i "
         "preference as D835 (Gilteritinib > Quizartinib).",
         "high"),
        (re.compile(r"F691[LMI]"),
         "FLT3-TKD F691L (gatekeeper)",
         "**F691L gatekeeper** — confers resistance to Gilteritinib + "
         "Quizartinib + most TKIs. Emerges on therapy. **Switch to "
         "Crenolanib** (in-development) or Type-2 Sorafenib + chemo, "
         "or proceed directly to allo-SCT in CR with maintenance trial.",
         "high"),
    ],
    # CEBPA bZIP — ELN 2022 Favorable for any single-allele bZIP
    "CEBPA": [
        # bZIP-flagged via is_bzip; codon parsing for backwards compat:
        # Common bZIP variants: K313 region, in-frame insertions/deletions
        # in bZIP domain. We don't parse those by codon here — use is_bzip.
    ],
    # NPM1 — exon 12 frameshift (TCTG insertion most common, type A)
    "NPM1": [
        (re.compile(r"W288[Cf]|L287|TCTG|exon\s*12", re.I),
         "NPM1 exon 12 frameshift",
         "**NPM1 exon 12 frameshift** — canonical AML hotspot (~95% of "
         "NPM1 mutations). Type A (TCTG insertion) most common (~80%). "
         "Confers cytoplasmic NPM1 mislocalization. **MRD-monitorable** "
         "via NPM1mut RT-qPCR (sensitivity 1e-5/1e-6). Co-existence with "
         "FLT3-ITD modulates ELN 2022 risk to Intermediate.",
         "high"),
    ],
    # KIT D816 / N822 — independent adverse modifier in CBF-AML
    "KIT": [
        (re.compile(r"D816[VFY]|N822|D419"),
         "KIT D816 / N822",
         "**KIT D816V/N822** — adverse modifier in t(8;21) and inv(16) "
         "CBF-AML; downgrades 5-yr OS from ~70% (CBF-AML alone) to ~40-50%. "
         "Some centers add midostaurin or escalate to allo-SCT in CR1 "
         "based on KIT status. Target with imatinib/dasatinib in trials.",
         "medium"),
    ],
}


def classify_hotspot(gene: str, protein_codon: str | None) -> HotspotResult:
    """Match a (gene, codon) pair against the hotspot table."""
    if not protein_codon:
        return HotspotResult(
            is_hotspot=False,
            hotspot_label=None,
            interpretation="Codon not extracted from report — verify hotspot "
                           "status manually if clinical action depends on it.",
            confidence="low",
        )
    patterns = _HOTSPOT_PATTERNS.get(gene.upper(), [])
    for regex, label, interp, conf in patterns:
        if regex.search(protein_codon):
            return HotspotResult(
                is_hotspot=True,
                hotspot_label=label,
                interpretation=interp,
                confidence=conf,
            )
    return HotspotResult(
        is_hotspot=False,
        hotspot_label=None,
        interpretation=f"{gene} {protein_codon} — non-hotspot variant in this "
                       f"gene. Prognostic impact less defined than canonical "
                       f"hotspots; cross-reference COSMIC + cBioPortal for "
                       f"recurrence frequency.",
        confidence="low",
    )


def hotspot_summary_for_report(mutations: list) -> list[dict]:
    """Run hotspot classification across all calls in a panel; return
    clinician-facing summary records (used by the report renderer)."""
    out = []
    for m in mutations:
        result = classify_hotspot(m.gene, getattr(m, "protein_codon", None))
        if result.is_hotspot or result.confidence != "low":
            out.append({
                "gene": m.gene,
                "codon": getattr(m, "protein_codon", None),
                "is_hotspot": result.is_hotspot,
                "hotspot_label": result.hotspot_label,
                "interpretation": result.interpretation,
                "confidence": result.confidence,
            })
    return out
