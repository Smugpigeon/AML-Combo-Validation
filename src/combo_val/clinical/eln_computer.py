"""Compute ELN 2017 risk category from karyotype + mutation panel.

Implements the 2017 ELN risk stratification (simplified):

FAVORABLE:
  - t(8;21)(q22;q22); RUNX1-RUNX1T1
  - inv(16) or t(16;16); CBFB-MYH11
  - Biallelic CEBPA mutation
  - NPM1-mut WITHOUT FLT3-ITD or with low FLT3-ITD allelic ratio (<0.5)

INTERMEDIATE:
  - NPM1-mut AND high FLT3-ITD allelic ratio (≥0.5)
  - Wild-type NPM1 without FLT3-ITD (or with low AR) and non-adverse cytogenetics

ADVERSE:
  - Complex karyotype (≥3 abnormalities)
  - Monosomy 5 / del(5q) / monosomy 7 / del(7q)
  - t(v;11q23); KMT2A-rearranged (except t(9;11))
  - Wild-type NPM1 with high FLT3-ITD allelic ratio
  - RUNX1-mut, ASXL1-mut, TP53-mut
  - del(17p) / TP53 loss

Returns the ELN category as a string, plus an ordinal value (0/1/2) aligned
with BeatAML's `clin_eln_ordinal` training encoding.

Note: This is the 2017 criteria. ELN 2022 introduced some refinements
(e.g., removing NPM1 + high FLT3-ITD from adverse); for kit deployment we
stay with 2017 to match the BeatAML training label distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

from combo_val.clinical.karyotype_parser import KaryotypeFlags, parse_karyotype


@dataclass(frozen=True)
class ELNResult:
    category: str       # "Favorable" / "Intermediate" / "Adverse"
    ordinal: float      # 0.0 / 1.0 / 2.0
    rationale: list[str]  # human-readable list of rules fired


def _has_mutation(mutations: list, gene: str) -> bool:
    return any(m.gene.upper() == gene.upper() for m in mutations)


def _flt3_itd(mutations: list) -> tuple[bool, float]:
    """Return (has_ITD, allelic_ratio)."""
    for m in mutations:
        if m.gene.upper() == "FLT3" and getattr(m, "is_ITD", False):
            ar = getattr(m, "allelic_ratio", None) or 0.0
            return True, float(ar)
    return False, 0.0


def _cebpa_biallelic(mutations: list) -> bool:
    return any(
        m.gene.upper() == "CEBPA" and getattr(m, "is_biallelic", False)
        for m in mutations
    )


def compute_eln2017(
    karyotype_text: str | None,
    mutations: list,
    fusions: list[str] | None = None,
) -> ELNResult:
    """Compute ELN 2017 risk. Any passed 'fusions' is treated as an override
    of what the karyotype text would imply."""
    k = parse_karyotype(karyotype_text)
    fusions_up = [f.upper() for f in (fusions or [])]
    rationale: list[str] = []

    # Fusion-driven calls
    has_cbf_t821 = k.t_8_21 or any("RUNX1-RUNX1T1" in f for f in fusions_up)
    has_cbf_inv16 = k.inv_16 or any("CBFB-MYH11" in f for f in fusions_up)
    has_apl = k.t_15_17 or any("PML-RARA" in f for f in fusions_up)
    has_kmt2a_non_t911 = any(
        "KMT2A" in f and "MLLT3" not in f and "9-11" not in f
        for f in fusions_up
    )
    flt3_itd, flt3_ar = _flt3_itd(mutations)
    has_npm1 = _has_mutation(mutations, "NPM1") and not flt3_itd_is_adverse_pair(flt3_itd, flt3_ar)

    # ---- ADVERSE (checked FIRST — adverse dominates other categories) ----
    if k.complex:
        rationale.append("Complex karyotype (≥3 abnormalities)")
        return ELNResult("Adverse", 2.0, rationale)
    if k.monosomy_5_or_7 or k.del_5q or k.del_7q:
        rationale.append("Monosomy/del 5 or 7")
        return ELNResult("Adverse", 2.0, rationale)
    if k.del_17p or _has_mutation(mutations, "TP53"):
        rationale.append("TP53 loss/mutation")
        return ELNResult("Adverse", 2.0, rationale)
    if has_kmt2a_non_t911:
        rationale.append("KMT2A-rearranged (non-t(9;11))")
        return ELNResult("Adverse", 2.0, rationale)
    if _has_mutation(mutations, "RUNX1") or _has_mutation(mutations, "ASXL1"):
        rationale.append("RUNX1 or ASXL1 mutation")
        return ELNResult("Adverse", 2.0, rationale)
    if flt3_itd and flt3_ar >= 0.5 and not _has_mutation(mutations, "NPM1"):
        rationale.append("FLT3-ITD high AR without NPM1")
        return ELNResult("Adverse", 2.0, rationale)

    # ---- FAVORABLE ----
    if has_cbf_t821:
        rationale.append("t(8;21) RUNX1-RUNX1T1 (core-binding factor)")
        return ELNResult("Favorable", 0.0, rationale)
    if has_cbf_inv16:
        rationale.append("inv(16)/t(16;16) CBFB-MYH11 (core-binding factor)")
        return ELNResult("Favorable", 0.0, rationale)
    if has_apl:
        # APL is technically its own category, but clinically treated
        # favorably with ATRA-based regimens. We classify as Favorable.
        rationale.append("t(15;17) PML-RARA (APL)")
        return ELNResult("Favorable", 0.0, rationale)
    if _cebpa_biallelic(mutations):
        rationale.append("Biallelic CEBPA mutation")
        return ELNResult("Favorable", 0.0, rationale)
    if _has_mutation(mutations, "NPM1") and not flt3_itd:
        rationale.append("NPM1-mut without FLT3-ITD")
        return ELNResult("Favorable", 0.0, rationale)
    if _has_mutation(mutations, "NPM1") and flt3_itd and flt3_ar < 0.5:
        rationale.append("NPM1-mut with low FLT3-ITD AR (<0.5)")
        return ELNResult("Favorable", 0.0, rationale)

    # ---- INTERMEDIATE (default) ----
    if _has_mutation(mutations, "NPM1") and flt3_itd and flt3_ar >= 0.5:
        rationale.append("NPM1-mut with high FLT3-ITD AR (≥0.5)")
        return ELNResult("Intermediate", 1.0, rationale)
    if k.t_9_11:
        rationale.append("t(9;11) KMT2A-MLLT3")
        return ELNResult("Intermediate", 1.0, rationale)

    rationale.append("No favorable/adverse rules fired — intermediate by default")
    return ELNResult("Intermediate", 1.0, rationale)


def flt3_itd_is_adverse_pair(has_itd: bool, ar: float) -> bool:
    return has_itd and ar >= 0.5


# Helper: map ELN string → ordinal (same mapping as beataml_etl's ELN_ORDINAL,
# re-exported here so feature_builder doesn't have to import from data package).
_ELN_ORDINAL_MAP = {
    "Favorable": 0.0,
    "FavorableOrIntermediate": 0.5,
    "Intermediate": 1.0,
    "IntermediateOrAdverse": 1.5,
    "Adverse": 2.0,
}


def ELN_ordinal_from_string(s: str) -> float:
    """Ordinal encoding of ELN 2017 category (matches training)."""
    return _ELN_ORDINAL_MAP.get(str(s), 1.0)  # unknown → Intermediate
