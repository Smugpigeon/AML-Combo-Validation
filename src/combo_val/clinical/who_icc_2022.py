"""WHO 2022 + ICC 2022 AML classification (issue #12).

Two parallel revised classification systems landed in 2022:

  - **WHO 2022** (Khoury JD et al., Leukemia 2022, PMID 35732831)
  - **ICC 2022** (Arber DA et al., Blood 2022, PMID 35797568)

Both supersede WHO 2016. They mostly agree, but differ on:

  - **Blast threshold**: WHO 2022 dropped the 20% blast cutoff for AML
    *with defining genetic abnormality* (any blast count is OK if the
    genetic abnormality is present). ICC 2022 keeps a 10% cutoff for
    most defining abnormalities and 20% for AML-MR / cytogenetic-only.
  - **AML with mutated NPM1**: both treat as a defining entity. WHO
    requires no blast cutoff; ICC requires ≥ 10% blasts.
  - **AML with mutated CEBPA**: WHO requires bZIP in-frame mutation
    (single allele OK); ICC also accepts bZIP single allele.
  - **AML, myelodysplasia-related (AML-MR)**: defined by either
    cytogenetic abnormalities OR specific gene mutations (BCOR, EZH2,
    SF3B1, SRSF2, STAG2, U2AF1, ZRSR2 — same list as ELN 2022 adverse).
  - **TP53**: ICC 2022 has dedicated AML / MDS-AML with TP53 mutation
    (multi-hit OR sole TP53 + complex karyotype) as separate entities;
    WHO folds them into AML-MR / unclassifiable.

This module computes BOTH classifications and returns the entity name
for each. Shown in patient report alongside ELN 2022 risk so clinicians
can map both to current standards.
"""

from __future__ import annotations

from dataclasses import dataclass

from combo_val.clinical.eln_2022 import ELN2022_MDS_RELATED_GENES
from combo_val.clinical.karyotype_parser import parse_karyotype


@dataclass(frozen=True)
class ClassificationResult:
    who_2022: str          # entity name per WHO 2022
    icc_2022: str          # entity name per ICC 2022
    concordant: bool       # True if WHO and ICC agree on the entity
    rationale: list[str]   # human-readable list of decision steps


def _has_mutation(mutations: list, gene: str) -> bool:
    return any(m.gene.upper() == gene.upper() for m in mutations)


def _flt3_itd(mutations: list) -> bool:
    return any(m.gene.upper() == "FLT3" and getattr(m, "is_ITD", False)
                for m in mutations)


def _cebpa_bzip(mutations: list) -> bool:
    """ELN 2022 / WHO 2022 / ICC 2022 all accept bZIP single allele OR
    explicit biallelic flag (legacy inputs)."""
    for m in mutations:
        if m.gene.upper() != "CEBPA":
            continue
        if getattr(m, "is_bzip", False) or getattr(m, "is_biallelic", False):
            return True
    return False


def _tp53_multi_hit(mutations: list, del_17p: bool) -> bool:
    tp53 = [m for m in mutations if m.gene.upper() == "TP53"]
    if not tp53:
        return False
    if any(getattr(m, "is_multi_hit", False) for m in tp53):
        return True
    if len(tp53) >= 2:
        return True
    if del_17p:
        return True
    for m in tp53:
        vaf = getattr(m, "vaf", None)
        if vaf is not None and float(vaf) >= 0.5:
            return True
    return False


def classify_who_icc_2022(
    karyotype_text: str | None,
    mutations: list,
    fusions: list[str] | None = None,
    blast_pct: float | None = None,
    prior_mds: bool | None = None,
) -> ClassificationResult:
    """Compute both WHO 2022 and ICC 2022 entity names.

    Args:
        blast_pct: BM blast percentage (matters for ICC 2022 thresholds;
                   WHO 2022 drops the 20% requirement for defining genetic
                   abnormalities).
    """
    k = parse_karyotype(karyotype_text)
    fusions_up = [f.upper() for f in (fusions or [])]
    rationale: list[str] = []

    # ------- Defining cytogenetics / fusions -------
    has_t821 = k.t_8_21 or any("RUNX1-RUNX1T1" in f for f in fusions_up)
    has_inv16 = k.inv_16 or any("CBFB-MYH11" in f for f in fusions_up)
    has_apl = k.t_15_17 or any("PML-RARA" in f for f in fusions_up)
    has_kmt2a = any(
        "KMT2A" in f for f in fusions_up
    )
    has_dek_nup214 = any("DEK-NUP214" in f for f in fusions_up)
    has_bcr_abl = any("BCR-ABL" in f for f in fusions_up)

    has_npm1 = _has_mutation(mutations, "NPM1")
    has_cebpa_bzip = _cebpa_bzip(mutations)
    flt3_itd = _flt3_itd(mutations)
    tp53_multi = _tp53_multi_hit(mutations, del_17p=k.del_17p)
    has_tp53 = _has_mutation(mutations, "TP53") or k.del_17p
    mds_gene = next(
        (m.gene.upper() for m in mutations
         if m.gene.upper() in ELN2022_MDS_RELATED_GENES),
        None,
    )

    # ICC has a 10% blast threshold for some entities; WHO doesn't
    icc_blast_ok_10 = blast_pct is None or blast_pct >= 10
    # ICC AML-MR cytogenetic-only requires 20%
    icc_blast_ok_20 = blast_pct is None or blast_pct >= 20

    # ------- Priority order: APL > recurrent fusions > NPM1/CEBPA > TP53/MR -------
    if has_apl:
        rationale.append("PML-RARA fusion → APL (distinct entity in both systems)")
        return ClassificationResult(
            who_2022="Acute promyelocytic leukemia with PML::RARA fusion",
            icc_2022="APL with t(15;17)(q22;q12)/PML::RARA",
            concordant=True,
            rationale=rationale,
        )

    if has_t821:
        rationale.append("t(8;21) RUNX1::RUNX1T1 — defining recurrent abnormality")
        return ClassificationResult(
            who_2022="AML with RUNX1::RUNX1T1 fusion",
            icc_2022="AML with t(8;21)(q22;q22.1)/RUNX1::RUNX1T1",
            concordant=True,
            rationale=rationale,
        )

    if has_inv16:
        rationale.append("inv(16)/t(16;16) CBFB::MYH11 — defining recurrent abnormality")
        return ClassificationResult(
            who_2022="AML with CBFB::MYH11 fusion",
            icc_2022="AML with inv(16)(p13.1q22)/CBFB::MYH11",
            concordant=True,
            rationale=rationale,
        )

    if has_kmt2a:
        rationale.append("KMT2A rearrangement — defining abnormality")
        return ClassificationResult(
            who_2022="AML with KMT2A rearrangement",
            icc_2022="AML with KMT2A-r",
            concordant=True,
            rationale=rationale,
        )

    if has_dek_nup214:
        rationale.append("DEK::NUP214 fusion — defining abnormality")
        return ClassificationResult(
            who_2022="AML with DEK::NUP214 fusion",
            icc_2022="AML with t(6;9)(p23;q34.1)/DEK::NUP214",
            concordant=True,
            rationale=rationale,
        )

    if has_bcr_abl:
        rationale.append("BCR::ABL1 fusion — defining abnormality")
        return ClassificationResult(
            who_2022="AML with BCR::ABL1 fusion",
            icc_2022="AML with t(9;22)(q34.1;q11.2)/BCR::ABL1",
            concordant=True,
            rationale=rationale,
        )

    # NPM1 — defining entity in both. WHO drops 20% blast req; ICC keeps 10%
    if has_npm1:
        rationale.append("NPM1 mutation — defining entity")
        if icc_blast_ok_10:
            return ClassificationResult(
                who_2022="AML with mutated NPM1",
                icc_2022="AML with mutated NPM1",
                concordant=True,
                rationale=rationale,
            )
        else:
            rationale.append(
                f"BM blast {blast_pct}% < 10% → ICC requires ≥ 10% for "
                f"AML with mutated NPM1; WHO has no threshold."
            )
            return ClassificationResult(
                who_2022="AML with mutated NPM1",
                icc_2022="MDS with mutated NPM1 (blast < 10%)",
                concordant=False,
                rationale=rationale,
            )

    # CEBPA bZIP — defining in both
    if has_cebpa_bzip:
        rationale.append("CEBPA bZIP in-frame mutation — defining entity")
        return ClassificationResult(
            who_2022="AML with mutated CEBPA (bZIP in-frame)",
            icc_2022="AML with mutated CEBPA (bZIP in-frame)",
            concordant=True,
            rationale=rationale,
        )

    # TP53 — ICC creates dedicated entity, WHO folds into AML-MR/NOS
    if tp53_multi or (has_tp53 and k.complex):
        rationale.append("TP53 multi-hit OR TP53+complex karyotype")
        return ClassificationResult(
            who_2022="AML, myelodysplasia-related (TP53 — folded into AML-MR)",
            icc_2022="AML with mutated TP53",
            concordant=False,
            rationale=rationale,
        )

    # MDS-related gene mutations — both treat as AML-MR
    if mds_gene is not None:
        rationale.append(
            f"MDS-related gene mutation: {mds_gene} → AML-MR in both systems"
        )
        return ClassificationResult(
            who_2022="AML, myelodysplasia-related (defining mutations)",
            icc_2022="AML with myelodysplasia-related gene mutations",
            concordant=True,
            rationale=rationale,
        )

    # Cytogenetic-only AML-MR (complex karyotype, monosomy 5/7, etc.)
    if k.complex or k.monosomy_5_or_7 or k.del_5q or k.del_7q:
        rationale.append("MDS-related cytogenetic abnormality")
        # ICC requires ≥ 20% blast for cytogenetic-only AML-MR
        if icc_blast_ok_20:
            return ClassificationResult(
                who_2022="AML, myelodysplasia-related (defining cytogenetics)",
                icc_2022="AML with myelodysplasia-related cytogenetic abnormalities",
                concordant=True,
                rationale=rationale,
            )
        else:
            rationale.append(
                f"BM blast {blast_pct}% < 20% → ICC requires ≥ 20% for "
                f"AML-MR (cytogenetic-only)."
            )
            return ClassificationResult(
                who_2022="AML, myelodysplasia-related",
                icc_2022="MDS-AML with myelodysplasia-related changes (blast 10-19%)",
                concordant=False,
                rationale=rationale,
            )

    # Prior MDS history (without explicit cytogenetic / mutation defining)
    if prior_mds is True:
        rationale.append("Prior MDS history → AML-MR (clinical history pathway)")
        return ClassificationResult(
            who_2022="AML, myelodysplasia-related (post-MDS)",
            icc_2022="AML with myelodysplasia-related changes (post-MDS)",
            concordant=True,
            rationale=rationale,
        )

    # Default: AML, NOS / not otherwise specified
    rationale.append("No defining genetic / cytogenetic abnormality — AML, NOS")
    return ClassificationResult(
        who_2022="AML, not otherwise specified (NOS)",
        icc_2022="AML, NOS (no recurrent genetic abnormality)",
        concordant=True,
        rationale=rationale,
    )
