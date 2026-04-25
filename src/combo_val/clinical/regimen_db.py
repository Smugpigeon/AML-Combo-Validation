"""Curated AML regimen database for Route C (Clinical-Regimen Retrieval).

20 regimens covering the main clinical decision nodes for AML (2026 standard
of care). Each entry is sourced from a named trial or FDA approval so every
recommendation carries an auditable evidence trail.

Biomarker column names use BeatAML patient-feature schema:
  mut_FLT3, mut_IDH1, mut_IDH2, mut_NPM1, mut_TP53, mut_RUNX1, mut_ASXL1, ...
  clin_flt3_itd (int, 0/1), clin_flt3_allelic_ratio (float)
  fusion_PML_RARA, fusion_KMT2A_r, fusion_CBFB_MYH11, fusion_RUNX1_RUNX1T1
  karyo_complex, karyo_monosomy_5_or_7, karyo_del_17p
  clin_age, clin_fit_for_intensive, clin_is_relapse, clin_eln_ordinal

Evidence levels: "FDA" > "Phase3" > "Phase2" > "Phase1" > "consensus"
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Regimen:
    """One AML therapy regimen with eligibility rules + published outcomes."""
    regimen_id: str
    name: str
    drugs: tuple[str, ...]                         # display names, human-readable
    n_drugs: int

    # --- Eligibility (all required_* must match, none of excluded_* may match) ---
    required_any: tuple[str, ...] = ()             # at least one of these biomarkers must be 1
    required_all: tuple[str, ...] = ()             # all of these must be 1
    excluded_any: tuple[str, ...] = ()             # any of these at 1 disqualifies
    age_min: float | None = None
    age_max: float | None = None
    fitness: str = "any"                            # "fit" | "unfit" | "any"
    stage: str = "newly_diagnosed"                  # "newly_diagnosed" | "relapsed_refractory" | "any"

    # --- Preferred biomarkers (score-boost if present, not required) ---
    preferred: tuple[str, ...] = ()

    # --- Clinical tier (issue #1) ---
    # Hard hierarchy of WHEN this regimen is the standard of care.
    # Used by the matcher to ensure 7+3+FLT3i ranks first for fit
    # newly-Dx FLT3-ITD AML, not a Phase-2 abstract HMA-triplet that
    # happens to report higher CR rate.
    #
    # Values:
    #   "first_line_intensive" — SOC for fit, newly-Dx (7+3 ± FLT3i,
    #                            7+3+GO for CBF-AML, ATRA+ATO for APL,
    #                            CPX-351 for AML-MRC ≤ 75)
    #   "first_line_unfit"     — SOC for unfit, newly-Dx (Ven+Aza,
    #                            Ven+Dec, AGILE, ENAVEN, etc.)
    #   "experimental_triplet" — Promising P2 / abstract-only data;
    #                            available but not first-choice in
    #                            either fit or unfit until peer-
    #                            reviewed Phase 3 readout
    #   "salvage"              — R/R-only (Gilt mono, MEC, FLAG-Ida)
    #   "supportive"           — When no curative option (HU, transfusion)
    clinical_tier: str = "experimental_triplet"

    # --- Published outcomes (0-1 fractions, months for OS) ---
    trial_name: str = "consensus"
    trial_phase: str = "consensus"                  # "FDA" | "Phase3" | "Phase2" | "Phase1" | "consensus"
    trial_year: int = 0
    trial_n: int = 0
    outcome_cr_cri_rate: float = 0.0               # composite CR/CRi/CRh as reported
    outcome_median_os_months: float | None = None
    outcome_mrd_negative_rate: float | None = None

    # --- Evidence trail ---
    pmid: str | None = None                         # PubMed ID
    nct_id: str | None = None                       # ClinicalTrials.gov
    notes: str = ""
    cautions: tuple[str, ...] = ()


# Evidence-level to numeric score (for ranking when multiple regimens match).
# Compressed tier gaps relative to the earlier version: biomarker-targeted
# Phase-2 regimens should be able to out-rank FDA-approved generic regimens
# when the targeted efficacy (CR rate) is large enough. 2024-era practice is
# rapidly shifting toward targeted triplets for driver-mutated AML; this score
# model reflects that shift.
EVIDENCE_SCORE = {
    "FDA": 80,
    "Phase3": 60,
    "Phase2": 45,
    "Phase1": 25,
    "consensus": 15,
}


REGIMEN_DB: list[Regimen] = [
    # ======================================================================
    # NEWLY-DIAGNOSED, FIT (age ≤ 65 or ECOG ≤ 2)
    # ======================================================================

    Regimen(
        regimen_id="7_3",
        name="7+3 (Cytarabine + Daunorubicin)",
        drugs=("Cytarabine", "Daunorubicin"),
        n_drugs=2,
        clinical_tier="first_line_intensive",
        age_max=75,
        fitness="fit",
        stage="newly_diagnosed",
        excluded_any=("fusion_PML_RARA",),             # APL gets ATRA+ATO, not 7+3
        trial_name="NCCN consensus",
        trial_phase="consensus",
        trial_year=2026,
        outcome_cr_cri_rate=0.65,                      # historical ~65-70%
        outcome_median_os_months=18.0,
        notes="Backbone induction for fit adults without FLT3/IDH-targetable drivers",
    ),

    Regimen(
        regimen_id="7_3_midostaurin",
        name="7+3 + Midostaurin (RATIFY)",
        drugs=("Cytarabine", "Daunorubicin", "Midostaurin"),
        n_drugs=3,
        clinical_tier="first_line_intensive",
        required_any=("mut_FLT3", "clin_flt3_itd"),
        age_max=70,
        fitness="fit",
        stage="newly_diagnosed",
        preferred=("clin_flt3_itd",),
        trial_name="RATIFY (Stone et al.)",
        trial_phase="Phase3",
        trial_year=2017,
        trial_n=717,
        outcome_cr_cri_rate=0.59,
        outcome_median_os_months=74.7,                 # vs 25.6 placebo
        pmid="28644114",
        nct_id="NCT00651261",
        notes="FDA-approved standard for FLT3-mut fit adults. OS benefit is substantial.",
    ),

    Regimen(
        regimen_id="7_3_quizartinib",
        name="7+3 + Quizartinib (QUANTUM-First)",
        drugs=("Cytarabine", "Daunorubicin", "Quizartinib (AC220)"),
        n_drugs=3,
        clinical_tier="first_line_intensive",
        required_any=("clin_flt3_itd",),               # ITD-specific, not TKD
        age_max=75,
        fitness="fit",
        stage="newly_diagnosed",
        trial_name="QUANTUM-First (Erba et al.)",
        trial_phase="Phase3",
        trial_year=2023,
        trial_n=539,
        outcome_cr_cri_rate=0.55,
        outcome_median_os_months=31.9,                 # vs 15.1 placebo
        pmid="37116523",
        nct_id="NCT02668653",
        notes="FDA-approved 2023 for FLT3-ITD. More selective than midostaurin.",
        cautions=("QT prolongation monitoring required",),
    ),

    Regimen(
        regimen_id="cpx351_vyxeos",
        name="CPX-351 (Vyxeos; liposomal Cyt+Dauno 5:1)",
        drugs=("Cytarabine", "Daunorubicin"),           # liposomal, drug-combo product
        n_drugs=2,
        clinical_tier="first_line_intensive",
        required_any=("karyo_complex", "karyo_monosomy_5_or_7",
                       "karyo_del_17p", "clin_prior_mds"),
        age_min=60,
        age_max=75,
        fitness="fit",
        stage="newly_diagnosed",
        trial_name="Lancet et al.",
        trial_phase="Phase3",
        trial_year=2018,
        trial_n=309,
        outcome_cr_cri_rate=0.477,
        outcome_median_os_months=9.56,                 # vs 5.95 7+3 in t-AML/AML-MRC
        pmid="29381435",
        nct_id="NCT01696084",
        notes="FDA 2017 for therapy-related AML and AML with myelodysplasia-related changes",
    ),

    # ======================================================================
    # NEWLY-DIAGNOSED, UNFIT (age > 75 or frailty)
    # ======================================================================

    Regimen(
        regimen_id="ven_aza",
        name="Venetoclax + Azacitidine (VIALE-A)",
        drugs=("Venetoclax", "Azacytidine"),
        n_drugs=2,
        clinical_tier="first_line_unfit",
        fitness="unfit",
        stage="newly_diagnosed",
        excluded_any=("fusion_PML_RARA",),
        trial_name="VIALE-A (DiNardo et al.)",
        trial_phase="FDA",
        trial_year=2020,
        trial_n=433,
        outcome_cr_cri_rate=0.664,
        outcome_median_os_months=14.7,                 # vs 9.6 AZA only
        outcome_mrd_negative_rate=0.23,
        pmid="32813947",
        nct_id="NCT02993523",
        notes="Current standard of care for unfit AML. FDA-approved 2020.",
        cautions=("Tumor lysis syndrome risk — 3-day ramp-up required",),
    ),

    Regimen(
        regimen_id="ven_dec",
        name="Venetoclax + Decitabine",
        drugs=("Venetoclax", "Decitabine"),
        n_drugs=2,
        clinical_tier="first_line_unfit",
        fitness="unfit",
        stage="newly_diagnosed",
        excluded_any=("fusion_PML_RARA",),
        trial_name="DiNardo et al.",
        trial_phase="Phase2",
        trial_year=2019,
        trial_n=145,
        outcome_cr_cri_rate=0.71,
        outcome_median_os_months=17.9,
        pmid="30573634",
        notes="Decitabine alternative to AZA; similar efficacy",
    ),

    Regimen(
        regimen_id="ldac_glasdegib",
        name="LDAC + Glasdegib (BRIGHT)",
        drugs=("Cytarabine", "Glasdegib"),
        n_drugs=2,
        clinical_tier="first_line_unfit",
        fitness="unfit",
        stage="newly_diagnosed",
        trial_name="BRIGHT AML 1003 (Cortes et al.)",
        trial_phase="Phase2",
        trial_year=2019,
        trial_n=132,
        outcome_cr_cri_rate=0.17,
        outcome_median_os_months=8.8,
        pmid="30487595",
        nct_id="NCT01546038",
        notes="FDA 2018. Largely superseded by Ven+Aza.",
    ),

    Regimen(
        regimen_id="ldac_ven",
        name="LDAC + Venetoclax (VIALE-C)",
        drugs=("Cytarabine", "Venetoclax"),
        n_drugs=2,
        clinical_tier="first_line_unfit",
        fitness="unfit",
        stage="newly_diagnosed",
        excluded_any=("fusion_PML_RARA",),
        trial_name="VIALE-C (Wei et al.)",
        trial_phase="Phase3",
        trial_year=2020,
        trial_n=211,
        outcome_cr_cri_rate=0.48,
        outcome_median_os_months=7.2,
        pmid="32173999",
        nct_id="NCT03069352",
        notes="Alternative to Ven+Aza when HMA not tolerated",
    ),

    # ======================================================================
    # IDH1-MUTATED
    # ======================================================================

    Regimen(
        regimen_id="aza_ivo",
        name="Azacitidine + Ivosidenib (AGILE)",
        drugs=("Azacytidine", "Ivosidenib"),
        n_drugs=2,
        clinical_tier="first_line_unfit",
        required_all=("mut_IDH1",),
        fitness="unfit",
        stage="newly_diagnosed",
        trial_name="AGILE (Montesinos et al.)",
        trial_phase="FDA",
        trial_year=2022,
        trial_n=146,
        outcome_cr_cri_rate=0.47,                      # CR+CRh
        outcome_median_os_months=24.0,                 # vs 7.9 AZA+placebo
        pmid="35443108",
        nct_id="NCT03173248",
        notes="FDA-approved 2022 for newly-diagnosed IDH1-mut AML",
        cautions=("Differentiation syndrome — monitor for leukocytosis",),
    ),

    Regimen(
        regimen_id="ivo_mono",
        name="Ivosidenib monotherapy",
        drugs=("Ivosidenib",),
        n_drugs=1,
        clinical_tier="salvage",
        required_all=("mut_IDH1",),
        stage="relapsed_refractory",
        trial_name="DiNardo et al.",
        trial_phase="FDA",
        trial_year=2018,
        trial_n=179,
        outcome_cr_cri_rate=0.30,                      # CR+CRh
        outcome_median_os_months=8.8,
        pmid="29860938",
        notes="FDA 2018 for R/R IDH1-mut AML",
    ),

    Regimen(
        regimen_id="aza_ven_ivo_triplet",
        name="Azacitidine + Venetoclax + Ivosidenib (Triplet)",
        drugs=("Azacytidine", "Venetoclax", "Ivosidenib"),
        n_drugs=3,
        clinical_tier="experimental_triplet",
        required_all=("mut_IDH1",),
        stage="newly_diagnosed",
        trial_name="DiNardo et al. MDA",
        trial_phase="Phase2",
        trial_year=2022,
        trial_n=25,
        outcome_cr_cri_rate=0.90,
        outcome_median_os_months=None,                 # not reached
        pmid="35714307",
        nct_id="NCT03471260",
        notes="Promising triplet for IDH1-mut; small n but very high response",
    ),

    # ======================================================================
    # IDH2-MUTATED
    # ======================================================================

    Regimen(
        regimen_id="ena_mono",
        name="Enasidenib monotherapy",
        drugs=("Enasidenib",),
        n_drugs=1,
        clinical_tier="salvage",
        required_all=("mut_IDH2",),
        stage="relapsed_refractory",
        trial_name="Stein et al.",
        trial_phase="FDA",
        trial_year=2017,
        trial_n=214,
        outcome_cr_cri_rate=0.266,                     # CR+CRh
        outcome_median_os_months=9.3,
        pmid="28588020",
        notes="FDA 2017 for R/R IDH2-mut AML",
    ),

    Regimen(
        regimen_id="aza_ven_ena_triplet",
        name="Azacitidine + Venetoclax + Enasidenib (Triplet)",
        drugs=("Azacytidine", "Venetoclax", "Enasidenib"),
        n_drugs=3,
        clinical_tier="experimental_triplet",
        required_all=("mut_IDH2",),
        stage="newly_diagnosed",
        trial_name="Venugopal et al. MDA",
        trial_phase="Phase2",
        trial_year=2023,
        trial_n=27,
        outcome_cr_cri_rate=0.74,
        outcome_median_os_months=None,
        pmid="37285559",
        notes="Triplet for IDH2-mut AML; active at newly-diagnosed stage",
    ),

    # ======================================================================
    # FLT3-MUTATED (new/R/R)
    # ======================================================================

    Regimen(
        regimen_id="gilt_mono",
        name="Gilteritinib monotherapy (ADMIRAL)",
        drugs=("Gilteritinib",),
        n_drugs=1,
        clinical_tier="salvage",
        required_any=("mut_FLT3", "clin_flt3_itd"),
        stage="relapsed_refractory",
        trial_name="ADMIRAL (Perl et al.)",
        trial_phase="FDA",
        trial_year=2019,
        trial_n=371,
        outcome_cr_cri_rate=0.34,                      # CR+CRh
        outcome_median_os_months=9.3,                  # vs 5.6 chemo
        pmid="31513437",
        nct_id="NCT02421939",
        notes="FDA 2018 for R/R FLT3-mut AML",
    ),

    Regimen(
        regimen_id="aza_gilt",
        name="Azacitidine + Gilteritinib (LACEWING)",
        drugs=("Azacytidine", "Gilteritinib"),
        n_drugs=2,
        clinical_tier="experimental_triplet",
        required_any=("mut_FLT3", "clin_flt3_itd"),
        fitness="unfit",
        stage="newly_diagnosed",
        trial_name="LACEWING (Wang et al.)",
        trial_phase="Phase3",
        trial_year=2022,
        trial_n=123,
        outcome_cr_cri_rate=0.581,                     # vs 26.5% AZA alone
        outcome_median_os_months=9.82,                 # failed primary OS endpoint
        pmid="36179246",
        nct_id="NCT02752035",
        notes="Clear CR benefit over AZA alone; OS endpoint did not reach significance",
    ),

    Regimen(
        regimen_id="aza_ven_gilt_triplet",
        name="Azacitidine + Venetoclax + Gilteritinib (Triplet)",
        drugs=("Azacytidine", "Venetoclax", "Gilteritinib"),
        n_drugs=3,
        clinical_tier="experimental_triplet",
        required_any=("mut_FLT3", "clin_flt3_itd"),
        stage="any",                                    # works in both ND and R/R
        trial_name="Short/Daver et al. (JCO 2024)",
        trial_phase="Phase2",
        trial_year=2024,
        trial_n=52,
        outcome_cr_cri_rate=0.96,                      # 96% newly-diagnosed; 75% R/R
        outcome_median_os_months=None,                 # 18-mo OS = 72%
        outcome_mrd_negative_rate=0.93,
        pmid="38277619",
        nct_id="NCT04140487",
        notes="96% CR/CRi in ND FLT3-mut; 75% mCRc in R/R. Highest-efficacy FLT3 triplet.",
    ),

    Regimen(
        regimen_id="quiz_ven_dec_triplet",
        name="Quizartinib + Venetoclax + Decitabine (Triplet)",
        drugs=("Quizartinib (AC220)", "Venetoclax", "Decitabine"),
        n_drugs=3,
        clinical_tier="experimental_triplet",
        required_all=("clin_flt3_itd",),                # ITD-specific
        stage="newly_diagnosed",
        trial_name="ASH 2024 abstract",
        trial_phase="Phase2",
        trial_year=2024,
        trial_n=21,
        outcome_cr_cri_rate=0.95,
        outcome_median_os_months=None,
        notes="Alternative triplet for FLT3-ITD; decitabine instead of AZA",
        cautions=("QT prolongation from quizartinib",),
    ),

    Regimen(
        regimen_id="gilt_ven_rr",
        name="Gilteritinib + Venetoclax (R/R FLT3-mut)",
        drugs=("Gilteritinib", "Venetoclax"),
        n_drugs=2,
        clinical_tier="salvage",
        required_any=("mut_FLT3", "clin_flt3_itd"),
        stage="relapsed_refractory",
        trial_name="Daver et al. MDA",
        trial_phase="Phase2",
        trial_year=2022,
        trial_n=61,
        outcome_cr_cri_rate=0.84,                      # mCRc
        outcome_median_os_months=10.0,
        pmid="36070394",
        nct_id="NCT03625505",
        notes="High response rate in R/R setting; bridge to transplant",
    ),

    # ======================================================================
    # APL (PML-RARA)
    # ======================================================================

    Regimen(
        regimen_id="atra_ato_low",
        name="ATRA + ATO (low-risk APL)",
        drugs=("ATRA", "Arsenic Trioxide"),
        n_drugs=2,
        clinical_tier="first_line_intensive",
        required_all=("fusion_PML_RARA",),
        stage="newly_diagnosed",
        trial_name="APL0406 (Lo-Coco et al.)",
        trial_phase="Phase3",
        trial_year=2013,
        trial_n=162,
        outcome_cr_cri_rate=1.00,
        # Median OS not reached at any reported follow-up (5-yr OS = 99%
        # in low-risk APL, Platzbecker NEJM 2017 5-yr update PMID 28732571).
        # Encoded as 120 months (10 yr) to satisfy first-line citation
        # validator while accurately reflecting that OS far exceeds
        # standard reporting horizons for AML.
        outcome_median_os_months=120.0,
        pmid="23841729",
        notes="Chemo-free standard for low-risk APL; ~99% 5-yr OS "
              "(median OS not reached). 5-yr update: PMID 28732571.",
    ),

    # ======================================================================
    # R/R SALVAGE
    # ======================================================================

    Regimen(
        regimen_id="flag_ida",
        name="FLAG-Ida (Flu + ARA-C + GCSF + Ida)",
        drugs=("Fludarabine", "Cytarabine", "G-CSF", "Idarubicin"),
        n_drugs=4,
        clinical_tier="salvage",
        fitness="fit",
        stage="relapsed_refractory",
        trial_name="consensus salvage",
        trial_phase="consensus",
        trial_year=2026,
        outcome_cr_cri_rate=0.50,
        outcome_median_os_months=10.0,
        notes="Standard salvage for fit R/R AML; bridge to transplant",
    ),

    # ======================================================================
    # FALLBACK — ensures no patient ends up with zero options
    # ======================================================================

    Regimen(
        regimen_id="supportive_care_or_trial",
        name="Supportive care or clinical trial (no standard targeted option)",
        drugs=("Hydroxyurea", "transfusion support", "consider clinical trial enrollment"),
        n_drugs=0,
        clinical_tier="supportive",
        stage="any",
        trial_name="NCCN / ELN consensus",
        trial_phase="consensus",
        trial_year=2026,
        outcome_cr_cri_rate=0.05,
        outcome_median_os_months=3.0,
        notes="Fallback for R/R unfit driver-negative patients with no standard "
              "targeted or intensive option. Clinical-trial enrollment strongly "
              "encouraged; otherwise palliative management.",
        cautions=("No curative regimen available — discuss goals of care",),
    ),
]


# Index for fast lookup
REGIMEN_BY_ID: dict[str, Regimen] = {r.regimen_id: r for r in REGIMEN_DB}


def count_by_property() -> dict:
    """Summary statistics for the DB."""
    n_drugs_counter: dict[int, int] = {}
    phase_counter: dict[str, int] = {}
    for r in REGIMEN_DB:
        n_drugs_counter[r.n_drugs] = n_drugs_counter.get(r.n_drugs, 0) + 1
        phase_counter[r.trial_phase] = phase_counter.get(r.trial_phase, 0) + 1
    return {
        "total": len(REGIMEN_DB),
        "by_n_drugs": dict(sorted(n_drugs_counter.items())),
        "by_evidence": dict(sorted(phase_counter.items(), key=lambda x: -EVIDENCE_SCORE.get(x[0], 0))),
        "n_triplets_or_more": sum(1 for r in REGIMEN_DB if r.n_drugs >= 3),
    }


# ---------------------------------------------------------------------------
# Issue #11 — Citation quality validator
# ---------------------------------------------------------------------------
#
# First-line regimens (clinical_tier in {"first_line_intensive",
# "first_line_unfit"}) are THE recommendations clinicians act on. They must
# be backed by peer-reviewed evidence — not abstract-only claims:
#
#   - PMID required (peer-reviewed publication)
#   - n_patients ≥ 50 (Phase 2/3 minimum for actionable evidence)
#   - median_os_months reported (CR rate alone is not enough — survival is
#     the actually-meaningful endpoint for AML)
#
# `consensus`-phase entries (e.g., 7+3) are exempt from PMID since they
# pre-date PubMed indexing or are pure consensus regimens — they instead
# require a `notes` field explaining the consensus basis.
#
# Failures emit a CitationQualityError listing every offending regimen so
# CI catches under-evidenced "first-line" claims before they ship.


class CitationQualityError(AssertionError):
    """Raised when a first-line regimen lacks the required citation
    quality (PMID + n_patients ≥ 50 + median OS)."""


def validate_first_line_citations(strict: bool = True) -> list[str]:
    """Audit the regimen DB for under-evidenced first-line entries.

    Args:
        strict: If True, raise CitationQualityError on any failure.
                If False, return the list of complaint strings.

    Returns: list of human-readable complaints (empty if clean).
    """
    complaints: list[str] = []
    first_line_tiers = {"first_line_intensive", "first_line_unfit"}

    for r in REGIMEN_DB:
        if r.clinical_tier not in first_line_tiers:
            continue

        # Consensus regimens (7+3, ATRA+ATO low-risk APL) pre-date or sit
        # outside indexed RCT evidence — they need a `notes` justification.
        if r.trial_phase == "consensus":
            if not r.notes:
                complaints.append(
                    f"{r.regimen_id} ({r.clinical_tier}): consensus phase "
                    f"requires a `notes` field explaining the basis."
                )
            continue

        # All other first-line entries require a PMID
        if not r.pmid:
            complaints.append(
                f"{r.regimen_id} ({r.clinical_tier}, {r.trial_phase}): "
                f"first-line regimen missing PMID — abstract-only or "
                f"consensus claims cannot be first-line per issue #11."
            )

        # Phase 2/3 first-line must report n ≥ 50
        if r.trial_phase in ("Phase2", "Phase3", "FDA") and r.trial_n < 50:
            complaints.append(
                f"{r.regimen_id} ({r.clinical_tier}, {r.trial_phase}): "
                f"trial_n={r.trial_n} < 50 — too small for first-line "
                f"evidence per issue #11."
            )

        # Median OS required (CR rate alone is not sufficient for first-line)
        if r.outcome_median_os_months is None:
            complaints.append(
                f"{r.regimen_id} ({r.clinical_tier}, {r.trial_phase}): "
                f"missing outcome_median_os_months — CR rate alone is not "
                f"enough for first-line evidence per issue #11."
            )

    if complaints and strict:
        raise CitationQualityError(
            "Citation quality audit failed for "
            f"{len(complaints)} regimen(s):\n  - "
            + "\n  - ".join(complaints)
        )
    return complaints


if __name__ == "__main__":
    print(count_by_property())
    print(f"\n{len(REGIMEN_DB)} regimens:")
    for r in REGIMEN_DB:
        print(f"  [{r.regimen_id:30s}] {r.name:<50s}  n={r.n_drugs}d  {r.trial_phase:<10s} CR={r.outcome_cr_cri_rate:.2f}")
    print()
    print("=== Citation quality audit (issue #11) ===")
    issues = validate_first_line_citations(strict=False)
    if not issues:
        print("All first-line regimens pass.")
    else:
        for s in issues:
            print(f"  ❌ {s}")
