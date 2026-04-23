"""Append 12 drug rows to drug_mechanism_v1.csv for Problem 3.

Each row is annotated from FDA labels + AML clinical trial literature +
DrugBank. Values in [0, 1] with common steps {0, 0.3, 0.5, 0.8, 1.0}.

Columns (39 mech features + 4 metadata = 43):
  drug_id, drug_name, moa_family,
  15 tgt_*  (target axis)
  7 cs_*    (cell-state axis)
  7 role_*  (regimen role)
  10 tox_*  (toxicity stacking)
  notes
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


COL_ORDER = [
    "drug_id", "drug_name", "moa_family",
    # 15 target axes
    "tgt_FLT3", "tgt_IDH1", "tgt_IDH2", "tgt_BCL2", "tgt_DNMT",
    "tgt_MENIN_HOX", "tgt_CD33", "tgt_SMO", "tgt_RARA",
    "tgt_DNA_SYNTHESIS", "tgt_TOPO_II", "tgt_TP53_PATHWAY",
    "tgt_SPLICEOSOME", "tgt_RAS_MAPK", "tgt_JAK_STAT",
    # 7 cell-state axes
    "cs_apoptosis_priming", "cs_differentiation_induction",
    "cs_DNA_damage", "cs_cell_cycle_block", "cs_stem_cell_targeting",
    "cs_hypomethylation", "cs_immune_engagement",
    # 7 role axes
    "role_intensive_backbone", "role_low_intensity_backbone",
    "role_targeted_anchor", "role_targeted_overlay",
    "role_modifier", "role_maintenance_post_cr", "role_alert_only",
    # 10 toxicity axes
    "tox_myelosuppression", "tox_hepatotoxicity", "tox_cardiotoxicity",
    "tox_QT_prolongation", "tox_differentiation_syndrome", "tox_TLS_risk",
    "tox_infection_neutropenic", "tox_GI_toxicity", "tox_renal",
    "tox_drug_interactions_CYP3A",
    "notes",
]


def _row(**kwargs) -> dict:
    """Build a row with defaults (0 for everything) and override provided."""
    row = {c: 0.0 for c in COL_ORDER}
    row["notes"] = ""
    row.update(kwargs)
    return row


NEW_DRUGS = [
    # ------------------------------------------------------------------
    # Trametinib — MEK1/2 inhibitor (FDA melanoma 2013; AML research only)
    # Primary rationale: in FLT3-ITD AML, constitutive RAS-MAPK is active;
    # MEK inhibition is a parallel pathway block. Synergy studies with
    # FLT3i and BCL2i reported in preclinical AML.
    # ------------------------------------------------------------------
    _row(drug_id="trametinib", drug_name="Trametinib", moa_family="MEK_inhibitor",
         tgt_RAS_MAPK=1.0,
         cs_cell_cycle_block=0.5, cs_differentiation_induction=0.3,
         cs_apoptosis_priming=0.3,
         role_targeted_overlay=0.8,
         tox_myelosuppression=0.3, tox_hepatotoxicity=0.3,
         tox_cardiotoxicity=0.5,                   # LVEF reduction ~7%
         tox_GI_toxicity=0.8,                      # diarrhea common
         tox_drug_interactions_CYP3A=0.3,
         notes="MEKi; parallel-pathway block for FLT3-ITD; LVEF + diarrhea dose-limiting"),

    # ------------------------------------------------------------------
    # Selumetinib — MEK1/2 (FDA NF1 pediatric 2020); weaker than trametinib
    # in AML; similar profile but slightly lower potency.
    # ------------------------------------------------------------------
    _row(drug_id="selumetinib", drug_name="Selumetinib", moa_family="MEK_inhibitor",
         tgt_RAS_MAPK=0.8,
         cs_cell_cycle_block=0.5, cs_differentiation_induction=0.3,
         cs_apoptosis_priming=0.3,
         role_targeted_overlay=0.5,
         tox_myelosuppression=0.3, tox_hepatotoxicity=0.3,
         tox_cardiotoxicity=0.3,
         tox_GI_toxicity=0.8,
         tox_drug_interactions_CYP3A=0.3,
         notes="Weaker MEKi than trametinib; AML use is research overlay"),

    # ------------------------------------------------------------------
    # Ruxolitinib — JAK1/2 (FDA MF 2011, PV 2014, GvHD 2019)
    # AML niche: post-MPN AML (JAK2-V617F commonly persists into blast crisis)
    # ------------------------------------------------------------------
    _row(drug_id="ruxolitinib", drug_name="Ruxolitinib", moa_family="JAK_inhibitor",
         tgt_JAK_STAT=1.0,
         cs_cell_cycle_block=0.3, cs_immune_engagement=0.5,
         role_targeted_overlay=0.5,
         tox_myelosuppression=0.8,                 # thrombocytopenia dose-limiting
         tox_hepatotoxicity=0.3,
         tox_infection_neutropenic=0.5,            # immunosuppressive
         tox_GI_toxicity=0.3,
         tox_drug_interactions_CYP3A=0.3,
         notes="MPN anchor; AML role is post-MPN secondary AML or JAK2-mut cases"),

    # ------------------------------------------------------------------
    # Sorafenib — multikinase (FLT3, RAF, VEGFR, c-KIT, PDGFR)
    # FDA HCC/RCC/thyroid; extensively used off-label in FLT3-mut AML
    # pre-quizartinib/gilteritinib era.
    # ------------------------------------------------------------------
    _row(drug_id="sorafenib", drug_name="Sorafenib", moa_family="multikinase_FLT3",
         tgt_FLT3=0.8,                             # active but less selective
         tgt_RAS_MAPK=0.5,                         # RAF upstream
         cs_apoptosis_priming=0.3, cs_differentiation_induction=0.3,
         cs_cell_cycle_block=0.3, cs_stem_cell_targeting=0.3,
         role_targeted_overlay=0.8,
         tox_myelosuppression=0.5, tox_hepatotoxicity=0.5,
         tox_cardiotoxicity=0.3, tox_QT_prolongation=0.3,
         tox_differentiation_syndrome=0.3,
         tox_GI_toxicity=0.5,                      # hand-foot syndrome
         tox_drug_interactions_CYP3A=0.5,
         notes="Historical FLT3 multikinase; less selective than quiz/gilt; hand-foot + diarrhea"),

    # ------------------------------------------------------------------
    # Dasatinib — BCR-ABL + SRC-family + c-KIT (FDA CML/Ph+ALL)
    # AML role: Ph+ AML (rare), c-KIT-mut AML, SRC-pathway research.
    # ------------------------------------------------------------------
    _row(drug_id="dasatinib", drug_name="Dasatinib", moa_family="SRC_ABL_inhibitor",
         tgt_RAS_MAPK=0.3,                         # via SRC/RAS
         cs_cell_cycle_block=0.3, cs_stem_cell_targeting=0.3,
         role_targeted_overlay=0.3,                # limited AML role
         tox_myelosuppression=0.8,                 # known issue
         tox_hepatotoxicity=0.3, tox_QT_prolongation=0.3,
         tox_cardiotoxicity=0.5,                   # pleural effusion
         tox_GI_toxicity=0.3,
         tox_drug_interactions_CYP3A=0.5,
         notes="BCR-ABL/SRC; AML use limited to Ph+ALL niche + c-KIT-mut; pleural effusion risk"),

    # ------------------------------------------------------------------
    # Imatinib — BCR-ABL (FDA CML/Ph+ALL/GIST 2001)
    # AML role: essentially only BCR-ABL+ AML (~1% of AML) or c-KIT-mut.
    # ------------------------------------------------------------------
    _row(drug_id="imatinib", drug_name="Imatinib", moa_family="BCR_ABL_inhibitor",
         cs_cell_cycle_block=0.2,                  # minor
         role_alert_only=0.5,                      # only for BCR-ABL+ AML
         tox_myelosuppression=0.5, tox_hepatotoxicity=0.3,
         tox_GI_toxicity=0.3,
         tox_drug_interactions_CYP3A=0.5,
         notes="CML primary agent; AML use only for rare BCR-ABL+ AML or c-KIT-mut"),

    # ------------------------------------------------------------------
    # Nilotinib — BCR-ABL 2nd-gen (FDA CML 2007)
    # Same AML niche as imatinib but more potent; QT prolongation is black box.
    # ------------------------------------------------------------------
    _row(drug_id="nilotinib", drug_name="Nilotinib", moa_family="BCR_ABL_inhibitor_2gen",
         cs_cell_cycle_block=0.3,
         role_alert_only=0.5,
         tox_myelosuppression=0.5, tox_hepatotoxicity=0.3,
         tox_QT_prolongation=0.8,                  # BLACK BOX
         tox_GI_toxicity=0.3,
         tox_drug_interactions_CYP3A=0.8,          # strong CYP3A inhibitor/substrate
         notes="More potent than imatinib; QT prolongation black box warning; strong CYP3A interactions"),

    # ------------------------------------------------------------------
    # Ponatinib — pan-TKI including BCR-ABL T315I + FLT3 activity
    # Black box: arterial thrombosis. AML role: limited, mostly for T315I CML
    # progressing to AML or rare FLT3-mut salvage.
    # ------------------------------------------------------------------
    _row(drug_id="ponatinib", drug_name="Ponatinib", moa_family="pan_TKI",
         tgt_FLT3=0.5,                             # has activity but not primary
         tgt_RAS_MAPK=0.3,
         cs_apoptosis_priming=0.3, cs_cell_cycle_block=0.3,
         cs_stem_cell_targeting=0.3,
         role_targeted_overlay=0.3,
         tox_myelosuppression=0.5, tox_hepatotoxicity=0.5,
         tox_cardiotoxicity=1.0,                   # BLACK BOX: arterial thrombosis
         tox_QT_prolongation=0.5,
         tox_GI_toxicity=0.5,
         tox_drug_interactions_CYP3A=0.5,
         notes="pan-TKI with FLT3 activity; arterial thrombosis black box; limited AML role"),

    # ------------------------------------------------------------------
    # Crenolanib — FLT3 TKI, Type I inhibitor active on BOTH ITD and TKD
    # (including D835Y quizartinib-resistance variants). Phase 3 in R/R AML.
    # ------------------------------------------------------------------
    _row(drug_id="crenolanib", drug_name="Crenolanib", moa_family="FLT3i_both_variants",
         tgt_FLT3=1.0,                             # primary, both ITD + TKD
         cs_apoptosis_priming=0.3, cs_differentiation_induction=0.3,
         cs_stem_cell_targeting=0.3,
         role_targeted_overlay=1.0,
         tox_myelosuppression=0.5,
         tox_hepatotoxicity=0.8,                   # LFT elevations common
         tox_QT_prolongation=0.3,                  # less than quizartinib
         tox_differentiation_syndrome=0.3,
         tox_GI_toxicity=0.5,
         tox_drug_interactions_CYP3A=0.5,
         notes="Unique: covers both FLT3-ITD and TKD (incl. D835Y); hepatotoxicity > quiz"),

    # ------------------------------------------------------------------
    # Crizotinib — MET/ALK/ROS1 (FDA NSCLC)
    # No direct AML target. Alert-only for MET-amplified FLT3i-resistance
    # rare scenarios.
    # ------------------------------------------------------------------
    _row(drug_id="crizotinib", drug_name="Crizotinib", moa_family="MET_ALK_inhibitor",
         role_alert_only=1.0,
         tox_myelosuppression=0.3, tox_hepatotoxicity=0.5,
         tox_cardiotoxicity=0.3, tox_QT_prolongation=0.5,
         tox_GI_toxicity=0.5,
         tox_drug_interactions_CYP3A=0.5,
         notes="NSCLC drug; alert-only for rare MET-amplified AML / FLT3i resistance"),

    # ------------------------------------------------------------------
    # Alisertib (MLN8237) — Aurora Kinase A inhibitor (investigational)
    # Phase 2/3 in R/R AML; mitotic spindle disruption = anti-proliferative
    # via aneuploidy-induced apoptosis.
    # ------------------------------------------------------------------
    _row(drug_id="alisertib", drug_name="Alisertib", moa_family="AURK_inhibitor",
         cs_cell_cycle_block=1.0,                  # primary mechanism
         cs_DNA_damage=0.5,                        # aneuploidy
         cs_apoptosis_priming=0.3,
         role_targeted_overlay=0.5,
         tox_myelosuppression=0.8,                 # classic
         tox_hepatotoxicity=0.3,
         tox_infection_neutropenic=0.5,
         tox_GI_toxicity=0.5,                      # mucositis
         tox_drug_interactions_CYP3A=0.3,
         notes="Aurora-A inhibitor; mitotic arrest primary mechanism; myelosuppression dose-limiting"),

    # ------------------------------------------------------------------
    # Pacritinib — JAK2/FLT3/IRAK1 (FDA MF 2022, esp. MF with thrombocytopenia)
    # Key advantage: less myelosuppressive than ruxolitinib. AML use research.
    # ------------------------------------------------------------------
    _row(drug_id="pacritinib", drug_name="Pacritinib", moa_family="JAK2_FLT3_IRAK1_inhibitor",
         tgt_JAK_STAT=0.8,                         # JAK2 selective
         tgt_FLT3=0.5,                             # secondary
         cs_cell_cycle_block=0.3, cs_immune_engagement=0.3,
         role_targeted_overlay=0.5,
         tox_myelosuppression=0.3,                 # less than rux (advantage)
         tox_hepatotoxicity=0.3,
         tox_QT_prolongation=0.5,                  # trial signal
         tox_cardiotoxicity=0.3,
         tox_GI_toxicity=0.5,                      # diarrhea
         tox_drug_interactions_CYP3A=0.3,
         notes="JAK2-selective; less myelosuppressive than ruxolitinib; AML overlay candidate"),
]


def main():
    csv_path = Path("src/combo_val/knowledge/drug_mechanism_v1.csv")
    existing = pd.read_csv(csv_path)

    # Build new rows
    new_df = pd.DataFrame(NEW_DRUGS)
    # Force column order to match existing
    new_df = new_df[existing.columns.tolist()]

    # Schema validation
    assert list(new_df.columns) == list(existing.columns)
    assert len(new_df) == 12

    # Check for duplicates against existing rows
    existing_ids = set(existing["drug_id"])
    dup = [d for d in new_df["drug_id"] if d in existing_ids]
    if dup:
        raise RuntimeError(f"Duplicate drug_ids: {dup}")

    # Validate ranges [0, 1] for mechanism cols
    mech_cols = [c for c in existing.columns
                 if c.startswith(("tgt_", "cs_", "role_", "tox_"))]
    bad = []
    for c in mech_cols:
        for idx, v in enumerate(new_df[c]):
            if not (0.0 <= float(v) <= 1.0):
                bad.append((new_df.iloc[idx]["drug_id"], c, v))
    if bad:
        raise ValueError(f"Out-of-range values: {bad}")

    merged = pd.concat([existing, new_df], ignore_index=True)
    merged.to_csv(csv_path, index=False)

    print(f"[annotate] added {len(new_df)} drugs; matrix now {merged.shape}")
    print(f"[annotate] all drug_ids: {merged['drug_id'].tolist()}")


if __name__ == "__main__":
    main()
