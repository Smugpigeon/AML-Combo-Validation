"""Export per-patient DNA-level tables (CSV + PNG figure) for clinical review.

Runs the kit on the two synthetic demo patients and writes:
  runs/dna_reports/patient_SYNTHETIC-001/
    ├── driver_mutations.csv         ← the table clinicians open in Excel
    ├── fusion_analysis.csv
    ├── cytogenetics.csv
    ├── targetability.csv
    ├── sample_qc.csv
    ├── dna_summary.json             ← full structured dict
    └── dna_profile.png              ← single-page publication figure
  runs/dna_reports/patient_SYNTHETIC-002/...
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib

from combo_val.clinical.demo_kit_run import _synthetic_rna_counts
from combo_val.clinical.dna_report import (
    export_dna_summary_csv,
    generate_patient_readme,
    render_dna_summary_figure,
)
from combo_val.clinical.kit_predict import predict_for_patient
from combo_val.clinical.kit_schema import KitInput, MutationCall


def _run_patient(patient_id: str, rna, kit, out_root: Path):
    out = predict_for_patient(rna, kit, top_k=5)
    paths = export_dna_summary_csv(out.dna_summary, patient_id, out_root)
    fig_path = out_root / f"patient_{patient_id}" / "dna_profile.png"
    render_dna_summary_figure(out.dna_summary, patient_id, fig_path)
    paths["figure"] = str(fig_path)

    # Per-patient README with key findings + file manifest + reading guide link
    readme_path = out_root / f"patient_{patient_id}" / "README.md"
    generate_patient_readme(
        out.dna_summary, patient_id, kit_output=out, out_path=readme_path,
    )
    paths["readme"] = str(readme_path)

    print(f"\n=== {patient_id} — ELN: {out.predicted_eln2017} ===")
    for name, p in paths.items():
        print(f"  {name:<25s}  {p}")
    return paths


def main():
    warnings.filterwarnings("ignore")
    bundle = joblib.load("data/canonical/beataml_rna_preprocessor.joblib")
    kept = bundle["kept_genes"]
    out_root = Path("runs/dna_reports")

    # --- Patient 1: FLT3-ITD + NPM1 (young, fit) ---
    rna1 = _synthetic_rna_counts(kept, "young_flt3")
    kit1 = KitInput(
        patient_id="SYNTHETIC-001",
        mutations=[
            MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.62, vaf=0.45),
            MutationCall(gene="NPM1", variant_type="missense", vaf=0.42),
            MutationCall(gene="DNMT3A", variant_type="frameshift", vaf=0.48),
        ],
        karyotype_text="46,XX[20]",
        fusions=[],
        wbc=95.0, platelet=32.0, hemoglobin=8.5, ldh=1240.0,
        alt=28.0, ast=35.0, albumin=3.5,
        age=45, sex="female", is_initial_diagnosis=True,
    )
    _run_patient("SYNTHETIC-001", rna1, kit1, out_root)

    # --- Patient 2: TP53 + complex karyotype (elderly, unfit) ---
    rna2 = _synthetic_rna_counts(kept, "elderly_tp53")
    kit2 = KitInput(
        patient_id="SYNTHETIC-002",
        mutations=[
            MutationCall(gene="TP53", variant_type="missense", vaf=0.55),
            MutationCall(gene="ASXL1", variant_type="nonsense", vaf=0.42),
        ],
        karyotype_text="45,XY,-7,del(5)(q13q33),+8,t(3;3)(q21;q26),del(17)(p13)[18]/46,XY[2]",
        fusions=[],
        wbc=12.0, platelet=25.0, hemoglobin=7.8, ldh=850.0,
        alt=22.0, ast=30.0, albumin=2.9,
        age=72, sex="male", prior_mds=True, is_initial_diagnosis=True,
    )
    _run_patient("SYNTHETIC-002", rna2, kit2, out_root)

    # --- Patient 3: APL (PML-RARA) — to show fusion-driven recommendations ---
    rna3 = _synthetic_rna_counts(kept, "young_flt3")  # placeholder RNA
    kit3 = KitInput(
        patient_id="SYNTHETIC-003-APL",
        mutations=[],
        karyotype_text="46,XX,t(15;17)(q22;q21)[20]",
        fusions=["PML-RARA"],
        wbc=2.5, platelet=30.0, hemoglobin=9.0, ldh=320.0,
        alt=25.0, ast=28.0, albumin=4.0,
        age=38, sex="female", is_initial_diagnosis=True,
    )
    _run_patient("SYNTHETIC-003-APL", rna3, kit3, out_root)


if __name__ == "__main__":
    main()
