"""Thin adapter from web submission → combo_val kit → persisted report.

Keeps the web layer decoupled from kit internals — if the kit's function
signatures change, we only update this file.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from app.config import get_settings


def _patient_dir(user_id: str, submission_id: str) -> Path:
    root = get_settings().STORAGE_ROOT / str(user_id) / str(submission_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_kit_prediction(
    user_id: str,
    submission_id: str,
    input_json: dict,
    rna_counts_csv: Optional[str] = None,
    rna_full_csv: Optional[str] = None,
) -> dict:
    """Run the full kit on one patient and write artifacts to disk.

    Args:
      user_id, submission_id: for scoping the output directory.
      input_json: dict matching KitInput schema — mutations (list of
        dicts), karyotype_text, fusions, labs (wbc, platelet, ...),
        age, sex, etc.
      rna_counts_csv: absolute path to a 2-col CSV (symbol, count) for
        the 5000-gene Layer-3 MLP panel. Required.
      rna_full_csv: absolute path to a full-transcriptome CSV for the
        RNA-Seq outlier analysis. Optional.

    Returns dict with keys:
      report_md_path, report_pdf_path, report_html_path,
      report_figure_path, dna_summary, rna_outlier,
      predicted_eln2017, top_regimen_name.
    """
    from combo_val.clinical.kit_predict import predict_for_patient
    from combo_val.clinical.kit_schema import KitInput, MutationCall
    from combo_val.clinical.dna_report import (
        export_dna_summary_csv, render_dna_summary_figure,
    )
    from combo_val.clinical.patient_report import export_clinical_report

    out_dir = _patient_dir(user_id, submission_id)

    # --- Load kit assets (preprocessor bundle + kept genes) ---
    assets_root = get_settings().KIT_ASSETS_ROOT
    bundle = joblib.load(assets_root / "beataml_rna_preprocessor.joblib")
    kept_genes = bundle["kept_genes"]

    # --- Parse RNA counts CSV ---
    if not rna_counts_csv or not Path(rna_counts_csv).exists():
        raise ValueError("rna_counts_csv is required (2 columns: symbol,count)")
    rna_df = pd.read_csv(rna_counts_csv)
    # Accept either (symbol, count) or (gene, value) column names
    if rna_df.shape[1] < 2:
        raise ValueError("rna_counts_csv must have at least 2 columns")
    rna_series = pd.Series(
        data=rna_df.iloc[:, 1].astype(float).values,
        index=rna_df.iloc[:, 0].astype(str).values,
    )
    rna_series.index.name = None

    # --- Parse optional full-transcriptome CSV ---
    rna_full_series: Optional[pd.Series] = None
    if rna_full_csv and Path(rna_full_csv).exists():
        full_df = pd.read_csv(rna_full_csv)
        rna_full_series = pd.Series(
            data=full_df.iloc[:, 1].astype(float).values,
            index=full_df.iloc[:, 0].astype(str).values,
        )
        rna_full_series.index.name = None

    # --- Build KitInput from the JSON payload ---
    mut_calls = [
        MutationCall(
            gene=m["gene"],
            variant_type=m.get("variant_type"),
            vaf=m.get("vaf"),
            is_ITD=bool(m.get("is_ITD", False)),
            is_TKD=bool(m.get("is_TKD", False)),
            allelic_ratio=m.get("allelic_ratio"),
            is_biallelic=bool(m.get("is_biallelic", False)),
            # Issues #4 + #9 + #13 — extended fields from LLM smart-paste
            # for ELN 2022 / WHO 2022 / hotspot codon classification.
            is_bzip=bool(m.get("is_bzip", False)),
            is_multi_hit=bool(m.get("is_multi_hit", False)),
            protein_codon=m.get("protein_codon"),
        )
        for m in input_json.get("mutations", [])
    ]
    kit = KitInput(
        patient_id=input_json.get("patient_label") or str(submission_id),
        mutations=mut_calls,
        karyotype_text=input_json.get("karyotype_text"),
        fusions=input_json.get("fusions", []),
        wbc=input_json.get("wbc"),
        platelet=input_json.get("platelet"),
        hemoglobin=input_json.get("hemoglobin"),
        ldh=input_json.get("ldh"),
        alt=input_json.get("alt"),
        ast=input_json.get("ast"),
        albumin=input_json.get("albumin"),
        creatinine=input_json.get("creatinine"),
        blast_pct_bm=input_json.get("blast_pct_bm"),
        blast_pct_pb=input_json.get("blast_pct_pb"),
        age=input_json.get("age"),
        sex=input_json.get("sex"),
        is_relapse=input_json.get("is_relapse"),
        prior_mds=input_json.get("prior_mds"),
        prior_chemo=input_json.get("prior_chemo"),
        is_initial_diagnosis=input_json.get("is_initial_diagnosis"),
        rna_expression_full=rna_full_series,
        intent_comment=input_json.get("intent_comment"),
    )

    # --- Run the kit ---
    kit_out = predict_for_patient(rna_series, kit, top_k=5)

    # --- Export DNA CSVs + figure ---
    export_dna_summary_csv(kit_out.dna_summary, kit.patient_id, out_dir.parent)
    # The export helper writes to out_dir.parent/patient_<id>/; our dir
    # is already the final location, so we point it at out_dir's parent
    # and the kit will make a subdir named patient_<id>. Simpler: render
    # figure + clinical report directly into our final dir.
    fig_path = out_dir / "dna_profile.png"
    render_dna_summary_figure(kit_out.dna_summary, kit.patient_id, fig_path)

    # --- Export clinical report (MD + HTML + PDF) ---
    report_paths = export_clinical_report(
        kit, kit_out, out_dir, also_render_pdf=True,
    )

    # --- Persist summary JSON for quick dashboard reads ---
    (out_dir / "dna_summary.json").write_text(
        json.dumps(kit_out.dna_summary, default=str, indent=2),
    )
    (out_dir / "rna_outlier.json").write_text(
        json.dumps(kit_out.rna_outlier, default=str, indent=2),
    )

    top_regimen = (kit_out.top_regimens or [{}])[0].get("name")

    return {
        "report_md_path": report_paths.get("markdown"),
        "report_pdf_path": report_paths.get("pdf"),
        "report_html_path": report_paths.get("html"),
        "report_figure_path": str(fig_path),
        "dna_summary": kit_out.dna_summary,
        "rna_outlier": kit_out.rna_outlier,
        "predicted_eln2017": kit_out.predicted_eln2017,
        "top_regimen_name": top_regimen,
        "fitness_flag": kit_out.fitness_flag,
        "cautions": kit_out.cautions,
        "confidence_notes": kit_out.confidence_notes,
    }
