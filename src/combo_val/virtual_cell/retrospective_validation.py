"""Leakage-resistant retrospective validation utilities for AML virtual cells.

The public scTherapy combination matrices contain zero-dose edges. Those edges
are real single-agent measurements from the same assay plate and can be used as
a monotherapy challenge without treating combination wells as monotherapy.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DRUG_ALIASES: Mapping[str, str] = {
    "afatinib": "Afatinib (BIBW-2992)",
    "alisertib": "Alisertib (MLN8237)",
    "alvocidib": "Flavopiridol",
    "bi2536": "BI-2536",
    "bortezomib": "Bortezomib (Velcade)",
    "dactolisib": "BEZ235",
    "dasatinib": "Dasatinib",
    "panobinostat": "Panobinostat",
    "selinexor": "Selinexor",
    "sirolimus": "Rapamycin",
    "trametinib": "Trametinib (GSK1120212)",
}


@dataclass(frozen=True)
class MonotherapyGatePolicy:
    """Predeclared thresholds for a public, directional feasibility gate."""

    minimum_patients: int = 3
    minimum_evaluable_patient_drug_rows: int = 12
    minimum_patients_with_positive_spearman: int = 2
    minimum_median_patient_spearman: float = 0.20
    minimum_median_increment_over_drug_mean: float = 0.00
    maximum_duplicate_response_range: float = 1e-6


def normalize_drug_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def model_drug_name(value: object) -> str | None:
    return DRUG_ALIASES.get(normalize_drug_name(value))


def normalize_patient_id(value: object) -> str:
    text = re.sub(r"\s+", "", str(value).strip().lower())
    match = re.search(r"(\d+)$", text)
    return f"patient{int(match.group(1))}" if match else text


def extract_monotherapy_edges(combo: pd.DataFrame) -> pd.DataFrame:
    """Extract zero-dose matrix edges and collapse exact repeated measurements."""

    required = {
        "Patient",
        "Drug1",
        "Drug2",
        "Dose1",
        "Dose2",
        "DoseUnit",
        "Response",
    }
    missing = sorted(required - set(combo.columns))
    if missing:
        raise KeyError(f"combination matrix columns missing: {missing}")

    frame = combo.copy()
    frame["Dose1"] = pd.to_numeric(frame["Dose1"], errors="coerce")
    frame["Dose2"] = pd.to_numeric(frame["Dose2"], errors="coerce")
    frame["Response"] = pd.to_numeric(frame["Response"], errors="coerce")

    left = frame.loc[
        frame["Dose1"].gt(0) & frame["Dose2"].eq(0),
        ["Patient", "Drug1", "Dose1", "DoseUnit", "Response"],
    ].rename(columns={"Drug1": "drug_id", "Dose1": "dose"})
    right = frame.loc[
        frame["Dose2"].gt(0) & frame["Dose1"].eq(0),
        ["Patient", "Drug2", "Dose2", "DoseUnit", "Response"],
    ].rename(columns={"Drug2": "drug_id", "Dose2": "dose"})
    edges = pd.concat([left, right], ignore_index=True)
    edges = edges.dropna(subset=["Patient", "drug_id", "dose", "Response"])
    edges["patient_id"] = edges["Patient"].map(normalize_patient_id)
    edges["drug_id"] = edges["drug_id"].astype(str).str.strip()
    edges["dose_unit"] = edges["DoseUnit"].astype(str).str.strip()
    edges["model_drug_id"] = edges["drug_id"].map(model_drug_name)

    group_columns = ["patient_id", "drug_id", "model_drug_id", "dose", "dose_unit"]
    grouped = edges.groupby(group_columns, dropna=False, sort=True)["Response"]
    collapsed = grouped.agg(
        observed_inhibition="median",
        repeated_edge_count="size",
        repeated_edge_min="min",
        repeated_edge_max="max",
    ).reset_index()
    collapsed["repeated_edge_range"] = (
        collapsed["repeated_edge_max"] - collapsed["repeated_edge_min"]
    )
    return collapsed.sort_values(["patient_id", "drug_id", "dose"]).reset_index(
        drop=True
    )


def _fixed_logdose_auc(dose: np.ndarray, response: np.ndarray) -> tuple[float, float]:
    """Return mean inhibition over 0.1--1000 nM and observed dose coverage."""

    order = np.argsort(dose)
    dose = np.asarray(dose, dtype=float)[order]
    response = np.clip(np.asarray(response, dtype=float)[order], 0.0, 100.0)
    unique_dose, inverse = np.unique(dose, return_inverse=True)
    if len(unique_dose) < 2:
        return float("nan"), 0.0
    unique_response = np.asarray(
        [np.median(response[inverse == index]) for index in range(len(unique_dose))]
    )
    log_dose = np.log10(unique_dose)
    lower, upper = -1.0, 3.0
    grid = np.linspace(lower, upper, 161)
    values = np.interp(
        grid,
        log_dose,
        unique_response,
        left=0.0,
        right=float(unique_response[-1]),
    )
    segment_width = np.diff(grid)
    area = np.sum((values[:-1] + values[1:]) * 0.5 * segment_width)
    coverage = max(0.0, min(upper, log_dose[-1]) - max(lower, log_dose[0])) / (
        upper - lower
    )
    return float(area / (upper - lower)), float(coverage)


def summarize_monotherapy_edges(edges: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient_id",
        "drug_id",
        "model_drug_id",
        "dose",
        "dose_unit",
        "observed_inhibition",
        "repeated_edge_range",
    }
    missing = sorted(required - set(edges.columns))
    if missing:
        raise KeyError(f"monotherapy edge columns missing: {missing}")

    rows: list[dict[str, object]] = []
    for (patient_id, drug_id, model_drug_id, dose_unit), group in edges.groupby(
        ["patient_id", "drug_id", "model_drug_id", "dose_unit"],
        dropna=False,
        sort=True,
    ):
        within_cap = group.loc[pd.to_numeric(group["dose"]).le(1000.0)].copy()
        if within_cap.empty:
            continue
        auc, coverage = _fixed_logdose_auc(
            within_cap["dose"].to_numpy(dtype=float),
            within_cap["observed_inhibition"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "patient_id": patient_id,
                "drug_id": drug_id,
                "model_drug_id": model_drug_id,
                "dose_unit": dose_unit,
                "n_positive_doses_le_1um": int(within_cap["dose"].nunique()),
                "minimum_positive_dose": float(within_cap["dose"].min()),
                "maximum_positive_dose_le_1um": float(within_cap["dose"].max()),
                "observed_dss_like_0_1um": auc,
                "logdose_coverage_fraction": coverage,
                "maximum_repeated_edge_range": float(
                    group["repeated_edge_range"].max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["patient_id", "drug_id"]).reset_index(
        drop=True
    )


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    values = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(values) < 3 or values["left"].nunique() < 2 or values["right"].nunique() < 2:
        return float("nan")
    return float(spearmanr(values["left"], values["right"]).statistic)


def evaluate_monotherapy_predictions(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    policy: MonotherapyGatePolicy | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate frozen patient-specific scores against sealed drug summaries."""

    policy = policy or MonotherapyGatePolicy()
    prediction_required = {
        "challenge_row_id",
        "patient_id",
        "drug_id",
        "predicted_sensitivity_score",
        "drug_mean_sensitivity_score",
    }
    outcome_required = {"challenge_row_id", "observed_dss_like_0_1um"}
    missing = sorted(prediction_required - set(predictions.columns))
    if missing:
        raise KeyError(f"prediction columns missing: {missing}")
    missing = sorted(outcome_required - set(outcomes.columns))
    if missing:
        raise KeyError(f"outcome columns missing: {missing}")

    merged = predictions.merge(
        outcomes,
        on="challenge_row_id",
        how="inner",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for patient_id, group in merged.groupby("patient_id", sort=True):
        model = _safe_spearman(
            group["predicted_sensitivity_score"],
            group["observed_dss_like_0_1um"],
        )
        baseline = _safe_spearman(
            group["drug_mean_sensitivity_score"],
            group["observed_dss_like_0_1um"],
        )
        rows.append(
            {
                "patient_id": patient_id,
                "n_drugs": int(len(group)),
                "patient_specific_spearman": model,
                "drug_mean_spearman": baseline,
                "spearman_increment": model - baseline,
            }
        )
    patient_metrics = pd.DataFrame(rows)
    finite = patient_metrics.dropna(
        subset=["patient_specific_spearman", "drug_mean_spearman"]
    )
    median_model = float(finite["patient_specific_spearman"].median()) if len(finite) else float("nan")
    median_increment = float(finite["spearman_increment"].median()) if len(finite) else float("nan")
    positive_patients = int((finite["patient_specific_spearman"] > 0).sum())
    blockers: list[str] = []
    if len(finite) < policy.minimum_patients:
        blockers.append("fewer than the predeclared minimum number of evaluable patients")
    if len(merged) < policy.minimum_evaluable_patient_drug_rows:
        blockers.append("insufficient evaluable patient-drug rows")
    if (
        "maximum_repeated_edge_range" in merged
        and pd.to_numeric(merged["maximum_repeated_edge_range"], errors="coerce").max()
        > policy.maximum_duplicate_response_range
    ):
        blockers.append("repeated zero-dose edges disagree beyond tolerance")
    if positive_patients < policy.minimum_patients_with_positive_spearman:
        blockers.append("too few patients have positive within-patient rank correlation")
    if not np.isfinite(median_model) or median_model < policy.minimum_median_patient_spearman:
        blockers.append("median patient-specific Spearman is below threshold")
    if not np.isfinite(median_increment) or median_increment < policy.minimum_median_increment_over_drug_mean:
        blockers.append("patient-specific model does not improve on the drug-mean baseline")

    summary: dict[str, object] = {
        "stage": "real_single_drug_viability_direction",
        "research_use_only": True,
        "n_evaluated_rows": int(len(merged)),
        "n_evaluable_patients": int(len(finite)),
        "patients_with_positive_spearman": positive_patients,
        "median_patient_specific_spearman": median_model,
        "median_drug_mean_spearman": (
            float(finite["drug_mean_spearman"].median()) if len(finite) else float("nan")
        ),
        "median_increment_over_drug_mean": median_increment,
        "policy": asdict(policy),
        "viability_direction_gate_pass": not blockers,
        "post_treatment_transcriptome_gate": "not_testable_from_public_data",
        "blockers": blockers,
        "limitations": [
            "only three public patients",
            "combination panels were selected by the original scTherapy study",
            "zero-dose edges validate viability, not a post-treatment transcriptome",
            "the current analysis is retrospective and cannot recreate a pristine blind",
        ],
    }
    return patient_metrics, summary


def combination_unlock_decision(
    identity_summary: Mapping[str, object],
    monotherapy_summary: Mapping[str, object],
) -> dict[str, object]:
    identity_pass = bool(
        identity_summary.get("retrospective_drug_validation_stage_unlocked", False)
    )
    monotherapy_pass = bool(monotherapy_summary.get("viability_direction_gate_pass", False))
    blockers: list[str] = []
    if not identity_pass:
        blockers.append("retrospective cell-identity replication gate did not pass")
    if not monotherapy_pass:
        blockers.append("real single-drug viability-direction gate did not pass")
    return {
        "research_use_only": True,
        "identity_gate_pass": identity_pass,
        "single_drug_gate_pass": monotherapy_pass,
        "combination_training_unlocked": not blockers,
        "clinical_combination_use_unlocked": False,
        "blockers": blockers,
        "boundary": (
            "A pass permits research-only combination benchmarking. It never permits "
            "automatic treatment selection or clinical dosing."
        ),
    }
