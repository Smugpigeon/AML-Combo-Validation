"""Evidence-gated AML cell identity auditing.

The identity gate separates descriptive transcriptomic evidence from
orthogonal evidence that can establish malignant or normal cell identity.
Blast-prevalence priors and expression clusters never count as independent
genomic evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

GENOMIC_CALL_COLUMNS = ("scdna_malignant_call", "cnv_malignant_call")
ORTHOGONAL_CALL_COLUMNS = ("flow_malignant_call", "cite_malignant_call")
AML_REFERENCE_CALL_COLUMNS = ("aml_reference_malignant_call",)

PATIENT_CONTEXT_COLUMNS = (
    "patient_id",
    "scrna_blast_pct",
    "clinical_blast_pct",
    "fab_type",
    "eln2022_risk",
    "potential_driver_mutations",
    "chromosomal_abnormalities",
)


@dataclass(frozen=True)
class IdentityGatePolicy:
    """Pre-registered engineering thresholds for entering perturbation work."""

    high_transcriptomic_suspicion: float = 0.60
    minimum_normal_lineage_confidence: float = 0.50
    major_state_fraction: float = 0.05
    minimum_validated_patient_fraction: float = 0.20
    minimum_validated_fraction_per_major_state: float = 0.10
    maximum_conflict_fraction: float = 0.05
    maximum_unresolved_fraction: float = 0.30
    maximum_unreconciled_blast_gap_pct: float = 15.0


def require_identity_gate_summary(
    summary: Mapping[str, object],
    *,
    patient_ids: Iterable[object] = (),
) -> None:
    """Refuse perturbation inference until the audited identity gate passes."""

    if not bool(summary.get("drug_perturbation_stage_unlocked", False)):
        raise RuntimeError(
            "drug perturbation stage is locked because cell identity is not proven"
        )
    inputs = summary.get("inputs", {})
    audited = {
        str(value).strip()
        for value in (inputs.get("patients", []) if isinstance(inputs, Mapping) else [])
    }
    requested = {str(value).strip() for value in patient_ids}
    missing = sorted(requested - audited)
    if missing:
        raise RuntimeError(
            "drug perturbation requested patients absent from identity audit: "
            + ", ".join(missing)
        )


def _nullable_call(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, bool | np.bool_):
        return float(value)
    if isinstance(value, int | float | np.integer | np.floating):
        return float(bool(value)) if float(value) in {0.0, 1.0} else np.nan
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "positive", "malignant", "leukemic"}:
        return 1.0
    if text in {"0", "false", "no", "negative", "normal", "non_malignant"}:
        return 0.0
    return np.nan


def _collect_calls(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return np.empty((len(frame), 0), dtype=float)
    return np.column_stack(
        [frame[column].map(_nullable_call).to_numpy(dtype=float) for column in available]
    )


def _known_call_count(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=int)
    return np.isfinite(values).sum(axis=1)


def annotate_cell_identity(
    annotations: pd.DataFrame,
    *,
    policy: IdentityGatePolicy | None = None,
) -> pd.DataFrame:
    """Assign evidence-bounded identity status to each observed cell.

    Validation requires cell-level genomic evidence, or agreement between an
    orthogonal phenotype call and an AML-specific reference call. General
    expression suspicion and blast priors remain provisional.
    """

    policy = policy or IdentityGatePolicy()
    required = {
        "sample_id",
        "virtual_state_id",
        "consensus_lineage",
        "lineage_confidence",
        "confident_normal_lineage_anchor",
        "transcriptomic_suspicion_index",
        "state_assignment_confidence",
    }
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise KeyError(f"cell identity columns missing: {missing}")

    out = annotations.copy()
    genomic = _collect_calls(out, GENOMIC_CALL_COLUMNS)
    orthogonal = _collect_calls(out, ORTHOGONAL_CALL_COLUMNS)
    aml_reference = _collect_calls(out, AML_REFERENCE_CALL_COLUMNS)
    all_independent = np.concatenate((genomic, orthogonal, aml_reference), axis=1)

    genomic_known = _known_call_count(genomic)
    orthogonal_known = _known_call_count(orthogonal)
    aml_reference_known = _known_call_count(aml_reference)
    independent_known = _known_call_count(all_independent)

    any_positive = (
        np.any(all_independent == 1.0, axis=1)
        if all_independent.shape[1]
        else np.zeros(len(out), dtype=bool)
    )
    any_negative = (
        np.any(all_independent == 0.0, axis=1)
        if all_independent.shape[1]
        else np.zeros(len(out), dtype=bool)
    )
    conflict = any_positive & any_negative

    genomic_positive = (
        np.any(genomic == 1.0, axis=1)
        if genomic.shape[1]
        else np.zeros(len(out), dtype=bool)
    )
    genomic_negative = (
        np.any(genomic == 0.0, axis=1)
        if genomic.shape[1]
        else np.zeros(len(out), dtype=bool)
    )
    orthogonal_positive = (
        np.any(orthogonal == 1.0, axis=1)
        if orthogonal.shape[1]
        else np.zeros(len(out), dtype=bool)
    )
    orthogonal_negative = (
        np.any(orthogonal == 0.0, axis=1)
        if orthogonal.shape[1]
        else np.zeros(len(out), dtype=bool)
    )
    reference_positive = (
        np.any(aml_reference == 1.0, axis=1)
        if aml_reference.shape[1]
        else np.zeros(len(out), dtype=bool)
    )
    reference_negative = (
        np.any(aml_reference == 0.0, axis=1)
        if aml_reference.shape[1]
        else np.zeros(len(out), dtype=bool)
    )

    validated_malignant = ~conflict & (
        genomic_positive | (orthogonal_positive & reference_positive)
    )
    validated_normal = ~conflict & (
        (genomic_negative & orthogonal_negative)
        | (genomic_negative & reference_negative)
        | (orthogonal_negative & reference_negative)
    )

    normal_anchor = out["confident_normal_lineage_anchor"].fillna(False).astype(bool)
    lineage_confidence = pd.to_numeric(out["lineage_confidence"], errors="coerce").fillna(0)
    suspicion = pd.to_numeric(
        out["transcriptomic_suspicion_index"], errors="coerce"
    ).fillna(0)
    reference_supported_normal = (
        ~conflict
        & ~validated_malignant
        & ~validated_normal
        & normal_anchor
        & (lineage_confidence >= policy.minimum_normal_lineage_confidence)
    )
    provisional_malignant = (
        ~conflict
        & ~validated_malignant
        & ~validated_normal
        & ~reference_supported_normal
        & ~normal_anchor
        & (suspicion >= policy.high_transcriptomic_suspicion)
    )

    status = np.full(len(out), "unresolved", dtype=object)
    status[reference_supported_normal] = "reference_supported_normal"
    status[provisional_malignant] = "provisional_malignant"
    status[validated_normal] = "validated_normal"
    status[validated_malignant] = "validated_malignant"
    status[conflict] = "conflicting_evidence"

    out["cell_identity_status"] = status
    out["identity_is_validated"] = np.isin(
        status, ["validated_malignant", "validated_normal"]
    )
    out["identity_conflict"] = conflict
    out["genomic_identity_evidence_count"] = genomic_known
    out["orthogonal_identity_evidence_count"] = orthogonal_known
    out["aml_reference_evidence_count"] = aml_reference_known
    out["independent_identity_evidence_count"] = independent_known
    out["blast_prior_counted_as_independent_evidence"] = False
    return out


def _context_frame(patient_summary: pd.DataFrame) -> pd.DataFrame:
    if "patient_id" not in patient_summary.columns:
        raise KeyError("patient summary requires patient_id")
    columns = [column for column in PATIENT_CONTEXT_COLUMNS if column in patient_summary]
    return patient_summary[columns].drop_duplicates("patient_id").copy()


def _manifest_lookup(evidence_manifest: pd.DataFrame | None) -> pd.DataFrame:
    if evidence_manifest is None:
        return pd.DataFrame(columns=["patient_id", "blast_discrepancy_reconciled"])
    if "patient_id" not in evidence_manifest.columns:
        raise KeyError("evidence manifest requires patient_id")
    manifest = evidence_manifest.copy()
    if "blast_discrepancy_reconciled" not in manifest:
        manifest["blast_discrepancy_reconciled"] = False
    manifest["blast_discrepancy_reconciled"] = manifest[
        "blast_discrepancy_reconciled"
    ].map(_nullable_call).fillna(0).astype(bool)
    return manifest.drop_duplicates("patient_id")


def _status_fraction(group: pd.DataFrame, status: str) -> float:
    return float((group["cell_identity_status"] == status).mean())


def build_identity_gate(
    annotations: pd.DataFrame,
    patient_summary: pd.DataFrame,
    *,
    evidence_manifest: pd.DataFrame | None = None,
    policy: IdentityGatePolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build patient- and state-level identity gates plus an evidence plan."""

    policy = policy or IdentityGatePolicy()
    cells = annotate_cell_identity(annotations, policy=policy)
    context = _context_frame(patient_summary)
    manifest = _manifest_lookup(evidence_manifest)

    state_rows: list[dict[str, object]] = []
    for (patient_id, state_id), group in cells.groupby(
        ["sample_id", "virtual_state_id"], sort=True
    ):
        patient_total = int((cells["sample_id"] == patient_id).sum())
        state_fraction = len(group) / patient_total
        validated_fraction = float(group["identity_is_validated"].mean())
        conflict_fraction = float(group["identity_conflict"].mean())
        is_major = state_fraction >= policy.major_state_fraction
        passes = (
            not is_major
            or (
                validated_fraction >= policy.minimum_validated_fraction_per_major_state
                and conflict_fraction <= policy.maximum_conflict_fraction
            )
        )
        state_rows.append(
            {
                "patient_id": patient_id,
                "virtual_state_id": state_id,
                "n_cells": len(group),
                "state_fraction": state_fraction,
                "is_major_state": is_major,
                "validated_identity_fraction": validated_fraction,
                "conflict_fraction": conflict_fraction,
                "provisional_malignant_fraction": _status_fraction(
                    group, "provisional_malignant"
                ),
                "reference_supported_normal_fraction": _status_fraction(
                    group, "reference_supported_normal"
                ),
                "unresolved_fraction": _status_fraction(group, "unresolved"),
                "state_identity_gate_pass": passes,
            }
        )
    state_gate = pd.DataFrame(state_rows)

    patient_rows: list[dict[str, object]] = []
    plan_rows: list[dict[str, object]] = []
    for patient_id, group in cells.groupby("sample_id", sort=True):
        patient_context = context.loc[context["patient_id"] == patient_id]
        row = patient_context.iloc[0] if not patient_context.empty else pd.Series(dtype=object)
        patient_manifest = manifest.loc[manifest["patient_id"] == patient_id]
        reconciled = bool(
            patient_manifest["blast_discrepancy_reconciled"].iloc[0]
        ) if not patient_manifest.empty else False

        validated_fraction = float(group["identity_is_validated"].mean())
        conflict_fraction = float(group["identity_conflict"].mean())
        unresolved_fraction = _status_fraction(group, "unresolved")
        major_states = state_gate.loc[
            (state_gate["patient_id"] == patient_id) & state_gate["is_major_state"]
        ]
        all_major_states_anchored = bool(
            not major_states.empty and major_states["state_identity_gate_pass"].all()
        )

        scrna_blast = pd.to_numeric(row.get("scrna_blast_pct"), errors="coerce")
        clinical_blast = pd.to_numeric(row.get("clinical_blast_pct"), errors="coerce")
        blast_gap = (
            float(abs(scrna_blast - clinical_blast))
            if pd.notna(scrna_blast) and pd.notna(clinical_blast)
            else np.nan
        )
        blast_block = bool(
            np.isfinite(blast_gap)
            and blast_gap > policy.maximum_unreconciled_blast_gap_pct
            and not reconciled
        )

        blockers: list[str] = []
        if validated_fraction < policy.minimum_validated_patient_fraction:
            blockers.append(
                "insufficient cell-level genomic or orthogonal identity anchors"
            )
        if not all_major_states_anchored:
            blockers.append("one or more major virtual states lack validated identity anchors")
        if conflict_fraction > policy.maximum_conflict_fraction:
            blockers.append("cell-level identity evidence conflict exceeds policy")
        if unresolved_fraction > policy.maximum_unresolved_fraction:
            blockers.append("unresolved cell fraction exceeds policy")
        if blast_block:
            blockers.append("scRNA and clinical blast fractions require reconciliation")

        gate_pass = not blockers
        only_missing_anchors = bool(blockers) and all(
            "anchor" in blocker for blocker in blockers
        )
        if gate_pass:
            gate_status = "pass"
        elif only_missing_anchors:
            gate_status = "provisional_not_proven"
        else:
            gate_status = "blocked"

        top_lineage = str(group["consensus_lineage"].value_counts().index[0])
        mean_state_confidence = float(
            pd.to_numeric(group.get("state_assignment_confidence"), errors="coerce").mean()
        )
        patient_rows.append(
            {
                "patient_id": patient_id,
                "n_cells": len(group),
                "top_lineage": top_lineage,
                "validated_identity_fraction": validated_fraction,
                "validated_malignant_fraction": _status_fraction(
                    group, "validated_malignant"
                ),
                "validated_normal_fraction": _status_fraction(group, "validated_normal"),
                "provisional_malignant_fraction": _status_fraction(
                    group, "provisional_malignant"
                ),
                "reference_supported_normal_fraction": _status_fraction(
                    group, "reference_supported_normal"
                ),
                "unresolved_fraction": unresolved_fraction,
                "conflict_fraction": conflict_fraction,
                "major_state_count": len(major_states),
                "major_states_with_identity_anchor": int(
                    major_states["state_identity_gate_pass"].sum()
                ),
                "mean_state_assignment_confidence": mean_state_confidence,
                "scrna_blast_pct": scrna_blast,
                "clinical_blast_pct": clinical_blast,
                "absolute_blast_gap_pct": blast_gap,
                "blast_discrepancy_reconciled": reconciled,
                "bulk_mutations_context_only": row.get("potential_driver_mutations", ""),
                "bulk_cytogenetics_context_only": row.get("chromosomal_abnormalities", ""),
                "identity_gate_status": gate_status,
                "identity_gate_pass": gate_pass,
                "gate_blockers": " | ".join(blockers),
            }
        )

        evidence_needs = [
            "cell-level CNV or targeted single-cell genotype mapped to barcodes",
            "AML-specific malignant/normal reference mapping",
            "matched flow or CITE-seq phenotype for orthogonal identity",
        ]
        if blast_block:
            evidence_needs.append(
                "reconcile sample source, sampling time, denominator, and flow blast gate"
            )
        if mean_state_confidence < 0.20:
            evidence_needs.append("refine low-confidence state assignment before perturbation")
        plan_rows.append(
            {
                "patient_id": patient_id,
                "priority": "identity_before_perturbation",
                "required_evidence": " | ".join(evidence_needs),
                "drug_perturbation_stage_unlocked": gate_pass,
            }
        )

    patient_gate = pd.DataFrame(patient_rows)
    evidence_plan = pd.DataFrame(plan_rows)
    summary: dict[str, object] = {
        "scope": "cell_identity_before_drug_perturbation",
        "research_use_only": True,
        "policy": asdict(policy),
        "n_patients": int(patient_gate["patient_id"].nunique()),
        "n_cells": int(len(cells)),
        "patient_gate_counts": {
            str(key): int(value)
            for key, value in patient_gate["identity_gate_status"].value_counts().items()
        },
        "all_patients_pass_identity_gate": bool(patient_gate["identity_gate_pass"].all()),
        "drug_perturbation_stage_unlocked": bool(patient_gate["identity_gate_pass"].all()),
        "combination_prediction_stage_unlocked": False,
        "independence_rule": (
            "blast priors, expression clusters, bulk mutations, and bulk cytogenetics "
            "do not independently validate cell identity"
        ),
    }
    return cells, patient_gate, state_gate, evidence_plan, summary
