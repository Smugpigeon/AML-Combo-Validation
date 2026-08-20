"""Defensive patient-state perturbation utilities for AML Virtual Cell v1.6.

This module does not generate a post-treatment transcriptome. It combines
outcome-blind single-drug anchors across observed baseline cell states, audits
the support domain, and disables pair-synergy models that fail discriminability
or grouped-split validation gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .challenge_firewall import assert_label_free


SUPPORT_ORDER = {"rejected": 0, "directional": 1, "supported": 2}


@dataclass(frozen=True)
class SupportPolicy:
    minimum_gene_coverage: float = 0.65
    full_gene_coverage: float = 0.90
    minimum_molecular_similarity: float = 0.30
    supported_molecular_similarity: float = 0.50
    rejected_qc_severities: tuple[str, ...] = ("ood", "far_ood")


@dataclass(frozen=True)
class PredictionSupport:
    level: str
    reasons: tuple[str, ...]
    qc_severity: str
    gene_coverage_fraction: float
    molecular_similarity: float | None
    model_in_vocabulary: bool


@dataclass(frozen=True)
class PairSplitAudit:
    n_rows: int
    n_train_rows: int
    n_validation_rows: int
    n_unique_pairs: int
    n_train_pairs: int
    n_validation_pairs: int
    pair_overlap_count: int


@dataclass(frozen=True)
class EmbeddingAudit:
    enabled: bool
    n_drugs: int
    unique_embedding_fraction: float
    zero_distance_pair_fraction: float
    median_pair_distance: float
    prediction_std: float | None
    reasons: tuple[str, ...]


def canonical_pair_key(drug_a: object, drug_b: object) -> str:
    left = str(drug_a).strip()
    right = str(drug_b).strip()
    return "||".join(sorted((left, right)))


def grouped_pair_split(
    frame: pd.DataFrame,
    *,
    drug_a_column: str,
    drug_b_column: str,
    validation_fraction: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, PairSplitAudit]:
    """Split by unordered drug pair so no pair appears on both sides."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    missing = [
        column
        for column in (drug_a_column, drug_b_column)
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(f"pair columns missing: {missing}")

    groups = np.asarray(
        [
            canonical_pair_key(a, b)
            for a, b in zip(
                frame[drug_a_column],
                frame[drug_b_column],
                strict=True,
            )
        ],
        dtype=object,
    )
    if len(set(groups)) < 2:
        raise ValueError("at least two unique drug pairs are required")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=random_state,
    )
    train_index, validation_index = next(splitter.split(frame, groups=groups))
    train_pairs = set(groups[train_index])
    validation_pairs = set(groups[validation_index])
    overlap = train_pairs & validation_pairs
    if overlap:
        raise AssertionError(f"drug-pair leakage detected: {sorted(overlap)[:5]}")

    audit = PairSplitAudit(
        n_rows=len(frame),
        n_train_rows=len(train_index),
        n_validation_rows=len(validation_index),
        n_unique_pairs=len(set(groups)),
        n_train_pairs=len(train_pairs),
        n_validation_pairs=len(validation_pairs),
        pair_overlap_count=len(overlap),
    )
    return train_index, validation_index, audit


def audit_embedding_discriminability(
    embeddings: np.ndarray,
    *,
    predictions: np.ndarray | None = None,
    distance_tolerance: float = 1e-6,
    minimum_unique_fraction: float = 0.90,
    maximum_zero_distance_fraction: float = 0.10,
    minimum_prediction_std: float = 0.05,
) -> EmbeddingAudit:
    """Detect representation collapse before a synergy head can be enabled."""
    values = np.asarray(embeddings, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("embeddings must have shape [n_drugs, n_features], n_drugs >= 2")
    if not np.isfinite(values).all():
        raise ValueError("embeddings contain non-finite values")

    rounded = np.round(values, decimals=6)
    unique_fraction = len(np.unique(rounded, axis=0)) / len(values)

    differences = values[:, None, :] - values[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=-1))
    upper = distances[np.triu_indices(len(values), k=1)]
    zero_fraction = float(np.mean(upper <= distance_tolerance))
    median_distance = float(np.median(upper))

    prediction_std: float | None = None
    reasons: list[str] = []
    if unique_fraction < minimum_unique_fraction:
        reasons.append(
            f"unique embedding fraction {unique_fraction:.3f} "
            f"< {minimum_unique_fraction:.3f}"
        )
    if zero_fraction > maximum_zero_distance_fraction:
        reasons.append(
            f"zero-distance pair fraction {zero_fraction:.3f} "
            f"> {maximum_zero_distance_fraction:.3f}"
        )

    if predictions is not None:
        prediction_values = np.asarray(predictions, dtype=float).reshape(-1)
        finite = prediction_values[np.isfinite(prediction_values)]
        if len(finite) < 2:
            reasons.append("fewer than two finite synergy predictions")
        else:
            prediction_std = float(np.std(finite))
            if prediction_std < minimum_prediction_std:
                reasons.append(
                    f"prediction std {prediction_std:.6f} "
                    f"< {minimum_prediction_std:.6f}"
                )

    return EmbeddingAudit(
        enabled=not reasons,
        n_drugs=len(values),
        unique_embedding_fraction=float(unique_fraction),
        zero_distance_pair_fraction=zero_fraction,
        median_pair_distance=median_distance,
        prediction_std=prediction_std,
        reasons=tuple(reasons),
    )


def require_valid_synergy_model(audit: EmbeddingAudit) -> None:
    if not audit.enabled:
        raise RuntimeError(
            "pair-synergy model disabled by discriminability gate: "
            + "; ".join(audit.reasons)
        )


def assess_prediction_support(
    *,
    qc_severity: str,
    gene_coverage_fraction: float,
    molecular_similarity: float | None,
    model_in_vocabulary: bool,
    policy: SupportPolicy | None = None,
) -> PredictionSupport:
    """Assign supported, directional, or rejected status."""
    policy = policy or SupportPolicy()
    qc = str(qc_severity).strip().lower()
    coverage = float(gene_coverage_fraction)
    similarity = (
        None if molecular_similarity is None or pd.isna(molecular_similarity)
        else float(molecular_similarity)
    )
    reasons: list[str] = []

    if qc in policy.rejected_qc_severities:
        reasons.append(f"RNA QC severity is {qc}")
    if not np.isfinite(coverage) or coverage < policy.minimum_gene_coverage:
        reasons.append(
            f"gene coverage {coverage:.3f} < {policy.minimum_gene_coverage:.3f}"
        )
    if similarity is None:
        reasons.append("molecular similarity is unavailable")
    elif similarity < policy.minimum_molecular_similarity:
        reasons.append(
            f"molecular similarity {similarity:.3f} "
            f"< {policy.minimum_molecular_similarity:.3f}"
        )

    if reasons:
        level = "rejected"
    elif (
        qc == "ok"
        and coverage >= policy.full_gene_coverage
        and model_in_vocabulary
        and similarity is not None
        and similarity >= policy.supported_molecular_similarity
    ):
        level = "supported"
    else:
        level = "directional"
        if qc != "ok":
            reasons.append(f"RNA QC is {qc}, not clean in-distribution")
        if coverage < policy.full_gene_coverage:
            reasons.append(
                f"gene coverage {coverage:.3f} is below full-support threshold "
                f"{policy.full_gene_coverage:.3f}"
            )
        if not model_in_vocabulary:
            reasons.append("drug is outside the fitted categorical vocabulary")
        if similarity is not None and similarity < policy.supported_molecular_similarity:
            reasons.append(
                f"molecular similarity {similarity:.3f} is extrapolative"
            )

    return PredictionSupport(
        level=level,
        reasons=tuple(reasons),
        qc_severity=qc,
        gene_coverage_fraction=coverage,
        molecular_similarity=similarity,
        model_in_vocabulary=bool(model_in_vocabulary),
    )


def combine_pair_support(
    drug_a: PredictionSupport,
    drug_b: PredictionSupport,
) -> tuple[str, tuple[str, ...]]:
    level = min(
        (drug_a.level, drug_b.level),
        key=lambda value: SUPPORT_ORDER[value],
    )
    reasons = tuple(
        f"drug_a: {reason}" for reason in drug_a.reasons
    ) + tuple(
        f"drug_b: {reason}" for reason in drug_b.reasons
    )
    return level, reasons


def state_aware_aggregate(
    frame: pd.DataFrame,
    *,
    patient_column: str = "patient_id",
    drug_column: str = "drug_id",
    prediction_column: str = "predicted_auc",
    state_fraction_column: str = "state_fraction",
    malignant_probability_column: str = "malignant_probability",
) -> pd.DataFrame:
    """Aggregate state-level single-drug AUC anchors into patient summaries."""
    required = {
        patient_column,
        drug_column,
        prediction_column,
        state_fraction_column,
        malignant_probability_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"state prediction columns missing: {missing}")
    assert_label_free(frame, allow_predicted=True, context="state prediction table")

    rows: list[dict[str, object]] = []
    grouped = frame.groupby([patient_column, drug_column], sort=True, dropna=False)
    for (patient_id, drug_id), group in grouped:
        predictions = pd.to_numeric(group[prediction_column], errors="coerce").to_numpy()
        fractions = pd.to_numeric(group[state_fraction_column], errors="coerce").to_numpy()
        malignant = pd.to_numeric(
            group[malignant_probability_column],
            errors="coerce",
        ).to_numpy()
        valid = (
            np.isfinite(predictions)
            & np.isfinite(fractions)
            & np.isfinite(malignant)
            & (fractions >= 0)
            & (malignant >= 0)
        )
        predictions = predictions[valid]
        raw_weights = fractions[valid] * np.clip(malignant[valid], 0.0, 1.0)
        if len(predictions) == 0 or raw_weights.sum() <= 0:
            continue
        weights = raw_weights / raw_weights.sum()
        mean = float(np.sum(weights * predictions))
        variance = float(np.sum(weights * (predictions - mean) ** 2))
        effective_states = float(1.0 / np.sum(weights * weights))
        rows.append(
            {
                patient_column: patient_id,
                drug_column: drug_id,
                "predicted_auc": mean,
                "between_state_sd": float(np.sqrt(max(variance, 0.0))),
                "between_state_range": float(predictions.max() - predictions.min()),
                "malignant_weight_mass": float(raw_weights.sum()),
                "n_states_used": int(len(predictions)),
                "effective_states": effective_states,
            }
        )

    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["patient_drug_rank"] = output.groupby(patient_column)["predicted_auc"].rank(
        method="dense",
        ascending=True,
    )
    return output.sort_values(
        [patient_column, "patient_drug_rank", drug_column],
        kind="mergesort",
    ).reset_index(drop=True)


def canonical_prediction_bytes(
    frame: pd.DataFrame,
    *,
    sort_columns: Sequence[str],
) -> bytes:
    missing = sorted(set(sort_columns) - set(frame.columns))
    if missing:
        raise KeyError(f"prediction sort columns missing: {missing}")
    assert_label_free(frame, allow_predicted=True, context="frozen prediction table")
    ordered = frame.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
    return ordered.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    ).encode("utf-8")


def freeze_predictions(
    frame: pd.DataFrame,
    *,
    output_path: Path,
    sort_columns: Sequence[str],
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = canonical_prediction_bytes(frame, sort_columns=sort_columns)
    digest = hashlib.sha256(payload).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    manifest = {
        "schema_version": "aml_virtual_cell_v16_predictions",
        "prediction_file": output_path.name,
        "bytes": len(payload),
        "sha256": digest,
        "metadata": metadata or {},
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def embedding_audit_dict(audit: EmbeddingAudit) -> dict[str, object]:
    return asdict(audit)
