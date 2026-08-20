"""Utilities for physically blinded virtual-cell challenge datasets.

The predictor must only receive the public tables. Outcome-bearing columns are
stored in a separate sealed directory and are joined by deterministic row IDs
only after a prediction artifact has been frozen.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

EXACT_OUTCOME_COLUMNS = frozenset(
    {
        "response",
        "cell_inhibition_pct",
        "prediction_label",
        "synergy_zip",
        "synergy_loewe",
        "synergy_bliss",
        "synergy_hsa",
        "viability",
        "toxicity",
        "outcome",
        "label",
        "auc",
        "ic50",
        "dss",
        "dss_like",
        "effect",
        "efficacy",
        "inhibition",
        "observed_dss_like",
        "observed_effect",
        "observed_inhibition",
        "known_treatment_response_events",
        "published_prediction_vs_later_treatment",
        "flow_evidence",
        "flow_statistical_note",
    }
)
OUTCOME_PATTERNS = (
    re.compile(r"^response\d*(?:_pct)?$"),
    re.compile(
        r"(?:^|_)observed_(?:response|viability|inhibition|toxicity|auc|ic50|dss|effect)"
        r"(?:_|$)"
    ),
    re.compile(r"(?:^|_)(?:outcome|synergy)(?:_|$)"),
    re.compile(r"(?:^|_)(?:dss|inhibition|efficacy|effect)(?:_|$)"),
)


@dataclass(frozen=True)
class TableSplit:
    public: pd.DataFrame
    sealed: pd.DataFrame
    outcome_columns: tuple[str, ...]
    row_id_column: str = "challenge_row_id"


@dataclass(frozen=True)
class FileDigest:
    relative_path: str
    bytes: int
    sha256: str


def normalize_column_name(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def find_outcome_columns(
    columns: Iterable[object],
    *,
    allow_predicted: bool = True,
) -> list[str]:
    """Return columns that appear to contain observed outcome information."""
    detected: list[str] = []
    for original in columns:
        name = str(original)
        normalized = normalize_column_name(name)
        if allow_predicted and normalized.startswith(("predicted_", "prediction_")):
            continue
        if normalized in EXACT_OUTCOME_COLUMNS:
            detected.append(name)
            continue
        if any(pattern.search(normalized) for pattern in OUTCOME_PATTERNS):
            detected.append(name)
    return detected


def assert_label_free(
    frame: pd.DataFrame,
    *,
    allow_predicted: bool = True,
    context: str = "prediction input",
) -> None:
    detected = find_outcome_columns(frame.columns, allow_predicted=allow_predicted)
    if detected:
        raise ValueError(
            f"{context} contains forbidden observed-outcome columns: "
            + ", ".join(sorted(detected))
        )


def _identity_tokens(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"identity columns missing from challenge table: {missing}")
    normalized = frame.loc[:, list(columns)].copy()
    for column in normalized:
        normalized[column] = normalized[column].map(
            lambda value: "" if pd.isna(value) else str(value).strip()
        )
    return normalized.apply(
        lambda row: json.dumps(row.to_dict(), sort_keys=True, separators=(",", ":")),
        axis=1,
    )


def add_stable_row_ids(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str],
    prefix: str,
    row_id_column: str = "challenge_row_id",
) -> pd.DataFrame:
    """Attach deterministic IDs, including a stable duplicate index."""
    if row_id_column in frame.columns:
        raise ValueError(f"{row_id_column} already exists")
    out = frame.copy()
    tokens = _identity_tokens(out, identity_columns)
    duplicate_index = tokens.groupby(tokens, sort=False).cumcount()
    out.insert(
        0,
        row_id_column,
        [
            f"{prefix}_{hashlib.sha256(f'{token}|{dup}'.encode()).hexdigest()[:20]}"
            for token, dup in zip(tokens, duplicate_index, strict=True)
        ],
    )
    if out[row_id_column].duplicated().any():
        raise AssertionError("stable challenge row IDs are not unique")
    return out


def split_challenge_table(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str],
    outcome_columns: Sequence[str],
    prefix: str,
    row_id_column: str = "challenge_row_id",
) -> TableSplit:
    """Split one table into public inputs and sealed outcomes."""
    missing = [column for column in outcome_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"outcome columns missing from challenge table: {missing}")

    detected = set(find_outcome_columns(frame.columns, allow_predicted=False))
    declared = set(outcome_columns)
    undeclared = sorted(detected - declared)
    if undeclared:
        raise ValueError(
            "outcome-like columns were not declared for sealing: " + ", ".join(undeclared)
        )

    with_ids = add_stable_row_ids(
        frame,
        identity_columns=identity_columns,
        prefix=prefix,
        row_id_column=row_id_column,
    )
    public_columns = [
        column
        for column in with_ids.columns
        if column not in declared
    ]
    public = with_ids.loc[:, public_columns].sort_values(row_id_column).reset_index(drop=True)
    sealed = with_ids.loc[:, [row_id_column, *outcome_columns]]
    sealed = sealed.sort_values(row_id_column).reset_index(drop=True)
    assert_label_free(public, allow_predicted=False, context=f"{prefix} public table")
    if not public[row_id_column].equals(sealed[row_id_column]):
        raise AssertionError("public and sealed row IDs are not aligned")
    return TableSplit(
        public=public,
        sealed=sealed,
        outcome_columns=tuple(outcome_columns),
        row_id_column=row_id_column,
    )


def canonical_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    ).encode("utf-8")


def dataframe_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(canonical_csv_bytes(frame)).hexdigest()


def file_digest(path: Path, *, relative_to: Path | None = None) -> FileDigest:
    payload = path.read_bytes()
    relative = path.relative_to(relative_to) if relative_to is not None else path
    return FileDigest(
        relative_path=relative.as_posix(),
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def verify_public_directory(path: Path) -> list[Path]:
    """Refuse a predictor input directory that contains outcome-bearing data."""
    path = path.resolve()
    if not path.is_dir():
        raise NotADirectoryError(path)
    if path.name.lower() == "sealed":
        raise ValueError("predictor cannot read a sealed challenge directory")
    csv_paths = sorted(path.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no public CSV tables found in {path}")
    for csv_path in csv_paths:
        assert_label_free(
            pd.read_csv(csv_path, nrows=5),
            allow_predicted=False,
            context=csv_path.name,
        )
    return csv_paths


def write_split(
    split: TableSplit,
    *,
    public_path: Path,
    sealed_path: Path,
) -> tuple[FileDigest, FileDigest]:
    public_path.parent.mkdir(parents=True, exist_ok=True)
    sealed_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_bytes(canonical_csv_bytes(split.public))
    sealed_path.write_bytes(canonical_csv_bytes(split.sealed))
    root = public_path.parent.parent
    return (
        file_digest(public_path, relative_to=root),
        file_digest(sealed_path, relative_to=root),
    )


def write_manifest(
    path: Path,
    *,
    files: Sequence[FileDigest],
    metadata: dict[str, object] | None = None,
) -> None:
    payload = {
        "schema_version": "sctherapy_blinded_challenge_v1",
        "files": [
            {
                "relative_path": item.relative_path,
                "bytes": item.bytes,
                "sha256": item.sha256,
            }
            for item in sorted(files, key=lambda item: item.relative_path)
        ],
        "metadata": metadata or {},
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
