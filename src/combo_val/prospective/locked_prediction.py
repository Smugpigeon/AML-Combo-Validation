"""Immutable hash-chained prediction store for prospective validation.

The data integrity contract:

  Once a prediction is generated for an enrolled patient, the prediction
  + the inputs that produced it + the kit version + the model checkpoint
  hash are written to a JSON file under
  `runs/prospective/<site_id>/<patient_id>/<timestamp>.json` AND a hash
  chain is updated.

  The hash chain prevents retroactive tampering — to "fix" yesterday's
  prediction you'd have to re-hash every subsequent prediction. Combined
  with periodic external anchoring (e.g., posting the latest hash to a
  public blockchain or git-signed tag), this gives a provable timeline:
  "this Top-1 was generated for this patient at time T, with this kit
  version, before treatment was given."

Why this matters for prospective validation:
  Without immutability you cannot prove the kit didn't see the outcome
  before "predicting." With immutability + IRB-approved enrollment +
  blinded outcome capture, the prospective study is methodologically
  defensible.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _git_commit_hash(repo_root: Path | None = None) -> str:
    """Get the kit's current git commit hash. Falls back to 'unknown' in
    detached or non-git contexts (e.g., pip install)."""
    repo_root = repo_root or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_sha256(path: Path | str) -> str:
    """SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _content_sha256(obj: dict) -> str:
    """Canonical-JSON SHA-256 — deterministic across Python runs."""
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass
class LockedPrediction:
    """A single immutable prediction record.

    Once written to disk + hash-chained, this object MUST NOT be modified.
    Any change invalidates the chain and is detectable by `verify_chain`.
    """
    # Identity
    patient_id: str                       # de-identified, e.g. "PUMC-2026-001"
    site_id: str                          # institution + IRB protocol, e.g. "PUMC_IRB_2026_HEM_42"
    enrolled_at: str                      # ISO-8601 UTC; from registry
    prediction_id: str                    # uuid4 hex (set at write time)

    # Provenance
    kit_version: str                      # e.g. "v0.5.0"
    kit_commit_hash: str                  # git rev-parse HEAD at predict time
    model_checkpoint_sha256: str          # SHA-256 of MLP weights file
    model_checkpoint_path: str            # relative path so cohort can be replayed
    backbone: str                         # e.g. "mlp", "st-v3-distill"

    # Inputs (hashed reference — actual inputs go to a separate file)
    input_hash: str                       # sha256 of canonical-JSON KitInput
    input_path: str                       # relative path to the saved input JSON

    # Predictions (full output, NOT just top-1)
    predicted_eln_2022: str               # "Favorable" | "Intermediate" | "Adverse"
    predicted_top1_regimen_id: str
    predicted_top3_regimen_ids: list[str]
    layer3_top3_combos: list[dict]        # [{drug1, drug2, predicted_combo_auc, ...}, ...]
    layer3_pearson_r_internal: float = 0.05  # documented for the chain

    # Hash chain
    prev_hash: str = ""                   # SHA-256 of prior LockedPrediction in this site's chain
    self_hash: str = ""                   # SHA-256 of THIS record (excl self_hash) — set at write

    def computable_hash(self) -> str:
        """Hash of this record EXCLUDING self_hash field."""
        d = asdict(self)
        d.pop("self_hash", None)
        return _content_sha256(d)


# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------


def _site_chain_dir(root: Path, site_id: str) -> Path:
    safe_site = "".join(c if c.isalnum() or c in "_-" else "_" for c in site_id)
    return root / "prospective" / safe_site


def _patient_dir(root: Path, site_id: str, patient_id: str) -> Path:
    safe_pt = "".join(c if c.isalnum() or c in "_-" else "_" for c in patient_id)
    return _site_chain_dir(root, site_id) / safe_pt


def _chain_index_path(root: Path, site_id: str) -> Path:
    """Per-site chain index — one JSON-Lines file holding (timestamp,
    patient_id, prediction_id, self_hash) per line. Append-only. """
    return _site_chain_dir(root, site_id) / "_chain.jsonl"


def _last_hash_in_chain(root: Path, site_id: str) -> str:
    """Read the last self_hash from the site's chain. Returns '' if empty."""
    idx = _chain_index_path(root, site_id)
    if not idx.exists():
        return ""
    last = ""
    with open(idx, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                last = row.get("self_hash", "")
            except json.JSONDecodeError:
                continue
    return last


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def lock_prediction(
    *,
    runs_root: Path,
    site_id: str,
    patient_id: str,
    enrolled_at: str,
    kit_version: str,
    model_checkpoint_path: Path,
    backbone: str,
    kit_input_dict: dict,
    kit_output_dict: dict,
) -> LockedPrediction:
    """Write a new locked prediction record + extend the site's hash chain.

    Args:
      runs_root: e.g. Path("runs"); the prospective/ subdirectory is
        created under this.
      site_id: "<institution>_<IRB_protocol>" — the site granularity.
      patient_id: de-identified patient label.
      enrolled_at: ISO-8601 UTC datetime — from the enrollment registry.
      kit_version: e.g. "v0.5.0".
      model_checkpoint_path: actual file used for inference; SHA-256'd.
      backbone: model backbone name from BACKBONE_REGISTRY.
      kit_input_dict: KitInput as a dict (will be canonical-JSON hashed +
        saved alongside).
      kit_output_dict: KitOutput as a dict (full output captured).

    Returns:
      The LockedPrediction object after being written + hash-chained.
    """
    import uuid

    runs_root = Path(runs_root)
    pt_dir = _patient_dir(runs_root, site_id, patient_id)
    pt_dir.mkdir(parents=True, exist_ok=True)

    # 1) Hash the input deterministically + write the full input alongside
    input_hash = _content_sha256(kit_input_dict)
    ts = _utcnow_iso().replace(":", "-").replace("+", "-")
    input_path = pt_dir / f"{ts}_input.json"
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(kit_input_dict, f, ensure_ascii=False, indent=2)

    # 2) Hash the model checkpoint
    if Path(model_checkpoint_path).exists():
        ckpt_hash = _file_sha256(model_checkpoint_path)
    else:
        ckpt_hash = "missing"

    # 3) Pull predictions out of kit_output_dict
    top_combos = kit_output_dict.get("top_combinations") or []
    top_regimens = kit_output_dict.get("top_regimens") or []
    eln_2022_dict = kit_output_dict.get("eln_2022") or {}

    rec = LockedPrediction(
        patient_id=patient_id,
        site_id=site_id,
        enrolled_at=enrolled_at,
        prediction_id=uuid.uuid4().hex,
        kit_version=kit_version,
        kit_commit_hash=_git_commit_hash(),
        model_checkpoint_sha256=ckpt_hash,
        model_checkpoint_path=str(model_checkpoint_path),
        backbone=backbone,
        input_hash=input_hash,
        input_path=str(input_path.relative_to(runs_root)),
        predicted_eln_2022=eln_2022_dict.get("category") or "Unknown",
        predicted_top1_regimen_id=(top_regimens[0].get("regimen_id")
                                    if top_regimens else "none"),
        predicted_top3_regimen_ids=[r.get("regimen_id", "") for r in top_regimens[:3]],
        layer3_top3_combos=[
            {k: c.get(k) for k in
             ("drug1", "drug2", "predicted_combo_auc", "mech_score",
              "clonal_coverage_score")}
            for c in top_combos[:3]
            if not c.get("suppressed")  # OOD-suppressed entries excluded
        ],
        prev_hash=_last_hash_in_chain(runs_root, site_id),
    )
    rec.self_hash = rec.computable_hash()

    # 4) Write the record + append to chain index atomically
    rec_path = pt_dir / f"{ts}_locked_prediction.json"
    tmp = rec_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(asdict(rec), f, ensure_ascii=False, indent=2)
    os.replace(tmp, rec_path)

    idx_path = _chain_index_path(runs_root, site_id)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(idx_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": ts, "patient_id": patient_id,
            "prediction_id": rec.prediction_id,
            "self_hash": rec.self_hash, "prev_hash": rec.prev_hash,
            "record_path": str(rec_path.relative_to(runs_root)),
        }, ensure_ascii=False) + "\n")

    return rec


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass
class ChainVerificationResult:
    site_id: str
    n_records: int
    chain_intact: bool
    failures: list[str] = field(default_factory=list)


def verify_chain(runs_root: Path | str, site_id: str) -> ChainVerificationResult:
    """Walk the site's chain index and verify each record's self_hash
    + prev_hash linkage. Used by analysis-trigger before running stats."""
    runs_root = Path(runs_root)
    idx = _chain_index_path(runs_root, site_id)
    failures: list[str] = []
    if not idx.exists():
        return ChainVerificationResult(site_id, 0, True, [])

    expected_prev = ""
    n = 0
    with open(idx, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1

            rec_path = runs_root / row["record_path"]
            if not rec_path.exists():
                failures.append(f"line {line_no}: record file missing: {rec_path}")
                continue

            with open(rec_path, "r", encoding="utf-8") as rf:
                stored = json.load(rf)

            # 1. prev_hash chain linkage
            if stored.get("prev_hash", "") != expected_prev:
                failures.append(
                    f"line {line_no}: prev_hash mismatch — chain expected "
                    f"{expected_prev[:8]}..., record has {stored.get('prev_hash','')[:8]}..."
                )

            # 2. self_hash recompute
            stored_self = stored.pop("self_hash", "")
            recomputed = _content_sha256(stored)
            if recomputed != stored_self:
                failures.append(
                    f"line {line_no}: self_hash mismatch — record content "
                    f"was modified after write (recomputed {recomputed[:8]}... vs "
                    f"stored {stored_self[:8]}...)"
                )

            expected_prev = stored_self  # for the next iteration

    return ChainVerificationResult(site_id, n, not failures, failures)
