"""B.2 — Aggregate ChEMBL drug-target binding into 18-axis coverage vectors.

Pipeline:
  data/canonical/chembl_drug_targets.csv  (drug × target × pIC50 records)
                ↓
  chembl_target_to_taxonomy.yaml          (UniProt → axis mapping)
                ↓
  per drug: 18-dim coverage vector       (max(normalize(pIC50)) per axis)

This is what lets us extend the drug pool beyond the 31 hand-annotated
drugs in target_taxonomy_v2.yaml — any drug with ChEMBL bioactivity data
can be added to the set-cover solver via a learned coverage vector.

For new drugs WITHOUT ChEMBL data (novel SMILES), Phase B.3 will train a
GIN regressor: SMILES → 18-dim coverage. This module is the data source
for that supervision (auxiliary task labels).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


_DEFAULT_TAXONOMY_MAP = (
    Path(__file__).parents[1] / "knowledge" / "chembl_target_to_taxonomy.yaml"
)
_DEFAULT_CHEMBL_CSV = (
    Path(__file__).parents[3] / "data" / "canonical" / "chembl_drug_targets.csv"
)


@dataclass
class TaxonomyMap:
    axis_to_uniprots: dict[str, list[str]]
    uniprot_to_axes: dict[str, list[str]] = field(default_factory=dict)
    bins: list[tuple[float, float]] = field(default_factory=list)  # (min_pIC50, coverage)

    def __post_init__(self):
        # Build inverse map UniProt → list of axes that include it
        if not self.uniprot_to_axes:
            inv: dict[str, list[str]] = defaultdict(list)
            for axis, uniprots in self.axis_to_uniprots.items():
                for u in uniprots:
                    inv[u].append(axis)
            self.uniprot_to_axes = dict(inv)

    def normalize_pIC50(self, pIC50: float) -> float:
        """Bin pIC50 → coverage value 0..1 using the curated thresholds."""
        for min_p, cov in self.bins:
            if pIC50 >= min_p:
                return cov
        return 0.0


def load_taxonomy_map(path: Path | str = _DEFAULT_TAXONOMY_MAP) -> TaxonomyMap:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    bins_raw = d.get("normalization", {}).get("bins", [])
    # Sort descending by min_pIC50 so first match wins for high pIC50 values
    bins = sorted(
        ((float(b["min_pIC50"]), float(b["coverage"])) for b in bins_raw),
        key=lambda x: -x[0],
    )
    axis_to_uniprots = {
        axis: list(spec.get("uniprots", []))
        for axis, spec in d.get("axis_to_uniprots", {}).items()
    }
    return TaxonomyMap(axis_to_uniprots=axis_to_uniprots, bins=bins)


def aggregate_drug_coverage(
    chembl_csv: Path | str = _DEFAULT_CHEMBL_CSV,
    taxonomy_map: Optional[TaxonomyMap] = None,
) -> dict[str, dict[str, float]]:
    """Returns {drug_id: {axis: coverage_0to1}}.

    For each drug:
      For each axis:
        coverage[drug, axis] = max over (UniProts in axis ∩ drug's known targets)
                                of normalize(pIC50)

    Drugs missing from ChEMBL or with no axis-relevant targets get an
    empty inner dict (all axes implicitly 0).
    """
    tm = taxonomy_map or load_taxonomy_map()
    chembl_csv = Path(chembl_csv)
    if not chembl_csv.exists():
        return {}

    # Group raw records: drug_id → uniprot → max pIC50
    raw: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(lambda: -1e9))
    with open(chembl_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            did = row.get("beataml_drug_id", "")
            uniprot = row.get("target_uniprot", "")
            pIC50_str = row.get("pIC50", "")
            if not (did and uniprot and pIC50_str):
                continue
            try:
                pIC50 = float(pIC50_str)
            except ValueError:
                continue
            if pIC50 > raw[did][uniprot]:
                raw[did][uniprot] = pIC50

    # Aggregate per axis
    out: dict[str, dict[str, float]] = {}
    for did, uniprot_pIC50s in raw.items():
        per_axis: dict[str, float] = {}
        for uniprot, pIC50 in uniprot_pIC50s.items():
            for axis in tm.uniprot_to_axes.get(uniprot, []):
                cov = tm.normalize_pIC50(pIC50)
                if cov > per_axis.get(axis, 0):
                    per_axis[axis] = cov
        out[did] = per_axis
    return out


def coverage_vector_18d(drug_id: str,
                         coverage_map: dict[str, dict[str, float]],
                         axis_order: list[str]) -> np.ndarray:
    """Returns the 18-dim coverage vector for drug_id, in axis_order."""
    per_axis = coverage_map.get(drug_id, {})
    return np.array(
        [per_axis.get(axis, 0.0) for axis in axis_order],
        dtype=np.float32,
    )
