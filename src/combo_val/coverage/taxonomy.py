"""Load target taxonomy + build drug × target coverage matrix.

The taxonomy defines targets and which drugs cover them (with explicit
example coverages 0..1). This module:
  1. Parses target_taxonomy_v2.yaml
  2. For each target, expands `covered_by_drug_examples` into a
     drug_id → coverage map (joined to the BeatAML 165-drug vocab + any
     extra drugs the user passes in via SMILES)
  3. Returns a DrugCoverageMatrix that the set-cover solver consumes
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


_DEFAULT_TAXONOMY = (
    Path(__file__).parents[1] / "knowledge" / "target_taxonomy_v2.yaml"
)
_DEFAULT_TOXICITY = (
    Path(__file__).parents[1] / "knowledge" / "drug_mechanism_v1.csv"
)


@dataclass
class TargetSpec:
    id: str
    tier: int
    description: str
    default_weight: float
    patient_evidence: dict
    covered_by_drug_examples: dict[str, float]
    covered_by_drug_classes: list[str]


@dataclass
class Taxonomy:
    targets: list[TargetSpec]
    toxicity_axes: list[str]
    constraints_default: dict
    version: str
    ida_aggregation: str

    def target_ids(self) -> list[str]:
        return [t.id for t in self.targets]

    def get(self, target_id: str) -> Optional[TargetSpec]:
        for t in self.targets:
            if t.id == target_id:
                return t
        return None


def load_taxonomy(path: Path | str = _DEFAULT_TAXONOMY) -> Taxonomy:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    targets = []
    for t in data.get("targets", []):
        targets.append(TargetSpec(
            id=t["id"],
            tier=int(t["tier"]),
            description=t.get("description", ""),
            default_weight=float(t.get("default_weight", 0.5)),
            patient_evidence=t.get("patient_evidence", {}),
            covered_by_drug_examples=t.get("covered_by_drug_examples", {}),
            covered_by_drug_classes=t.get("covered_by_drug_classes", []),
        ))
    return Taxonomy(
        targets=targets,
        toxicity_axes=data.get("toxicity_axes", []),
        constraints_default=data.get("constraints_default", {}),
        version=str(data.get("meta", {}).get("version", "?")),
        ida_aggregation=str(data.get("meta", {}).get("ida_aggregation", "bliss")),
    )


@dataclass
class DrugCoverageMatrix:
    """Coverage matrix C of shape (n_drugs, n_targets) with values in [0, 1].

    C[i, j] = how strongly drug i covers target j.
    """
    drug_ids: list[str]
    target_ids: list[str]
    coverage: np.ndarray             # (n_drugs, n_targets), float32
    drug_id_to_idx: dict[str, int]
    target_id_to_idx: dict[str, int]
    # toxicity (n_drugs, n_tox_axes)
    toxicity: np.ndarray
    toxicity_axes: list[str]


def build_coverage_matrix(taxonomy: Taxonomy,
                           drug_pool: list[str],
                           toxicity_csv: Path = _DEFAULT_TOXICITY,
                           ) -> DrugCoverageMatrix:
    """Construct DrugCoverageMatrix from taxonomy + drug list.

    Args:
      taxonomy: loaded Taxonomy
      drug_pool: list of drug_ids to include (e.g. BeatAML vocab + custom)
      toxicity_csv: existing tox annotation table

    Coverage values come from `covered_by_drug_examples` in the taxonomy
    (drugs not explicitly listed in any target get 0 coverage everywhere).
    """
    n_drugs = len(drug_pool)
    targets = taxonomy.targets
    n_targets = len(targets)
    drug_id_to_idx = {d: i for i, d in enumerate(drug_pool)}
    target_id_to_idx = {t.id: i for i, t in enumerate(targets)}

    coverage = np.zeros((n_drugs, n_targets), dtype=np.float32)
    for j, t in enumerate(targets):
        for drug_name, cov in t.covered_by_drug_examples.items():
            i = drug_id_to_idx.get(drug_name)
            if i is None:
                continue
            coverage[i, j] = max(coverage[i, j], float(cov))

    # Toxicity: load existing CSV, map by drug_name → tox vector
    tox_axes = taxonomy.toxicity_axes
    tox = np.zeros((n_drugs, len(tox_axes)), dtype=np.float32)
    if toxicity_csv.exists():
        with open(toxicity_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                drug_name = row.get("drug_name", "")
                if drug_name not in drug_id_to_idx:
                    continue
                i = drug_id_to_idx[drug_name]
                for ax_idx, ax in enumerate(tox_axes):
                    col = f"tox_{ax}"
                    val = row.get(col)
                    if val and val.replace(".", "").replace("-", "").isdigit():
                        tox[i, ax_idx] = float(val)

    return DrugCoverageMatrix(
        drug_ids=drug_pool,
        target_ids=[t.id for t in targets],
        coverage=coverage,
        drug_id_to_idx=drug_id_to_idx,
        target_id_to_idx=target_id_to_idx,
        toxicity=tox,
        toxicity_axes=tox_axes,
    )
