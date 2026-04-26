"""3-tier merged drug coverage source.

Resolves coverage[drug, axis] using a fallback chain:

  Tier 1: Expert taxonomy (covered_by_drug_examples in target_taxonomy_v2)
          — Authoritative, hand-curated by domain knowledge
  Tier 2: ChEMBL aggregated (chembl_drug_targets.csv → chembl_coverage)
          — Authoritative for drugs with known binding profiles
  Tier 3: GIN prediction (coverage_gin.pt model on SMILES)
          — Fallback for novel SMILES not in Tiers 1 or 2

Combination rule per (drug, axis):
  coverage[d, a] = max(tier1[d, a], tier2[d, a], tier3[d, a])

The max-merge is deliberate: each tier surfaces information the others
miss. Expert annotations capture mechanism-of-action drugs (HMA,
MENINi, IDHi). ChEMBL captures kinase off-targets. GIN extrapolates
to novel chemistry.

Usage:
    pool = MergedDrugPool(taxonomy=tx)
    cm = pool.build_coverage_matrix(["Quizartinib (AC220)", "Venetoclax"])
    # → DrugCoverageMatrix where each [drug, axis] is the max of all 3 tiers

    # Add a novel SMILES
    pool.register_novel_smiles({"NovelDrug1": "CC(=O)O..."})
    cm2 = pool.build_coverage_matrix(["NovelDrug1", "Quizartinib (AC220)"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from combo_val.coverage.chembl_coverage import (
    aggregate_drug_coverage, load_taxonomy_map,
)
from combo_val.coverage.taxonomy import (
    DrugCoverageMatrix, Taxonomy, build_coverage_matrix, load_taxonomy,
)


_DEFAULT_GIN_CKPT = (
    Path(__file__).parents[3] / "runs" / "coverage_predictor" / "coverage_gin.pt"
)


@dataclass
class MergedDrugPool:
    """Holds tier 1+2 coverage maps + (optional) GIN tier-3 predictor.

    Attributes:
      taxonomy: taxonomy v2 instance
      novel_smiles: {drug_id: SMILES} registered at runtime — these
        drugs go through GIN tier-3 only (no expert/chembl entry)
    """
    taxonomy: Taxonomy
    novel_smiles: dict[str, str] = field(default_factory=dict)
    _expert_coverage: dict[str, dict[str, float]] | None = None
    _chembl_coverage: dict[str, dict[str, float]] | None = None
    _gin_model: object | None = None
    _gin_axis_order: list[str] | None = None
    _gin_ckpt: Path | None = None

    def __post_init__(self):
        # Build expert coverage from taxonomy
        if self._expert_coverage is None:
            self._expert_coverage = {}
            for t in self.taxonomy.targets:
                for drug, cov in t.covered_by_drug_examples.items():
                    self._expert_coverage.setdefault(drug, {})[t.id] = cov
        # Try to load ChEMBL aggregation (lazy — file may not exist in tests)
        if self._chembl_coverage is None:
            try:
                self._chembl_coverage = aggregate_drug_coverage()
            except Exception:
                self._chembl_coverage = {}

    @property
    def expert_drugs(self) -> set[str]:
        return set(self._expert_coverage or {})

    @property
    def chembl_drugs(self) -> set[str]:
        return set(self._chembl_coverage or {})

    @property
    def known_drugs(self) -> list[str]:
        """Drugs with Tier 1 OR Tier 2 coverage (not requiring GIN)."""
        return sorted(self.expert_drugs | self.chembl_drugs)

    def register_novel_smiles(self, smiles_dict: dict[str, str]) -> None:
        self.novel_smiles.update(smiles_dict)

    def _ensure_gin_loaded(self,
                           ckpt: Path | None = None) -> bool:
        """Lazy-load the GIN coverage predictor. Returns True on success."""
        if self._gin_model is not None:
            return True
        path = ckpt or self._gin_ckpt or _DEFAULT_GIN_CKPT
        if not Path(path).exists():
            return False
        try:
            from combo_val.coverage.gin_coverage_predictor import load_predictor
            model, axis_order = load_predictor(path)
            self._gin_model = model
            self._gin_axis_order = axis_order
            self._gin_ckpt = path
            return True
        except Exception:
            return False

    def coverage_for_drug(self, drug_id: str,
                           smiles: str | None = None) -> dict[str, float]:
        """Returns {axis: coverage} for a single drug, max-merged across tiers."""
        merged: dict[str, float] = {}
        for axis, cov in (self._expert_coverage.get(drug_id, {}) or {}).items():
            if cov > merged.get(axis, 0):
                merged[axis] = cov
        for axis, cov in (self._chembl_coverage.get(drug_id, {}) or {}).items():
            if cov > merged.get(axis, 0):
                merged[axis] = cov
        # Tier 3 — GIN, only if SMILES provided AND drug is novel (not in 1/2)
        if smiles and drug_id not in self.expert_drugs and drug_id not in self.chembl_drugs:
            if self._ensure_gin_loaded():
                pred = self._gin_model.predict_smiles([smiles])[0]
                for ax, val in zip(self._gin_axis_order or [], pred):
                    if float(val) > merged.get(ax, 0):
                        merged[ax] = float(val)
        return merged

    def build_coverage_matrix(
        self, drug_ids: list[str],
        smiles_lookup: dict[str, str] | None = None,
    ) -> DrugCoverageMatrix:
        """Build a DrugCoverageMatrix for an arbitrary drug list.

        For each drug:
          1. Start with expert taxonomy coverage (if known to taxonomy).
          2. Overlay max(., ChEMBL coverage) for axes in chembl_coverage.
          3. For any drug not in 1 or 2, run GIN inference if SMILES given
             via smiles_lookup (or self.novel_smiles).

        toxicity stays from drug_mechanism_v1.csv (Tier 1 only) since we
        don't have ChEMBL toxicity profiles.
        """
        smiles_lookup = dict(smiles_lookup or {})
        for d, s in self.novel_smiles.items():
            smiles_lookup.setdefault(d, s)

        # Start with the expert-pool matrix (handles axis order + toxicity)
        cm = build_coverage_matrix(self.taxonomy, drug_ids)
        # Now overlay tier-2 (ChEMBL) for any drug present in chembl_coverage
        for d in drug_ids:
            if d not in cm.drug_id_to_idx:
                continue
            di = cm.drug_id_to_idx[d]
            chembl_axes = self._chembl_coverage.get(d, {})
            for axis, cov in chembl_axes.items():
                if axis not in cm.target_id_to_idx:
                    continue
                ai = cm.target_id_to_idx[axis]
                if cov > cm.coverage[di, ai]:
                    cm.coverage[di, ai] = cov

        # Tier-3: novel SMILES → GIN
        novel = [d for d in drug_ids
                  if d not in self.expert_drugs
                  and d not in self.chembl_drugs
                  and d in smiles_lookup]
        if novel and self._ensure_gin_loaded():
            smiles_list = [smiles_lookup[d] for d in novel]
            preds = self._gin_model.predict_smiles(smiles_list)
            for d, pred in zip(novel, preds):
                if d not in cm.drug_id_to_idx:
                    continue
                di = cm.drug_id_to_idx[d]
                for ax_id, val in zip(self._gin_axis_order or [], pred):
                    if ax_id not in cm.target_id_to_idx:
                        continue
                    ai = cm.target_id_to_idx[ax_id]
                    if float(val) > cm.coverage[di, ai]:
                        cm.coverage[di, ai] = float(val)
        return cm

    def coverage_provenance(self, drug_id: str) -> str:
        """Return which tier provides coverage for this drug."""
        if drug_id in self.expert_drugs and drug_id in self.chembl_drugs:
            return "expert+chembl"
        if drug_id in self.expert_drugs:
            return "expert"
        if drug_id in self.chembl_drugs:
            return "chembl"
        if drug_id in self.novel_smiles:
            return "gin (novel SMILES)"
        return "unknown"
