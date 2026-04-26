"""Set-cover combination optimizer with Bliss-IDA aggregation.

Problem:
  Given:
    - active_targets: {target_id → weight}  (from patient inference)
    - drug_pool: list of n_drugs drugs with coverage[i, j] for each target
    - constraints: max_arity, toxicity_ceiling, coverage_threshold
  Find:
    - top-K minimal combinations G ⊆ drug_pool such that
      Σ_t weight_t · covers(G, t) >= coverage_threshold
      AND |G| <= max_arity
      AND Σ_{d ∈ G} toxicity[d, ax] <= ceiling for all ax

  where covers(G, t) = 1 - ∏_{d ∈ G} (1 - coverage[d, t])  (Bliss-IDA)

Algorithm:
  1. Greedy seed: start empty G; at each step pick d that maximizes
     marginal coverage gain, subject to constraints. Stop when threshold
     reached or no feasible drug.
  2. Local search refinement: try swapping each drug in G with each in
     pool. Keep swaps that improve total score without violating
     constraints.
  3. Multi-start: run greedy + LS from K random initial subsets. Keep
     diverse top-K (by Jaccard).

Returns: list of CombinationResult, sorted by descending coverage / arity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np

from combo_val.coverage.taxonomy import DrugCoverageMatrix


@dataclass
class CombinationResult:
    drug_ids: list[str]                     # drug names
    drug_indices: list[int]                  # indices into pool
    total_weighted_coverage: float           # main objective
    coverage_per_target: dict[str, float]    # {target_id: covers(G, t)}
    arity: int
    toxicity_per_axis: dict[str, float]
    feasible: bool                           # all constraints satisfied
    constraint_violations: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)


def bliss_coverage_per_target(
    coverage_matrix: np.ndarray,             # (n_drugs, n_targets)
    drug_indices: list[int],                  # selected combo
) -> np.ndarray:                              # (n_targets,) covers(G, t) ∈ [0, 1]
    """1 - ∏_{d ∈ G} (1 - coverage[d, t])"""
    if not drug_indices:
        return np.zeros(coverage_matrix.shape[1], dtype=np.float64)
    sub = coverage_matrix[drug_indices, :]            # (k, n_targets)
    one_minus = 1.0 - np.clip(sub, 0.0, 1.0)
    not_covered = np.prod(one_minus, axis=0)          # ∏ (1 − cov)
    return 1.0 - not_covered


def total_weighted_coverage(
    coverage_per_target: np.ndarray,         # (n_targets,)
    weights: np.ndarray,                     # (n_targets,)
) -> float:
    """Σ_t weight_t · covers(G, t) / Σ_t weight_t  (normalized to [0,1])."""
    w_sum = weights.sum()
    if w_sum <= 0:
        return 0.0
    return float(np.dot(weights, coverage_per_target) / w_sum)


def total_toxicity(toxicity_matrix: np.ndarray,
                    drug_indices: list[int]) -> np.ndarray:
    """Σ_{d ∈ G} toxicity[d, ax] per axis."""
    if not drug_indices:
        return np.zeros(toxicity_matrix.shape[1], dtype=np.float64)
    return toxicity_matrix[drug_indices, :].sum(axis=0)


def violates_toxicity(toxicity_per_axis: np.ndarray,
                       toxicity_axes: list[str],
                       toxicity_ceiling: dict[str, float]) -> list[str]:
    """Returns list of axes where total exceeds ceiling."""
    out = []
    for ax_name, ax_total in zip(toxicity_axes, toxicity_per_axis):
        ceiling = toxicity_ceiling.get(ax_name)
        if ceiling is not None and ax_total > ceiling:
            out.append(f"{ax_name}: {ax_total:.2f} > {ceiling:.2f}")
    return out


def greedy_select(
    coverage_matrix: np.ndarray,
    toxicity_matrix: np.ndarray,
    weights: np.ndarray,
    target_ids: list[str],
    drug_ids: list[str],
    toxicity_axes: list[str],
    toxicity_ceiling: dict[str, float],
    max_arity: int,
    coverage_threshold: float,
    excluded_drug_indices: Optional[set[int]] = None,
    seed_indices: Optional[list[int]] = None,
) -> list[int]:
    """Greedily pick drugs maximizing marginal weighted coverage gain.

    Stops when coverage_threshold reached OR max_arity reached OR no
    feasible drug remains."""
    excluded = set(excluded_drug_indices or set())
    selected = list(seed_indices or [])
    n_drugs = coverage_matrix.shape[0]
    rng_iters = max_arity * 2  # safety upper bound

    for _ in range(rng_iters):
        if len(selected) >= max_arity:
            break
        cur_cov = bliss_coverage_per_target(coverage_matrix, selected)
        cur_tot = total_weighted_coverage(cur_cov, weights)
        if cur_tot >= coverage_threshold:
            break

        best_gain = 0.0
        best_idx = None
        for d in range(n_drugs):
            if d in selected or d in excluded:
                continue
            # Marginal new coverage
            cand = selected + [d]
            cand_cov = bliss_coverage_per_target(coverage_matrix, cand)
            cand_tot = total_weighted_coverage(cand_cov, weights)
            gain = cand_tot - cur_tot
            if gain <= 1e-6:
                continue
            # Check toxicity feasibility
            cand_tox = total_toxicity(toxicity_matrix, cand)
            if violates_toxicity(cand_tox, toxicity_axes, toxicity_ceiling):
                continue
            if gain > best_gain:
                best_gain = gain
                best_idx = d

        if best_idx is None:
            break
        selected.append(best_idx)

    return selected


def local_search_refine(
    initial_selected: list[int],
    coverage_matrix: np.ndarray,
    toxicity_matrix: np.ndarray,
    weights: np.ndarray,
    toxicity_axes: list[str],
    toxicity_ceiling: dict[str, float],
    max_iter: int = 50,
    excluded_drug_indices: Optional[set[int]] = None,
) -> list[int]:
    """Try swapping each drug in selected with each unused drug. Keep
    swaps that strictly improve total coverage without violating
    constraints."""
    excluded = set(excluded_drug_indices or set())
    selected = list(initial_selected)
    n_drugs = coverage_matrix.shape[0]

    for _ in range(max_iter):
        cur_cov = bliss_coverage_per_target(coverage_matrix, selected)
        cur_tot = total_weighted_coverage(cur_cov, weights)
        improved = False
        best_swap = None
        best_new_tot = cur_tot

        for slot, d_old in enumerate(selected):
            for d_new in range(n_drugs):
                if d_new in selected or d_new in excluded:
                    continue
                trial = list(selected)
                trial[slot] = d_new
                trial_cov = bliss_coverage_per_target(coverage_matrix, trial)
                trial_tot = total_weighted_coverage(trial_cov, weights)
                if trial_tot <= best_new_tot:
                    continue
                trial_tox = total_toxicity(toxicity_matrix, trial)
                if violates_toxicity(trial_tox, toxicity_axes, toxicity_ceiling):
                    continue
                best_new_tot = trial_tot
                best_swap = (slot, d_new)
                improved = True

        if not improved:
            break
        slot, d_new = best_swap
        selected[slot] = d_new

    return selected


def find_top_combinations(
    active_targets: dict[str, float],
    coverage_matrix: DrugCoverageMatrix,
    constraints: dict,
    excluded_drug_ids: Optional[list[str]] = None,
    n_solutions: int = 5,
    multistart: int = 5,
    seed: int = 42,
) -> list[CombinationResult]:
    """Top-N minimal combinations covering active_targets ≥ threshold.

    Uses multi-start greedy + local-search; deduplicates by drug-set
    identity; returns sorted by (coverage_desc, arity_asc)."""
    rng = np.random.default_rng(seed)
    targets = coverage_matrix.target_ids
    n_targets = len(targets)
    weights = np.zeros(n_targets, dtype=np.float64)
    for tid, w in active_targets.items():
        if tid in coverage_matrix.target_id_to_idx:
            weights[coverage_matrix.target_id_to_idx[tid]] = float(w)

    if weights.sum() <= 0:
        return []

    excluded_set = set()
    if excluded_drug_ids:
        for d in excluded_drug_ids:
            if d in coverage_matrix.drug_id_to_idx:
                excluded_set.add(coverage_matrix.drug_id_to_idx[d])

    max_arity = int(constraints.get("max_arity", 4))
    threshold = float(constraints.get("coverage_threshold", 0.70))
    tox_ceiling = constraints.get("toxicity_ceiling", {})

    seen_combos: set[tuple[int, ...]] = set()
    results: list[CombinationResult] = []

    def _evaluate(selected: list[int]) -> CombinationResult:
        cov_per = bliss_coverage_per_target(coverage_matrix.coverage, selected)
        total = total_weighted_coverage(cov_per, weights)
        tox = total_toxicity(coverage_matrix.toxicity, selected)
        violations = violates_toxicity(
            tox, coverage_matrix.toxicity_axes, tox_ceiling,
        )
        cov_per_dict = {
            tid: float(cov_per[i]) for i, tid in enumerate(targets)
            if active_targets.get(tid, 0) > 0
        }
        tox_dict = {
            ax: float(tox[i])
            for i, ax in enumerate(coverage_matrix.toxicity_axes)
        }
        rationale = []
        for tid, w in sorted(active_targets.items(),
                              key=lambda x: -x[1]):
            if tid not in cov_per_dict:
                continue
            cov_val = cov_per_dict[tid]
            top_drug_idx = max(
                (i for i in selected),
                key=lambda i: coverage_matrix.coverage[
                    i, coverage_matrix.target_id_to_idx[tid]
                ],
                default=None,
            )
            if top_drug_idx is None:
                continue
            top_drug = coverage_matrix.drug_ids[top_drug_idx]
            top_drug_cov = float(coverage_matrix.coverage[
                top_drug_idx, coverage_matrix.target_id_to_idx[tid]
            ])
            if top_drug_cov > 0:
                rationale.append(
                    f"{tid} (w={w:.2f}): covered {cov_val:.2f}, "
                    f"primarily by {top_drug} (cov={top_drug_cov:.2f})"
                )
        return CombinationResult(
            drug_ids=[coverage_matrix.drug_ids[i] for i in selected],
            drug_indices=list(selected),
            total_weighted_coverage=float(total),
            coverage_per_target=cov_per_dict,
            arity=len(selected),
            toxicity_per_axis=tox_dict,
            feasible=(not violations) and (total >= threshold),
            constraint_violations=violations,
            rationale=rationale,
        )

    # Multi-start: pure greedy + greedy seeded with random 1 drug
    starts: list[Optional[list[int]]] = [None]   # pure greedy
    n_drugs_pool = coverage_matrix.coverage.shape[0]
    available = [i for i in range(n_drugs_pool) if i not in excluded_set]
    if available:
        for _ in range(multistart - 1):
            starts.append([int(rng.choice(available))])

    for seed_indices in starts:
        try:
            selected = greedy_select(
                coverage_matrix.coverage, coverage_matrix.toxicity, weights,
                targets, coverage_matrix.drug_ids,
                coverage_matrix.toxicity_axes, tox_ceiling,
                max_arity=max_arity, coverage_threshold=threshold,
                excluded_drug_indices=excluded_set,
                seed_indices=seed_indices,
            )
            selected = local_search_refine(
                selected,
                coverage_matrix.coverage, coverage_matrix.toxicity,
                weights, coverage_matrix.toxicity_axes, tox_ceiling,
                excluded_drug_indices=excluded_set,
            )
        except Exception:
            continue
        if not selected:
            continue
        key = tuple(sorted(selected))
        if key in seen_combos:
            continue
        seen_combos.add(key)
        results.append(_evaluate(selected))

    # Sort: feasible first, then highest coverage, then smallest arity
    results.sort(
        key=lambda r: (-int(r.feasible), -r.total_weighted_coverage, r.arity)
    )
    return results[:n_solutions]
