"""Multi-Target Coverage Explorer route.

GET  /coverage-explorer        — interactive HTMX page (drug picker + patient mut input)
POST /api/v1/coverage/compute  — JSON endpoint: returns ranked combos for the inputs
GET  /api/v1/coverage/taxonomy — returns target taxonomy + drug pool metadata for UI

Design principle: this is the *expert* page — exposes the IDA framework
directly for researchers / hematology MDT discussions. No patient
de-identification gating because no patient data is stored — everything
is computed from per-request input + cached taxonomy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.deps import current_user_optional
from app.i18n import make_template_context
from app.models import User


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

router = APIRouter(tags=["coverage"])


# ---------------------------------------------------------------------------
# GET /coverage-explorer  (HTMX page)
# ---------------------------------------------------------------------------


@router.get("/coverage-explorer", response_class=HTMLResponse)
def coverage_explorer_page(request: Request,
                            user: User | None = Depends(current_user_optional)):
    return templates.TemplateResponse(
        request, "coverage_explorer.html",
        make_template_context(request, user=user),
    )


# ---------------------------------------------------------------------------
# GET /api/v1/coverage/taxonomy  (UI bootstrap)
# ---------------------------------------------------------------------------


@router.get("/api/v1/coverage/taxonomy")
def get_taxonomy_metadata():
    """Returns 18-target taxonomy + the canonical drug pool for the UI."""
    try:
        from combo_val.coverage.taxonomy import load_taxonomy
    except ImportError:
        raise HTTPException(500, "coverage module unavailable")
    tx = load_taxonomy()
    targets = []
    all_drugs: set[str] = set()
    for t in tx.targets:
        targets.append({
            "id": t.id,
            "tier": t.tier,
            "description": t.description,
            "default_weight": t.default_weight,
            "covered_by_examples": t.covered_by_drug_examples,
        })
        all_drugs.update(t.covered_by_drug_examples.keys())
    return {
        "version": tx.version,
        "framework": "Palmer-Sorger IDA (Cell 2017, PMID 29245013)",
        "targets": targets,
        "drug_pool": sorted(all_drugs),
        "constraints_default": tx.constraints_default,
        "toxicity_axes": tx.toxicity_axes,
    }


# ---------------------------------------------------------------------------
# POST /api/v1/coverage/compute  (run the solver)
# ---------------------------------------------------------------------------


class CoverageComputeRequest(BaseModel):
    # Patient inputs
    mutations: list[str] = Field(
        default_factory=list,
        description="HUGO gene symbols mutated in this patient (e.g. ['FLT3', 'NPM1'])",
    )
    fusions: list[str] = Field(
        default_factory=list,
        description="Fusion identifiers (e.g. ['PML_RARA', 'KMT2A_r'])",
    )
    flt3_itd: bool = False
    flt3_tkd: bool = False
    tp53_multihit: bool = False
    karyo_del_17p: bool = False
    age: float | None = None
    is_relapse: bool = False
    # RNA program scores (optional)
    rna_programs: dict[str, float] = Field(default_factory=dict)
    # Drug pool restriction (None = use full merged pool — Expert ∪ ChEMBL)
    drug_pool: list[str] | None = None
    # Novel drugs added at request time: {label: SMILES} — go through GIN
    # tier-3 inference, then join the merged pool for this request only.
    novel_drugs: dict[str, str] = Field(
        default_factory=dict,
        description="Map of {drug_label: SMILES} for novel drugs not in "
                     "Expert taxonomy or ChEMBL. Coverage predicted via "
                     "GIN. drug_pool can include these labels.",
    )
    # Constraints (None = taxonomy defaults)
    max_arity: int | None = None
    coverage_threshold: float | None = None
    excluded_drugs: list[str] = Field(default_factory=list)
    n_solutions: int = 5


class CombinationDict(BaseModel):
    drug_ids: list[str]
    arity: int
    weighted_coverage: float
    coverage_per_target: dict[str, float]
    toxicity_per_axis: dict[str, float]
    feasible: bool
    constraint_violations: list[str]
    rationale: list[str]


class CoverageComputeResponse(BaseModel):
    active_targets: dict[str, float]
    drug_pool_size: int
    constraints: dict
    framework: str
    taxonomy_version: str
    top_combinations: list[CombinationDict]


@router.post("/api/v1/coverage/compute", response_model=CoverageComputeResponse)
def compute_coverage(req: CoverageComputeRequest):
    try:
        from combo_val.coverage.set_cover import find_top_combinations
        from combo_val.coverage.taxonomy import load_taxonomy
        from combo_val.coverage.merged_pool import MergedDrugPool
        from combo_val.coverage.patient_targets import infer_active_targets
    except ImportError:
        raise HTTPException(500, "coverage module unavailable")

    tx = load_taxonomy()

    # Build patient feature dict from request
    pf: dict[str, Any] = {}
    for g in req.mutations:
        pf[f"mut_{g.upper()}"] = 1
    for fus in req.fusions:
        pf[f"fusion_{fus.upper().replace('-', '_')}"] = 1
    if req.flt3_itd:
        pf["clin_flt3_itd"] = 1
    if req.flt3_tkd:
        pf["clin_flt3_tkd"] = 1
    if req.tp53_multihit:
        pf["tp53_multihit"] = True
    if req.karyo_del_17p:
        pf["karyo_del_17p"] = 1
    if req.age is not None:
        pf["clin_age"] = float(req.age)
    if req.is_relapse:
        pf["clin_is_relapse"] = 1

    active = infer_active_targets(pd.Series(pf), tx,
                                    rna_programs=req.rna_programs)
    if not active:
        return CoverageComputeResponse(
            active_targets={}, drug_pool_size=0, constraints={},
            framework="Palmer-Sorger IDA (Cell 2017, PMID 29245013)",
            taxonomy_version=tx.version, top_combinations=[],
        )

    # 3-tier merged drug pool: Expert ∪ ChEMBL ∪ GIN-novel
    merged = MergedDrugPool(taxonomy=tx)
    if req.novel_drugs:
        merged.register_novel_smiles(req.novel_drugs)

    if req.drug_pool:
        pool = list(req.drug_pool)
        # Include any novel drug names in the pool that weren't passed
        for d in req.novel_drugs:
            if d not in pool:
                pool.append(d)
    else:
        pool = list(merged.known_drugs) + list(req.novel_drugs.keys())

    cm = merged.build_coverage_matrix(pool, smiles_lookup=req.novel_drugs)

    # Constraints (defaults from taxonomy, overridden by request)
    constraints = dict(tx.constraints_default)
    if req.max_arity is not None:
        constraints["max_arity"] = int(req.max_arity)
    if req.coverage_threshold is not None:
        constraints["coverage_threshold"] = float(req.coverage_threshold)

    try:
        results = find_top_combinations(
            active, cm, constraints,
            excluded_drug_ids=req.excluded_drugs or None,
            n_solutions=int(req.n_solutions),
            multistart=8,
        )
    except Exception as e:
        raise HTTPException(500, f"solver failure: {type(e).__name__}: {e}")

    return CoverageComputeResponse(
        active_targets=active,
        drug_pool_size=len(pool),
        constraints=constraints,
        framework="Palmer-Sorger IDA (Cell 2017, PMID 29245013)",
        taxonomy_version=tx.version,
        top_combinations=[
            CombinationDict(
                drug_ids=r.drug_ids, arity=r.arity,
                weighted_coverage=r.total_weighted_coverage,
                coverage_per_target=r.coverage_per_target,
                toxicity_per_axis=r.toxicity_per_axis,
                feasible=r.feasible,
                constraint_violations=r.constraint_violations,
                rationale=r.rationale,
            )
            for r in results
        ],
    )
