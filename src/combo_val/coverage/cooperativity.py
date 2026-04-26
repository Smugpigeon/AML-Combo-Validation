"""D.2 — Mutation cooperativity rules.

Co-mutation patterns where the COMBINED biology is different from the
sum of individual mutations. Each rule fires when a specific co-mut
pattern is detected and adjusts axis weights accordingly.

Rules curated from clinical literature, sources cited inline.

Applied AFTER infer_active_targets but BEFORE set-cover. Modifies the
{axis: weight} dict in-place by multiplying active axis weights or
zeroing out axes that are biologically unavailable (e.g. TP53 multi-hit
makes BCL2 dependence unreliable — Ven loses efficacy).

Rule application order: rules are applied sequentially. A later rule
can stack on top of an earlier one (e.g., FLT3-ITD high AR rule +
NPM1+DNMT3A rule both fire on a co-mutated patient).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass
class CooperativityRule:
    """A single cooperativity rule.

    Predicate: takes patient feature dict → bool (does pattern fire?)
    Modify: takes (active_targets dict) → modifies in place
    """
    name: str
    description: str
    citation: str
    predicate: Callable[[dict], bool]
    modify: Callable[[dict], None]
    rationale: str  # human-readable, for report


def _has_mut(pf: dict, gene: str) -> bool:
    return bool(pf.get(f"mut_{gene.upper()}", 0) and float(pf[f"mut_{gene.upper()}"]) > 0.5)


def _has_fusion(pf: dict, fus: str) -> bool:
    return bool(pf.get(f"fusion_{fus.upper().replace('-', '_')}", 0))


# ---------------------------------------------------------------------------
# Curated rules
# ---------------------------------------------------------------------------


def _scale(active, axis: str, factor: float, min_w: float = 0.0,
            max_w: float = 1.0):
    if axis in active:
        active[axis] = float(min(max_w, max(min_w, active[axis] * factor)))


def _ensure(active, axis: str, weight: float):
    """Set axis to AT LEAST weight (don't overwrite if already higher)."""
    if active.get(axis, 0) < weight:
        active[axis] = float(weight)


def _suppress(active, axis: str, factor: float = 0.3):
    """Knock down an axis weight (e.g. TP53 mut → Ven less effective)."""
    if axis in active:
        active[axis] = float(active[axis] * factor)


COOPERATIVITY_RULES: list[CooperativityRule] = [

    # ── 1. NPM1 + DNMT3A → strong MENIN dependence ──
    # NPM1mut/DNMT3A R882 co-mutation drives potent HOX/MEIS1 program,
    # makes MENIN inhibitor especially effective. Issa Nature 2023
    # (PMID 36922593) showed CR > 70% in NPM1+DNMT3A R882 with Revumenib.
    CooperativityRule(
        name="NPM1_DNMT3A_MENIN_strong",
        description="NPM1 + DNMT3A co-mut → MENIN axis weight ↑",
        citation="Issa Nature 2023 (PMID 36922593)",
        predicate=lambda pf: _has_mut(pf, "NPM1") and _has_mut(pf, "DNMT3A"),
        modify=lambda a: _scale(a, "MENIN_NPM1", 1.25),
        rationale="NPM1+DNMT3A 共突变是最经典的 MENINi 适应症 — Revumenib"
                   " 在此亚群 CR > 70% (Issa Nat 2023). MENIN axis 权重提升 1.25×."
    ),

    # ── 2. TP53 multi-hit → BCL2 dependence unreliable ──
    # TP53-mut AML: Ven+Aza CR drops to ~30-40% (vs 65-70% in TP53-wt),
    # and durable response near zero. Pre-clinical: TP53 mut bypass BAX
    # apoptosis activation regardless of BCL-2 inhibition (Pollyea Blood
    # 2023 PMID 37733992). Practical: don't recommend Ven-anchored regimens.
    CooperativityRule(
        name="TP53_multihit_BCL2_unreliable",
        description="TP53 multi-hit → BCL2 axis weight strongly suppressed",
        citation="Pollyea Blood 2023 PMID 37733992",
        predicate=lambda pf: bool(pf.get("tp53_multihit") or
                                   (_has_mut(pf, "TP53") and pf.get("karyo_del_17p"))),
        modify=lambda a: _suppress(a, "BCL2", factor=0.30),
        rationale="TP53 multi-hit AML 对 Ven 临床应答差 (CR ~30-40%, 持久"
                   "应答近零, Pollyea Blood 2023). BCL2 axis 权重压到 0.3×."
    ),

    # ── 3. FLT3-ITD high AR → secondary axis activation ──
    # High allelic ratio FLT3-ITD activates downstream AKT/STAT5
    # signaling AND increases MCL-1 — a known Ven-resistance mechanism.
    # Need to add MCL1 + JAK_STAT to coverage when high AR.
    CooperativityRule(
        name="FLT3_ITD_high_AR_secondary_axes",
        description="FLT3-ITD high AR (≥0.5) activates STAT5 + MCL1 axes",
        citation="Smith Cancer Discov 2018 PMID 30087161; Carter Nat Comm 2020 PMID 32041960",
        predicate=lambda pf: bool(pf.get("clin_flt3_itd") and
                                   float(pf.get("clin_flt3_allelic_ratio") or 0) >= 0.5),
        modify=lambda a: (_ensure(a, "JAK_STAT", 0.40),
                          _ensure(a, "MCL1", 0.35)),
        rationale="FLT3-ITD 高 AR (≥0.5) 通过 AKT/STAT5 + MCL-1 上调 driving "
                   "Ven-resistance (Smith CD 2018, Carter NC 2020). 加 JAK_STAT "
                   "+ MCL1 axes 进 cover."
    ),

    # ── 4. CBF-AML + KIT → KIT axis priority ──
    # t(8;21) or inv(16) with KIT D816V: KIT is the primary "second hit",
    # adding KIT inhibitor to standard 7+3+GO improves outcomes. The
    # default weight gives KIT 0.7; cooperativity bumps to 1.0.
    CooperativityRule(
        name="CBF_AML_KIT_priority",
        description="CBF-AML (t(8;21) or inv(16)) + KIT mut → KIT axis weight ↑",
        citation="Paschka JCO 2013 PMID 23833002",
        predicate=lambda pf: (_has_mut(pf, "KIT") and
                               (_has_fusion(pf, "RUNX1_RUNX1T1") or
                                _has_fusion(pf, "CBFB_MYH11"))),
        modify=lambda a: _ensure(a, "KIT_RTK", 1.0),
        rationale="CBF-AML + KIT mut = KIT 是核心 driver. KIT axis 权重提升 1.0."
    ),

    # ── 5. APL (PML-RARA) → differentiation_block dominates, others muted ──
    # APL is a fundamentally different disease — ATRA+ATO is curative.
    # Suppress all other axes so set cover doesn't recommend bunch of
    # off-target drugs.
    CooperativityRule(
        name="APL_differentiation_dominates",
        description="PML-RARA fusion → differentiation_block axis maxed; suppress others",
        citation="Lo-Coco NEJM 2013 (APL0406, PMID 23841729)",
        predicate=lambda pf: _has_fusion(pf, "PML_RARA"),
        modify=lambda a: (_ensure(a, "differentiation_block", 1.0),
                          _suppress(a, "BCL2", factor=0.5),
                          _suppress(a, "DNMT", factor=0.5)),
        rationale="APL 是独立 entity — ATRA+ATO 标准。differentiation_block "
                   "axis 拉到 1.0,BCL2/DNMT 弱化避免 cover 推 Ven/Aza."
    ),

    # ── 6. Prior HMA failure → DNMT axis suppressed ──
    # Patients with previous HMA exposure (typically MDS → AML) develop
    # tachyphylaxis. HMA in this context CR rate < 20%.
    CooperativityRule(
        name="prior_HMA_failure_DNMT_tachyphylaxis",
        description="Prior MDS/HMA history → DNMT axis weight suppressed",
        citation="Jabbour JCO 2017 PMID 28771409 (HMA failure outcomes)",
        predicate=lambda pf: bool(pf.get("clin_prior_mds") or pf.get("clin_prior_hma")),
        modify=lambda a: _suppress(a, "DNMT", factor=0.4),
        rationale="HMA-pretreated patient (prior MDS or HMA failure) has "
                   "DNMT axis tachyphylaxis. Weight scaled to 0.4×."
    ),

    # ── 7. NPM1 + FLT3-ITD high AR + DNMT3A → triple-driver MENIN priority ──
    # The "M classic NPM1+" triplet from the original papers (e.g.
    # Falini 2015). All three synergize the HOX/MEIS1 program. MENINi
    # particularly potent here.
    CooperativityRule(
        name="NPM1_FLT3_DNMT3A_triplet_MENIN",
        description="NPM1 + FLT3-ITD + DNMT3A triple → strongest MENIN axis",
        citation="Krivtsov Cancer Cell 2006; Issa Nature 2023",
        predicate=lambda pf: (_has_mut(pf, "NPM1") and _has_mut(pf, "FLT3") and
                               _has_mut(pf, "DNMT3A") and pf.get("clin_flt3_itd")),
        modify=lambda a: _scale(a, "MENIN_NPM1", 1.30),
        rationale="经典 NPM1+FLT3-ITD+DNMT3A 三 driver — HOX/MEIS1 program "
                   "极强 → MENIN axis 1.30× 加权."
    ),
]


def apply_cooperativity_rules(patient_features: pd.Series | dict,
                                active_targets: dict[str, float]
                                ) -> tuple[dict[str, float], list[str]]:
    """Apply all cooperativity rules in order. Returns (modified active_targets,
    list of rule names that fired with rationales).
    """
    if isinstance(patient_features, pd.Series):
        pf = patient_features.to_dict()
    else:
        pf = dict(patient_features)

    fired: list[str] = []
    out = dict(active_targets)
    for rule in COOPERATIVITY_RULES:
        try:
            if rule.predicate(pf):
                rule.modify(out)
                fired.append(f"{rule.name}: {rule.rationale}")
        except Exception:
            # Rule predicate or modify crashed — skip silently
            continue
    return out, fired
