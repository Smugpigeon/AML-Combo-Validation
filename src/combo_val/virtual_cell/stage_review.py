"""Adversarial and neutral reviews for each virtual-cell evidence gate."""

from __future__ import annotations

from collections.abc import Mapping


def _identity_review(summary: Mapping[str, object]) -> dict[str, object]:
    passed = bool(summary.get("retrospective_drug_validation_stage_unlocked", False))
    return {
        "stage": "cell_identity",
        "strongest_opposing_case": (
            "ScType, CopyKAT and SCEVAN all derive evidence from the same scRNA-seq "
            "matrix. Agreement can reproduce a published annotation workflow but "
            "cannot establish genotype-linked malignant identity."
        ),
        "omitted_or_weak_facts": [
            "No same-cell DNA barcode or genotype-to-cell linkage is available.",
            "No CITE-seq or flow label is paired to each sequenced cell.",
            "Reported blast percentages are patient-level priors, not cell labels.",
        ],
        "optimistic_assumptions": [
            "T-cell references are uncontaminated and representative.",
            "RNA-derived copy-number calls remain stable in low-CNA AML clones.",
            "Major virtual states are not dominated by doublets or stressed normal cells.",
        ],
        "irreversible_or_opportunity_cost": (
            "Training a response model on incorrectly labelled cells would propagate "
            "identity error into every downstream drug and combination score."
        ),
        "worst_consequence": (
            "A normal or reactive state could be presented as a leukemic vulnerability."
        ),
        "strongest_supporting_evidence": (
            f"The pinned published RNA ensemble was evaluated on "
            f"{summary.get('n_cells', 'unknown')} cells across "
            f"{len(summary.get('patients', []))} selected patients."
        ),
        "neutral_verdict": (
            "Permit public retrospective single-drug validation only; keep clinical "
            "and patient-specific identity claims locked."
            if passed
            else "Stop before drug-response validation because identity reproduction failed."
        ),
        "current_stronger_side": "conditional_support" if passed else "opposition",
        "largest_unknown": (
            "Whether RNA-ensemble malignant calls agree with an orthogonal, same-cell "
            "genomic or immunophenotypic label."
        ),
        "evidence_that_would_reverse_the_verdict": (
            "Prospectively barcoded scDNA+scRNA, genotype-linked TARGET-seq, or "
            "cell-matched CITE/flow evidence with predeclared concordance thresholds."
        ),
    }


def _monotherapy_review(summary: Mapping[str, object]) -> dict[str, object]:
    passed = bool(summary.get("viability_direction_gate_pass", False))
    median_rho = summary.get("median_patient_specific_spearman", "unknown")
    increment = summary.get("median_increment_over_drug_mean", "unknown")
    return {
        "stage": "real_single_drug_direction",
        "strongest_opposing_case": (
            "The assay panels were selected by the source study, only three patients "
            "are available, and zero-dose matrix edges measure viability rather than "
            "the predicted post-treatment cell state."
        ),
        "omitted_or_weak_facts": [
            "This is retrospective and cannot recreate a pristine prospective blind.",
            "The source study's panel selection can enrich apparently predictable drugs.",
            "No paired post-treatment scRNA-seq is available to test state movement.",
        ],
        "optimistic_assumptions": [
            "Pseudobulk projection across platforms preserves patient-specific signal.",
            "Drug-name aliases map equivalent compounds and assay contexts.",
            "Within-patient ranking is informative despite the small drug intersection.",
        ],
        "irreversible_or_opportunity_cost": (
            "Unlocking combinations after a weak single-drug result would multiply "
            "model uncertainty and consume scarce organoid wells."
        ),
        "worst_consequence": (
            "A generic drug potency ranking could be mistaken for patient specificity."
        ),
        "strongest_supporting_evidence": (
            f"Frozen predictions achieved median within-patient Spearman {median_rho} "
            f"with median increment {increment} over a drug-mean baseline."
        ),
        "neutral_verdict": (
            "Permit research-only combination benchmarking, but not clinical use; "
            "post-treatment transcriptomic validation remains required."
            if passed
            else "Keep combination training locked and collect independent drug-response data."
        ),
        "current_stronger_side": "conditional_support" if passed else "opposition",
        "largest_unknown": (
            "Whether a real perturbation moves malignant cells toward the model-predicted "
            "state rather than merely lowering bulk viability."
        ),
        "evidence_that_would_reverse_the_verdict": (
            "A preregistered external cohort with paired baseline and post-drug single-cell "
            "profiles, dose/time controls, and patient-matched viability."
        ),
    }


def _combination_review(summary: Mapping[str, object]) -> dict[str, object]:
    passed = bool(summary.get("combination_training_unlocked", False))
    benchmark_unlocked = bool(summary.get("combination_benchmark_unlocked", False))
    blockers = summary.get("blockers", [])
    return {
        "stage": "patient_specific_combination",
        "strongest_opposing_case": (
            "Single-agent direction does not identify interaction, dose ratio, order, "
            "toxicity or causal synergy. A combination model can learn marginal potency "
            "and still fail on true interaction."
        ),
        "omitted_or_weak_facts": [
            "No prospective patient-level pair or triplet holdout is yet available.",
            "Ex vivo synergy is not equivalent to clinical response or tolerability.",
            "The current public cohort is too small for a clinical claim.",
        ],
        "optimistic_assumptions": [
            "Single-drug embeddings transfer to combinations.",
            "Observed ex vivo interaction generalizes across dose and scheduling.",
            "Patient-specific gains exceed a strong drug/pair-mean baseline."
        ],
        "irreversible_or_opportunity_cost": (
            "Prematurely naming a best combination can bias experimental allocation and "
            "clinical review toward an under-validated hypothesis."
        ),
        "worst_consequence": (
            "An ineffective or toxic combination could be over-prioritized because its "
            "components were individually active."
        ),
        "strongest_supporting_evidence": (
            "Both prerequisite research gates passed."
            if passed
            else (
                "The identity and single-drug gates support combination benchmarking, "
                "but the training-data gate remains locked."
                if benchmark_unlocked
                else f"No supporting unlock; active blockers: {blockers}."
            )
        ),
        "neutral_verdict": (
            "Allow versioned research benchmarking against pair-mean and Bliss/Loewe "
            "baselines; keep treatment selection and dosing locked."
            if passed
            else (
                "Allow descriptive combination baselines only; do not train or publish "
                "patient-specific combination predictions."
                if benchmark_unlocked
                else "Do not train or publish patient-specific combination predictions yet."
            )
        ),
        "current_stronger_side": "conditional_support" if passed else "opposition",
        "largest_unknown": (
            "Out-of-patient generalization of interaction residuals beyond the stronger "
            "single-agent and pair-mean baselines."
        ),
        "evidence_that_would_reverse_the_verdict": (
            "A grouped-by-patient external challenge showing calibrated interaction gains, "
            "followed by prospective organoid and safety validation."
        ),
    }


def build_stage_review(
    stage: str,
    summary: Mapping[str, object],
) -> dict[str, object]:
    """Return the requested adversarial review without manufacturing balance."""

    builders = {
        "identity": _identity_review,
        "monotherapy": _monotherapy_review,
        "combination": _combination_review,
    }
    if stage not in builders:
        raise ValueError(f"unsupported review stage: {stage}")
    review = builders[stage](summary)
    review["review_method"] = (
        "strongest opposition, omitted facts, optimistic assumptions, irreversible "
        "cost, worst consequence, strongest support, then neutral verdict"
    )
    review["source_gate_pass"] = {
        "identity": bool(
            summary.get("retrospective_drug_validation_stage_unlocked", False)
        ),
        "monotherapy": bool(summary.get("viability_direction_gate_pass", False)),
        "combination": bool(summary.get("combination_training_unlocked", False)),
    }[stage]
    return review


def render_stage_review_markdown(review: Mapping[str, object]) -> str:
    lines = [
        f"# Virtual-cell gate review: {review['stage']}",
        "",
        "Research use only. This review does not authorize treatment selection.",
        "",
        "## Strongest opposing case",
        str(review["strongest_opposing_case"]),
        "",
        "## Omitted or weak facts",
    ]
    lines.extend(f"- {item}" for item in review["omitted_or_weak_facts"])
    lines.extend(["", "## Optimistic assumptions"])
    lines.extend(f"- {item}" for item in review["optimistic_assumptions"])
    lines.extend(
        [
            "",
            "## Irreversible or opportunity cost",
            str(review["irreversible_or_opportunity_cost"]),
            "",
            "## Worst consequence",
            str(review["worst_consequence"]),
            "",
            "## Strongest supporting evidence",
            str(review["strongest_supporting_evidence"]),
            "",
            "## Neutral verdict",
            str(review["neutral_verdict"]),
            "",
            "## Largest unknown",
            str(review["largest_unknown"]),
            "",
            "## Evidence that would reverse the verdict",
            str(review["evidence_that_would_reverse_the_verdict"]),
            "",
        ]
    )
    return "\n".join(lines)
