# Hard-gated virtual-cell sequence: current evidence result

**Date:** 2026-08-21  
**Code:** `codex/virtual-cell-identity-gate` at `561bc40`  
**Scope:** public retrospective research only; not treatment selection

## Decision

The pipeline stopped at the cell-identity gate with status
`blocked_at_identity`. No single-drug response prediction was generated, no
combination model was trained, and no patient-specific combination was
released.

The current objects are useful patient-conditioned transcriptomic state
representations. They are **not yet validated patient-specific virtual cells
for drug-response prediction**.

## Predeclared sequence

1. Reproduce cell identity before using any drug outcome.
2. If and only if identity passes, freeze single-drug predictions and test them
   against real viability outcomes and a strong drug-mean baseline.
3. If and only if both prior gates pass, assess whether combination data are
   sufficient for grouped-by-patient model development.
4. Patient-specific combination prediction remains locked until an external
   held-out evaluation beats additive and pair-mean baselines.

The retrospective RNA identity policy required at least two algorithms per
cell, at least two-thirds agreement, patient call coverage at least 0.80,
major-state coverage at least 0.70, major-state purity at least 0.60, at least
20 known-normal references, and no more than 20 percentage points of
disagreement with the reported scRNA blast fraction.

## Pinned identity reproduction

The reproduction used the official scTherapy workflow with ScType, SCEVAN and
CopyKAT in the pinned container
`kmnader/sctherapy_v5@sha256:2ccea16a2740109279e812636a77a2245cf86061b980f8bc7aa3fc4d7f31f055`
and upstream scTherapy commit
`4366320fe81b161dc750957b1a05b39a504ecc75`.

These three calls share the same scRNA-seq input. Their agreement reproduces an
RNA-based annotation workflow but is not independent genomic proof. The strict
orthogonal identity gate therefore remains false even for a patient that passes
the retrospective RNA gate. The method context is described in the
[Nature Communications study](https://www.nature.com/articles/s41467-024-52980-5)
and the [official scTherapy repository](https://github.com/kris-nader/scTherapy).

## Identity results

| Patient | Cells | Official run | RNA call coverage | Malignant fraction among called cells | Reported scRNA blast | Point gap | Retrospective RNA gate |
|---|---:|---|---:|---:|---:|---:|---|
| patient12 | 3,610 | complete | 98.98% | 73.89% | 65.3% | 8.59 pp | pass |
| patient5 | 2,365 | complete | 85.03% | 80.31% | 54.4% | 25.91 pp | fail |
| patient6 | 5,677 | failed | unavailable | unavailable | 40.0% | unavailable | fail |

`patient6` failed inside the pinned official SCEVAN path with
`arguments imply differing number of rows: 6773, 3386`. This is recorded as an
algorithm reproduction failure, not as zero normal cells or a biological
identity result. There were zero true barcode mismatches; 5,677 cells lack an
identity call record because the patient-level workflow did not complete.

The major-state results were:

| Patient | State | Fraction of patient | Coverage | Dominant call | Purity | Pass |
|---|---|---:|---:|---|---:|---|
| patient12 | VS01 | 91.72% | 99.03% | malignant | 74.81% | yes |
| patient5 | VS01 | 11.97% | 96.82% | healthy | 85.77% | yes |
| patient5 | VS09 | 77.42% | 82.14% | malignant | 99.60% | yes |
| patient6 | VS01 | 7.10% | unavailable | unavailable | unavailable | no |
| patient6 | VS05 | 46.56% | unavailable | unavailable | unavailable | no |
| patient6 | VS08 | 36.02% | unavailable | unavailable | unavailable | no |

### Sensitivity of the patient5 point estimate

For patient5, the ensemble produced 1,615 malignant, 396 healthy and 354
unresolved calls. The reported 80.31% malignant fraction uses called cells as
the denominator. If unresolved cells are retained as an uncertainty interval,
the possible whole-sample malignant fraction is 68.29%-83.26%; the reported
54.4% blast fraction is 13.89 percentage points below that interval.

Therefore, patient5 fails the predeclared point-estimate rule, but that failure
is sensitive to how unresolved cells are handled. It should not be described as
a definitive biological contradiction. The global gate remains failed
independently because patient6 has no completed official identity result.

## Combination-data readiness

The combination matrix was audited even though model training remained locked.

| Criterion | Observed | Required | Result |
|---|---:|---:|---|
| Patients | 3 | at least 20 | fail |
| Positive-dose combination wells | 648 | at least 500 | pass |
| Unique unordered pairs | 63 | at least 30 | pass |
| Median patients per pair | 1 | at least 3 | fail |
| Largest patient contribution | 33.33% | at most 25% | fail |
| Response standard deviation | 35.98 | at least 5 | pass |

The matrix is suitable for descriptive dose-surface and baseline analyses, but
not for training or publishing a patient-specific combination predictor.

## Adversarial review

### Strongest opposing case

All three identity algorithms derive from the same RNA matrix, patient6 cannot
be reproduced by the pinned official path, patient5 is sensitive to the
unresolved-cell denominator, and there is no same-cell DNA or orthogonal
immunophenotype. Training now could turn a reactive or normal state into a
spurious leukemic vulnerability.

### Strongest supporting case

patient12 passes all retrospective RNA thresholds; patient5 shows a coherent
healthy VS01 and highly pure malignant VS09; the workflow is container-pinned,
input-hashed and hard-gated; and drug outcomes were not used to rescue the
identity result.

### Neutral verdict

The opposition is currently stronger. Keep the project at the
identity-validation stage. The negative result is useful because it prevents a
method-reproduction result from being mislabeled as a validated virtual cell.

### Largest unknown

Whether the RNA-derived malignant calls agree with a same-cell genomic or
orthogonal immunophenotypic label, and whether a real perturbation moves those
validated malignant cells in the predicted direction.

### Evidence that would reverse the verdict

1. Reproduce or formally repair the patient6 SCEVAN failure without changing
   thresholds or excluding the patient, then rerun all selected patients.
2. Add genotype-linked TARGET-seq, paired scDNA+scRNA, or cell-matched CITE-seq
   or flow evidence under predeclared concordance thresholds.
3. After identity passes, use an independent cohort with baseline and
   post-treatment single-cell profiles, dose/time controls and matched
   viability. Freeze predictions before outcomes are opened.
4. Before combination training, obtain at least 20 patients with repeated pairs
   across patients and cap any one patient's contribution at 25%.
5. Require grouped-by-patient external validation against additive,
   drug-mean and pair-mean baselines before releasing patient-specific scores.

## Audit corrections made during this run

Two reporting defects were found by reviewing the outputs rather than trusting
the first summary:

- `healthy` was initially omitted from the non-malignant label normalization,
  which understated coverage and falsely displayed 100% malignant among called
  cells. The mapping and regression test were fixed in `651f538`.
- A failed patient-level identity workflow was initially counted as barcode
  mismatch and zero normal references. Missing call records, true barcode gaps
  and upstream run failures are now separated in `561bc40`.

Neither correction was used to relax a gate. Both make the blocked result more
accurate and auditable.

## Evidence files

- [Pipeline status](results_20260821/PIPELINE_STATUS.json)
- [Patient identity gate](results_20260821/stage1_identity/retrospective_patient_identity_gate.csv)
- [State identity gate](results_20260821/stage1_identity/retrospective_state_identity_gate.csv)
- [Identity summary and input hashes](results_20260821/stage1_identity/retrospective_identity_gate_summary.json)
- [Identity adversarial review](results_20260821/reviews/identity_adversarial_review.md)
- [Combination readiness](results_20260821/stage3_combination_readiness.json)
- [Combination unlock decision](results_20260821/combination_unlock_decision.json)
- [Combination adversarial review](results_20260821/reviews/combination_adversarial_review.md)
- [Official run status](results_20260821/identity_reproduction/run_status.json)
- [Pinned R session](results_20260821/identity_reproduction/R_SESSION_INFO.txt)

The full server result is frozen at:

`/lhcos-data/aml_virtual_cell/derived/virtual_cell_gated_sequence_20260820/gated_sequence_final_20260821`

