# AML Virtual Cell Research Sequence

The project follows a hard-gated sequence. Later stages cannot compensate for
failed earlier stages.

```text
Stage 1: prove cell identity
        |
        v
Stage 2: prove predicted drug-direction changes
        |
        v
Stage 3: train and unlock patient-specific combination prediction
```

## Stage 1: cell identity

**Question:** Which cells are malignant, normal, or unresolved, and which
expression states contain those cells?

**Required evidence:** barcode-linked single-cell DNA/CNV, or concordant
orthogonal phenotype and AML-specific reference evidence.

**Outputs:** cell, state, and patient identity audits with explicit blockers.

**Current status:** locked. None of the three public patients passes the gate.
See `CELL_IDENTITY_GATE.md`.

## Stage 2: perturbation direction

**Question:** Does a frozen model predict the direction and distribution of a
real single-drug response better than simple baselines?

This stage has two distinct tracks:

1. A retrospective, outcome-blind challenge on the three public scTherapy
   patients. Existing labels remain sealed until prediction hashes are frozen.
2. A prospective hospital cohort with drug, dose, time, replicate, viability,
   malignant fraction, normal-cell toxicity, and selected post-treatment
   single-cell measurements.

The required baselines are no change, global drug mean, state-conditioned mean,
regularized linear models, and nearest-neighbor/optimal-transport models.

**Current status:** locked by Stage 1. BeatAML AUC anchors remain external
directional priors, not patient-cell transition labels.

## Stage 3: patient-specific combinations

**Question:** Does a pair model improve over additive, Bliss/HSA/ZIP, pair-mean,
and pair/cell baselines on unseen patients and unseen unordered drug pairs?

Unlocking requires:

- A passing Stage 2 result with patient-level uncertainty.
- Grouped unordered-pair splits with zero pair overlap.
- Leave-patient and leave-center evaluation.
- Non-collapsed drug representations.
- Normal-cell selectivity and dose/schedule support.

**Current status:** locked. The prior synergy checkpoint failed representation
and split gates, so no synergy score is emitted.

## Separate clinical protocol

MDT agreement, CR, MRD, toxicity, and survival belong to a later clinical
decision-support protocol. MDT agreement measures workflow concordance, not
whether a virtual cell correctly predicted biology. It must not be used as the
primary endpoint for Stages 1 or 2.

## Stop rules

- If identity cannot be anchored, stop perturbation modeling for that state.
- If measured perturbation signal does not exceed assay variation, improve the
  assay before changing the model.
- If a complex model does not beat frozen simple baselines, retain the simpler
  model and do not increase parameter count.
- If single-drug direction fails, do not train a combination model.
