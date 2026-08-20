# Drug Perturbation Validation Protocol

Research use only. This protocol is biologic validation of a virtual-cell
transition model. It is separate from clinical decision concordance.

## Prerequisite

Every included patient must pass the identity gate in
`identity_gate_summary.json`. The prediction command checks this file and
refuses to run when the gate is locked or when requested patients were not part
of the audit.

## Track A: retrospective three-patient challenge

The public `patient5`, `patient6`, and `patient12` cases are a feasibility
challenge, not a generalization cohort.

1. Keep combo, flow, and single-agent outcomes in the sealed directory.
2. Build identity evidence using baseline data only.
3. Freeze single-drug and state-resistance predictions with SHA-256.
4. Unblind only after the prediction manifest is immutable.
5. Report each patient separately; dose wells are not independent patients.
6. Treat `patient12` as exploratory because its reported flow validation is
   underpowered.

No new post-treatment transcriptome can be inferred from these public files.
The track can validate ranking direction only where drug identities and outcomes
are available.

## Track B: prospective sample-sparing perturbation cohort

Use two assay tiers for newly collected, consented research samples:

1. Screen the frozen candidate panel using multiple doses, vehicle controls,
   positive controls, recorded exposure time, and replicate aliquots with flow
   or image-based viability.
2. Select representative sensitive, resistant, and neutral conditions for
   post-treatment scRNA/CITE-seq rather than sequencing every well.

Record at minimum:

```text
patient_id, sample_id, source, collection_time
drug_id, dose, dose_unit, exposure_hours, batch_id, replicate_id
viability, malignant_fraction, normal_fraction, normal_cell_toxicity
baseline_state_id, post_state_id, post_expression
```

Assay harmonization is part of the model: medium, processing delay, fresh versus
frozen status, tissue source, dose grid, and response metric must be recorded.

## Prediction targets

The model may predict:

- Differential-expression direction and effect size.
- Post-treatment state proportions.
- Malignant-cell viability and residual resistant-state mass.
- Normal-cell toxicity.
- Calibrated uncertainty and out-of-domain status.

It must not predict CR, MRD negativity, or clinical dose from this protocol.

## Evaluation

Use patient-, drug-, dose/time-, and center-held-out evaluations. Report:

- DEG recovery and correlation of perturbation-induced log-fold changes.
- Distribution distance for post-treatment cells.
- Error in state composition and malignant fraction.
- Viability and selectivity error.
- Within-patient drug ranking and top-k recovery.
- Calibration and abstention performance.

Complex models must be compared on identical splits against no-change, global
mean, drug mean, state mean, regularized linear, and nearest-neighbor baselines.
Bootstrap independent patients, not cells or wells.

## Unlock decision

Stage 3 remains locked until a pre-registered analysis shows that the
state-aware model improves over simple baselines on independent patients, the
predicted resistant state is experimentally enriched, and normal-cell
selectivity is measured. The required cohort size must be determined from the
pilot effect size and assay variance; the three public patients cannot satisfy
this requirement.
