# AML Virtual Cell v1.6: Defensive Perturbation Layer

Research use only. This layer prioritizes ex-vivo experiments. It does not
generate a causal post-treatment cell, prescribe treatment, or replace clinical
review.

## Why this release exists

The v1.5 atlas represents observed baseline AML cells as lineage annotations,
14 program scores, and 12 cross-patient expression states. v1.6 adds a guarded
bridge from those baseline states to single-drug response anchors.

The bridge is deliberately conservative:

1. Aggregate single-drug AUC predictions across patient states.
2. Weight each state by its patient fraction and malignant-cell prior.
3. Report between-state heterogeneity.
4. Grade every prediction as supported, directional, or rejected.
5. Keep pair synergy disabled until it passes grouped-pair and representation
   discriminability gates.

Lower predicted AUC means greater ex-vivo sensitivity. These values are external
BeatAML-derived priors, not measured inhibition percentages.

## Required identity gate

Drug perturbation is now downstream of `CELL_IDENTITY_GATE.md`. Expression
clusters, blast priors, and bulk variants do not independently prove which
cells are malignant. Generate `identity_gate_summary.json` with
`scripts/audit_rraml_cell_identity.py`. The prediction command refuses to run
unless the audited patients pass that gate.

## Physical label firewall

The original scTherapy candidate package placed candidate inputs and response
columns in the same directory. A script could avoid reading those columns, but
the blind was procedural rather than technical.

Run:

```bash
python scripts/prepare_sctherapy_blinded_challenge.py \
  --combo-matrix selected_rr3_combo_dose_matrices.csv \
  --flow-validation selected_rr3_flow_validation.csv \
  --single-agent-labels selected_rr3_single_agent_label_distribution.csv \
  --out-dir challenge_v1
```

This creates:

```text
challenge_v1/
  public/
    combo_candidates.csv
    flow_cases.csv
    single_agent_cases.csv
    MANIFEST.json
  sealed/
    combo_outcomes.csv
    flow_outcomes.csv
    single_agent_outcomes.csv
  MANIFEST.json
```

Prediction code must receive only `challenge_v1/public`. It rejects columns
that look like observed response, viability, inhibition, toxicity, synergy,
AUC, IC50, outcome, or label fields. Public and sealed tables are linked by a
deterministic `challenge_row_id`.

The source single-agent table has no drug identity. It can audit label
distribution, but it cannot validate drug ranking.

## State-level prediction contract

The outcome-blind state table must include:

| Column | Meaning |
|---|---|
| `patient_id` | Stable patient identifier |
| `drug_id` | Canonical drug identifier |
| `predicted_auc` | Single-drug BeatAML anchor; lower is more sensitive |
| `state_fraction` | Fraction of this patient's cells in the state |
| `malignant_probability` | State/cell malignant prior |
| `qc_severity` | `ok`, `pipeline_mismatch`, `borderline`, `ood`, or `far_ood` |
| `gene_coverage_fraction` | Fraction of the BeatAML gene panel observed |
| `molecular_similarity` | Nearest training-drug structural similarity |
| `model_in_vocabulary` | Whether the drug was fitted categorically |

Freeze predictions with:

```bash
python scripts/build_rraml_virtual_cell_v16_predictions.py \
  --public-dir challenge_v1/public \
  --identity-gate-summary identity_audit/identity_gate_summary.json \
  --state-predictions state_drug_anchors.csv \
  --pair-candidates challenge_v1/public/combo_candidates.csv \
  --out-dir predictions_v16
```

The command writes canonical CSV bytes and a SHA-256 manifest before any sealed
outcome is read.

## Support policy

A prediction is rejected when any of the following is true:

- RNA QC is `ood` or `far_ood`.
- Gene coverage is below 65%.
- Molecular similarity is missing or below 0.30.

Full support requires all of the following:

- RNA QC is `ok`.
- Gene coverage is at least 90%.
- Drug is in the fitted categorical vocabulary.
- Molecular similarity is at least 0.50.

Everything between these boundaries is directional. Cross-platform scTherapy
to BeatAML predictions with approximately 67% gene coverage therefore remain
directional even when quantile normalization corrects the distribution shift.

## Why pair synergy is disabled

The checkpoint audit found two independent failures:

- Only 2 distinct embeddings among 11 in-vocabulary challenge drugs; 45 of 55
  pairwise distances were effectively zero.
- The hematologic validation used a random row split. Approximately 84% of
  validation rows reused a drug pair present in training. A pair/cell
  memorization baseline reached Pearson 0.487 versus 0.498 for the neural model.

The v1.6 output therefore contains pair support status but no claimed synergy
probability.

A future synergy checkpoint must pass all gates:

- Unordered drug-pair grouped split with zero pair overlap.
- Separate leave-cell-line-out evaluation.
- At least 90% unique drug embeddings on the audit panel.
- At most 10% effectively zero pairwise embedding distances.
- Non-constant prediction distribution.
- Improvement over global, cell-mean, pair-mean, and pair/cell baselines.

## Evaluation after freezing

Only after the prediction SHA-256 has been recorded should the sealed evaluator
be run:

```bash
python scripts/evaluate_rraml_virtual_cell_v16.py \
  --predictions frozen_predictions.csv \
  --prediction-manifest frozen_predictions.csv.manifest.json \
  --sealed-outcomes challenge_v1/sealed/outcomes.csv \
  --score-column predicted_score \
  --outcome-column observed_response \
  --group-column patient_id \
  --out evaluation.json
```

The evaluator refuses a modified prediction file. Bootstrap units should be
patients or unordered drug pairs, not individual dose wells.

## Interpretation boundary

v1.6 can answer:

- Which supported single drugs are directionally favored for a patient?
- Which virtual states appear least sensitive?
- How heterogeneous is the predicted response across malignant states?
- Which pairs are sufficiently in-domain to justify ex-vivo testing?

v1.6 cannot answer:

- What post-treatment transcriptome will occur?
- What dose or schedule is clinically optimal?
- Whether a pair is synergistic in this patient.
- Whether a patient will achieve CR or MRD negativity.
