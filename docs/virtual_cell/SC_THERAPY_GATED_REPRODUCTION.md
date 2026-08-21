# Gated scTherapy Reproduction and Drug-Direction Challenge

Research use only. This workflow does not select treatment, dose, or schedule.

## Purpose

This workflow implements the required evidence order:

1. Reproduce cell identity calls.
2. Freeze predictions and test real single-drug viability direction.
3. Unlock combination benchmarking only if both earlier gates pass.

No later result can compensate for a failed earlier gate.

## Two different identity gates

The project intentionally keeps two identity standards separate.

### Strict identity gate

The strict gate requires same-cell genomic evidence or concordant orthogonal
phenotype plus AML-specific reference evidence. It can support patient-cell
identity claims. The current three public patients do not have this evidence,
so this gate remains locked.

### Published RNA-ensemble reproduction gate

The public retrospective gate reproduces the published scTherapy workflow:

- general ScType annotation and T-cell normal references;
- AML ScType malignant/healthy markers;
- SCEVAN RNA-derived copy-number classification;
- CopyKAT RNA-derived copy-number classification;
- agreement audit at cell, virtual-state, and patient levels.

All three methods use the same scRNA-seq matrix. This is an RNA-ensemble
reproduction, not single-cell DNA validation. A pass unlocks only the public
retrospective drug-direction challenge.

## Frozen upstream implementation

- scTherapy commit: `4366320fe81b161dc750957b1a05b39a504ecc75`
- Docker image: `kmnader/sctherapy_v5@sha256:2ccea16a2740109279e812636a77a2245cf86061b980f8bc7aa3fc4d7f31f055`
- `identify_healthy_mal_v5.R` SHA-256: `647d80e224298389ce52da2a265d1dd437c82b60145d45e71fd1df9d6ceba08c`
- AML marker workbook SHA-256: `142088466aacb167913e3c4a91bd96f4b69f3fa816fb27086bed6e1cdb419f11`
- ScType commit: `630e15cf1e51f2612eda4ad0406dfb17503fa8c9`
- SCEVAN helper commit: `06bf2bc390235f00abda6c0fbc5cbc09c5f5a420`
- ScType database SHA-256: `f9adcf25f056184107150f0413241c0ccd311cf98309bc4d412071ae38ccabad`
- ScType helper SHA-256 values: `7f9acae84a524e5d01b523245bf4ca9344bfecfc38d40c0241645441ec725376`,
  `f1f6fc04b0edf88b8a17f8cc997eda129bfc84a5e71b1c9f074bd6aebd3749cc`,
  and `8df238b663f51d41e9db838861a5f4978635ad33bb68ad58d22b74b726455a27`
- SCEVAN helper SHA-256: `26cb94cf317c86d731cdcde918ecdcbd82c5227843c9fa175baf10bc97201671`

The local R wrapper calls the pinned upstream functions and does not vendor or
rewrite their algorithm.

## SCEVAN semantic audit and corrected mode

The released scTherapy helper has an important semantic distinction. With
`all_pred=FALSE`, it executes SCEVAN but writes `SCEVAN_output=malignant` for
every cell outside the supplied T-cell normal-reference set. That column is
therefore a released-code effective rule, not the tumour/normal classification
returned by SCEVAN. The audit keeps this behavior for reproduction but never
labels it as a true SCEVAN call.

Two modes are generated and evaluated separately:

- `released_code_effective`: reproduces the released writeback rule exactly;
- `true_scevan_corrected`: uses the actual SCEVAN tumour/normal classification.

The corrected path uses the pinned pure-Python SCEVAN implementation at commit
`c7917cb47df9ab46465f9f94afbe801a71bd74e6`. Before using it for the patient
whose native R run fails, parity is required on both R-computable reference
patients with predeclared thresholds: call agreement at least 0.95, adjusted
Rand index at least 0.95, malignant-call Jaccard at least 0.90, and common
evaluable-cell fraction at least 0.95. The observed values were 1.00 for every
metric in both reference patients.

This repair does not create independent genomic evidence. ScType, CopyKAT, and
SCEVAN remain three analyses of the same RNA matrix.

## Resource-safe component execution

Run one patient per process. On the current 7.3 GiB server, two-worker SCEVAN
and CopyKAT jobs can lose one forked worker and return half-length intermediate
objects. Such results are failures, not partial classifications. The supported
fallback is:

- true SCEVAN through the parity-gated Python implementation;
- CopyKAT as an independent component with `--ncores 1`;
- no patient sampling, cell removal, or threshold relaxation.

After all component files exist, assemble both semantic modes and compute both
identity gates. `compare_sctherapy_identity_mode_gates.py` permits the frozen
single-drug challenge only when both mode-level gates pass and every patient's
gate decision agrees.

## Released-wrapper reference command

The command below reproduces the released all-in-one wrapper. It is retained
for semantic auditing and is not the supported low-memory command: on the
current server, one forked worker can disappear and invalidate the whole
patient result. Use the component execution described above for production
reproduction.

```bash
docker run --rm --cpus 2 --memory 6g \
  -v /lhcos-data/aml_virtual_cell:/work \
  -v /home/ubuntu/sctherapy_aux_pinned:/pinned:ro \
  kmnader/sctherapy_v5@sha256:2ccea16a2740109279e812636a77a2245cf86061b980f8bc7aa3fc4d7f31f055 \
  Rscript /work/code/AML-Combo-Validation-vcell-gated/scripts/run_sctherapy_identity_ensemble.R \
  --input-dir /work/public_references/sctherapy \
  --output-dir /work/derived/virtual_cell_gated_sequence_20260820/identity_ensemble \
  --upstream-script /work/public_references/sctherapy_upstream_4366320/identify_healthy_mal_v5.R \
  --custom-marker /work/public_references/sctherapy_upstream_4366320/sctype_aml_cellmarker20_cosmic.xlsx \
  --sctype-dir /pinned \
  --scevan-script /pinned/scevan_mod.R \
  --patients patient5,patient6,patient12 \
  --ncores 2
```

## Stages 1-3 command

After the three official identity-call CSV files exist, the Python orchestrator
runs the remaining hard-gated sequence:

```bash
python scripts/run_sctherapy_gated_sequence.py \
  --identity-call-dir /lhcos-data/aml_virtual_cell/derived/virtual_cell_gated_sequence_20260820/identity_ensemble \
  --cell-annotations /lhcos-data/aml_virtual_cell/derived/virtual_cell_v15_final_20260820/selected_rr3_cell_state_annotations.parquet \
  --patient-manifest /lhcos-data/aml_virtual_cell/derived/sctherapy_selected_rr3_release/selected_rr3_patient_manifest.csv \
  --h5ad /lhcos-data/aml_virtual_cell/derived/sctherapy_selected_rr3_h5ad/sctherapy_selected_rr3_raw.h5ad \
  --preprocessor /path/to/beataml_rna_preprocessor.joblib \
  --split-safe-rna-preprocessor /path/to/rna_pca_split_safe_preprocessor.npz \
  --model-dir /path/to/frozen_split_safe_model \
  --combo-matrix /lhcos-data/aml_virtual_cell/derived/sctherapy_selected_rr3_release/selected_rr3_combo_dose_matrices.csv \
  --out-dir /lhcos-data/aml_virtual_cell/derived/virtual_cell_gated_sequence_20260820/gated_sequence
```

## Stage 2 endpoint

The source combination matrices contain zero-dose edges. The workflow extracts
only wells where exactly one drug dose is positive and collapses repeated edge
measurements. Combination wells are never treated as monotherapy.

The primary public endpoint is a fixed log-dose integrated inhibition score
over 0.1-1000 nM. Predictions are generated from a frozen split-safe BeatAML
ensemble, hashed, and only then joined to sealed outcomes.

The RNA projection uses the PCA genes, mean, and components fitted on the
recorded BeatAML training split only. The model input is additionally blocked
when more than 20% of RNA PCs exceed an absolute training-standardized z-score
of 5, or any RNA PC exceeds 25. This prevents a severe bulk-versus-single-cell
platform shift from being mistaken for patient-specific biology.

The gate requires:

- at least 3 evaluable patients and 12 patient-drug rows;
- positive within-patient Spearman correlation in at least 2 patients;
- median within-patient Spearman at least 0.20;
- non-negative median increment over the drug-mean baseline;
- no contradictory duplicate zero-dose edges.

This tests viability direction. It does not prove a post-treatment
transcriptomic state transition because paired post-treatment scRNA-seq is not
available in this public source.

## Stage 3 boundary

A pass unlocks research-only combination benchmarking against strong additive,
drug-mean, pair-mean, Bliss, Loewe, HSA, and ZIP baselines. It does not unlock:

- automatic treatment selection;
- dose or schedule recommendations;
- clinical efficacy claims;
- patient-specific prescribing.

Every stage writes a structured adversarial review containing the strongest
opposing case, omitted facts, optimistic assumptions, irreversible cost, worst
consequence, strongest supporting evidence, neutral verdict, largest unknown,
and evidence that would reverse the verdict.
