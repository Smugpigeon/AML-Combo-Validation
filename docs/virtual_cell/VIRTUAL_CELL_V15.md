# AML Virtual Cell v1.5

## Purpose

AML Virtual Cell v1.5 is a research-only baseline state model for public AML
single-cell transcriptomes. It expands the original BeatAML patient-vector
pilot into a hierarchical representation:

```text
observed cell -> broad lineage + biological programs + virtual state
             -> state-level baseline vector
             -> patient composition and heterogeneity vector
             -> frozen input contract for later perturbation prediction
```

It does not claim to reconstruct a causal post-treatment cell and does not
produce a treatment prescription.

## Data used in the frozen reference

- 12 public scTherapy AML samples.
- 48,100 observed baseline cells.
- 21,135 genes and a STATE-compatible 2,000-HVG matrix.
- Three focused R/R AML samples: `patient5`, `patient6`, and `patient12`.
- 11,652 focused cells in total.
- Clinical blast-fraction priors were available only for the three focused
  samples.
- Drug-combination, flow-cytometry, and single-agent validation labels were not
  read during atlas construction.

## Model layers

### 1. Frozen expression normalization

The STATE-preprocessed log-expression matrix is converted back to linear
proportions, normalized to 10,000 counts per cell, and log-transformed again.
This produces the scale expected by CellTypist while preserving the sparse
matrix.

### 2. Broad-lineage evidence

Each cell receives two independent lineage signals:

- curated hematopoietic marker modules;
- CellTypist `Immune_All_Low.pkl`, with majority voting inside each source
  patient/cluster.

The output records concordant, discordant, marker-supported,
CellTypist-supported, and low-confidence calls. Disagreement is retained as
`ambiguous`; it is not silently overwritten.

### 3. Biological program scores

Fourteen transcriptomic programs are calculated, including stem/progenitor,
proliferation, apoptosis priming, anti-apoptotic escape, interferon,
inflammatory JAK/STAT, hypoxia, glycolysis, oxidative phosphorylation,
DNA-damage repair, MAPK feedback, oxidative stress, p53 stress, and myeloid
differentiation.

For gene `g`, the frozen reference score is:

```text
z_g = clip((x_g - median_g) / robust_scale_g, -8, 8)
```

The module score is the mean of available `z_g` values. These are relative
transcriptomic activities, not direct pathway flux or phosphoprotein activity.

### 4. Cross-patient virtual states

The 2,000-HVG matrix is projected to 30 components using truncated SVD,
standardized, and partitioned into 12 cross-patient expression-state
prototypes. `VS01` to `VS12` are stable identifiers ordered by reference-state
size.

Two uncertainty measures are kept separately:

- geometric assignment margin: `1 - nearest_distance / second_distance`;
- initialization stability: agreement after label-aligned repeated clustering.

The 12-state reference explains 63.2% of the input HVG variance. Across four
additional initializations, mean adjusted mutual information is 0.600 and mean
cell-level initialization agreement is 0.569. The atlas is therefore useful as
a reproducible research representation, but its states should not be described
as definitive biological clones.

### 5. Transcriptomic suspicion and blast prior

The transcriptomic suspicion index combines stem/progenitor, proliferation,
anti-apoptotic, DNA-repair, and myeloid-differentiation signals while reducing
the score of confident normal lymphoid/erythroid anchors.

When a reported scRNA blast fraction is available, the cell ranking is retained
and a single logit offset is fitted so that mean probability equals the known
prevalence:

```text
p_prior = sigmoid(logit(p_expression) + intercept)
```

This is prior-anchored representation, not independent malignant-cell
validation.

## Focused-patient results

| Patient | Cells | Mean state initialization stability | Effective states | Blast prior |
|---|---:|---:|---:|---:|
| patient5 | 2,365 | 0.693 | 2.35 | 54.4% |
| patient6 | 5,677 | 0.666 | 3.70 | 40.0% |
| patient12 | 3,610 | 0.428 | 1.53 | 65.3% |

`patient6` sits at the edge of the 12-patient reference. Its main shift drivers
are high `VS05` and `VS08` fractions plus B-cell/plasma-cell components. This is
a transparent cohort-relative observation, not a clinical out-of-distribution
diagnosis.

## Build the frozen reference

```bash
PYTHONPATH=src python scripts/build_rraml_virtual_cell_v15.py \
  --adata /path/to/sctherapy_all12_state_preprocessed_2000hvg.h5ad \
  --patient-manifest /path/to/selected_rr3_patient_manifest.csv \
  --out-dir /path/to/virtual_cell_v15 \
  --selected-samples patient5,patient6,patient12 \
  --n-states 12 \
  --n-components 30 \
  --celltypist-model Immune_All_Low.pkl
```

## Infer a new patient batch

The input must be processed with the same frozen 2,000-HVG schema. The script
rejects mismatched genes or column order instead of silently producing an
incomparable state.

```bash
PYTHONPATH=src python scripts/infer_rraml_virtual_cell_v15.py \
  --adata /path/to/new_patient_state_preprocessed.h5ad \
  --model /path/to/virtual_state_atlas_model.joblib \
  --patient-manifest /path/to/new_patient_manifest.csv \
  --out-dir /path/to/new_patient_virtual_cell_v15 \
  --write-h5ad
```

## Main artifacts

- `virtual_state_atlas_model.joblib`: frozen SVD, scaler, state prototypes,
  stable label mapping, marker scales, and reference-shift model.
- `selected_rr3_virtual_cell_v15.h5ad`: focused full-transcriptome cells plus
  virtual-cell annotations.
- `cell_state_annotations.parquet`: all 48,100 cell-level annotations.
- `patient_state_vectors.csv`: patient-level compositional and program vectors.
- `virtual_state_prototypes.csv`: state-level lineage/program descriptions.
- `virtual_state_descriptive_markers.csv`: descriptive state markers without
  invalid cell-level pseudo-replicate p-values.
- `selected_rr3_state_perturbation_inputs.csv`: frozen baseline input contract
  for later drug/dose/time perturbation models.
- `patient_reference_shift_drivers.csv`: interpretable cohort-shift drivers.
- `ARTIFACT_MANIFEST.csv`: file sizes and SHA-256 values.

## Verified reproducibility

The three focused patients were passed back through the frozen inference model:

- 11,652/11,652 rows recovered;
- virtual-state agreement: 100%;
- consensus-lineage agreement: 100%;
- maximum absolute program-score difference: 0;
- maximum absolute transcriptomic-suspicion difference: 0.

## Remaining hard limits

1. No matched baseline/post-treatment single-cell profiles are available.
2. Virtual states are expression states, not scDNA-confirmed clones.
3. Broad lineage references can misclassify AML blasts.
4. Bone-marrow microenvironment, PK/PD, protein, MRD, and longitudinal clinical
   outcomes are incomplete.
5. A future perturbation head must be frozen before the held-out combination and
   flow-validation labels are opened.

This AI workflow is a research and engineering assistant for state
characterization and assay prioritization. It is not a prescription engine and
does not replace physician judgment or multidisciplinary review.
