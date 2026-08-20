# BeatAML Virtual Patient-State Pilot

> The reusable single-cell upgrade is documented in
> [`VIRTUAL_CELL_V15.md`](VIRTUAL_CELL_V15.md). It adds a 12-patient baseline
> state atlas, frozen marker scales, uncertainty/stability outputs, and a strict
> new-patient inference command while preserving the no-post-treatment-data
> boundary described below.

## What this module does

The pilot builds a shared AML patient-state model from the canonical BeatAML
feature and ex-vivo drug-response tables:

```text
RNA principal components + DNA mutations + clinical/cytogenetic features
                              |
                              v
                 32-dimensional patient state
                              +
                    learned drug perturbation
                              |
                              v
                    predicted single-drug AUC
```

Each patient receives a distinct latent vector, but all patients use the same
trained encoder. This is the appropriate starting point for a cohort of this
size; fitting a separate neural network per patient would not be identifiable.

## Scientific boundary

This is a **virtual patient-state perturbation pilot**, not a complete virtual
cell. BeatAML provides rich baseline bulk RNA/DNA/clinical features and ex-vivo
functional drug response, but not paired single-cell pre/post-treatment
transcriptomes for every patient. Therefore the pilot predicts a response scalar
(AUC), not a generated post-treatment cell state.

It is research-use-only. A low predicted AUC is a candidate sensitivity signal,
not a treatment recommendation.

## Data currently available locally

- 613 patients with 104 canonical baseline features.
- 55,826 patient-drug AUC records from 487 aligned patients and 165 drugs.
- 555,583 quality-controlled dose-level viability records from 569 patients and
  166 inhibitors.
- Official BeatAML sample mapping and drug-family tables.

See `DATA_ACQUISITION_MANIFEST.csv` for source URLs, access restrictions,
checksums, and deferred large datasets.

## Model and evaluation

- Patient encoder: `104 -> 128 -> 32`.
- Drug tower: one learned 32-dimensional embedding per known drug.
- Composition: additive state composition plus multiplicative interaction.
- Objective: Huber loss on training-patient, per-drug standardized AUC.
- Split: 70/15/15 by patient, never by patient-drug row.
- Uncertainty: three independently initialized ensemble members.
- Calibration: one validation-fitted blend coefficient shrinks or retains the
  neural patient-specific residual relative to the drug-mean baseline.
- Required comparator: drug-mean baseline on the identical held-out patients.

## Train

For publication-grade internal evaluation, first fit the RNA PCA on training
patients only:

```bash
/usr/bin/python3 scripts/build_beataml_split_safe_features.py \
  --out-dir data/derived/virtual_cell/beataml_split42
```

Then train on the split-safe feature table. The current Mac has PyTorch in
`/usr/bin/python3`:

```bash
/usr/bin/python3 scripts/train_beataml_virtual_cell_pilot.py \
  --features data/derived/virtual_cell/beataml_split42/beataml_patient_features_split_safe.csv \
  --device cpu \
  --out-dir runs/virtual_cell_beataml_pilot_split_safe
```

For a quick pipeline check:

```bash
/usr/bin/python3 scripts/train_beataml_virtual_cell_pilot.py \
  --smoke \
  --device cpu \
  --out-dir runs/virtual_cell_beataml_pilot_smoke
```

Render the patient-state map, first-batch response heatmap, nearest-state audit,
and machine-readable patient cards:

```bash
/usr/bin/python3 scripts/render_beataml_virtual_cell_pilot.py \
  --run-dir runs/virtual_cell_beataml_pilot_split_safe \
  --features data/derived/virtual_cell/beataml_split42/beataml_patient_features_split_safe.csv
```

## Batch inference

The input may contain additional columns and arbitrary column order. Every model
feature must be present by name.

```bash
/usr/bin/python3 scripts/infer_beataml_virtual_cell_pilot.py \
  --patient-features path/to/patient_features.csv \
  --model-dir runs/virtual_cell_beataml_pilot/model \
  --out-dir runs/virtual_cell_inference
```

## Outputs

- `metrics.json`: baseline, neural, blended, uncertainty, and leakage audit.
- `MODEL_CARD.md`: concise scope, held-out performance, and limitations.
- `heldout_test_predictions.csv`: complete held-out audit table.
- `patient_latent_states.csv`: baseline state vector for every feature patient.
- `first_batch_patients.csv`: first held-out patient batch.
- `first_batch_drug_predictions.csv`: ranked single-drug predictions with
  ensemble uncertainty and observed AUC where available.
- `model/model_seed_*.pt`: self-contained inference checkpoints.
- `report/virtual_patient_state_map.png`: two-dimensional view of the learned
  patient-state manifold.
- `report/first_batch_response_heatmap.png`: first-batch single-drug response map.
- `report/first_batch_state_neighbors.csv`: nearest training states for audit.
- `report/first_batch_virtual_patient_cards.jsonl`: one machine-readable record
  per first-batch patient.

The completed split-safe internal results are recorded in `PILOT_RESULTS.md`.
Commands that run directly inside the compact delivery ZIP are recorded in
`BUNDLE_USAGE.md`.

## What is still needed for a true virtual cell

1. Paired pre/post-treatment scRNA-seq or high-quality bulk RNA for state-change
   supervision.
2. Drug dose, exposure time, and combination matrices.
3. Normal hematopoietic controls for therapeutic-window modeling.
4. Longitudinal R/R samples and clinical outcomes such as CR, MRD, and survival.
5. External and prospective validation before any clinical routing claim.

## Primary public resources

- [BeatAML 2.0 public data](https://biodev.github.io/BeatAML2/)
- [BeatAML study](https://www.nature.com/articles/s41586-018-0623-z)
- [scTherapy processed single-cell objects](https://doi.org/10.5281/zenodo.13340927)
- [TuPro AML processed single-cell/CyTOF data](https://doi.org/10.5281/zenodo.13837019)
- [STATE reference implementation](https://github.com/ArcInstitute/state)
- [CPA method](https://pmc.ncbi.nlm.nih.gov/articles/PMC10258562/)

## Server-side public data acquisition

Use the resumable downloader for the public scTherapy and TuPro Zenodo objects:

```bash
python code/scripts/download_virtual_cell_public_data.py \
  --root /lhcos-data/aml_virtual_cell/public_references \
  --datasets sctherapy,tupro
```

On a memory-constrained host, use `--skip-cytof` to omit the 5.2 GB TuPro CyTOF
object. Every completed file is checked against the MD5 published by Zenodo;
interrupted `.part` files are resumed on the next run.

The compressed TuPro scRNA RDS expands to roughly 10 GB in memory. Convert it
once into reusable sample-level pseudobulk and cell-composition references:

```bash
Rscript scripts/summarize_tupro_singlecell.R \
  --input /lhcos-data/aml_virtual_cell/public_references/tupro/TuPro_AML_scRNA_counts_all_samples_SCE.RDS \
  --out-dir /lhcos-data/aml_virtual_cell/derived/tupro_sample_level
```

The conversion writes raw pseudobulk counts, `log2(CPM + 1)`, the 3,000 most
variable genes, sample QC, cell-type proportions, and a compact RDS. The source
sample identifiers are retained because they are required for reproducible
alignment to the public TuPro metadata; no clinical treatment recommendation is
produced.

The public scTherapy Seurat objects can be converted to sparse STATE-compatible
H5AD files without densifying the count matrix:

```bash
Rscript scripts/export_sctherapy_seurat.R \
  --input-dir /lhcos-data/aml_virtual_cell/public_references/sctherapy \
  --out-dir /lhcos-data/aml_virtual_cell/derived/sctherapy_matrix_market \
  --limit 5

/home/ubuntu/venvs/arc-state-py312/bin/python \
  scripts/build_sctherapy_h5ad.py \
  --export-dir /lhcos-data/aml_virtual_cell/derived/sctherapy_matrix_market \
  --out-dir /lhcos-data/aml_virtual_cell/derived/sctherapy_h5ad_raw
```

These files contain observed baseline cells. The available Seurat metadata have
QC and clustering fields but no drug, dose, exposure-time, or paired post-drug
state fields, so they are useful for patient-state initialization and external
cellular structure only, not as direct transition supervision.

## Defensive perturbation layer

See [VIRTUAL_CELL_V16.md](VIRTUAL_CELL_V16.md) for the physically blinded
challenge workflow, state-aware single-drug aggregation, support-domain gates,
and the criteria required before pair synergy can be re-enabled.

Machine-readable audit values are in [V16_AUDIT.json](V16_AUDIT.json).

