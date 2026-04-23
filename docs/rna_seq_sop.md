# RNA-Seq SOP for AML Combo-Prediction Kit

This SOP defines **how a new patient's RNA-Seq data must be prepared** to be
compatible with the BeatAML-trained model. Deviations invalidate predictions.

## 1. Input reference pipeline (what BeatAML used)

BeatAML 2.0 RNA-Seq was processed as follows (Bottomly et al., Cancer Cell
2022; Tyner et al., Nature 2018):

| Step | Tool | Version | Parameters that matter |
|---|---|---|---|
| Sequencing | Illumina HiSeq 2500 / NovaSeq | — | Paired-end, 100bp |
| QC | FastQC | 0.11.x | Trim adapters; min q30 |
| Alignment | STAR | 2.7.x | `--outFilterMultimapNmax 10`, `--outSAMtype BAM SortedByCoordinate`, `--sjdbOverhang 99` |
| Reference | GENCODE | v37 (hg38) | GTF with basic biotypes |
| Quantification | featureCounts | 2.0.x | `-s 2` (reverse-stranded; BeatAML uses dUTP protocol) |
| Gene naming | HUGO symbols | — | Ensembl IDs mapped via GENCODE GTF; one row per HGNC symbol |
| Normalization | log-like shifted counts | — | Matrix in the public xlsx spans ~[−9, +14] (not raw count, not standard TPM) |

The model sees whatever this pipeline outputs for the 5000 genes it
variance-selected during training. These 5000 HUGO symbols are persisted in
`data/canonical/beataml_rna_preprocessor.joblib` (`bundle["kept_genes"]`).

## 2. What the kit accepts

The `build_patient_features_from_raw()` function accepts a `pandas.Series`:

```python
rna_counts: pd.Series      # index = HUGO gene symbol, values = expression
```

Rules:
- **Index is HUGO symbols**, not Ensembl IDs (`ENSG00000xxxxxx`), not UCSC,
  not mouse orthologs. Ensembl → symbol conversion must happen upstream.
- Values are **non-negative**. If your pipeline can produce negative values
  (e.g., after VST centering), add a shift so the minimum is ≥ 0.
- Values may be **raw counts, TPM, FPKM, or log-transformed** — the quantile
  normalization step handles shape differences. But you should NOT
  arbitrarily switch schemes mid-study.
- Missing gene coverage is tolerated up to ~30%. Missing genes are filled
  with 0 and a warning is emitted. Coverage < 70% triggers a low-confidence
  flag.

## 3. Recommended pipeline for a new lab (clinical-grade)

If your lab does not already have an RNA-Seq pipeline configured to match
BeatAML, use:

```bash
# 1. Align
STAR --runThreadN 8 \
     --genomeDir /path/to/STAR-GENCODE-v37 \
     --readFilesIn R1.fq.gz R2.fq.gz \
     --readFilesCommand zcat \
     --outFilterMultimapNmax 10 \
     --outSAMtype BAM SortedByCoordinate \
     --sjdbOverhang 99 \
     --outFileNamePrefix ${SAMPLE}_

# 2. Quantify
featureCounts \
    -p --countReadPairs \
    -s 2 \
    -a /path/to/gencode.v37.annotation.gtf \
    -o ${SAMPLE}_counts.tsv \
    ${SAMPLE}_Aligned.sortedByCoord.out.bam

# 3. Convert Ensembl → HUGO symbol (join featureCounts output with GTF)
python -c "
import pandas as pd
counts = pd.read_csv('${SAMPLE}_counts.tsv', sep='\t', skiprows=1)
# Columns: Geneid, Chr, Start, End, Strand, Length, <sample_bam>
# Convert Geneid (Ensembl) → HUGO via the GTF's gene_name attribute
# (see scripts/ensembl_to_hugo.py if you need a helper)
"
```

The result is a `gene_symbol → count` TSV file the kit can consume.

## 4. Automatic OOD detection (how the kit catches pipeline mismatches)

Every prediction runs **four sequential QC checks** on the RNA input:

| Layer | Check | Purpose |
|---|---|---|
| 1 | Input variance (log-count std > 0.5) | Catches empty, constant, or corrupt inputs |
| 2 | Gene coverage (% of 5000 trained genes present) | Warns below 70%, critical below 30% |
| 3 | AML marker-gene rank (CD34, KIT, HOXA9, MEIS1 z-score) | Catches non-AML samples even after QN |
| 4 | Mahalanobis distance in PC space, raw AND post-QN | Distinguishes pipeline difference from biology difference |

Severity levels emitted in `KitOutput.confidence_notes`:

| Severity | Meaning | Recommended action |
|---|---|---|
| `ok` | Input matches training distribution | Trust predictions |
| `borderline` | Raw or QN Mahalanobis > 95%ile | Use rankings, not quantitative AUC |
| `pipeline_mismatch` | Raw > 99%ile, QN < 95%ile | Pipeline differs but QN corrected it — still verify PC loadings |
| `ood` | QN distance > 99%ile OR marker biology breaks | Predictions unreliable; verify sample identity |
| `far_ood` | Variance broken OR QN distance > 99.9%ile | Input rejected; do not use |

## 5. Validation examples (synthetic)

Verified with `demo_kit_run.py` and `test_clinical_kit.py`:

| Input | Expected | Actual |
|---|---|---|
| Real BeatAML sample | ok | ✓ ok (Mahalanobis 6.3) |
| Lognormal synthetic counts | pipeline_mismatch (QN saves) | ✓ pipeline_mismatch (raw 36 → QN 4.5) |
| All zeros | far_ood (variance) | ✓ far_ood |
| Constant non-zero | far_ood (variance) | ✓ far_ood |
| Scaled real sample (×100 + 50) | pipeline_mismatch | ✓ pipeline_mismatch |
| Markers forced to zero | ood (marker biology) | ✓ ood |
| 500 of 5000 genes (10% coverage) | far_ood (variance from filling zeros) | ✓ far_ood |
| Real sample with QN off | ok (same as with QN) | ✓ ok |

## 6. What to do if your pipeline produces a `pipeline_mismatch`

This is **common and expected** if your lab's pipeline differs from BeatAML's
in any way. The quantile normalization step inside `project_and_qc()` will
align distributions, and predictions should be usable. But:

1. Run 3-5 **known AML samples** through your pipeline and through BeatAML's
   pipeline side-by-side. If the kit's predictions for these samples agree
   with published AML targeted-therapy response (e.g., FLT3-ITD patient →
   predicted responder to Gilteritinib), trust your pipeline.
2. If predictions **systematically disagree**, either (a) switch to the exact
   BeatAML pipeline or (b) fit a re-calibration transform (Layer 4 of the
   OOD-handling plan; see `docs/kit_readiness.md`).

## 7. What to do if you get `ood` or `far_ood`

1. **Check the gene ID column**. Is it HUGO symbols (`FLT3`, `CD34`) or
   Ensembl IDs (`ENSG00000...`)? The kit needs HUGO.
2. **Check the sample** — is it actually AML? Peripheral blood, bone marrow,
   or leukemic blast fraction? Non-malignant samples will fail.
3. **Check library size** — samples with < 5M mapped reads often fail QC.
4. **Check FastQC** — adapter contamination, low-quality reads, or high
   duplicate rate cause distribution shifts.
5. **Do not use predictions** from a `far_ood` sample.

## 8. Persistence and versioning

The preprocessor bundle (`beataml_rna_preprocessor.joblib`) is part of the
model artifact set and is version-locked with the trained Baseline A MLP.
Its `schema_version` field (currently `beataml_rna_pca_v3_qc_markers`) is
bumped whenever the feature schema changes. The kit refuses to run if the
MLP checkpoint's `feature_cols` length doesn't match the bundle's.
