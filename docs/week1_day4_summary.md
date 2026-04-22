# Week 1 Day 4 — TCGA-LAML independent-cohort ETL

## Summary

Fetched TCGA Acute Myeloid Leukemia PanCancer Atlas 2018 cohort (n=173
patients with RNA-Seq V2 data) from cBioPortal's public REST API, built the
same 80-dim patient feature schema as BeatAML, and saved canonical tables for
Week 5 independent-cohort validation.

## Key numbers

| Stage | Count |
|---|---|
| TCGA-LAML samples with mRNA | 173 |
| Unique patients | 173 (1:1 mapping) |
| Genes fetched via cBioPortal | 30,000 → 16,596 retained after pivot |
| Mutation rows (25 curated genes) | 357 (145 unique samples) |
| Clinical attributes available | 6 (OS_MONTHS, OS_STATUS, SUBTYPE, SAMPLE_COUNT, CANCER_TYPE_ACRONYM, IN_PANCANPATHWAYS_FREEZE) |
| Deceased at last follow-up | 114/173 (65.9%) |
| Median OS (months) | 11.0 |
| RNA 50-PC cumulative variance | 79.1% |

## Top mutation prevalence (25-gene panel)

| Gene | TCGA-LAML | BeatAML (comparison) |
|---|---:|---:|
| FLT3 | **30.1%** | 35.2% |
| NPM1 | **27.7%** | 32.0% |
| DNMT3A | **24.9%** | 17.5% |
| IDH2 | 10.4% | ~10% |
| RUNX1 | 9.8% | 13.4% |

Good match with BeatAML and with AML literature (FLT3 ~30%, NPM1 ~30%,
DNMT3A ~20%). DNMT3A is slightly higher in TCGA — expected since TCGA-LAML is
biased toward older, de-novo cases where DNMT3A is more common.

## Important caveat: clinical fields are BeatAML-median placeholders

cBioPortal's PanCancer Atlas API only exposes OS_MONTHS, OS_STATUS, SUBTYPE
for TCGA-LAML patients. AGE, SEX, RACE, ELN risk are listed as available
attributes but return empty values (privacy or data-sparsity reasons).

To preserve the 80-dim feature schema for later transfer, we fill the 5
clinical columns with BeatAML medians (age=62, ELN ordinal=1.0,
blast_pct=70%, secondary_aml=0.16, fit_for_intensive=0.5). This means the
clinical features carry **no TCGA-specific information**; all personalization
for TCGA predictions comes from RNA PCA + mutation features.

**Practical implication**: Week 5 TCGA validation should lean on the
mechanism-prior combo scoring (uses mutations + expression signatures), not
on direct transfer of the BeatAML-trained Baseline A MLP (which expects real
clinical features).

## Outputs

```
data/canonical/
├── tcga_laml_patient_features.csv    # 173 × 81 (patient_id + 80 features)
├── tcga_laml_clinical.csv            # 173 × 4 (patient_id + OS + SUBTYPE)
└── tcga_laml_manifest.json           # PCA variance, mutation prevalence, config

data/raw/tcga_laml/                   # cBioPortal API response cache (1.9 GB)
├── sample_ids.json
├── clinical_data.json
├── mutations.json
├── all_genes.json
├── expr_batch_000.json  …  expr_batch_007.json    # 8 expression batches
└── expression.parquet                              # pivoted gene × sample
```

(Raw cache is gitignored; only canonical CSVs get committed.)

## Implementation notes

cBioPortal REST API quirks discovered:
1. `POST /clinical-data/fetch` expects `entityId`, not `sampleId` or
   `patientId` in the identifier body.
2. `POST /genes/fetch` takes a plain array body `["FLT3","NPM1"]`, and the
   `geneIdType=HUGO_GENE_SYMBOL` must be in the query string (not body).
3. Expression fetch is batched at 4000 Entrez IDs × 173 samples per request
   (~100 MB JSON per batch). Total wall time ~17 min (network-bound). Full
   cache is 1.9 GB; subsequent runs reload from cache in <1 s.
4. TCGA sample IDs embed the patient ID: `TCGA-AB-3008-03` → patient
   `TCGA-AB-3008`. No extra API call needed for mapping.

## Tests

Existing 46 tests still pass. No TCGA-specific tests added yet — the
data comes directly from a public API with well-known schema, and the
code-under-test is mostly plumbing (JSON reshape + PCA on the same
pipeline as BeatAML, which already has tests).

## Next (Day 5)

End-to-end integration smoke test: load all canonical tables, verify
schema compatibility between BeatAML (613 patients), TCGA (173 patients),
and DrugComb (186 strict pairs), and produce a one-page data-quality
dashboard confirming Week 1 is complete.
