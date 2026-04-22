# Week 1 Day 1 — Repo scaffolding + BeatAML canonical tables

## Done

- Initialized fresh git repo at `~/Desktop/AML-combo-validation/`
- Directory structure per README
- Migrated from `AML-CRAFT`:
  - 4 BeatAML 2.0 raw files → `data/raw/BeatAML2.0/`
  - `mechanism_vocab.yaml` + `drug_mechanism_v1.csv` → `src/combo_val/knowledge/`
  - `parse_beataml_variants.py` → `src/combo_val/data/`
  - Hallmark GMT → `data/raw/`
- Wrote simplified `beataml_etl.py` (no HyGReM-NC, no multi-omics graph)
- Produced 3 canonical artifacts at `data/canonical/`:
  - `beataml_patient_features.csv` — 613 patients × 80 features
  - `beataml_drug_response_long.csv` — 55,826 (patient × drug) rows, 487 patients × 165 drugs
  - `beataml_feature_manifest.json` — provenance

## Feature quality check (passes data-quality gate — 613 ≥ 450)

**Mutation frequencies match AML epidemiology**:

| Gene | BeatAML this ETL | Literature expected |
|---|:---:|:---:|
| FLT3 | 29.2% | ~30% |
| NPM1 | 25.8% | ~28% |
| DNMT3A | 14.7% | ~20% (slightly low — curated panel) |
| NRAS | 12.4% | ~12% |
| RUNX1 | 10.6% | ~10% |
| IDH2 | 9.6% | ~10% |
| TET2 | 9.1% | ~10% |
| ASXL1 | 7.7% | ~10% |
| TP53 | 7.0% | ~10% |

**RNA PCA**: top 50 components capture 67% of variance in the 5000 most-variable genes. Consistent with the ~10-intrinsic-dim finding from AML-CRAFT.

**Clinical**:
- Median age 61 (expected)
- ELN distribution roughly matches BeatAML cohort composition
- Feature coverage: all 613 patients have complete mutation + clinical; RNA PCA coverage is 613/805 (78% — the 192 missing are patients without RNA-Seq samples)

## Next steps (Week 1 Day 2-5)

### Day 2 — single-drug predictor skeleton (Baseline A)

Write `src/combo_val/baselines/single_drug_mlp.py`:
- Multi-task MLP with shared representation
- Task per drug (165 heads)
- Input: 80-dim patient feature
- Output: predicted AUC per drug

Target: MAE < 30 on held-out set (AUC scale 0-300), Spearman > 0.4.

### Day 3 — DrugComb AML subset (user action required)

The DrugComb v2 dump (`DrugComb_v2.0.csv`) is ~300MB. Download manually:

```bash
cd ~/Desktop/AML-combo-validation/data/raw/drugcomb
curl -L -o drugcomb_summary_v1_6.csv https://drugcomb.fimm.fi/download/summary_v_1_6.csv
# Or (v2 dump, if available):
# curl -L -o drugcomb_v2.csv.gz https://drugcomb.org/api/download/v2/drugcomb_v2.csv.gz
# gunzip drugcomb_v2.csv.gz
```

Then filter to AML cell lines (HL-60, MV4-11, KASUMI-1, MOLM-13, MOLM-14, OCI-AML2, OCI-AML3, NB4, U937, KG-1, THP-1) — scripted as Day 3 work.

### Day 4 — TCGA-LAML public alignment

Reuse existing `align_tcga_public_to_templates.py` from `AML-CRAFT` with minor path surgery. Target: ~160 patients with expression + mutation + OS/clinical for independent validation.

### Day 5 — sanity integration test

End-to-end smoke: load all 3 canonical tables, dry-run the pipeline stubs, confirm wiring. Commit.

## Known limitations at Day 1

1. **Mutation file comes from `variantSummary` parser** (2,210 events, 731 patients) — not the official WES VCFs (dbGaP controlled). Downstream results should be labeled "mutation coverage ≈ official WES" for reviewer transparency.
2. **RNA coverage 613/805 (78%)** — patients without RNA-Seq can't be in the training set. Acceptable for this cohort size but note in Limitations.
3. **No epigenomic or proteomic modalities** — BeatAML doesn't ship them. Not a regression; neither AML-CRAFT nor this project plans to use them.
4. **Clinical features are minimal (5 dim)** — age, ELN, blast, secondary flag, fitness. Could add cytogenetics as one-hot if needed but ELN already encodes most of that information.
