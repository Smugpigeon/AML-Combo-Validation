# Week 1 Day 3 — DrugComb AML Subset ETL

## Summary

Full ETL on the 1.4M-row DrugComb v1.5 summary table (`summary_v_1_5.csv`, 1.3 GB)
completed in **~6 seconds** via 100K-row streaming chunks. Output is three
canonical tables that Week 3 will consume.

## Key numbers

| Stage | Count |
|---|---|
| Total DrugComb rows scanned | 1,432,351 |
| Rows on AML cell lines (13 canonical lines) | 13,877 |
| **True combination rows** (drug1 ≠ drug2, both present) | **4,938** |
| Monotherapy rows (drug2 = nan) | 8,939 |
| Unique drug names in AML subset | 769 |
| Drug names aligned to BeatAML vocab | 88 (80 exact + 8 fuzzy) |
| **Strict combo pairs** (both drugs in BeatAML) | **186** |
| Loose combo pairs (≥1 drug in BeatAML) | 1,708 |
| Monotherapy rows with BeatAML drug | 1,603 |

## Critical finding: combination data is ALMANAC-HL-60 only

The 186 strict pairs all live on a single cell line — **HL-60** — and all come
from the **NCI ALMANAC** screen. Other AML lines (MV4-11, MOLM-13, THP-1, etc.)
appear in DrugComb but only as monotherapy screens from CTRPv2/GDSC1/FIMM/CCLE/gCSI.

### Implications for Week 3

This constrains the combo predictor design. We cannot learn how
*patient biology* modulates combination response from one cell line. The
architecture must factor:

1. **Single-drug response** `→` from BeatAML (613 patients × 165 drugs, already trained in Baseline A)
2. **Combo adjustment** `→` learned from ALMANAC-HL-60 (186 pairs)
3. **Patient extrapolation** `→` mechanism prior + drug-drug interaction embedding

Revised factorization:

```
combo_pred(patient, d1, d2) = single_pred(patient, d1)
                            + single_pred(patient, d2)
                            + combo_adjustment(d1, d2)          ← ALMANAC-HL-60 residual
                            + mechanism_prior(patient, d1, d2)  ← knowledge-graph term
```

The HL-60 residual teaches pair chemistry; the mechanism prior + per-patient
single-drug predictions carry the patient-personalization.

## Outputs

```
data/canonical/
├── drugcomb_aml_pairs.csv              # 186 strict pairs (both drugs in BeatAML)
├── drugcomb_aml_pairs_any_match.csv    # 1,708 loose pairs (≥1 drug in BeatAML)
├── drugcomb_aml_monotherapy.csv        # 1,603 single-drug AML cell-line screens
├── drugcomb_drug_alignment.csv         # 769 names × source: no_match / exact / fuzzy
└── drugcomb_filter_manifest.json       # reproducibility manifest
```

## Cell line coverage

| Cell line | Total rows | Combo rows | Mono rows |
|---|---:|---:|---:|
| HL-60 | 5,661 | 1,709 (ALMANAC) | 3,952 |
| HEL | 1,167 | 0 | 1,167 |
| OCI-AML5 | 828 | 0 | 828 |
| THP-1 | 748 | 0 | 748 |
| MV4-11 | 732 | 0 | 732 |
| OCI-AML2 | 705 | 0 | 705 |
| OCI-AML3 | 699 | 0 | 699 |
| KASUMI-1 | 674 | 0 | 674 |
| MOLM-13 | 673 | 0 | 673 |
| NB4 | 623 | 0 | 623 |
| SET-2 | 552 | 0 | 552 |
| U-937 | 495 | 0 | 495 |
| KG-1 | 320 | 0 | 320 |

## Tests

46/46 pass:
- `tests/test_drug_name_normalizer.py` (26 tests)
- `tests/test_drugcomb_etl.py` (20 tests including end-to-end synthetic ETL)

## Issues resolved

- **Salt-form regex bug**: original `r"\bmono?hydrochloride\b"` made the `o`
  optional instead of treating `mono` as an optional prefix. Fixed to
  `r"\b(?:mono|di|tri)?hydrochloride\b"`.
- **Extended salt list**: added `citrate, tartrate, phosphate, tosylate,
  ditosylate, fumarate, succinate, acetate, potassium, (TN)`. Recovered ~54
  additional pair rows (Erlotinib, Pazopanib, Lapatinib salts).
- **Monotherapy contamination**: DrugComb v1.5 unified table mixes monotherapy
  (CTRPv2/GDSC/FIMM/CCLE/gCSI) with combination (ALMANAC). Added explicit
  `is_combo_row` filter so pair counts don't double-count single-drug screens.

## Next (Day 4)

TCGA-LAML alignment — reuse existing `align_tcga_public_to_templates.py` from
AML-CRAFT. Output: `data/canonical/tcga_laml_validation.csv` with ~200 patients'
expression + mutation aligned to BeatAML template space for independent-cohort
validation in Week 5.
