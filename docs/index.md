---
title: AML Combo Validation
description: Mechanism-aware combination therapy prediction for AML
---

# AML Combo Validation

**A mechanism-aware, patient-specific combination therapy predictor for acute myeloid leukemia.**

Does mechanism-informed combination prediction outperform best-predicted single drug for AML patients? This project answers that question, finds the precision-medicine subpopulation where combinations win, and ships an end-to-end clinical kit pipeline for new patients.

---

## Headline findings

| Metric | Value |
|---|---|
| **Single-drug baseline** (per-patient Spearman) | **ρ = 0.704** — gate ≥ 0.40 passed |
| **FLT3-mutant AML** (n=179) combo vs best single | **Δ = +16.67 AUC units** [95% CI 14.98–18.19] |
| **FLT3-mutant** win rate | **89.9%** of patients |
| **FLT3 wild-type** (n=434) | Δ = −14.14 (no combo advantage) |
| **Permutation p-value** | < 0.001 |
| **TCGA-LAML validation** | Top picks match published clinical combos |
| **Clinical drugs annotated** | 20/20 |
| **Tests passing** | 86/86 |

### Top combination recommendations

For FLT3-mutant AML patients, the model's most-frequent rank-1 picks are:

1. **Gilteritinib + Venetoclax** (115 patients) — matches NCT03625505, LACEWING rationale
2. **Quizartinib + Venetoclax** (100 patients) — matches NCT03441555
3. **Trametinib + Venetoclax** (82 patients) — parallel-pathway MEK + BCL2 block

All align with ongoing or completed AML clinical trials.

---

## Documents

### 📘 Research
- [Pre-registered thesis](00_thesis) — one-line hypothesis + four acceptable outcomes
- [Manuscript draft](manuscript) — full paper with methods, results, discussion
- [Week-4 head-to-head summary](week4_summary) — Δ distribution + subgroup stats
- [Week-5 TCGA independent-cohort validation](week5_summary)

### 🧪 Clinical kit
- [Kit readiness report](kit_readiness) — 104-dim features, ELN computer, top-3 combo API
- [RNA-Seq SOP for upstream pipeline](rna_seq_sop) — STAR + featureCounts requirements and OOD-detection layers
- [Induction-response validation](induction_validation) — honest negative finding vs 7+3 CR

### 🔬 Research pipelines (3+ drug extensions)
- [Path A — Clonal coverage × IDA](path_a_validation) — N-drug scoring via patient-specific clonal decomposition
- [Path B — Set Transformer](pathB_set_transformer) — arbitrary-arity neural architecture
- [Path D — Higher-Order Factorization feasibility](path_d_hofm_feasibility) — tensor-decomposition benchmark
- [Route C — Clinical regimen retrieval](route_c_regimen_retrieval) — trial-matched evidence layer

### 📂 Drug knowledge base
- [Drug mechanism annotation v2](drug_annotation_v2) — 32 drugs × 39 mechanism axes

### 📅 Development log
- [Week 1 — Day 1: scaffold](week1_day1_summary)
- [Week 1 — Day 2: Baseline A single-drug MLP](week1_day2_summary)
- [Week 1 — Day 3: DrugComb ETL](week1_day3_summary)
- [Week 1 — Day 4: TCGA-LAML ETL](week1_day4_summary)
- [Week 1 — Day 5: integration + Week-3 smoke test](week1_day5_summary)

---

## Using the kit

For a new AML patient:

```python
from combo_val.clinical.kit_schema import KitInput, MutationCall
from combo_val.clinical.kit_predict import predict_for_patient, pretty_print_kit_output
import pandas as pd

kit = KitInput(
    patient_id="P-2026-0001",
    mutations=[
        MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.58),
        MutationCall(gene="NPM1"),
    ],
    karyotype_text="46,XX[20]",
    wbc=90, platelet=35, hemoglobin=8.1, ldh=1100,
    age=48, sex="female",
    blast_pct_bm=72,
    is_initial_diagnosis=True,
)

# gene_symbol → raw count, matching BeatAML pipeline (STAR + featureCounts)
rna_counts = pd.read_csv("patient_counts.tsv", sep="\t").set_index("gene")["count"]

out = predict_for_patient(rna_counts, kit)
print(pretty_print_kit_output(out))
```

**Output:** ELN 2017 risk class, top-5 combination recommendations with predicted AUC + mechanism scores, top-5 single-drug picks, driver-specific cautions, RNA-Seq QC confidence notes.

---

## Data sources

| Source | Role | Size |
|---|---|---|
| [BeatAML 2.0](https://www.nature.com/articles/s41586-022-05140-y) (Bottomly et al., Cancer Cell 2022) | Training + internal validation | 613 patients × 165 drugs × ~55K AUC measurements |
| [DrugComb v1.5](https://drugcomb.fimm.fi) (Zagidullin et al., NAR 2019) | Combination synergy training | 186 strict AML pairs (ALMANAC on HL-60) |
| [TCGA-LAML PanCancer Atlas 2018](https://www.cbioportal.org/study/summary?id=laml_tcga_pan_can_atlas_2018) | Independent cohort | 173 patients |

---

## Citation

*Manuscript in preparation.* For now, cite the repository:

```
Tom, E. (2026). AML Combo Validation: mechanism-aware combination therapy prediction for AML.
https://github.com/Smugpigeon/AML-Combo-Validation
```

---

## Contact

**Erick Tom** — [ericktom94720@gmail.com](mailto:ericktom94720@gmail.com)

[Code on GitHub →](https://github.com/Smugpigeon/AML-Combo-Validation)
