# Route A — Clinical kit readiness (new-patient pipeline)

## What changed

Added a clinical-kit layer that lets a new AML patient's raw data flow through
the full prediction pipeline. Prior to this, the pipeline only worked on the
613 BeatAML patients whose 80-dim feature vectors were already computed and
cached.

### New `src/combo_val/clinical/` package

| File | Purpose |
|---|---|
| `kit_schema.py` | `KitInput` / `KitOutput` / `MutationCall` dataclasses — the contract a new patient's data must satisfy |
| `karyotype_parser.py` | Parse ISCN karyotype strings → structured flags (complex, -5/-7, del(17p), t(8;21), inv(16), t(15;17), t(9;11)) |
| `eln_computer.py` | Compute ELN 2017 risk category from karyotype + mutation panel (no external dependency) |
| `feature_builder.py` | `build_patient_features_from_raw(rna_counts, kit) → 104-dim vector` |
| `kit_predict.py` | End-to-end: features → MLP predictions → combo scoring → KitOutput |
| `demo_kit_run.py` | Runnable demo on two synthetic patients |

### ETL changes (`src/combo_val/data/beataml_etl.py`)

1. **Persist RNA preprocessor** (`data/canonical/beataml_rna_preprocessor.joblib`):
   - 5000 variance-selected gene symbols
   - 50 × 5000 PCA component matrix (float32)
   - 5000-dim PCA mean vector
   - Clinical column medians for imputation
   - Full `feature_columns` list (for schema-version handshake)

2. **Extended clinical features: 5 → 29 columns** (total feature vector 80 → 104):
   - Demographics: sex, is_relapse
   - Labs (log where skewed): WBC, platelet, Hb, LDH, ALT, AST, albumin
   - FLT3 detail: ITD flag, TKD flag, allelic ratio (was: single mut_FLT3)
   - CEBPA: biallelic flag (was: bundled into mut_CEBPA)
   - Fusions: PML-RARA, KMT2A-r, CBFB-MYH11, RUNX1-RUNX1T1, other (was: unused)
   - Karyotype: complex, monosomy 5 or 7, del(17p) (was: unused)
   - Disease state: prior MDS, prior chemo, is initial diagnosis (was: bundled)

   29-column coverage (of 613 feature patients):

   | Group | Coverage after BeatAML-median imputation |
   |---|---|
   | Demographics | 100% (sex via consensus_sex) |
   | CBC core (WBC/plt/Hb) | 70–80% observed, rest imputed |
   | LDH / liver enzymes | 40–60% observed, rest imputed |
   | FLT3-ITD flag | 99.7% observed |
   | Fusions | 17.8% non-empty → one-hot 5 classes |
   | Karyotype flags | 89.4% parseable |

## Retraining outcomes

Baseline A retrained on 104-dim features:

| Version | n_features | per-patient ρ | MAE | Notes |
|---|---:|---:|---:|---|
| v1 | 80 | 0.7036 ± 0.018 | 35.04 | original Week 2 |
| **v2** | **104** | **0.7005 ± 0.018** | **35.10** | current primary model |

Extended features did **not** improve predictive accuracy — the labs have
~40–60% missingness and median imputation adds noise. The gain is in
**kit enablement**, not in prediction quality. Baseline A remains well
above the pre-registered ≥ 0.40 gate.

Week 4 head-to-head re-runs with the v2 model preserve the qualitative
finding:

| | v1 80-dim | v2 104-dim |
|---|---|---|
| FLT3-mut Δ | +16.67 [14.98, 18.19] | **+11.60** [9.27, 13.85] |
| FLT3-mut pct combo wins | 89.9% | 80.4% |
| FLT3-wt Δ | −14.14 [−15.27, −13.05] | −17.55 [−19.34, −15.72] |
| FLT3-wt pct combo wins | 6.2% | 10.8% |

Effect size for FLT3-mut shrinks slightly under v2. Still highly significant
and in the same direction. Top-ranked combos change: v2's #1 and #2 picks
are now **Gilteritinib + Venetoclax** (115 patients) and **Quizartinib +
Venetoclax** (100 patients) — a cleaner clinical match than v1's ranking.

## Demo output (synthetic patient walk-through)

`python -m combo_val.clinical.demo_kit_run` produces two reports:

Patient 1 — 45y female, NPM1-mut + FLT3-ITD (AR 0.62):

```
Predicted ELN 2017: Intermediate   Fitness: fit_for_intensive
Driver flags: FLT3_ITD, NPM1
TOP COMBOS
 1. Gilteritinib      + Venetoclax           predicted AUC = 224.1  (mech +1.03)
 2. Quizartinib       + Venetoclax           predicted AUC = 226.2  (mech +1.03)
 3. Midostaurin       + Venetoclax           predicted AUC = 242.8  (mech +1.03)
CAUTIONS
 ⚠ High LDH (1240 U/L) — elevated TLS risk
 ⚠ FLT3-ITD positive — monitor QTc on quizartinib/gilteritinib
```

Patient 2 — 72y male, TP53 + complex karyotype (-7, del(5q), +8, t(3;3), del(17p)):

```
Predicted ELN 2017: Adverse        Fitness: unfit
Driver flags: TP53
CAUTIONS
 ⚠ Low platelets (25 ×10⁹/L) — bleeding precautions
 ⚠ TP53 mutation — conventional induction poorly effective; consider trial enrollment
```

ELN inference is correct in both cases. The combo picks and cautions are
biologically sensible.

## Remaining limitations (what the kit still CAN'T do)

1. **Absolute AUC values are not calibrated for out-of-distribution inputs**.
   The synthetic lognormal RNA counts used in the demo produce AUC estimates
   near 220–290 — much higher than BeatAML patients (60–150 range). For real
   deployment, RNA-Seq must be processed through the same count / TPM pipeline
   BeatAML used (STAR + featureCounts → raw counts), not a generic pipeline.

2. **Ex-vivo prediction ≠ clinical response**. Route B's negative finding
   still stands: even the actual observed ex-vivo cytarabine AUC doesn't
   predict 7+3 CR in BeatAML (ROC-AUC 0.53, CI crosses 0.5). The kit predicts
   ex-vivo cell kill, not clinical remission.

3. **Clinical kit must be validated prospectively on organoid / primary
   sample measurements**, not retrospectively against outcomes patients
   received under legacy therapy.

4. **Only 8 of the 20 clinically-relevant drugs have mechanism-axis
   annotation**. Midostaurin, Quizartinib, Gilteritinib, Venetoclax,
   Azacitidine, Cytarabine, Ivosidenib, Enasidenib — good. Trametinib,
   Selumetinib, Ruxolitinib, Sorafenib, Dasatinib, etc. are NOT annotated
   and the mech prior contributes zero for them. This is why Trametinib-
   containing pairs sometimes win via baseline AUC alone.

## How a clinician / operator uses the kit

```python
from combo_val.clinical.kit_schema import KitInput, MutationCall
from combo_val.clinical.kit_predict import predict_for_patient, pretty_print_kit_output

# 1. Gather inputs from NGS report, cytogenetics report, CBC, chemistry panel
kit = KitInput(
    patient_id="P-2026-04-23-0001",
    mutations=[
        MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.58, vaf=0.43),
        MutationCall(gene="NPM1", variant_type="missense", vaf=0.41),
    ],
    karyotype_text="46,XX[20]",
    wbc=90.0, platelet=35.0, hemoglobin=8.1, ldh=1100.0,
    alt=30.0, ast=38.0, albumin=3.4, creatinine=0.9,
    blast_pct_bm=72.0, blast_pct_pb=60.0,
    age=48, sex="female",
    is_relapse=False, prior_mds=False,
    is_initial_diagnosis=True,
)

# 2. Pass RNA-Seq counts (gene_symbol → count)
rna_counts = pd.read_csv("patient_counts.tsv", sep="\t").set_index("gene")["count"]

# 3. Predict
out = predict_for_patient(rna_counts, kit)

# 4. Format for clinical report
print(pretty_print_kit_output(out))
```

## Tests

80 pass (was 59 before Route A; 21 new clinical-kit tests cover):
- Karyotype parser (8 tests): normal, t(8;21), inv(16), t(15;17), complex, -7, del(17p), empty input
- ELN 2017 computer (10 tests): Favorable (CBF, APL, NPM1+no-FLT3, low-AR, CEBPA biallelic); Intermediate (high-AR+NPM1); Adverse (complex, TP53, RUNX1, high-AR+no-NPM1)
- ELN string→ordinal mapping (1 test)
- Feature builder end-to-end (1 test): 104-dim output, correct ELN prediction
- Feature builder handles missing labs (1 test): median imputation

## Files committed

```
src/combo_val/clinical/
├── __init__.py
├── kit_schema.py
├── karyotype_parser.py
├── eln_computer.py
├── feature_builder.py
├── kit_predict.py
└── demo_kit_run.py

data/canonical/
└── beataml_rna_preprocessor.joblib     (~1 MB: genes + PCA + medians)

docs/
└── kit_readiness.md                    (this file)

tests/
└── test_clinical_kit.py                (21 new tests)
```

## Next (if user wants more)

- **Add mechanism annotation for the remaining 12 clinical-drug-filter drugs**
  (Trametinib, Selumetinib, Ruxolitinib, Sorafenib, Dasatinib, etc.) so the
  mech prior contributes uniformly across the top-10 rec set.
- **Expand `consensusAMLFusions` encoding** — BeatAML only lists 17.8% of
  patients with fusions; most cohorts have higher rates via better cytogenetic
  workup.
- **Provide a CLI entry point** (`combo-val predict --ngs-report.json --cbc.csv
  --karyotype.txt --rna-counts.tsv`) so wet-lab operators can run the kit
  without writing Python.
- **Calibrate the predicted AUC scale against real BeatAML distribution**
  so a clinician sees "predicted AUC 50 = strongly sensitive" rather than
  needing to compare relative rankings.
