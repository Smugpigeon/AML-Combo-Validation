# Prospective Validation Protocol — AML Combo-Prediction Kit Layer-3

**Study short title**: AMLCK-Prospective-1
**Kit version under study**: v0.5.x (frozen for the duration of recruitment)
**Sponsor**: amlcombo.org research project (PI: Eric Tom, ericktom94720@gmail.com)
**Status**: Drafting — pending PI commitment + IRB submission

---

## 1. Background and rationale

The AML Combo-Prediction Kit (`AML-Combo-Validation`, v0.5) generates
three layers of clinical decision support for newly-diagnosed AML
patients:

- **Layer 1**: Evidence-based regimen retrieval (NCCN / ELN-anchored,
  21 regimens with PMID + tier; this layer is well-grounded)
- **Layer 2**: Path A clonal-coverage IDA (Independent Drug Action)
  triplet ranking
- **Layer 3**: BeatAML 2.0-trained MLP that predicts pairwise drug
  combination AUC and ranks Top-3 combos for the patient

In internal hold-out validation on the BeatAML 2.0 cohort (Route B), the
Layer-3 predicted AUC vs the patients' actual induction CR rate had
**Pearson r ≈ 0.05** (i.e., statistically uncorrelated). This is the
single largest scientific question hanging over the kit.

Two interpretations are possible:

(a) BeatAML 2.0 ex-vivo AUC simply does not transfer to in-vivo CR rate
    in a clinically useful way → Layer-3 is fundamentally limited and
    should be removed from the kit.

(b) The BeatAML 2.0 internal hold-out is too small / not representative
    enough to detect a true effect that exists in real Chinese / Asian
    AML populations → Layer-3 is salvageable and worth deploying after
    additional validation.

**This study is designed to distinguish (a) from (b).**

---

## 2. Objectives

### Primary objective

Assess the strength of association between the Layer-3 MLP's Top-1 combo
prediction (made BEFORE treatment, immutably locked) and the patient's
12-month overall survival (OS) in a prospectively enrolled real-world
Chinese AML cohort (N = 200 with completed 12-month follow-up).

**Primary endpoint**: Pearson correlation coefficient `r` between the
Layer-3 Top-1 predicted combo AUC and the patient's 12-month OS
(continuous: 0-365 days alive; censored at 365).

**Decision rule**:
- `r ≥ 0.30` (95% CI lower bound > 0.15) → **Layer-3 GO**: kit version
  v0.6 promotes Layer-3 to a non-research output with calibrated
  confidence intervals.
- `0.15 ≤ r < 0.30` → **Layer-3 INDETERMINATE**: extend recruitment to
  N = 400 or pivot to a more permissive secondary endpoint.
- `r < 0.15` → **Layer-3 NO-GO**: kit version v0.6 deletes Layer-3 from
  the codebase entirely; project pivots to clinical decision support
  (Path A in the v0.4 reviewer dialog).

### Secondary objectives

1. **ELN 2022 classification accuracy**: Cohen's kappa between kit's
   ELN 2022 classification and a panel of ≥2 hematologists' independent
   ELN classification, on the same 200 patients.
2. **Top-1 regimen agreement**: Proportion of cases where kit's Top-1
   regimen matches what the MDT actually prescribed.
3. **Calibration**: Brier score and ECE of kit's predicted CR
   probability (derived from ELN class) vs observed induction CR rate.
4. **Subgroup analysis**: Primary endpoint within `(age ≤ 60, fit) ×
   (NPM1+, NPM1-)` strata.

### Exploratory objectives

- Survival comparison: Kaplan-Meier OS at 12 / 18 / 24 months stratified
  by whether the actually-given regimen matched the kit's Top-1
  recommendation (log-rank test, with full disclosure that this is
  observational, not randomized).

---

## 3. Study design

**Type**: Multi-center, prospective, observational cohort study with
locked-prediction-then-outcome-capture design.

**Important**: This is NOT a randomized trial. Treatment decisions remain
entirely with the local MDT. The kit's predictions are recorded for
research purposes only; clinicians are NOT instructed to follow them.

**Population**:
- Newly-diagnosed AML (any subtype, including APL)
- Adults ≥ 18 years
- Treated at a participating site
- Written informed consent for prospective-validation data capture
- Sufficient diagnostic data to run the kit (mutation panel, karyotype,
  basic labs)

**Exclusion**:
- Secondary AML following another active malignancy with ongoing
  cytotoxic therapy
- Patient unable to consent or follow up

---

## 4. Sample size justification

### Primary endpoint sample size

We want to detect a Pearson `r ≥ 0.20` with 80% power at α = 0.05
(two-sided), using Fisher's z-transformation:

```
n = ( (Z_{1-α/2} + Z_{1-β}) / arctanh(r) )² + 3
n = ( (1.96 + 0.84) / arctanh(0.20) )² + 3
n = ( 2.80 / 0.2027 )² + 3
n = (13.81)² + 3
n ≈ 194
```

We **target N = 200** patients with 12-month follow-up complete to
absorb the sample-size formula's lower bound, censoring (deaths within
the 12-month window are NOT censored — they ARE the outcome we're
correlating to predicted AUC), and plausible up-to-15% loss to follow-up.

**Inflation for loss-to-follow-up**: Recruitment target = 240 to
guarantee ≥ 200 with month-12 outcome.

### Multi-site contribution

Assuming 2-3 participating sites at recruitment rates of 4-8 newly-Dx
AML / month / site:
- 3 sites × 6 patients/month × 12 months = ~216 enrolled in year 1
- 12-month follow-up adds 12 months → primary analysis at month 24
  from study start

### Interim analysis (Bayesian futility)

At **N = 50 patients with 12-month outcome**, perform a Bayesian futility
check:

- Posterior P(`r` ≥ 0.30 | data) using a Beta(1,1) flat prior on
  Fisher's-z-transformed correlation
- **Stop for futility** if posterior P(`r` ≥ 0.30) < 0.10
- **Continue** otherwise

---

## 5. Statistical analysis plan (SAP)

### 5.1 Primary analysis

For each enrolled patient `i` with 12-month OS data:
1. Pull `predicted_layer3_top1_combo_auc[i]` from the locked prediction
   record (`runs/prospective/<site>/<pt>/*_locked_prediction.json`)
2. Pull `actual_12mo_OS_days[i]` from the month-12 outcome record
3. Compute Pearson `r` over all `i` with both available

Report `r`, two-sided p-value, and Fisher 95% CI.

### 5.2 Secondary analyses

- **ELN kappa**: Use `combo_val.validation.external_validation.compute_eln_concordance`
- **Top-1 match**: Use `compute_regimen_top1_match`
- **Calibration**: Use `compute_brier_score` + `compute_ece` with the
  ELN-prior CR probability (acknowledged limitation: prior is fixed, not
  individualized)

### 5.3 Subgroup analyses (pre-specified)

- (age ≤ 60) × (fit-for-intensive)
- NPM1+ vs NPM1-
- FLT3-ITD+ vs FLT3-ITD-
- ELN 2022 Favorable vs Intermediate vs Adverse

For each subgroup with ≥ 30 patients, recompute primary `r` + 95% CI.

### 5.4 Survival analysis (exploratory)

Kaplan-Meier OS curves stratified by kit-Top-1 == actual regimen:
- Yes (kit recommendation == clinician choice)
- No (kit recommendation ≠ clinician choice)

Log-rank test reported with full caveat: this is observational and
prone to confounding by indication.

### 5.5 Handling of missing data

- Patients missing month-12 outcome at study close: censor at last known
  alive date, sensitivity analysis with worst-case imputation (assumed
  dead at last known follow-up day).
- Patients missing locked prediction (e.g., kit failed at enrollment):
  excluded from analysis, reported in CONSORT-style flow diagram.

### 5.6 Pre-registration

Before recruiting the first patient, this SAP is registered at:
- ClinicalTrials.gov (NCT TBD on PI commitment)
- ChiCTR (Chinese registry) for any China sites
- OSF Registries (open-science timestamp)

---

## 6. Data integrity

### Locked predictions

Per `combo_val.prospective.locked_prediction`:
- Prediction generated → immutably written + SHA-256 hash + linked into
  per-site hash chain BEFORE patient is treated
- Hash chain prevents retroactive editing (any change detectable by
  `verify_chain()`)
- Periodic anchoring of latest hash to a public timestamp service
  (proposed: GPG-signed git tag pushed to public origin every Friday)

### Outcome capture

- Site coordinators fill outcome forms via the platform's
  `outcome_capture` API (or via web UI if the site uses amlcombo.org)
- Outcome data is BLINDED to the kit prediction during entry (the form
  does NOT pre-fill from prediction record)

---

## 7. Risks and benefits

### Risks to participants

**Direct treatment risks**: Zero. The kit prediction is observational;
treatment is entirely per local MDT.

**Privacy risks**: Standard. Data stored under de-identified research IDs.
Kit input + locked prediction stored on amlcombo.org server (Hetzner DE,
EU jurisdiction) or self-hosted institutional server. PHI detector
(`amlcombo_web/app/phi_detector.py`) gates LLM-feature uploads. See
`docs/COMPLIANCE.md` for full data flow.

### Benefits to participants

No direct benefit (kit prediction does not influence their treatment).
Potential societal benefit: contributes to validation of an AI-decision-
support tool.

---

## 8. Roles

- **Sponsor / coordinator**: Eric Tom (ericktom94720@gmail.com)
- **Site PIs**: TBD (target: 2-3 sites in China; potentially 1 EU)
- **Local IRB**: each site files independently
- **Data Safety Monitoring Board (DSMB)**: ≥ 3 independent hematologists
  (not on the engineering team), reviews interim analysis at N = 50
- **Statistician**: TBD; SAP locked before unblinding any outcome data

---

## 9. Timeline (target)

| Quarter | Milestone |
|---------|-----------|
| 2026-Q2 | Protocol finalized, SAP locked, ClinicalTrials.gov / ChiCTR registration |
| 2026-Q3 | First site IRB approval; first patient enrolled |
| 2026-Q4 | Recruitment ramp |
| 2027-Q2 | N = 50 reached → Bayesian futility check (interim DSMB review) |
| 2027-Q4 | Recruitment complete (N = 240 enrolled) |
| 2028-Q4 | All N = 200 with 12-month follow-up complete; primary analysis |
| 2029-Q1 | Manuscript submission; kit v0.6 GO/NO-GO decision on Layer-3 |

---

## 10. Publication plan

**Primary publication**: Prospective validation results, regardless of
direction (positive, negative, or indeterminate). Pre-committed in the
ClinicalTrials.gov entry.

**Authorship**: ICMJE criteria. Site PIs by recruitment volume; engineer
last author + corresponding.

**Data sharing**: Aggregate locked predictions + outcomes (de-identified)
deposited on Zenodo with DOI under CC-BY-4.0 at publication.

---

## 11. Document control

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-04-26 | Initial draft (responding to v0.4 reviewer dialog Path B) |

This protocol must be signed by all site PIs and locked before any
prospective patient is enrolled.
