# CLINICAL_REVIEW.md — Traceability of Hardcoded Clinical Rules

This document is the **clinical sign-off ledger** for every hardcoded clinical
rule in the AML combo-prediction kit. Per the post-v0.3 reviewer concern,
no rule may be deployed in clinical workflow without a documented source AND
a sign-off status.

**Sign-off levels** (used in the tables below):

| Level | Meaning |
|-------|---------|
| ⭐⭐⭐ **Guideline-anchored** | Rule directly from NCCN / ELN / WHO / ICC / FDA label. PMID provided. No clinical judgment by engineer. |
| ⭐⭐ **Trial-anchored** | Rule synthesized from a published phase 3 trial (PMID provided). No clinical judgment by engineer beyond reading the trial. |
| ⭐ **Operational synthesis** | Rule composed by the engineer from multiple sources where no single guideline gives an answer. **Requires hematologist review before clinical use.** |
| ⚠ **Engineer's clinical intuition** | Rule has no direct guideline / trial source; it is the developer's best judgment. **MUST NOT be used clinically until reviewed by a board-certified hematologist.** |
| ✅ **Reviewed by [Name, Date]** | A board-certified hematologist has signed off on this rule. |

**Status today (2026-04-26)**: No rules have ⭐⭐⭐ "Reviewed by" sign-off
status. The kit is research-only until institutional review is documented
(see [section 5](#section-5--how-to-record-a-sign-off) below).

---

## 1. Regimen database — `clinical_tier` assignments

[`src/combo_val/clinical/regimen_db.py`](../src/combo_val/clinical/regimen_db.py)

The `clinical_tier` field is the SINGLE most important rule in the kit — it
drives the ±200 ranking bonus that decides Top-1 recommendation. Every
assignment below MUST have a sign-off before clinical deployment.

### `first_line_intensive` (5 regimens — fit, newly diagnosed)

| regimen_id | Source | Tier sign-off |
|-----------|--------|---------------|
| `7_3` (NCCN consensus) | NCCN AML Guideline 2024 v.4.2024 §AML-D 1 of 8 | ⭐⭐⭐ |
| `7_3_midostaurin` (RATIFY) | Stone NEJM 2017, **PMID 28644114**, n=717, FDA label | ⭐⭐⭐ |
| `7_3_quizartinib` (QUANTUM-First) | Erba Lancet 2023, **PMID 37116523**, n=539, FDA 2023 | ⭐⭐⭐ |
| `cpx351_vyxeos` (CPX-351) | Lancet JCO 2018, **PMID 29381435**, n=309, FDA 2017 (sAML/MDS-AML) | ⭐⭐⭐ |
| `atra_ato_low` (APL0406) | Lo-Coco NEJM 2013, **PMID 23841729**, n=263, NCCN APL standard | ⭐⭐⭐ |

### `first_line_unfit` (5 regimens — unfit, newly diagnosed)

| regimen_id | Source | Tier sign-off |
|-----------|--------|---------------|
| `ven_aza` (VIALE-A) | DiNardo NEJM 2020, **PMID 32813947**, n=433, FDA label | ⭐⭐⭐ |
| `ven_dec` | DiNardo Blood 2019, **PMID 30573634**, n=145, NCCN-listed alternative | ⭐⭐ |
| `ldac_glasdegib` (BRIGHT AML 1003) | Cortes Leukemia 2019, **PMID 30487595**, n=88 (NB: small) | ⭐⭐ |
| `ldac_ven` (VIALE-C) | Wei Blood 2020, **PMID 32173999**, n=211 | ⭐⭐⭐ |
| `aza_ivo` (AGILE) | Montesinos NEJM 2022, **PMID 35443108**, n=146, IDH1mut, FDA 2022 | ⭐⭐⭐ |

### `experimental_triplet` (5 regimens — phase 2, signal-promising)

| regimen_id | Source | Tier sign-off |
|-----------|--------|---------------|
| `aza_ven_ivo_triplet` | DiNardo MDA Lancet Oncol 2022, **PMID 35714307**, n=33 IDH1mut | ⭐ |
| `aza_ven_ena_triplet` | Venugopal MDA Blood Adv 2023, **PMID 37285559**, n=42 IDH2mut | ⭐ |
| `aza_gilt` (LACEWING) | Wang JCO 2022, **PMID 36179246**, n=170 (negative for OS) | ⭐⭐ |
| `aza_ven_gilt_triplet` | Short/Daver JCO 2024, **PMID 38277619**, n=52 FLT3mut | ⭐ |
| `quiz_ven_dec_triplet` | **ASH 2024 abstract only — no PMID, n=21** | ⚠ |

> **Engineer note on `quiz_ven_dec_triplet`**: Tier set to
> `experimental_triplet` deliberately so it cannot beat first-line tiers in
> ranking. Pre-v0.3 it was effectively top-1 for fit FLT3-ITD newly-Dx and
> caused issue #1. **No tier upgrade** without a peer-reviewed PMID and
> n ≥ 50.

### `salvage` (5 regimens — relapsed/refractory)

| regimen_id | Source | Tier sign-off |
|-----------|--------|---------------|
| `ivo_mono` (Ivosidenib R/R) | DiNardo NEJM 2018, **PMID 29860938**, n=125, FDA 2018 | ⭐⭐⭐ |
| `ena_mono` (Enasidenib R/R) | Stein Blood 2017, **PMID 28588020**, n=239, FDA 2017 | ⭐⭐⭐ |
| `gilt_mono` (ADMIRAL) | Perl NEJM 2019, **PMID 31513437**, n=371, FDA 2018 | ⭐⭐⭐ |
| `gilt_ven_rr` | Daver MDA Cancer Discov 2022, **PMID 36070394**, n=61 R/R FLT3 | ⭐⭐ |
| `flag_ida` | NCCN / consensus salvage; multiple supporting PMIDs | ⭐ |

### `supportive` (1 regimen)

| regimen_id | Source | Tier sign-off |
|-----------|--------|---------------|
| `supportive_care_or_trial` | NCCN / ELN consensus fallback | ⭐⭐⭐ |

---

## 2. Ranking algorithm — `_score()` in `regimen_matcher.py`

[`src/combo_val/clinical/regimen_matcher.py`](../src/combo_val/clinical/regimen_matcher.py)

### Hardcoded weights

| Constant | Value | Source / rationale | Sign-off |
|----------|-------|--------------------|----------|
| Tier-context match bonus | **±200 pts** | Engineer choice. Must dominate any combination of CR-rate + biomarker bonuses so a Phase-2 abstract triplet can never beat a phase-3 FDA-approved first-line. | ⚠ |
| Targeted-FDA bonus | **+50 pts** | Engineer choice. Lets biomarker-specific FDA approvals (e.g., AGILE for IDH1) beat generic FDA SOC (Ven+Aza). | ⚠ |
| CR-rate weighting cap by evidence | FDA: 100 pts / Phase 3: 80 / Phase 2: 50 / Abstract: 25 | Engineer choice. Prevents a 95% CR rate from a 21-patient ASH abstract from outscoring a 59% CR from a 717-patient phase 3. | ⚠ |
| Biomarker required-all match | +30 pts each | Engineer choice. | ⚠ |
| Biomarker excluded-any violation | hard reject (eligible=False) | NCCN-style hard exclusion (e.g., APL → no 7+3) | ⭐⭐ |
| Age out of range | hard reject | Trial inclusion criteria as published | ⭐⭐⭐ |
| Fitness mismatch | hard reject | Standard clinical practice (fit/unfit definition) | ⭐⭐ |
| Stage mismatch (newly-Dx vs R/R) | hard reject | Standard practice | ⭐⭐⭐ |

> **Critical operational synthesis (⚠)**: The 200/50/100/80/50/25 numbers are
> NOT from any guideline. They were chosen by the engineer to satisfy
> qualitative "must-not-happen" cases (issue #1: Quiz+Ven+Dec must not beat
> RATIFY for fit FLT3-ITD newly-Dx). They produce the right ordering on the
> 5 test patients but have NEVER been validated against an independent set
> of patient cases scored by ≥3 hematologists.

---

## 3. ELN 2017 / ELN 2022 / WHO 2022 / ICC 2022 classification rules

| Module | Source | Sign-off |
|--------|--------|----------|
| `eln_computer.py` (ELN 2017) | Döhner Blood 2017, **PMID 27895058** | ⭐⭐⭐ |
| `eln_2022.py` (ELN 2022) | Döhner Blood 2022, **PMID 35797463** | ⭐⭐⭐ |
| `who_icc_2022.py` (WHO 2022 + ICC 2022) | WHO 2022 (PMID 35732831) + ICC 2022 (PMID 35797568) | ⭐⭐⭐ |

### Engineer-introduced thresholds in ELN 2022 implementation

| Rule | Source / rationale | Sign-off |
|------|--------------------|----------|
| TP53 multi-hit: ≥2 distinct TP53 calls | Bernard Nat Med 2020 (PMID 32747829) — **not in ELN 2022 verbatim** | ⭐ |
| TP53 multi-hit: single TP53 + VAF ≥ 0.5 | Bernard Nat Med 2020 LOH-by-VAF heuristic | ⭐ |
| TP53 multi-hit: single TP53 + del(17p) | ELN 2022 implies LOH counts | ⭐⭐ |
| MDS-related genes: BCOR/EZH2/SF3B1/SRSF2/STAG2/U2AF1/ZRSR2 → Adverse | ELN 2022 verbatim list | ⭐⭐⭐ |
| CEBPA bZIP single allele → Favorable | ELN 2022 verbatim | ⭐⭐⭐ |
| CEBPA `is_biallelic=True` → treat as bZIP-equivalent (back-compat) | Engineer choice (most biallelic CEBPA cases involve bZIP per Tarlock Blood 2022) | ⭐ |

> **Reviewer concern Q7 (post-v0.3)**: TP53 multi-hit definition simplified
> to VAF ≥ 0.5 may produce false-positives in high-tumor-purity samples.
> Stricter definition (require LOH/CNV evidence) NOT YET IMPLEMENTED.

---

## 4. Hotspot codon interpretations

[`src/combo_val/clinical/hotspot_codons.py`](../src/combo_val/clinical/hotspot_codons.py)

| Hotspot | Source | Clinical interpretation | Sign-off |
|---------|--------|-------------------------|----------|
| DNMT3A R882 | Patel NEJM 2012 (PMID 22417203) | "Independent adverse, MRD-persistence common" | ⭐ |
| IDH1 R132 | NCCN AML 2024 / FDA Ivosidenib label | Ivosidenib FDA target | ⭐⭐⭐ |
| IDH2 R140 vs R172 | NCCN / FDA Enasidenib label | Both Enasidenib targets; R140 more common | ⭐⭐⭐ |
| FLT3-TKD D835 / I836 | NCCN / FDA Gilteritinib label | Type-1 FLT3i (Gilt) preferred over Type-2 (Quiz) | ⭐⭐ |
| FLT3 F691L | Schmalbrock Blood 2021 (PMID 33945614) | Gatekeeper resistance to Gilt + Quiz; consider Crenolanib / SCT | ⭐⭐ |
| NPM1 exon 12 frameshift (W288Cfs*12) | Falini NEJM 2005 (PMID 16091632) | TCTG insertion most common | ⭐⭐⭐ |
| KIT D816V/N822 | NCCN — CBF-AML adverse modifier | Independent adverse in CBF-AML | ⭐⭐ |

> **Reviewer concern Q6 (post-v0.3)**: DNMT3A R882 "independent adverse"
> may not hold post-allo-SCT (Bezerra Blood 2020 PMID 32687572). Current
> interpretation does NOT stratify by treatment arm. Needs hematologist
> review on whether to soften the language for fit + SCT-eligible patients.

---

## 5. Allo-SCT recommendation tiers

[`src/combo_val/clinical/patient_report.py`](../src/combo_val/clinical/patient_report.py) `_allo_sct_section()`

| Trigger | Recommendation strength | Source | Sign-off |
|---------|------------------------|--------|----------|
| ELN 2022 Adverse + fit | 🔴 Strongly recommended CR1 | ELN 2022 + DGHO/EBMT consensus | ⭐⭐⭐ |
| FLT3-ITD any AR + NPM1 + fit | 🔴 Strongly recommended CR1 | SORMAIN trial (PMID 32663460), GIMEMA AML0310 | ⭐⭐ |
| ELN 2022 Intermediate + high-risk co-mut + fit | 🟡 Recommended | NCCN / ELN consensus | ⭐⭐ |
| ELN 2022 Intermediate (no high-risk) + fit | 🟡 Discuss MDT | NCCN / ELN consensus | ⭐⭐ |
| ELN 2022 Favorable | 🟢 Not routine in CR1 | ELN 2022 / NCCN | ⭐⭐⭐ |
| APL CR1 | 🟢 Not standard | NCCN APL guideline | ⭐⭐⭐ |
| 30-day SCT center referral | 🟡 from EBMT timing recs | EBMT 2023 transplant indications | ⭐ (China-specific timing not addressed) |

> **Reviewer concern Q10 (post-v0.3)**: 30-day timeline is from western EBMT
> standards. China MUD search + bed availability typically 6-12 weeks. Not
> currently differentiated.

---

## 6. Cautions / drug-interaction warnings

[`src/combo_val/clinical/patient_report.py`](../src/combo_val/clinical/patient_report.py) `_drug_interaction_warnings()`

| Caution | Source | Sign-off |
|---------|--------|----------|
| Venetoclax + posaconazole/itraconazole/voriconazole → 75% reduction | Ven FDA label | ⭐⭐⭐ |
| Venetoclax + fluconazole → 50% reduction | Ven FDA label | ⭐⭐⭐ |
| Quizartinib + CYP3A4i → 30 mg/day; FDA black box for QTc | Quiz FDA label (PI 2023) | ⭐⭐⭐ |
| Midostaurin self-induction (steady-state AUC ↓ 75% over 28d) | Mido FDA label / RATIFY supplementary | ⭐⭐⭐ |
| Ivosidenib differentiation syndrome management (dexamethasone 10 mg IV q12h) | Ivo FDA label | ⭐⭐⭐ |
| Daunorubicin lifetime cumulative ≤ 550 mg/m² | NCCN / FDA daunorubicin label | ⭐⭐⭐ |
| Hyperleukocytic AML (WBC > 50): hydroxyurea 50 mg/kg PO BID | NCCN AML 2024 / Röllig Blood 2015 (PMID 25677356) | ⭐⭐⭐ |
| Hyperleukocytic AML (WBC > 100): leukapheresis emergency | NCCN AML 2024 — note: leukapheresis efficacy controversial | ⭐⭐ |

---

## 7. MRD modality auto-selection (`_mrd_monitoring_section`)

| Patient feature | Selected MRD modality | Source | Sign-off |
|-----------------|----------------------|--------|----------|
| PML-RARA | RT-qPCR (sensitivity 1e-4) | Sanz Blood 2019 APL guideline (PMID 30792272) | ⭐⭐⭐ |
| CBF-AML (RUNX1-RUNX1T1 / CBFB-MYH11) | Fusion RT-qPCR (sensitivity 1e-4 to 1e-5) | ELN 2021 MRD (PMID 33591443) + Yin Blood 2012 | ⭐⭐⭐ |
| NPM1mut | NPM1 RT-qPCR, NPM1mut/ABL ratio | ELN 2021 MRD; AMLSG / NCRI consensus | ⭐⭐⭐ |
| FLT3-ITD (always added) | NGS-MRD (ClonoSEQ etc.) | Levis Blood 2018 (PMID 30209064) | ⭐⭐⭐ |
| Default fallback | Multiparameter flow cytometry (DfN, ≥0.1% = MRD+) | ELN 2021 MRD | ⭐⭐⭐ |

> **Reviewer concern Q9 (post-v0.3)**: Many Chinese hospitals lack NPM1
> RT-qPCR routine capability and use MFC instead. Modality selection is
> currently NOT user-configurable. Needs settings hook.

---

## 8. How to record a sign-off

When a board-certified hematologist completes review of a rule (or a group
of rules), update this file as follows:

1. Replace the ⚠ / ⭐ / ⭐⭐ / ⭐⭐⭐ symbol with **✅ Reviewed by [Name],
   MD [credential], [Institution], YYYY-MM-DD**
2. Add a footnote explaining any modifications requested:
   ```
   [^1]: Dr. Wang (PUMC Hematology) requested that the +200 tier bonus be
         reduced to +150 for elderly fit patients to avoid pushing 70+ y.o.
         FLT3-ITD into 7+3+Mido (which RATIFY did not enroll). Implemented
         in commit abc1234.
   ```
3. Open a PR linking the sign-off and any code changes.
4. Record reviewer credentials in `docs/CLINICAL_REVIEWERS.md` (a separate
   roster file).

The kit's web UI shows reviewer status at `/about` (TODO: implement) so end
users can see at a glance which rules are guideline-anchored vs operational
synthesis.

---

## 9. Engineer disclosure

Every rule in this kit was first drafted by **Eric Tom (engineer, no
medical credentials)** by reading the cited guideline / trial / FDA label
and translating it into code. The current state of this document is the
honest accounting of which rules are direct quotations of the source
(⭐⭐⭐), which are synthesized across sources (⭐), and which are the
engineer's clinical judgment with no source backing (⚠).

**Until ⚠ rules have ✅ sign-off, this kit must not be used for clinical
decisions.** Research / methodology demonstration use only.

— Last updated: 2026-04-26
