# PI Outreach Letter — AMLCK-Prospective-1

Cold-outreach template for inviting a hematology PI to participate as a
site in the prospective validation study. Two versions: short (email,
~200 words) and long (formal letter, ~600 words).

---

## A. Short version (email)

> **Subject**: 邀请合作 — AML AI 用药决策工具的前瞻性验证研究

[Dear Dr. X / 王主任 您好,]

我是 [your name],  [背景一句话, e.g. "in collaboration with [合作机构]
maintain an open-source AML clinical decision support kit"]. We have
deployed an integrated tool ([amlcombo.org](https://amlcombo.org)) that
combines ELN 2017/2022 + WHO/ICC 2022 classification + evidence-based
regimen retrieval + an experimental Layer-3 ML combo predictor.

Our internal validation found the Layer-3 ML predictor has only Pearson
r ≈ 0.05 vs clinical CR — we don't know if it's a real limitation of
ex-vivo data or a power issue with the BeatAML 2.0 cohort. We're
launching a prospective validation study to find out.

**The ask**: would your center consider being one of 2-3 sites enrolling
N≈80-100 newly-Dx AML patients (over 12-18 months) under an
**observational, no-treatment-influence** protocol? We provide all
infrastructure (kit + locked-prediction storage + outcome forms); your
team enrolls + fills 30-day / 6-month / 12-month outcome forms. No data
leaves your hospital.

Protocol: https://github.com/Smugpigeon/AML-Combo-Validation/blob/main/docs/PROSPECTIVE_VALIDATION_PROTOCOL.md

If interested, I can send the IRB packet + come visit / video call.

Best regards,
[your name]
[email]

---

## B. Long version (formal letter)

> [Hospital letterhead]
>
> [Date]
>
> Dr. [Family Name] [Given Name], MD/PhD
> Director, Department of Hematology
> [Hospital Name]
>
> Dear Dr. [Family Name],

### Background

I am Eric Tom, the principal developer of `AML-Combo-Validation`
([amlcombo.org](https://amlcombo.org)), an open-source AML clinical
decision support tool. Over the past year we have built a multi-layer
recommendation system covering:

- **Layer 1**: ELN 2017 + ELN 2022 + WHO 2022 + ICC 2022 classification
  with explicit transition notes between systems
- **Layer 2**: Path A clonal-coverage IDA-framework triplet ranking
- **Layer 3**: BeatAML 2.0-trained MLP for pairwise drug combination
  AUC prediction, with FDA/Phase-3 evidence-tiered regimen ranking
- Pre-induction workup + Allo-SCT decision tree + MRD monitoring plan
  + CYP3A4 / drug-interaction warnings, all per ELN 2021 / NCCN 2024

The kit has been reviewed by hematology research collaborators in 4
rounds, addressing 13 GitHub issues spanning ELN 2022 transition,
hyperleukocytic warnings, CEBPA bZIP detection, and Layer-3 OOD
suppression. Full traceability of every clinical rule (with PMID +
sign-off status) is maintained in
[`docs/CLINICAL_REVIEW.md`](https://github.com/Smugpigeon/AML-Combo-Validation/blob/main/docs/CLINICAL_REVIEW.md).

### The scientific question

Our internal validation on the BeatAML 2.0 hold-out (Route B) found
that the Layer-3 MLP's predicted combo AUC has **Pearson r ≈ 0.05**
with clinical induction CR rate. This is the single largest scientific
question hanging over the kit. Two interpretations:

(a) BeatAML 2.0 ex-vivo AUC simply does not transfer to in-vivo CR rate,
    and Layer-3 should be removed from the kit; or

(b) The internal hold-out underestimates the real-world signal because
    of cohort homogeneity (BeatAML is ~80% North American Caucasian),
    and validation in a Chinese AML cohort would reveal a stronger
    correlation.

### The ask

We are launching a multi-center, prospective, **observational** study
to distinguish (a) from (b):

- N = 200 newly-Dx AML patients with 12-month follow-up complete (target
  recruitment N = 240 for ≥85% follow-up rate)
- 2-3 participating sites; we are seeking commitment from your center
  for ≈ 80-100 patients over 12-18 months of recruitment
- **Strictly observational**: kit predictions are immutably logged
  before treatment, but kept blind from the MDT — your team's treatment
  decisions are 100% unaffected by the kit
- Per-patient time burden on your team: ~10-15 minutes at enrollment
  (run kit, file consent), ~5 minutes at 30 days / 6 months / 12 months
  (fill outcome form, mostly auto-filled from EHR)
- Primary endpoint: Pearson `r` between kit's Layer-3 Top-1 predicted
  combo AUC and patient's 12-month overall survival

Full statistical analysis plan + sample size justification is in
[`docs/PROSPECTIVE_VALIDATION_PROTOCOL.md`](https://github.com/Smugpigeon/AML-Combo-Validation/blob/main/docs/PROSPECTIVE_VALIDATION_PROTOCOL.md).
The IRB submission packet (in both English and 中文) is in
[`docs/templates/IRB_SUBMISSION_PACKET.md`](https://github.com/Smugpigeon/AML-Combo-Validation/blob/main/docs/templates/IRB_SUBMISSION_PACKET.md).

### What we provide

- Kit + per-site infrastructure setup (we offer Docker / Python deploy
  options — full self-host so no patient data leaves your hospital,
  or hosted on amlcombo.org if your IRB allows)
- IRB submission packet (already drafted in 中文 for mainland China sites)
- Statistical analysis plan (locked before patient 1)
- Site coordinator training (1-day video session)
- Pre-printed consent forms in CN / EN
- Per-site monthly progress dashboard
- 1-day in-person or video site initiation visit

### Authorship and publication

- Site PI = co-author by ICMJE criteria (recruitment volume + outcome
  data contribution)
- Pre-committed to publishing **regardless of direction** — positive,
  negative, or indeterminate result, all published. Pre-registered at
  ClinicalTrials.gov / ChiCTR before recruitment opens.
- Manuscript target: Blood / JCO / Lancet Hematology

### What we need to know from you

If you are open to participating, please let us know:

1. Approximate annual newly-Dx AML volume at your center
2. Available site coordinator capacity (~5 hr/month over the
   recruitment period)
3. Whether your IRB requires the protocol in 中文 / English / both
4. Preferred deployment model: hosted on amlcombo.org, or self-host
   in your hospital network
5. Whether your team has any specific subgroup interests (e.g., Chinese
   APL biology, CBF-AML t(8;21) prevalence) that should be added to
   the SAP as pre-specified subgroup analyses

I am available for a video call or in-person visit at your convenience.
You may reply to this email or contact me at [phone].

Best regards,

[Eric Tom]
[Position / affiliation]
ericktom94720@gmail.com
[phone]

---

## C. Site onboarding checklist (after PI commits)

Once a PI commits, we walk through this checklist together:

### Week 1 — IRB preparation
- [ ] Confirm protocol language (CN / EN) and any local IRB-required
      modifications
- [ ] PI signs the protocol-version-lock document (kit version frozen
      as `v0.5.x`, commit hash recorded)
- [ ] Sponsor (amlcombo.org) prepares the site-specific IRB packet with
      blanks filled in (site name, PI name, IRB contact, consent forms)
- [ ] PI submits to local IRB

### Week 2-4 — Infrastructure setup
- [ ] Decide deployment: amlcombo.org SaaS vs self-host Docker
- [ ] If self-host: provision a Docker host (4 vCPU / 8GB RAM minimum),
      receive deployment instructions
- [ ] Train site coordinator on `prospective.locked_prediction.lock_prediction()`
      + `outcome_capture.write_outcome()` API (web UI or Python API)
- [ ] Add site to `data/prospective/site_registry.json` via PR
- [ ] Test enrollment with a synthetic patient (verify hash chain)

### Week 4-8 — IRB approval expected
- [ ] Receive IRB approval letter
- [ ] Update site_registry.json with `irb_approval_date`
- [ ] Coordinator dry-run with 2 synthetic patients

### Week 8+ — Recruitment opens
- [ ] First real patient enrolled (signed consent + locked prediction)
- [ ] Monthly status emails to sponsor (recruitment count + any AEs)
- [ ] Quarterly sponsor video check-in

### Month-12 outcome milestones
- [ ] Coordinator sets calendar reminders at +30d / +6mo / +12mo for
      each enrolled patient
- [ ] Outcome forms filed within 14 days of each milestone
- [ ] Sponsor monthly reconciliation: any patient missing outcome at
      milestone gets a query email
