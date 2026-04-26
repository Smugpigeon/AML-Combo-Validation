# V6 IRB Amendment Template

For each participating site PI to file with their local IRB **after**
v0.6 deploys. This is a model-architecture amendment to an already-
approved protocol — it does NOT change risk profile, consent language,
or data-flow boundaries.

**Filing target**: minor amendment / expedited review (not full board)

---

## 1. Cover letter

> [Date]
>
> Re: Amendment to study **AMLCK-Prospective-1**
> IRB protocol number: **[your local IRB #]**
> Original approval date: **[YYYY-MM-DD]**
>
> Dear IRB Chair,
>
> I am submitting an amendment to the above study to update the
> investigational tool from **kit version v0.5.x** to **kit version
> v0.6.x**. The amendment changes the underlying machine-learning model
> architecture but does NOT change:
>
> - the patient population or eligibility criteria
> - the consent process or consent form text
> - the data collection schedule (Day 30 / Month 6 / Month 12 / Month 18)
> - the data storage location (Hetzner Falkenstein DE)
> - the risk profile (still minimal-risk observational)
> - the role of the kit (research output, blinded from MDT decisions)
>
> Detail of the change is in §2 below.
>
> Sincerely,
> [PI Name]

---

## 2. Detailed description of the change

### 2.1 What is changing

| Element | v0.5 (current) | v0.6 (proposed amendment) |
|---------|---------------|---------------------------|
| Drug encoder | `nn.Embedding(165, 64)` learned from scratch | **DrugGIN** — Graph Isomorphism Network over RDKit-derived atom-bond graph (Xu et al. ICLR 2019) |
| Gene context | 25 mutation flags as independent features | **GeneContextGAT** — Graph Attention Network over STRING v12 protein-protein interaction subgraph (Veličković et al. ICLR 2018) |
| Combo head | Factorized: `0.5·(AUC_i + AUC_j) − k·mech_score` | **Set-Transformer** ISAB×2 + PMA(k=1) over GIN drug node embeddings (Lee et al. ICML 2019) |
| Pretraining | None | **DrugComb cross-cancer pretrain** (≈30k drug pairs) before BeatAML fine-tune |
| Internal Pearson r (BeatAML hold-out) | 0.05 (per Route B) | **[INSERT FROM PHASE 5 RESULT]** |

### 2.2 What is NOT changing

- **Study population**: same — newly-diagnosed AML adults at this site
- **Recruitment process**: same — voluntary opt-in via consent form
- **Data collected**: same fields (mutation panel, karyotype, lab values,
  treatment regimen, day-30 / 6-month / 12-month outcome forms)
- **Data storage**: same — Hetzner Falkenstein DE / encrypted at rest
- **Treatment decisions**: still 100% per local MDT, kit predictions
  blinded
- **Outcome capture**: identical schedule + forms
- **Risk to participants**: still minimal-risk observational

### 2.3 Why the change

The v0.5 kit was built around an `nn.Embedding(165, 64)` drug encoder
learned from scratch on BeatAML 2.0 single-drug AUC labels. Internal
validation showed Pearson r ≈ 0.05 between predicted combo AUC and
clinical induction CR rate — i.e., no meaningful signal. The
prospective validation study was paused before patient enrollment
specifically to determine whether (a) the BeatAML ex-vivo → clinical
CR transfer is fundamentally limited (in which case Layer-3 should be
removed) or (b) the ML architecture was the bottleneck.

We chose option (b)'s architecture upgrade based on three pieces of
evidence:

1. v0.6 GIN encoder reaches Pearson r = [INSERT] for single-drug AUC
   prediction on BeatAML hold-out (vs r = [INSERT] for v0.5
   `nn.Embedding`)
2. The GIN encoder generalizes to drugs not in BeatAML by featurizing
   their SMILES — useful for future drug-development applications
3. The GAT gene encoder learns protein-pathway-aware mutation
   embeddings (e.g., NRAS-KRAS-PTPN11 RAS pathway co-aggregation)

The v0.6 architecture change rationale + ablation evidence is in
[`docs/V6_MIGRATION_PLAN.md`](../V6_MIGRATION_PLAN.md) and
[`runs/v6_ablation_smoke/ablation_table.md`](../../runs/v6_ablation_smoke/ablation_table.md).

### 2.4 Risk-benefit re-assessment

**Risks**: NO CHANGE.
- Same data collection, same storage, same de-identification
- Kit predictions still blinded from MDT decisions per protocol §3
- Architecture change is invisible to participants — they cannot tell
  v0.5 from v0.6

**Benefits**: SLIGHT INCREASE (theoretical — to be confirmed by the
prospective study itself).
- Higher-quality ML predictions if the prospective endpoint validates
- Stronger contribution to the AML pharmacogenomics literature
- Future drug-development applications (zero-shot inference on novel
  compounds via SMILES)

---

## 3. Documents attached / updated

- [ ] Updated protocol document (v2): kit version + model description
      sections only
- [ ] Updated investigator brochure (v2): Section C of IRB packet
      reflects v0.6 architecture
- [ ] Statistical analysis plan (SAP) v2: sample size recalculation if
      target Pearson r changes (likely smaller N because expected effect
      size is higher)
- [ ] Locked predictions data flow diagram: NO CHANGE (same hash-chain
      infrastructure, kit_version field flips from "v0.5.x" → "v0.6.x")

---

## 4. Things that do NOT require re-consent

Per ICH-GCP E6(R2) §4.8, re-consent is required when:
- Material change to risk profile → **NO change here**
- Material change to study procedures → **NO change here** (data flow,
  schedule, treatments unchanged)
- Information becomes available that may affect willingness to
  continue → **NO** — kit accuracy is not material to participation
  since predictions are blinded from treatment

Already-enrolled patients (if any pre-amendment) **continue** under v0.5
locked predictions; **new** enrollees post-amendment use v0.6. The
analysis pipeline filters by kit_version field so the cohorts don't
mix.

---

## 5. Estimated review burden

This is an architecture-only amendment with no risk profile change.
Estimated IRB review: **expedited (single-reviewer)** within 2-3 weeks.
No full-board review required.

---

## 6. Document control

| Version | Date | Status |
|---------|------|--------|
| 0.1 | 2026-04-26 | Template — fill in [INSERT] fields after Phase 5 result |
