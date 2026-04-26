# V6 Migration Plan — DrugGIN + GeneGAT + Set-Transformer Combo Head

This is the operational plan for transitioning the kit's Layer-3
backbone from `mlp` (single-drug MLP + factorized 0.5·(AUC_i + AUC_j) −
k·mech_score) to `gnn-v6` (DrugGIN + GeneGAT + Set-Transformer combo).

Authoring this plan is the v0.6 Phase 6 commitment from the post-v0.5
reviewer dialog (Path C: revise architecture before resuming
prospective validation).

---

## 1. Decision tree (per Phase 5 ablation outcome)

```
Phase 5 ablation runs scripts/v6_ablation.py:
   → 4 arms × 5 folds × 20 epochs on BeatAML 55,826 single-drug
   → Reports per-arm Pearson r vs baseline

      ┌─ v0.6.0_full Pearson r ≥ 0.20 ─→ GO   (Phase 6 begins)
      │
      ├─ 0.10 ≤ r < 0.20             ─→ REVIEW (DSMB-style assessment)
      │
      └─ r < 0.10                     ─→ NO-GO (delete Layer-3, Path A)
```

The decision is **automated** — `scripts/v6_ablation.py` writes
`runs/v6_ablation/go_no_go_decision.json` based on the threshold logic.
Engineering may not override.

---

## 2. Phase 6 implementation steps (only if Phase 5 = GO)

### 6.1 Wire `gnn-v6` into kit_predict dispatch

`src/combo_val/clinical/kit_predict.py` currently has the early-rejection
gate:

```python
if spec.get("kind") == "gnn-v6":
    raise NotImplementedError(...)
```

Replace with proper dispatch:

```python
elif spec["kind"] == "gnn-v6":
    from combo_val.encoders.v6_full_model import V6FullModel
    model = _load_v6(spec["checkpoint"], spec["ppi_subgraph"], gene_order)
    drug_input = [smiles_map[d] for d in drug_vocab_filt]
    drug_embs = model.encode_drugs([drug_input], n_max=len(drug_vocab_filt))
    # ... compute pairwise combo_auc via the Set-Transformer head over all pairs
```

A new helper `_load_v6()` parallel to existing `_load_mlp()` /
`load_set_drug_predictor()`.

### 6.2 SMILES bundle in package data

Add `data/canonical/drug_smiles.csv` to the package data so the kit
ships with the per-drug SMILES table. Without this the GNN encoder has
no way to look up SMILES at inference time.

### 6.3 Default backbone change?

Open question: does v0.6 release flip the default `backbone="mlp"` to
`backbone="gnn-v6"`?

**Recommendation**: No. Keep `mlp` as default for one minor release
(v0.6.0). Make `gnn-v6` opt-in via `predict_for_patient(backbone="gnn-v6")`
and via the audit_mode UI toggle. Flip default to `gnn-v6` in v0.6.1
ONLY after at least 30 days of internal usage with no kit-output
regressions reported.

### 6.4 Tests

New test file `tests/test_v6_kit_integration.py`:

- `test_v6_predict_for_patient_runs` — smoke
- `test_v6_top1_for_flt3_canonical` — clinical sanity
- `test_v6_handles_missing_smiles_gracefully` — fallback behavior
- `test_v6_ood_suppression_still_works` — Layer-3 OOD gate preserved
- `test_v6_inference_latency_budget` — < 60s per patient

### 6.5 Web app caveat update

Update `amlcombo_web/app/locales/zh.json` + `en.json` and the patient
report banner to reflect "v0.6 GNN candidate, internal r = X.XX,
prospective validation pending":

```json
{
  "v6_banner": "v0.6 — GNN-based Layer-3 (DrugGIN + GeneGAT). Internal Pearson r = {r:.2f}; prospective validation pending."
}
```

---

## 3. Phase 7 deployment

Standard deploy flow (same as v0.4 / v0.5):

```bash
git push origin main          # → triggers no automation; manual deploy
ssh amlcombo "cd ~/AML-combo-validation && git pull && \
              cd amlcombo_web && \
              docker compose build web worker && \
              docker compose up -d"
```

Smoke test on amlcombo.org:

```python
out = predict_for_patient(rna, kit, backbone="gnn-v6")
assert out.predicted_top1_regimen_id  # not empty
assert out.top_combinations[0]["layer3_backbone"].startswith("v0.6-GNN")
```

---

## 4. Phase 8 — re-engaging prospective study

If v0.6 lands successfully, the v0.5 prospective protocol
(`docs/PROSPECTIVE_VALIDATION_PROTOCOL.md`) needs amendment:

1. Lock v0.6.0 commit hash → `kit_version: "v0.6.0"` + `kit_commit_hash: "<sha>"`
2. SAP §5: replace internal r ≈ 0.05 baseline with the actual Phase 5
   internal r number; recompute sample size for the new effect size
   (likely smaller N because we're testing a stronger model)
3. IRB amendment: file a "model architecture change" amendment with
   each participating site's IRB. Not a new study — same ethics, same
   risk profile, just upgraded model.
4. Update PI outreach letter: explain why v0.5 prospective was paused
   and v0.6 is now the candidate.

---

## 5. Versioning + locked-prediction schema

`src/combo_val/prospective/locked_prediction.py` already records
`kit_commit_hash` and `model_checkpoint_sha256`. After v0.6 cutover:

- All v0.5 locked predictions remain immutable + verifiable in the
  hash chain
- New patients enrolled under v0.6 amendment will have
  `kit_version: "v0.6.0"` field and a different `model_checkpoint_sha256`
- The Phase 5 cohort assembly script
  (`combo_val.prospective.outcome_capture.assemble_cohort_for_analysis`)
  filters by `kit_version` so analyses don't mix v0.5 and v0.6 patients

---

## 6. Rollback plan

If a v0.6 production bug emerges:

1. `git revert <v0.6 merge commit>` on origin/main
2. SSH deploy: `git pull && docker compose build && up -d`
3. Web users see v0.5 behavior again immediately
4. v0.6-locked prospective predictions remain valid (immutable) but new
   enrollment temporarily pauses

Risk window: < 30 minutes from rollback decision to live.

---

## 7. Document control

| Version | Date | Status |
|---------|------|--------|
| 0.1 | 2026-04-26 | Initial — Phase 5 ablation pending |
