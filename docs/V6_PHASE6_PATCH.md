# Phase 6 — kit_predict.py dispatch patch (apply after Phase 5 GO)

This is the surgical patch to wire `gnn-v6` into the Layer-3 dispatch.
Apply ONLY when Phase 5 ablation outputs `decision: "GO"` in
`runs/v6_ablation_smoke/go_no_go_decision.json` AND the corresponding
checkpoint is at `runs/v6_endtoend/final_model.pt` (or update the path
in BACKBONE_REGISTRY['gnn-v6']['checkpoint']).

## 1. Remove the early-rejection gate

In `src/combo_val/clinical/kit_predict.py`, **remove** this block (added
in commit a40944e):

```python
if spec.get("kind") == "gnn-v6":
    raise NotImplementedError(
        f"Backbone '{backbone}' is registered but pre-validation ..."
    )
```

## 2. Add v0.6 dispatch in the kind-switch

In the same file, find the dispatch:

```python
if spec["kind"] == "st":
    st_predictor = load_set_drug_predictor(...)
    ...
elif spec["kind"] == "mlp+synergy" and synergy_predictor is not None:
    ...
else:
    # "mlp" default: factorized + mech prior
    combo_auc = (
        0.5 * (pred_auc_filt[:, None] + pred_auc_filt[None, :])
        - mech_prior_scale * mech_scores
    )
```

**Add a new branch** before the `else: # "mlp" default` line:

```python
elif spec["kind"] == "gnn-v6":
    from combo_val.clinical._v6_loader import (
        load_smiles_table, load_v6_model, predict_combo_auc_via_gnn_v6,
    )
    smiles_table = load_smiles_table(spec["smiles_table"])
    v6_model, v6_meta = load_v6_model(
        Path(spec["checkpoint"]),
        Path(spec["ppi_subgraph"]),
        Path("data/canonical/beataml_patient_features.csv"),
    )
    combo_auc = predict_combo_auc_via_gnn_v6(
        v6_model, smiles_table, drug_vocab_filt, features,
    )
    layer3_backbone = (
        spec["label"] + f" (Phase-5 arm={v6_meta['arm']!r}, "
        f"epoch={v6_meta['epoch']})"
    )
```

## 3. Status flip

In `BACKBONE_REGISTRY['gnn-v6']`, change:

```python
"status": "candidate-pre-validation",
```

to:

```python
"status": "production-after-phase5-go",
"phase5_internal_pearson_r": <fill from go_no_go_decision.json>,
"activated_in_commit": "<this commit's hash after merge>",
```

## 4. Tests to update / add

`tests/test_backbone_registry.py`:
- Remove `test_gnn_v6_registered_but_dispatch_blocked` (gate is gone)
- Add `test_gnn_v6_predict_returns_real_combo_matrix` smoke

`tests/test_v6_kit_integration.py` (new file):
- `test_v6_top1_for_flt3_canonical` — clinical sanity
- `test_v6_inference_latency_under_60s`
- `test_v6_handles_missing_smiles_per_drug`
- `test_v6_ood_suppression_still_fires`

## 5. Default backbone NOT flipped

Per V6_MIGRATION_PLAN.md §6.3, the default `backbone="mlp"` stays in
v0.6.0. Users opt into `gnn-v6` explicitly via:

```python
out = predict_for_patient(rna, kit, backbone="gnn-v6")
```

Default flip to `gnn-v6` is the v0.6.1 release decision after ≥30 days
of opt-in usage with no regressions.

## 6. Web app caveat banner update

`amlcombo_web/app/locales/en.json` + `zh.json`: add a v0.6 banner
displayed when `audit_mode=True` and the patient's report shows the
gnn-v6 prediction. Don't show in default reports until Phase 8
prospective re-engagement closes the loop.
