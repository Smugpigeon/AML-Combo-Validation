"""End-to-end kit prediction: KitInput + RNA counts → KitOutput (top combos).

Three-layer recommendation stack, all patient-specific:

  Layer 1 — Evidence (Route C — regimen retrieval)
    match_patient() → top 5 trial-matched regimens with published CR/OS.
    Covers drugs outside BeatAML vocab (ATRA, ATO, GO).

  Layer 2 — Biology (Path A — Clonal-Coverage × IDA)
    Decompose patient into clonal archetypes, score every combo (any arity)
    by Bliss-IDA coverage. Scales to N drugs without retraining. Biology-
    grounded, fully interpretable.

  Layer 3 — Prediction (Baseline A MLP + factorized combo)
    Continuous predicted AUC for all drug pairs in the clinical filter.
    Each pair also annotated with its Path A coverage score for audit.

Pipeline:
  1. feature_builder.build_patient_features_from_raw → 104-dim vector + QC
  2. Trained Baseline A MLP → predicted AUC for all 165 drugs
  3. mechanism_prior.compute_combo_mech_scores → 2-drug pair scores
  4. clonal_coverage → patient clones + N-arity coverage scores
  5. regimen_matcher.match_patient → trial-matched regimens
  6. Assemble + rank + emit KitOutput
"""

from __future__ import annotations

import warnings
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from combo_val.baselines.single_drug_mlp import SingleDrugMLP, SingleDrugMLPConfig
from combo_val.clinical.dna_report import build_dna_summary
from combo_val.clinical.eln_computer import compute_eln2017
from combo_val.clinical.feature_builder import build_patient_features_from_raw
from combo_val.clinical.kit_schema import KitInput, KitOutput
from combo_val.combo.mechanism_prior import (
    BEATAML_TO_MECH_ID,
    compute_combo_mech_scores,
)
from combo_val.combo.set_drug_inference import load_set_drug_predictor


# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------
#
# Each backbone is a tuple of (checkpoint_path, kind), where kind determines
# how it's loaded / used:
#
#   "mlp"                    Baseline A SingleDrugMLP — factorized combo +
#                            mechanism prior. Default; aligns with clinical
#                            literature (FLT3i+BCL2i at rank 1 for FLT3-mut).
#
#   "st"                     Set Transformer family. Covers v2 stable and all
#                            v3 variants (bliss / distill / pair-ft) — same
#                            architecture, different training data. Loaded via
#                            SetDrugInference.
#
#   "mlp+synergy"            MLP for singles + Route 4 SynergyHead for pair
#                            residuals. combo_auc = 0.5*(a1+a2) + synergy.

BACKBONE_REGISTRY: dict[str, dict] = {
    "mlp": {
        "kind": "mlp",
        "checkpoint": "runs/baseline_single_drug_mlp/final_model.pt",
        "label": "BaselineA-MLP + mech-prior",
        "description": "Single-drug MLP + factorized (0.5·(AUC_i+AUC_j) − k·mech_score)",
    },
    "st-v2": {
        "kind": "st",
        "checkpoint": "runs/set_drug_predictor_v2_stable/final_model.pt",
        "label": "SetTransformer-v2-stable",
        "description": "Multi-seed stable ST; single-drug-trained only",
    },
    "st-v3-bliss": {
        "kind": "st",
        "checkpoint": "runs/set_drug_predictor_v3_bliss/final_model.pt",
        "label": "SetTransformer-v3-BlissIDA",
        "description": "ST fine-tuned on synthetic Bliss-IDA pair labels (mechanism-blind)",
    },
    "st-v3-distill": {
        "kind": "st",
        "checkpoint": "runs/set_drug_predictor_v3_distilled/final_model.pt",
        "label": "SetTransformer-v3-Distill",
        "description": "ST distilled from Path A clonal-coverage teacher (biology-aligned ranking)",
    },
    "st-v3-186pair": {
        "kind": "st",
        "checkpoint": "runs/set_drug_predictor_v3_pair_ft/final_model.pt",
        "label": "SetTransformer-v3-186Pair",
        "description": "ST fine-tuned on 186 real ALMANAC-HL60 pair measurements",
    },
    "mlp+synergy": {
        "kind": "mlp+synergy",
        "checkpoint": "runs/baseline_single_drug_mlp/final_model.pt",
        "synergy_checkpoint": "runs/set_drug_predictor_v3_route4/synergy_head.pt",
        "label": "BaselineA-MLP + Route4-SynergyHead",
        "description": "MLP singles + learned synergy residual from 186 ALMANAC pairs",
    },
}


# Clinical filter matches Week 4 — drops pan-cytotoxic pipeline drugs
# that can artificially dominate single-drug picks.
CLINICALLY_RELEVANT_AML_DRUGS = (
    "Venetoclax",
    "Azacytidine",
    "Cytarabine",
    "Midostaurin",
    "Quizartinib (AC220)",
    "Gilteritinib",
    "Ivosidenib",
    "Enasidenib",
    "Sorafenib",
    "Crenolanib",
    "Dasatinib",
    "Nilotinib",
    "Imatinib",
    "Ponatinib",
    "Ruxolitinib (INCB018424)",
    "Trametinib (GSK1120212)",
    "Selumetinib (AZD6244)",
    "Alisertib (MLN8237)",
    "Pacritinib",
    "Crizotinib (PF-2341066)",
)


def _load_mlp(checkpoint_path: Path) -> tuple[SingleDrugMLP, dict]:
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    cfg_dict = ckpt["cfg"]
    # Rehydrate config
    from dataclasses import fields as _fields
    kwargs = {f.name: cfg_dict[f.name] for f in _fields(SingleDrugMLPConfig)
              if f.name in cfg_dict}
    cfg = SingleDrugMLPConfig(**kwargs)
    model = SingleDrugMLP(
        n_patient_features=ckpt["n_patient_features"],
        n_drugs=ckpt["n_drugs"],
        cfg=cfg,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _check_kit_cautions(kit: KitInput, driver_flags: dict) -> list[str]:
    cautions: list[str] = []
    if kit.ldh is not None and kit.ldh > 1000:
        cautions.append(f"High LDH ({kit.ldh:.0f} U/L) — elevated TLS risk with cytoreductive therapy")
    if driver_flags.get("FLT3_ITD"):
        cautions.append("FLT3-ITD positive — monitor QTc on FLT3 inhibitors (quizartinib, gilteritinib)")
    if kit.platelet is not None and kit.platelet < 30:
        cautions.append(f"Low platelets ({kit.platelet:.0f} ×10⁹/L) — bleeding precautions")
    if kit.age is not None and kit.age >= 75:
        cautions.append(f"Age ≥ 75 — intensive induction carries excess mortality; consider Ven+Aza")
    if driver_flags.get("TP53"):
        cautions.append("TP53 mutation — conventional induction poorly effective; consider trial enrollment")
    return cautions


def predict_for_patient(
    rna_counts: pd.Series,
    kit: KitInput,
    checkpoint_path: Path = Path("runs/baseline_single_drug_mlp/final_model.pt"),
    preprocessor_path: Path = Path("data/canonical/beataml_rna_preprocessor.joblib"),
    top_k: int = 5,
    drug_filter: tuple[str, ...] | None = CLINICALLY_RELEVANT_AML_DRUGS,
    mech_prior_scale: float = 30.0,
    backbone: str = "mlp",
    # Deprecated aliases
    set_transformer_checkpoint: Path | None = None,
    prefer_set_transformer: bool | None = None,
) -> KitOutput:
    """Run the full kit prediction pipeline for one new patient.

    Layer 3 backbone (``backbone`` kwarg, default ``"mlp"``):

      Clinical-recommender backbones (aligned with published evidence):
        "mlp"             Baseline A MLP + factorized combo + mech prior.
                          Default. Canonical FLT3i+BCL2i at rank 1 for FLT3-mut.

      Diagnostic / research backbones (report-only — do NOT replace "mlp" for
      clinical recommendation; each one's failure mode is documented):
        "st-v2"           Set Transformer, single-drug training only.
        "st-v3-bliss"     ST + synthetic Bliss-IDA (mechanism-blind).
        "st-v3-distill"   ST distilled from Path A clonal-coverage teacher.
        "st-v3-186pair"   ST fine-tuned on 186 real ALMANAC-HL60 pair
                          measurements (HL-60 is FLT3-wt → no FLT3-mut rules).
        "mlp+synergy"     MLP singles + Route-4 SynergyHead pair residual.

    Availability gracefully falls back to "mlp" if the requested backbone's
    checkpoint is missing.
    """
    # Back-compat: legacy prefer_set_transformer / set_transformer_checkpoint
    if prefer_set_transformer is True and backbone == "mlp":
        backbone = "st-v2"
    if set_transformer_checkpoint is not None:
        warnings.warn(
            "set_transformer_checkpoint is deprecated; use backbone='st-v2' "
            "and rely on BACKBONE_REGISTRY for the checkpoint path.",
            DeprecationWarning,
            stacklevel=2,
        )

    if backbone not in BACKBONE_REGISTRY:
        available = ", ".join(BACKBONE_REGISTRY.keys())
        raise ValueError(
            f"Unknown backbone '{backbone}'. Available: {available}"
        )
    spec = BACKBONE_REGISTRY[backbone]
    if not Path(spec["checkpoint"]).exists():
        warnings.warn(
            f"Backbone '{backbone}' checkpoint missing at {spec['checkpoint']}; "
            f"falling back to 'mlp'.",
            RuntimeWarning,
            stacklevel=2,
        )
        backbone = "mlp"
        spec = BACKBONE_REGISTRY["mlp"]

    # --- 1. Features ---
    features, diag = build_patient_features_from_raw(
        rna_counts, kit, preprocessor_path=preprocessor_path,
    )

    # --- 2. Layer 3 predictions — select by backbone kind ---
    synergy_predictor = None  # only set for mlp+synergy

    if spec["kind"] == "st":
        st_predictor = load_set_drug_predictor(Path(spec["checkpoint"]))
        if st_predictor is None:
            raise RuntimeError(
                f"Failed to load ST backbone '{backbone}' — checkpoint at "
                f"{spec['checkpoint']} exists but SetDrugInference returned None."
            )
        layer3_backbone = spec["label"]
        drug_vocab: list[str] = st_predictor.drug_vocab
        feature_cols_ckpt = st_predictor.feature_cols
        if len(feature_cols_ckpt) != len(features):
            raise ValueError(
                f"Feature-schema mismatch (ST '{backbone}'): checkpoint expects "
                f"{len(feature_cols_ckpt)} features, builder produced {len(features)}."
            )
        pred_auc = st_predictor.predict_singles(features)
    else:
        # "mlp" OR "mlp+synergy" — both use MLP for singles
        layer3_backbone = spec["label"]
        mlp_ckpt_path = Path(spec["checkpoint"])
        model, ckpt = _load_mlp(mlp_ckpt_path)
        drug_vocab = ckpt["drug_vocab"]
        feature_cols_ckpt = ckpt["feature_cols"]
        scaler_mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
        scaler_scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
        if len(feature_cols_ckpt) != len(features):
            raise ValueError(
                f"Feature-schema mismatch: MLP checkpoint expects "
                f"{len(feature_cols_ckpt)} features, builder produced {len(features)}. "
                f"Rerun beataml_etl + retrain Baseline A so both match."
            )
        standardized = (features - scaler_mean) / np.where(scaler_scale > 0, scaler_scale, 1.0)
        pf_tensor = torch.tensor(standardized, dtype=torch.float32)
        with torch.no_grad():
            pred_auc = model.predict_all_drugs_for_patient(pf_tensor, torch.device("cpu")).numpy()

        # If mlp+synergy, also load the Route 4 synergy head
        if spec["kind"] == "mlp+synergy":
            from combo_val.combo.synergy_inference import SynergyInference
            syn_path = Path(spec["synergy_checkpoint"])
            if not syn_path.exists():
                warnings.warn(
                    f"Synergy head missing at {syn_path}; falling back to pure MLP.",
                    RuntimeWarning, stacklevel=2,
                )
            else:
                synergy_predictor = SynergyInference(syn_path, device="cpu")
                layer3_backbone = spec["label"]

    # For ST backbones, we still need the MLP's drug_vocab alignment is
    # identical (both train on same beataml_drug_response_long.csv → sorted()).
    # Confirmed equal by construction; no extra validation needed.

    # --- 3. Drug filter ---
    if drug_filter:
        mask = np.array([d in drug_filter for d in drug_vocab])
    else:
        mask = np.ones(len(drug_vocab), dtype=bool)
    keep_idx = np.where(mask)[0]
    drug_vocab_filt = [drug_vocab[i] for i in keep_idx]
    pred_auc_filt = pred_auc[keep_idx]

    # --- 4. Single-drug top-k ---
    single_order = np.argsort(pred_auc_filt)
    top_single = [
        {
            "rank": r + 1,
            "drug": drug_vocab_filt[i],
            "predicted_auc": round(float(pred_auc_filt[i]), 2),
        }
        for r, i in enumerate(single_order[:top_k])
    ]

    # --- 5. Combo scoring ---
    # Mechanism prior (for audit, displayed on each combo regardless of backbone)
    pf_df = pd.DataFrame([features], columns=feature_cols_ckpt, index=[kit.patient_id])
    mech_scores = compute_combo_mech_scores(pf_df, drug_vocab_filt)  # (1, n_d, n_d)
    mech_scores = mech_scores[0]                                      # (n_d, n_d)

    n_d = len(drug_vocab_filt)
    if spec["kind"] == "st":
        # ST family: learned pair AUC direct from the set transformer.
        filt_st_indices = [st_predictor.drug_to_int[d] for d in drug_vocab_filt]
        combo_auc = st_predictor.predict_pairs(features, filt_st_indices)
        combo_auc = 0.5 * (combo_auc + combo_auc.T)   # enforce symmetry
    elif spec["kind"] == "mlp+synergy" and synergy_predictor is not None:
        # Route 4: MLP singles + learned synergy residual from ALMANAC pairs.
        filt_syn_indices = [
            synergy_predictor.drug_to_int[d] for d in drug_vocab_filt
            if d in synergy_predictor.drug_to_int
        ]
        # Build full synergy matrix (n_d, n_d) padding missing drugs with 0
        syn_matrix = np.zeros((n_d, n_d), dtype=np.float32)
        if len(filt_syn_indices) == n_d:
            syn_matrix = synergy_predictor.predict_synergy_matrix(filt_syn_indices)
        else:
            # Partial coverage: build matrix in the subset, leave others at 0
            subset_syn = synergy_predictor.predict_synergy_matrix(filt_syn_indices)
            subset_lookup = {
                d: i for i, d in enumerate(drug_vocab_filt)
                if d in synergy_predictor.drug_to_int
            }
            subset_order = [subset_lookup[d] for d in drug_vocab_filt
                            if d in subset_lookup]
            for a_i, a_full in enumerate(subset_order):
                for b_i, b_full in enumerate(subset_order):
                    syn_matrix[a_full, b_full] = subset_syn[a_i, b_i]
        combo_auc = (
            0.5 * (pred_auc_filt[:, None] + pred_auc_filt[None, :])
            + syn_matrix                       # synergy_loewe sign convention: negative = synergistic
        )
    else:
        # "mlp" default: factorized + mech prior.
        combo_auc = (
            0.5 * (pred_auc_filt[:, None] + pred_auc_filt[None, :])
            - mech_prior_scale * mech_scores
        )

    # Upper triangle (no self-pairs, no duplicates)
    tri_i, tri_j = np.triu_indices(n_d, k=1)
    pair_auc = combo_auc[tri_i, tri_j]
    pair_mech = mech_scores[tri_i, tri_j]
    order = np.argsort(pair_auc)
    top_combos = []
    for r, k in enumerate(order[:top_k]):
        d1 = drug_vocab_filt[tri_i[k]]
        d2 = drug_vocab_filt[tri_j[k]]
        top_combos.append({
            "rank": r + 1,
            "drug1": d1,
            "drug2": d2,
            "predicted_combo_auc": round(float(pair_auc[k]), 2),
            "single_auc_d1": round(float(pred_auc_filt[tri_i[k]]), 2),
            "single_auc_d2": round(float(pred_auc_filt[tri_j[k]]), 2),
            "mech_score": round(float(pair_mech[k]), 3),
            "both_mech_annotated": (
                d1 in BEATAML_TO_MECH_ID and d2 in BEATAML_TO_MECH_ID
            ),
            "layer3_backbone": layer3_backbone,
        })

    # --- 6. Assemble output ---
    driver_flags = {
        "FLT3_ITD": any(m.gene.upper() == "FLT3" and m.is_ITD for m in kit.mutations),
        "FLT3_TKD": any(m.gene.upper() == "FLT3" and m.is_TKD for m in kit.mutations),
        "NPM1": any(m.gene.upper() == "NPM1" for m in kit.mutations),
        "IDH1": any(m.gene.upper() == "IDH1" for m in kit.mutations),
        "IDH2": any(m.gene.upper() == "IDH2" for m in kit.mutations),
        "TP53": any(m.gene.upper() == "TP53" for m in kit.mutations),
        "RUNX1": any(m.gene.upper() == "RUNX1" for m in kit.mutations),
        "CEBPA_biallelic": any(m.gene.upper() == "CEBPA" and m.is_biallelic for m in kit.mutations),
    }
    fitness_flag = "fit_for_intensive" if (kit.age is not None and kit.age <= 65) else "unfit"
    confidence_notes = []

    # OOD / pipeline-mismatch warnings — pass through QC's own messages
    # (which are already severity-aware and contain accurate numerics).
    qc = diag.get("qc", {})
    severity = qc.get("ood_severity", "ok")
    severity_badge = {
        "far_ood": "✗",
        "ood": "⚠",
        "pipeline_mismatch": "⚠",
        "borderline": "~",
        "ok": "ⓘ",
    }.get(severity, "ⓘ")
    for msg in qc.get("warning_messages", []):
        confidence_notes.append(f"{severity_badge} {msg}")

    if diag["n_imputed_fields"] > 0:
        confidence_notes.append(
            f"{diag['n_imputed_fields']} clinical field(s) imputed with BeatAML medians: "
            f"{', '.join(diag['imputed_fields'][:6])}"
            + ("..." if diag["n_imputed_fields"] > 6 else "")
        )

    # ---- Route C (Layer 1): regimen retrieval from curated trial DB ----
    # Runs independently of the MLP so it can ALWAYS produce a recommendation
    # even for drugs the MLP has no vocab for (ATRA/ATO, Decitabine, etc.).
    from combo_val.clinical.regimen_matcher import match_patient as _match_regimen

    # Build a feature dict keyed by the saved feature_cols (patient_features
    # order) so the matcher sees the same schema as training.
    patient_feat_dict = {
        col: float(features[i]) for i, col in enumerate(feature_cols_ckpt)
    }
    regimen_matches = _match_regimen(patient_feat_dict, top_k=top_k)
    top_regimens = []
    for rank, m in enumerate(regimen_matches, start=1):
        s = m.as_summary()
        s["rank"] = rank
        top_regimens.append(s)

    # ---- Path A (Layer 2): Clonal-Coverage × IDA ----
    # Patient-specific clonal decomposition + N-arity coverage scoring.
    # Provides a biology-grounded ranking independent of MLP training data.
    from combo_val.combo.clonal_coverage import (
        build_patient_clone_matrix,
        build_drug_clone_coverage,
        rank_top_combos_per_patient,
        score_combo_for_patient,
    )

    pf_df_for_clones = pd.DataFrame(
        [patient_feat_dict], index=[kit.patient_id], columns=feature_cols_ckpt,
    )
    patient_clones = build_patient_clone_matrix(pf_df_for_clones)
    drug_clone_cov = build_drug_clone_coverage(list(drug_vocab_filt))

    # Top doublets and triplets by PURE clonal coverage (biology-only)
    def _coverage_ranking_to_dicts(df: pd.DataFrame) -> list[dict]:
        out = []
        drug_cols = [c for c in df.columns if c.startswith("drug")]
        for _, row in df.iterrows():
            entry = {
                "rank": int(row["rank"]),
                "drugs": [str(row[c]) for c in drug_cols if pd.notna(row[c])],
                "coverage_score": round(float(row["score"]), 3),
                "arity": int(row["arity"]),
            }
            out.append(entry)
        return out

    cov_k2 = rank_top_combos_per_patient(
        patient_clones, drug_clone_cov, arity=2, top_k=top_k,
    )
    cov_k3 = rank_top_combos_per_patient(
        patient_clones, drug_clone_cov, arity=3, top_k=top_k,
    )

    # Annotate each MLP combo with its Path A coverage score (audit)
    clones_arr = patient_clones.iloc[0].to_numpy(dtype=np.float64)
    drug_cov_np = drug_clone_cov.to_numpy(dtype=np.float64)
    drug_to_idx = {d: i for i, d in enumerate(drug_clone_cov.index)}
    for c in top_combos:
        i1 = drug_to_idx.get(c["drug1"])
        i2 = drug_to_idx.get(c["drug2"])
        if i1 is not None and i2 is not None:
            cov, _ = score_combo_for_patient(
                clones_arr, drug_cov_np[[i1, i2]],
            )
            c["clonal_coverage_score"] = round(float(cov), 3)
        else:
            c["clonal_coverage_score"] = None

    # Per-patient clone structure (biology explanation)
    present_clones = patient_clones.iloc[0]
    present_clones = present_clones[present_clones > 0].sort_values(ascending=False)

    clonal_coverage = {
        "patient_clones": {
            str(k): round(float(v), 2) for k, v in present_clones.items()
        },
        "n_clones_present": int(len(present_clones)),
        "dominant_clones": [str(k) for k in present_clones.head(3).index],
        "top_doublets_by_coverage": _coverage_ranking_to_dicts(cov_k2),
        "top_triplets_by_coverage": _coverage_ranking_to_dicts(cov_k3),
    }

    # ---- DNA-level profile (core genes, mutations, targetability) ----
    dna_summary = build_dna_summary(kit, computed_eln=diag["eln_predicted"])

    return KitOutput(
        patient_id=kit.patient_id,
        predicted_eln2017=diag["eln_predicted"],
        top_combinations=top_combos,
        top_single_drugs=top_single,
        top_regimens=top_regimens,
        clonal_coverage=clonal_coverage,
        dna_summary=dna_summary,
        driver_flags=driver_flags,
        fitness_flag=fitness_flag,
        cautions=_check_kit_cautions(kit, driver_flags),
        confidence_notes=confidence_notes,
    )


def pretty_print_kit_output(out: KitOutput) -> str:
    """Format a KitOutput for clinician-facing reporting."""
    lines = [
        f"╔═══ Patient {out.patient_id} — AML Combo-Prediction Kit Report ═══",
        f"║ Predicted ELN 2017: {out.predicted_eln2017:<12s}  Fitness: {out.fitness_flag}",
        f"║ Driver flags: {', '.join(k for k, v in out.driver_flags.items() if v) or 'none detected'}",
    ]

    # ---- Layer 2 (biology summary, headline) — patient clone structure ----
    cc = out.clonal_coverage or {}
    clones = cc.get("patient_clones") or {}
    if clones:
        clones_str = ", ".join(
            f"{name} ({w:.1f})" for name, w in list(clones.items())[:5]
        )
        lines.append(
            f"║ Clonal structure ({cc.get('n_clones_present', 0)} present): {clones_str}"
        )

    # Detect which backbone produced the Layer 3 output (defaults to MLP for
    # older saved KitOutputs that don't carry the annotation).
    backbone = (
        out.top_combinations[0].get("layer3_backbone", "BaselineA-MLP")
        if out.top_combinations else "BaselineA-MLP"
    )
    lines += [
        "║",
        f"║ LAYER 3 — TOP COMBINATIONS ({backbone}, ★ = both drugs mech-annotated)",
    ]
    for c in out.top_combinations:
        mark = "★" if c["both_mech_annotated"] else " "
        cov = c.get("clonal_coverage_score")
        cov_str = f"  cov={cov:+.2f}" if cov is not None else ""
        lines.append(
            f"║  {c['rank']}.{mark} {c['drug1']:<22s} + {c['drug2']:<22s}  "
            f"AUC = {c['predicted_combo_auc']:6.1f}  "
            f"(mech {c['mech_score']:+.2f}{cov_str})"
        )

    lines += [
        "║",
        "║ TOP SINGLE DRUGS (reference)",
    ]
    for s in out.top_single_drugs:
        lines.append(
            f"║  {s['rank']}. {s['drug']:<30s}  predicted AUC = {s['predicted_auc']:6.1f}"
        )

    # ---- Layer 2 (biology, detail) — Path A coverage rankings ----
    if cc.get("top_doublets_by_coverage") or cc.get("top_triplets_by_coverage"):
        lines += [
            "║",
            "║ LAYER 2 — BIOLOGY (Path A clonal-coverage, biology-only ranking)",
        ]
        doublets = cc.get("top_doublets_by_coverage") or []
        triplets = cc.get("top_triplets_by_coverage") or []
        if doublets:
            lines.append("║  Doublets (covers N clones via Bliss-IDA):")
            for d in doublets[:3]:
                drugs_str = " + ".join(d["drugs"])
                lines.append(
                    f"║    {d['rank']}. {drugs_str:<50s}  coverage = {d['coverage_score']:.3f}"
                )
        if triplets:
            lines.append("║  Triplets:")
            for d in triplets[:3]:
                drugs_str = " + ".join(d["drugs"])
                lines.append(
                    f"║    {d['rank']}. {drugs_str:<50s}  coverage = {d['coverage_score']:.3f}"
                )

    # ---- Layer 1 (evidence) — Route C trial-matched regimens ----
    if out.top_regimens:
        lines += [
            "║",
            "║ LAYER 1 — EVIDENCE (trial-matched regimens with published CR/OS)",
        ]
        for r in out.top_regimens:
            drugs_str = " + ".join(r["drugs"])
            cr_str = f"CR/CRi {100 * r['published_cr_cri_rate']:.0f}%"
            os_str = (f" · median OS {r['published_median_os_months']:.1f}mo"
                      if r.get("published_median_os_months") is not None else "")
            evidence = f"[{r['trial_phase']} {r['trial_name']}]"
            pmid = f" PMID {r['pmid']}" if r.get("pmid") else ""
            lines.append(
                f"║  {r['rank']}. {drugs_str}"
            )
            lines.append(
                f"║     {cr_str}{os_str}  {evidence}{pmid}"
            )
            if r.get("cautions"):
                for c in r["cautions"]:
                    lines.append(f"║       ⚠ {c}")
    if out.cautions:
        lines += ["║", "║ CAUTIONS"]
        for c in out.cautions:
            lines.append(f"║  ⚠ {c}")
    # DNA-level profile — core genes, mutations, targetability (audit table)
    if out.dna_summary:
        from combo_val.clinical.dna_report import pretty_print_dna_summary
        lines.append(pretty_print_dna_summary(out.dna_summary))

    if out.confidence_notes:
        lines += ["║", "║ CONFIDENCE NOTES"]
        for c in out.confidence_notes:
            lines.append(f"║  ⓘ {c}")
    lines.append("╚" + "═" * 68)
    return "\n".join(lines)
