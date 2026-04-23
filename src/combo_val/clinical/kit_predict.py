"""End-to-end kit prediction: KitInput + RNA counts → KitOutput (top combos).

Chains together:
  1. feature_builder.build_patient_features_from_raw → 104-dim vector
  2. Trained Baseline A MLP → predicted AUC for all 165 drugs
  3. mechanism_prior.compute_combo_mech_scores for ALL drug pairs
  4. Factorized combo AUC: 0.5 * (AUC_d1 + AUC_d2) - mech_prior_scale * mech_score
  5. Rank, format, emit KitOutput

This is the "kit API" a clinical operator would call after running the
NGS + CBC + karyotype workup on a new patient.
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
from combo_val.clinical.eln_computer import compute_eln2017
from combo_val.clinical.feature_builder import build_patient_features_from_raw
from combo_val.clinical.kit_schema import KitInput, KitOutput
from combo_val.combo.mechanism_prior import (
    BEATAML_TO_MECH_ID,
    compute_combo_mech_scores,
)


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
) -> KitOutput:
    """Run the full kit prediction pipeline for one new patient."""
    # --- 1. Features ---
    features, diag = build_patient_features_from_raw(
        rna_counts, kit, preprocessor_path=preprocessor_path,
    )

    # --- 2. MLP predictions ---
    model, ckpt = _load_mlp(checkpoint_path)
    drug_vocab: list[str] = ckpt["drug_vocab"]
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
    # Build a 1-row patient features frame for mech-prior scoring.
    # Use the COLUMNS from the checkpoint feature_cols so the mech builder
    # can find mut_FLT3 etc.
    pf_df = pd.DataFrame([features], columns=feature_cols_ckpt, index=[kit.patient_id])
    mech_scores = compute_combo_mech_scores(pf_df, drug_vocab_filt)  # (1, n_d, n_d)
    mech_scores = mech_scores[0]                                      # (n_d, n_d)

    # Combo AUC = 0.5 * (auc_i + auc_j) - mech_scale * mech_score (lower = better)
    n_d = len(drug_vocab_filt)
    combo_auc = 0.5 * (pred_auc_filt[:, None] + pred_auc_filt[None, :]) - mech_prior_scale * mech_scores
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
    if diag["rna_gene_coverage_pct"] < 70:
        confidence_notes.append(
            f"RNA-Seq gene coverage only {diag['rna_gene_coverage_pct']:.0f}% of the "
            f"5000-gene training panel; predictions may be less reliable"
        )
    if diag["n_imputed_fields"] > 0:
        confidence_notes.append(
            f"{diag['n_imputed_fields']} clinical field(s) imputed with BeatAML medians: "
            f"{', '.join(diag['imputed_fields'][:6])}"
            + ("..." if diag["n_imputed_fields"] > 6 else "")
        )

    return KitOutput(
        patient_id=kit.patient_id,
        predicted_eln2017=diag["eln_predicted"],
        top_combinations=top_combos,
        top_single_drugs=top_single,
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
        "║",
        "║ TOP RECOMMENDED COMBINATIONS (lower predicted AUC = more cell kill)",
    ]
    for c in out.top_combinations:
        mark = "★" if c["both_mech_annotated"] else " "
        lines.append(
            f"║  {c['rank']}.{mark} {c['drug1']:<22s} + {c['drug2']:<22s}  "
            f"predicted combo AUC = {c['predicted_combo_auc']:6.1f}  "
            f"(mech score = {c['mech_score']:+.2f})"
        )
    lines += [
        "║",
        "║ TOP SINGLE DRUGS (reference)",
    ]
    for s in out.top_single_drugs:
        lines.append(
            f"║  {s['rank']}. {s['drug']:<30s}  predicted AUC = {s['predicted_auc']:6.1f}"
        )
    if out.cautions:
        lines += ["║", "║ CAUTIONS"]
        for c in out.cautions:
            lines.append(f"║  ⚠ {c}")
    if out.confidence_notes:
        lines += ["║", "║ CONFIDENCE NOTES"]
        for c in out.confidence_notes:
            lines.append(f"║  ⓘ {c}")
    lines.append("╚" + "═" * 68)
    return "\n".join(lines)
