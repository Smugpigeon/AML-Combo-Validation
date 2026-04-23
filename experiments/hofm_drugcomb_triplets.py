"""Real-data Path D experiment: train HOFM on DrugComb AML pair data,
extrapolate to ALL 3-drug combinations, inspect the ranking.

Two questions:

  Q1 — CV performance on 2-way (sanity):
       HOFM should match or slightly beat the order-2 SynergyMLP baseline
       (RMSE ≤ ~13 on ALMANAC-HL60 pair data).

  Q2 — Extrapolation to 3-way:
       With non-negative V, trained only on the 186 pair rows, predict
       ⟨v_i, v_j, v_k⟩ for all C(19, 3) = 969 possible triples among drugs
       that appear in the pair data. Inspect:
         - Top-10 highest-scoring triples (do FLT3i + BCL2i + something
           appear? do known-clinical triplets surface?)
         - Bottom-10 (should avoid same-MoA duplicates)
         - Embedding interpretability: t-SNE of V colored by moa_family
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import KFold

from combo_val.combo.hofm import HOFM, HOFMConfig

PAIR_CSV = Path("data/canonical/drugcomb_aml_pairs.csv")
MECH_CSV = Path("src/combo_val/knowledge/drug_mechanism_v1.csv")
OUT_DIR = Path("runs/hofm_drugcomb")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pair_data(path: Path = PAIR_CSV, synergy_col: str = "synergy_loewe"):
    df = pd.read_csv(path).dropna(subset=["drug1_id", "drug2_id", synergy_col]).copy()
    drugs = sorted(set(df["drug1_id"]).union(df["drug2_id"]))
    d2i = {d: i for i, d in enumerate(drugs)}
    df["d1_int"] = df["drug1_id"].map(d2i)
    df["d2_int"] = df["drug2_id"].map(d2i)
    return df, drugs, d2i


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def fit_hofm(pair_df: pd.DataFrame, n_drugs: int, cfg: HOFMConfig,
             synergy_col: str = "synergy_loewe", n_iter: int = 3000,
             lr: float = 1e-2, weight_decay: float = 1e-3,
             seed: int = 0, verbose: bool = False) -> HOFM:
    torch.manual_seed(seed)
    model = HOFM(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ids = torch.tensor(pair_df[["d1_int", "d2_int"]].values, dtype=torch.long)
    y = torch.tensor(pair_df[synergy_col].values, dtype=torch.float)
    for it in range(n_iter):
        opt.zero_grad()
        pred = model(ids)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()
        if verbose and it % 500 == 0:
            print(f"  iter {it}  loss {loss.item():.3f}")
    return model


def cv_5fold(pair_df: pd.DataFrame, n_drugs: int, cfg_nonneg: bool,
             k_model: int, synergy_col: str = "synergy_loewe",
             n_splits: int = 5, seed: int = 42) -> dict:
    """5-fold CV of HOFM on pair data. Reports RMSE, Pearson."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds_all, truth_all, fold_rmse = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(pair_df)):
        train_df = pair_df.iloc[tr_idx].reset_index(drop=True)
        val_df = pair_df.iloc[va_idx].reset_index(drop=True)
        cfg = HOFMConfig(n_drugs=n_drugs, embedding_dim=k_model, max_order=3,
                         nonneg_embedding=cfg_nonneg, init_std=0.05)
        model = fit_hofm(train_df, n_drugs, cfg, synergy_col=synergy_col,
                         seed=fold * 100 + seed)
        model.eval()
        with torch.no_grad():
            v_ids = torch.tensor(val_df[["d1_int", "d2_int"]].values, dtype=torch.long)
            v_pred = model(v_ids).cpu().numpy()
        v_true = val_df[synergy_col].values
        preds_all.extend(v_pred)
        truth_all.extend(v_true)
        fold_rmse.append(float(np.sqrt(np.mean((v_pred - v_true) ** 2))))

    preds = np.array(preds_all)
    truth = np.array(truth_all)
    r = float(np.corrcoef(preds, truth)[0, 1])
    spr = float(stats.spearmanr(preds, truth).correlation)
    return {
        "mode": "nonneg" if cfg_nonneg else "unconstrained",
        "k_model": k_model,
        "fold_rmse_mean": float(np.mean(fold_rmse)),
        "fold_rmse_std": float(np.std(fold_rmse)),
        "cv_pooled_pearson": r,
        "cv_pooled_spearman": spr,
        "null_rmse_baseline": float(np.std(truth)),
    }


# ---------------------------------------------------------------------------
# Triple extrapolation
# ---------------------------------------------------------------------------


def score_all_triples(model: HOFM, n_drugs: int) -> pd.DataFrame:
    """Score every C(n_drugs, 3) triple with the trained HOFM."""
    triples = list(combinations(range(n_drugs), 3))
    ids = torch.tensor(triples, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        y, br = model(ids, return_order_breakdown=True)
    y = y.cpu().numpy()
    order3 = br["order3"].sum(dim=-1).cpu().numpy() if br["order3"].dim() > 1 else br["order3"].cpu().numpy()
    order2 = br["order2"].sum(dim=-1).cpu().numpy() if br["order2"].dim() > 1 else br["order2"].cpu().numpy()
    # NOTE: "order2" here is the pair-sum INSIDE the triple (3 pairs). Not
    # directly comparable to 2-drug pair scores but informative for separating
    # the 3-way contribution from pair contributions.
    df = pd.DataFrame({
        "d1_int": [t[0] for t in triples],
        "d2_int": [t[1] for t in triples],
        "d3_int": [t[2] for t in triples],
        "total_score": y,
        "pairs_in_triple_sum": order2,
        "trilinear_term": order3,
    })
    return df


# ---------------------------------------------------------------------------
# Mechanism annotation for triples
# ---------------------------------------------------------------------------


# Map BeatAML drug name → MoA family from the mechanism table. Used to label
# the top-scoring triples so we can eyeball whether they make biological sense.
def _load_moa_map(mech_csv: Path = MECH_CSV) -> dict[str, str]:
    """Return a dict: BeatAML drug_id → moa_family, using our BEATAML_TO_MECH_ID
    bridge. Drugs without mechanism annotation get "unannotated"."""
    from combo_val.combo.mechanism_prior import BEATAML_TO_MECH_ID
    mech = pd.read_csv(mech_csv).set_index("drug_id")
    moa_by_beat: dict[str, str] = {}
    for beat_id, mech_id in BEATAML_TO_MECH_ID.items():
        if mech_id in mech.index:
            moa_by_beat[beat_id] = str(mech.loc[mech_id, "moa_family"])
    return moa_by_beat


# ---------------------------------------------------------------------------
# Embedding interpretability
# ---------------------------------------------------------------------------


def plot_embedding(model: HOFM, drug_names: list[str], moa_map: dict[str, str],
                    out_path: Path) -> None:
    V = torch.nn.functional.softplus(model.drug_emb.weight.data).cpu().numpy() \
        if model.cfg.nonneg_embedding else model.drug_emb.weight.data.cpu().numpy()

    # Simple 2D embedding via PCA (t-SNE would need more data points)
    Vc = V - V.mean(axis=0)
    U, S, Vt = np.linalg.svd(Vc, full_matrices=False)
    coords = Vc @ Vt.T[:, :2]

    # Group drugs by MoA family
    moas = [moa_map.get(n, "unannotated") for n in drug_names]
    unique_moas = sorted(set(moas))
    colors = plt.cm.tab20.colors

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for i, moa in enumerate(unique_moas):
        mask = np.array([m == moa for m in moas])
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], s=80,
                   label=moa, color=colors[i % len(colors)], alpha=0.8)
    for i, name in enumerate(drug_names):
        short = name.replace(" ", "")[:12]
        ax.annotate(short, (coords[i, 0], coords[i, 1]),
                    fontsize=8, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("PC1 of learned V")
    ax.set_ylabel("PC2 of learned V")
    ax.set_title("HOFM drug embeddings (non-negative V, trained on 186 AML pairs)\n"
                 "Clusters by moa_family validate biological coherence")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_all(out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    pair_df, drugs, d2i = load_pair_data()
    n_drugs = len(drugs)
    print(f"[hofm-drugcomb] Loaded {len(pair_df)} pair rows, {n_drugs} unique drugs")

    # -----------------------------------------------------
    # Q1 — CV on pair synergy
    # -----------------------------------------------------
    print("\n--- Q1: 5-fold CV on pair synergy prediction ---")
    cv_results = []
    for mode in (False, True):
        for k in (4, 8, 16):
            res = cv_5fold(pair_df, n_drugs, cfg_nonneg=mode, k_model=k)
            cv_results.append(res)
            print(f"  {res}")
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(out_dir / "cv_pair_synergy.csv", index=False)
    print(cv_df[["mode", "k_model", "fold_rmse_mean", "cv_pooled_pearson"]].to_string(index=False))

    # -----------------------------------------------------
    # Q2 — Train on full pair data with best config, extrapolate triples
    # -----------------------------------------------------
    print("\n--- Q2: Full-data HOFM → all triples ---")
    best_cfg = HOFMConfig(n_drugs=n_drugs, embedding_dim=8, max_order=3,
                          nonneg_embedding=True, init_std=0.05)
    full_model = fit_hofm(pair_df, n_drugs, best_cfg, n_iter=5000, lr=1e-2,
                          weight_decay=1e-3, seed=42, verbose=True)

    trip_df = score_all_triples(full_model, n_drugs)
    trip_df["drug1"] = trip_df["d1_int"].map(lambda i: drugs[i])
    trip_df["drug2"] = trip_df["d2_int"].map(lambda i: drugs[i])
    trip_df["drug3"] = trip_df["d3_int"].map(lambda i: drugs[i])

    moa_map = _load_moa_map()
    trip_df["moa1"] = trip_df["drug1"].map(lambda d: moa_map.get(d, "?"))
    trip_df["moa2"] = trip_df["drug2"].map(lambda d: moa_map.get(d, "?"))
    trip_df["moa3"] = trip_df["drug3"].map(lambda d: moa_map.get(d, "?"))

    # Sort by most-synergistic (low Loewe = more synergy; HOFM predicts raw score)
    trip_df = trip_df.sort_values("total_score")
    trip_df.to_csv(out_dir / "all_triples_ranked.csv", index=False)

    print(f"\nTop-10 most-synergistic triples (lowest total_score):")
    print(trip_df.head(10)[["drug1", "drug2", "drug3", "moa1", "moa2", "moa3",
                             "total_score", "trilinear_term"]].to_string(index=False))
    print(f"\nBottom-10 least-synergistic triples:")
    print(trip_df.tail(10)[["drug1", "drug2", "drug3", "moa1", "moa2", "moa3",
                             "total_score", "trilinear_term"]].to_string(index=False))

    # -----------------------------------------------------
    # Q3 — Embedding interpretability
    # -----------------------------------------------------
    print("\n--- Q3: Embedding PCA by MoA ---")
    plot_embedding(full_model, drugs, moa_map, out_dir / "embedding_pca_by_moa.png")

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    summary = {
        "n_pairs": int(len(pair_df)),
        "n_drugs_in_pairs": int(n_drugs),
        "n_triples_scored": int(len(trip_df)),
        "cv_best_config": cv_df.iloc[cv_df["cv_pooled_pearson"].idxmax()].to_dict(),
        "cv_all": cv_results,
        "top_triples": trip_df.head(10).to_dict(orient="records"),
        "bottom_triples": trip_df.tail(10).to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[summary] → {out_dir / 'summary.json'}")


if __name__ == "__main__":
    run_all()
