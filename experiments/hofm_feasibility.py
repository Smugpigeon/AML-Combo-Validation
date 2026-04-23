"""Path D feasibility experiment: can HOFM extrapolate 2-way → 3-way?

Five-phase experimental program:

  Phase A  — IDENTIFIABILITY SWEEP
             Unconstrained vs non-negative HOFM on exact HOFM-generative data.
             Quantifies the rotation-invariance bug and its fix.

  Phase B  — EMBEDDING-DIM SENSITIVITY
             Model k vs true generative k*. Does overspecification help or hurt?

  Phase C  — PAIR-COVERAGE CURVE
             As fraction of pair data shrinks, how does 3-way recovery degrade?

  Phase D  — NOISE ROBUSTNESS
             Gaussian noise on pair training targets, measure 3-way r.

  Phase E  — SPIKE-IN TRIPLES
             Mix a small fraction of simulated triples into training.
             How many triples are needed to restore r→0.95?

All phases output to runs/hofm_feasibility/{phase}_results.csv and figures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from combo_val.combo.hofm import HOFM, HOFMConfig


OUT_DIR = Path("runs/hofm_feasibility")


# ---------------------------------------------------------------------------
# Synthetic data + trainer
# ---------------------------------------------------------------------------


def _make_V_true(n_drugs: int, k: int, nonneg_truth: bool, seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    if nonneg_truth:
        return torch.rand(n_drugs, k, generator=rng) * 0.7
    return torch.randn(n_drugs, k, generator=rng) * 0.4


def _gen_pair_data(V_true: torch.Tensor, pair_coverage: float, noise_std: float,
                    seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exhaustive pair list, optionally subsampled to `pair_coverage` fraction."""
    n_drugs = V_true.shape[0]
    all_pairs = list(combinations(range(n_drugs), 2))
    rng = np.random.default_rng(seed)
    if pair_coverage < 1.0:
        n_keep = max(3, int(len(all_pairs) * pair_coverage))
        idx = rng.choice(len(all_pairs), size=n_keep, replace=False)
        pairs = [all_pairs[i] for i in idx]
    else:
        pairs = all_pairs
    ids = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor(
        [float((V_true[i] * V_true[j]).sum()) for i, j in pairs], dtype=torch.float
    )
    if noise_std > 0:
        torch.manual_seed(seed + 9999)
        y = y + torch.randn_like(y) * noise_std
    return ids, y


def _gen_triple_data(V_true: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    """Exhaustive triple list + ground-truth ⟨v*_i, v*_j, v*_k⟩ values."""
    n_drugs = V_true.shape[0]
    triples = list(combinations(range(n_drugs), 3))
    ids = torch.tensor(triples, dtype=torch.long)
    truth = np.array(
        [float((V_true[i] * V_true[j] * V_true[k]).sum()) for i, j, k in triples]
    )
    return ids, truth


def _train_eval(V_true: torch.Tensor, pair_ids: torch.Tensor, pair_y: torch.Tensor,
                spike_triple_ids: torch.Tensor | None, spike_triple_y: torch.Tensor | None,
                k_model: int, nonneg: bool, init_std: float,
                n_iter: int, lr: float, seed: int) -> dict:
    """Train HOFM on (pairs ∪ optional spike triples); eval 3-way recovery."""
    torch.manual_seed(seed)
    n_drugs = V_true.shape[0]
    cfg = HOFMConfig(n_drugs=n_drugs, embedding_dim=k_model, max_order=3,
                     nonneg_embedding=nonneg, init_std=init_std)
    model = HOFM(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_iter):
        opt.zero_grad()
        pair_pred = model(pair_ids)
        loss = ((pair_pred - pair_y) ** 2).mean()
        if spike_triple_ids is not None:
            tr_pred = model(spike_triple_ids)
            loss = loss + ((tr_pred - spike_triple_y) ** 2).mean()
        loss.backward()
        opt.step()

    model.eval()
    # Evaluate on ALL triples — compare model's order-3 ANOVA component to
    # brute-force ⟨v*_i, v*_j, v*_k⟩.
    trip_ids, trip_truth = _gen_triple_data(V_true)
    with torch.no_grad():
        _, br = model(trip_ids, return_order_breakdown=True)
    pred_triples = br["order3"].cpu().numpy()
    r = float(np.corrcoef(pred_triples, trip_truth)[0, 1])

    # Also measure how well we identified V_true itself (up to permutation +
    # rotation). Use canonical correlation of V_model vs V_true.
    if nonneg:
        V_model = torch.nn.functional.softplus(model.drug_emb.weight.data).cpu().numpy()
    else:
        V_model = model.drug_emb.weight.data.cpu().numpy()

    return {
        "pair_loss": float(((model(pair_ids).detach() - pair_y) ** 2).mean().item()),
        "triple_pearson": r,
        "triple_rmse": float(np.sqrt(np.mean((pred_triples - trip_truth) ** 2))),
        "n_triples_evaluated": int(len(trip_truth)),
    }


# ---------------------------------------------------------------------------
# Phase A — Identifiability sweep: unconstrained vs non-negative
# ---------------------------------------------------------------------------


def phase_A_identifiability_sweep(n_drugs: int = 12, k: int = 4, n_seeds: int = 10,
                                   n_iter: int = 3000) -> pd.DataFrame:
    rows = []
    for mode in ("unconstrained", "nonneg"):
        nonneg = mode == "nonneg"
        for seed in range(n_seeds):
            V_true = _make_V_true(n_drugs, k, nonneg_truth=nonneg, seed=seed)
            pair_ids, pair_y = _gen_pair_data(V_true, pair_coverage=1.0, noise_std=0.0, seed=seed)
            res = _train_eval(V_true, pair_ids, pair_y, None, None,
                              k_model=k, nonneg=nonneg, init_std=0.05,
                              n_iter=n_iter, lr=5e-2, seed=seed)
            rows.append({"mode": mode, "seed": seed, **res})
    df = pd.DataFrame(rows)
    print("[Phase A] Identifiability sweep")
    print(df.groupby("mode")[["pair_loss", "triple_pearson"]].agg(["mean", "std", "min", "max"]))
    return df


# ---------------------------------------------------------------------------
# Phase B — Embedding-dim sensitivity
# ---------------------------------------------------------------------------


def phase_B_dim_sensitivity(n_drugs: int = 12, k_true: int = 4,
                             k_models: tuple = (2, 4, 8, 16, 32),
                             n_seeds: int = 6, n_iter: int = 3000) -> pd.DataFrame:
    rows = []
    for k_m in k_models:
        for seed in range(n_seeds):
            V_true = _make_V_true(n_drugs, k_true, nonneg_truth=True, seed=seed)
            pair_ids, pair_y = _gen_pair_data(V_true, pair_coverage=1.0, noise_std=0.0, seed=seed)
            res = _train_eval(V_true, pair_ids, pair_y, None, None,
                              k_model=k_m, nonneg=True, init_std=0.05,
                              n_iter=n_iter, lr=5e-2, seed=seed + k_m * 100)
            rows.append({"k_true": k_true, "k_model": k_m, "seed": seed, **res})
    df = pd.DataFrame(rows)
    print("[Phase B] Embedding dim sensitivity (non-negative)")
    print(df.groupby("k_model")[["pair_loss", "triple_pearson"]].agg(["mean", "std"]))
    return df


# ---------------------------------------------------------------------------
# Phase C — Pair-coverage curve
# ---------------------------------------------------------------------------


def phase_C_pair_coverage(n_drugs: int = 15, k: int = 4,
                           coverages: tuple = (0.3, 0.5, 0.7, 0.9, 1.0),
                           n_seeds: int = 6, n_iter: int = 3000) -> pd.DataFrame:
    rows = []
    for cov in coverages:
        for seed in range(n_seeds):
            V_true = _make_V_true(n_drugs, k, nonneg_truth=True, seed=seed)
            pair_ids, pair_y = _gen_pair_data(V_true, pair_coverage=cov, noise_std=0.0, seed=seed)
            res = _train_eval(V_true, pair_ids, pair_y, None, None,
                              k_model=k, nonneg=True, init_std=0.05,
                              n_iter=n_iter, lr=5e-2, seed=seed + int(cov * 1000))
            rows.append({"pair_coverage": cov, "n_pairs_used": len(pair_ids),
                         "seed": seed, **res})
    df = pd.DataFrame(rows)
    print("[Phase C] Pair coverage curve")
    print(df.groupby("pair_coverage")[["pair_loss", "triple_pearson"]].agg(["mean", "std"]))
    return df


# ---------------------------------------------------------------------------
# Phase D — Noise robustness
# ---------------------------------------------------------------------------


def phase_D_noise_robustness(n_drugs: int = 12, k: int = 4,
                              noise_levels: tuple = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8),
                              n_seeds: int = 6, n_iter: int = 3000) -> pd.DataFrame:
    rows = []
    for noise in noise_levels:
        for seed in range(n_seeds):
            V_true = _make_V_true(n_drugs, k, nonneg_truth=True, seed=seed)
            pair_ids, pair_y = _gen_pair_data(V_true, pair_coverage=1.0, noise_std=noise, seed=seed)
            res = _train_eval(V_true, pair_ids, pair_y, None, None,
                              k_model=k, nonneg=True, init_std=0.05,
                              n_iter=n_iter, lr=5e-2, seed=seed + int(noise * 1000))
            rows.append({"noise_std": noise, "seed": seed, **res})
    df = pd.DataFrame(rows)
    print("[Phase D] Noise robustness")
    print(df.groupby("noise_std")[["pair_loss", "triple_pearson"]].agg(["mean", "std"]))
    return df


# ---------------------------------------------------------------------------
# Phase E — Spike-in triples
# ---------------------------------------------------------------------------


def phase_E_spike_in_triples(n_drugs: int = 12, k: int = 4,
                              spike_counts: tuple = (0, 3, 10, 30, 60, 100),
                              n_seeds: int = 6, n_iter: int = 3000,
                              mode: str = "unconstrained") -> pd.DataFrame:
    """How many simulated triples unlock 3-way recovery even when nonneg=False?

    Tests the alternative fix: add triple training data."""
    rows = []
    for n_spike in spike_counts:
        for seed in range(n_seeds):
            V_true = _make_V_true(n_drugs, k, nonneg_truth=(mode == "nonneg"), seed=seed)
            pair_ids, pair_y = _gen_pair_data(V_true, 1.0, 0.0, seed)

            # Sample random triples
            all_triples = list(combinations(range(n_drugs), 3))
            if n_spike > 0:
                rng = np.random.default_rng(seed + 2000)
                idx = rng.choice(len(all_triples), size=min(n_spike, len(all_triples)),
                                 replace=False)
                spike_triples = [all_triples[i] for i in idx]
                spike_ids = torch.tensor(spike_triples, dtype=torch.long)
                spike_y = torch.tensor(
                    [float((V_true[i] * V_true[j] * V_true[k_]).sum())
                     for i, j, k_ in spike_triples], dtype=torch.float
                )
            else:
                spike_ids = None
                spike_y = None

            res = _train_eval(V_true, pair_ids, pair_y, spike_ids, spike_y,
                              k_model=k, nonneg=(mode == "nonneg"),
                              init_std=0.05, n_iter=n_iter, lr=5e-2,
                              seed=seed + int(n_spike * 31))
            rows.append({"mode": mode, "n_spike_triples": n_spike,
                         "n_total_triples": len(all_triples),
                         "spike_fraction": n_spike / len(all_triples),
                         "seed": seed, **res})
    df = pd.DataFrame(rows)
    print(f"[Phase E] Spike-in triples ({mode} mode)")
    print(df.groupby("n_spike_triples")[["pair_loss", "triple_pearson"]].agg(["mean", "std"]))
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_phase_A(df_A: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    positions = [0, 1]
    for pos, mode in zip(positions, ["unconstrained", "nonneg"]):
        values = df_A[df_A["mode"] == mode]["triple_pearson"].values
        ax.scatter([pos] * len(values), values, s=50, alpha=0.6,
                   color="#C44E52" if mode == "unconstrained" else "#55A868")
        ax.plot([pos - 0.15, pos + 0.15], [values.mean(), values.mean()],
                color="black", lw=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Unconstrained V", "Non-negative V (softplus)"])
    ax.set_ylabel("Pearson r (3-way predicted vs ground truth)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Phase A — Rotation-invariance bug and its fix\n"
                 "(pair-only training; 12 drugs; 10 seeds)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase_A_identifiability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT_DIR / 'phase_A_identifiability.png'}")


def _plot_phase_B(df_B: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    grouped = df_B.groupby("k_model")["triple_pearson"]
    means = grouped.mean()
    stds = grouped.std()
    ax.errorbar(means.index, means.values, yerr=stds.values, marker="o",
                color="#4C72B0", capsize=4)
    ax.axvline(4, color="gray", linestyle="--", alpha=0.5, label="True k*=4")
    ax.set_xlabel("Model embedding dim k")
    ax.set_ylabel("Pearson r (3-way)")
    ax.set_xscale("log", base=2)
    ax.set_title("Phase B — Model dim sensitivity (non-negative V)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase_B_embedding_dim.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT_DIR / 'phase_B_embedding_dim.png'}")


def _plot_phase_C(df_C: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    grouped = df_C.groupby("pair_coverage")["triple_pearson"]
    means = grouped.mean()
    stds = grouped.std()
    ax.errorbar(means.index, means.values, yerr=stds.values, marker="o",
                color="#55A868", capsize=4)
    ax.set_xlabel("Fraction of all pairs used for training")
    ax.set_ylabel("Pearson r (3-way)")
    ax.set_title("Phase C — Pair coverage (non-negative V)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase_C_pair_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT_DIR / 'phase_C_pair_coverage.png'}")


def _plot_phase_D(df_D: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    grouped = df_D.groupby("noise_std")["triple_pearson"]
    means = grouped.mean()
    stds = grouped.std()
    ax.errorbar(means.index, means.values, yerr=stds.values, marker="o",
                color="#8172B2", capsize=4)
    ax.set_xlabel("Pair-target noise σ")
    ax.set_ylabel("Pearson r (3-way)")
    ax.set_title("Phase D — Noise robustness (non-negative V)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase_D_noise_robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT_DIR / 'phase_D_noise_robustness.png'}")


def _plot_phase_E(df_E_uncon: pd.DataFrame, df_E_nonneg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.2))
    for df_E, mode, color in [(df_E_uncon, "unconstrained", "#C44E52"),
                               (df_E_nonneg, "nonneg", "#55A868")]:
        grouped = df_E.groupby("n_spike_triples")["triple_pearson"]
        means = grouped.mean()
        stds = grouped.std()
        ax.errorbar(means.index, means.values, yerr=stds.values, marker="o",
                    label=mode, color=color, capsize=4)
    ax.set_xlabel("# simulated triples added to training")
    ax.set_ylabel("Pearson r (3-way)")
    ax.set_title("Phase E — Spike-in triples alternative fix")
    ax.set_ylim(-0.8, 1.05)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(title="Parametrization")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase_E_spike_in_triples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT_DIR / 'phase_E_spike_in_triples.png'}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_all(out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    # Faster plots
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

    print("=" * 70)
    print("Phase A — Unconstrained vs non-negative (identifiability)")
    print("=" * 70)
    df_A = phase_A_identifiability_sweep()
    df_A.to_csv(out_dir / "phase_A_results.csv", index=False)
    _plot_phase_A(df_A)

    print("\n" + "=" * 70)
    print("Phase B — Embedding dim sensitivity")
    print("=" * 70)
    df_B = phase_B_dim_sensitivity()
    df_B.to_csv(out_dir / "phase_B_results.csv", index=False)
    _plot_phase_B(df_B)

    print("\n" + "=" * 70)
    print("Phase C — Pair-coverage curve")
    print("=" * 70)
    df_C = phase_C_pair_coverage()
    df_C.to_csv(out_dir / "phase_C_results.csv", index=False)
    _plot_phase_C(df_C)

    print("\n" + "=" * 70)
    print("Phase D — Noise robustness")
    print("=" * 70)
    df_D = phase_D_noise_robustness()
    df_D.to_csv(out_dir / "phase_D_results.csv", index=False)
    _plot_phase_D(df_D)

    print("\n" + "=" * 70)
    print("Phase E — Spike-in triples")
    print("=" * 70)
    df_E_uncon = phase_E_spike_in_triples(mode="unconstrained")
    df_E_uncon.to_csv(out_dir / "phase_E_unconstrained.csv", index=False)
    df_E_nonneg = phase_E_spike_in_triples(mode="nonneg")
    df_E_nonneg.to_csv(out_dir / "phase_E_nonneg.csv", index=False)
    _plot_phase_E(df_E_uncon, df_E_nonneg)

    # Final summary
    summary = {
        "phase_A_unconstrained_mean_r": float(
            df_A[df_A["mode"] == "unconstrained"]["triple_pearson"].mean()
        ),
        "phase_A_unconstrained_std_r": float(
            df_A[df_A["mode"] == "unconstrained"]["triple_pearson"].std()
        ),
        "phase_A_nonneg_mean_r": float(
            df_A[df_A["mode"] == "nonneg"]["triple_pearson"].mean()
        ),
        "phase_A_nonneg_std_r": float(
            df_A[df_A["mode"] == "nonneg"]["triple_pearson"].std()
        ),
        "phase_B_best_k_mean_r": float(
            df_B.groupby("k_model")["triple_pearson"].mean().max()
        ),
        "phase_C_min_coverage_for_r_gt_0.8": float(
            df_C.groupby("pair_coverage")["triple_pearson"].mean().loc[
                lambda s: s > 0.8
            ].index.min() if any(df_C.groupby("pair_coverage")["triple_pearson"].mean() > 0.8)
            else -1
        ),
        "phase_D_r_at_noise_0.2": float(
            df_D[df_D["noise_std"] == 0.2]["triple_pearson"].mean()
        ),
        "phase_E_unconstrained_r_at_30_spikes": float(
            df_E_uncon[df_E_uncon["n_spike_triples"] == 30]["triple_pearson"].mean()
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    run_all()
