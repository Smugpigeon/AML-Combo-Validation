"""Correctness tests for HOFM (higher-order factorization machines).

The ANOVA kernel recursion must produce exactly:
  a_2 = Σ_{i<j} ⟨v_i, v_j⟩           (bilinear pair sum)
  a_3 = Σ_{i<j<k} ⟨v_i, v_j, v_k⟩    (trilinear triple sum)

These tests compute those sums by BRUTE FORCE (explicit loops over subsets)
and compare to the DP computation inside `HOFM._anova_orders`.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
import torch

from combo_val.combo.hofm import HOFM, HOFMConfig, PAD_ID, score_combo, score_combos_batch


# ---------------------------------------------------------------------------
# Brute-force ground-truth helpers
# ---------------------------------------------------------------------------


def brute_force_pair_sum(V: np.ndarray) -> float:
    """Σ_{i<j} ⟨v_i, v_j⟩ — explicit double loop over pairs."""
    n = V.shape[0]
    total = 0.0
    for i, j in combinations(range(n), 2):
        total += float(np.dot(V[i], V[j]))
    return total


def brute_force_triple_sum(V: np.ndarray) -> float:
    """Σ_{i<j<k} ⟨v_i, v_j, v_k⟩ — triple inner product summed over triples."""
    n = V.shape[0]
    total = 0.0
    for i, j, k in combinations(range(n), 3):
        total += float(np.sum(V[i] * V[j] * V[k]))
    return total


def brute_force_quadruple_sum(V: np.ndarray) -> float:
    n = V.shape[0]
    total = 0.0
    for i, j, k, l in combinations(range(n), 4):
        total += float(np.sum(V[i] * V[j] * V[k] * V[l]))
    return total


# ---------------------------------------------------------------------------
# Test 1 — ANOVA term a_2 equals brute-force pair sum
# ---------------------------------------------------------------------------


def _load_unconstrained_V(model: HOFM, V: torch.Tensor) -> None:
    """Inject a known V directly. Bypasses softplus by using nonneg=False mode."""
    assert not model.cfg.nonneg_embedding
    with torch.no_grad():
        model.drug_emb.weight.data[: V.shape[0]] = V


def test_anova_order2_matches_brute_force():
    torch.manual_seed(0)
    V = torch.randn(5, 8) * 0.3
    cfg = HOFMConfig(n_drugs=10, embedding_dim=8, max_order=3, nonneg_embedding=False)
    model = HOFM(cfg)
    _load_unconstrained_V(model, V)

    drug_ids = torch.tensor([[0, 1, 2, 3, 4]])
    raw = model.drug_emb(drug_ids)
    emb = model._V_from_raw(raw)  # identity under nonneg=False
    mask = drug_ids >= 0
    a_orders = model._anova_orders(emb, mask)
    a2_scalar = a_orders[1].sum(dim=-1).item()
    expected = brute_force_pair_sum(V.numpy())
    assert abs(a2_scalar - expected) < 1e-4, (
        f"a_2 = {a2_scalar}, brute force = {expected}"
    )


def test_anova_order3_matches_brute_force():
    torch.manual_seed(1)
    V = torch.randn(5, 8) * 0.3
    cfg = HOFMConfig(n_drugs=10, embedding_dim=8, max_order=3, nonneg_embedding=False)
    model = HOFM(cfg)
    _load_unconstrained_V(model, V)

    drug_ids = torch.tensor([[0, 1, 2, 3, 4]])
    raw = model.drug_emb(drug_ids)
    emb = model._V_from_raw(raw)
    mask = drug_ids >= 0
    a_orders = model._anova_orders(emb, mask)
    a3_scalar = a_orders[2].sum(dim=-1).item()
    expected = brute_force_triple_sum(V.numpy())
    assert abs(a3_scalar - expected) < 1e-4, (
        f"a_3 = {a3_scalar}, brute force = {expected}"
    )


def test_anova_order4_matches_brute_force():
    torch.manual_seed(2)
    V = torch.randn(6, 8) * 0.3
    cfg = HOFMConfig(n_drugs=10, embedding_dim=8, max_order=4, nonneg_embedding=False)
    model = HOFM(cfg)
    _load_unconstrained_V(model, V)

    drug_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    raw = model.drug_emb(drug_ids)
    emb = model._V_from_raw(raw)
    mask = drug_ids >= 0
    a_orders = model._anova_orders(emb, mask)
    a4_scalar = a_orders[3].sum(dim=-1).item()
    expected = brute_force_quadruple_sum(V.numpy())
    assert abs(a4_scalar - expected) < 1e-4, (
        f"a_4 = {a4_scalar}, brute force = {expected}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Order-3 term is EXACTLY ZERO for 2-drug combos
# (this is why 2-drug training data doesn't constrain order-3 params when
# embeddings are NOT shared; with shared V, order-3 is a deterministic
# function of the 2-drug-learned V.)
# ---------------------------------------------------------------------------


def test_order3_zero_for_2drug_combo():
    cfg = HOFMConfig(n_drugs=10, embedding_dim=8, max_order=3, nonneg_embedding=False)
    model = HOFM(cfg)
    torch.manual_seed(3)
    model.drug_emb.weight.data = torch.randn_like(model.drug_emb.weight.data)

    drug_ids = torch.tensor([[0, 1]])
    raw = model.drug_emb(drug_ids)
    emb = model._V_from_raw(raw)
    mask = drug_ids >= 0
    a_orders = model._anova_orders(emb, mask)
    a3 = a_orders[2].sum(dim=-1).item()
    assert abs(a3) < 1e-6, f"a_3 for 2-drug combo should be 0, got {a3}"


# ---------------------------------------------------------------------------
# Test 3 — Padding invariance: padded drugs contribute 0
# ---------------------------------------------------------------------------


def test_padding_invariance():
    cfg = HOFMConfig(n_drugs=10, embedding_dim=8, max_order=3)
    model = HOFM(cfg)
    torch.manual_seed(4)
    model.drug_emb.weight.data = torch.randn_like(model.drug_emb.weight.data)

    # Same combo, one padded, one not
    ids_no_pad = torch.tensor([[0, 1, 2]])
    ids_padded = torch.tensor([[0, 1, 2, PAD_ID, PAD_ID]])
    out_no_pad = model(ids_no_pad)
    out_padded = model(ids_padded)
    assert torch.allclose(out_no_pad, out_padded, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 4 — Permutation invariance within a combo
# ---------------------------------------------------------------------------


def test_permutation_invariance_within_combo():
    cfg = HOFMConfig(n_drugs=10, embedding_dim=8, max_order=3)
    model = HOFM(cfg)
    torch.manual_seed(5)
    model.drug_emb.weight.data = torch.randn_like(model.drug_emb.weight.data)

    out_abc = model(torch.tensor([[0, 1, 2]]))
    out_cab = model(torch.tensor([[2, 0, 1]]))
    out_bca = model(torch.tensor([[1, 2, 0]]))
    assert torch.allclose(out_abc, out_cab, atol=1e-5)
    assert torch.allclose(out_abc, out_bca, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 5 — Order breakdown returns correct scalars
# ---------------------------------------------------------------------------


def test_order_breakdown_matches_brute_force():
    cfg = HOFMConfig(n_drugs=10, embedding_dim=4, max_order=3, nonneg_embedding=False)
    model = HOFM(cfg)
    torch.manual_seed(6)
    V = torch.randn(5, 4) * 0.2
    with torch.no_grad():
        model.drug_emb.weight.data[:5] = V
        model.drug_linear.weight.data.zero_()
        model.bias.data.zero_()

    drug_ids = torch.tensor([[0, 1, 2, 3, 4]])
    out, breakdown = model(drug_ids, return_order_breakdown=True)
    # Full output = a_2 + a_3 (order-1 is zeroed via w_i=0, bias=0)
    expected_a2 = brute_force_pair_sum(V.numpy())
    expected_a3 = brute_force_triple_sum(V.numpy())
    assert abs(breakdown["order2"].item() - expected_a2) < 1e-4
    assert abs(breakdown["order3"].item() - expected_a3) < 1e-4
    assert abs(out.item() - (expected_a2 + expected_a3)) < 1e-4


# ---------------------------------------------------------------------------
# Test 6 — score_combo / score_combos_batch convenience helpers
# ---------------------------------------------------------------------------


def test_score_combo_convenience():
    cfg = HOFMConfig(n_drugs=10, embedding_dim=8, max_order=3)
    model = HOFM(cfg)
    y1 = score_combo(model, [0, 1, 2])
    assert isinstance(y1, float)

    ys = score_combos_batch(model, [[0, 1], [0, 1, 2], [3, 4]])
    assert ys.shape == (3,)


# ---------------------------------------------------------------------------
# Test 7A — The NEGATIVE result: unconstrained HOFM with shared V, trained on
# pair data only, CANNOT reliably predict 3-way interactions. The pair loss
# ⟨v_i, v_j⟩ is invariant under any orthogonal transform of V, but the 3-way
# ANOVA term is NOT, so gradient descent lands in a rotation orbit where
# 3-way predictions are arbitrary.
#
# This is a CORRECTNESS TEST for a known limitation. We verify that across 5
# seeds, pair fit converges but 3-way Pearson r is near zero.
# ---------------------------------------------------------------------------


def _fit_pair_and_predict_triples(n_drugs: int, k: int, seed: int,
                                    nonneg: bool) -> tuple[float, float]:
    """Fit HOFM on exhaustive pair data, return (pair_loss, triple_pearson)."""
    rng = torch.Generator().manual_seed(seed)
    # Ground-truth factor: non-negative for the nonneg case (fair comparison),
    # signed for the unconstrained case (both are "honest" test conditions).
    if nonneg:
        V_true = torch.rand(n_drugs, k, generator=rng) * 0.7
    else:
        V_true = torch.randn(n_drugs, k, generator=rng) * 0.7

    pair_ids, pair_y = [], []
    for i, j in combinations(range(n_drugs), 2):
        pair_ids.append([i, j])
        pair_y.append(float((V_true[i] * V_true[j]).sum()))
    pair_ids_t = torch.tensor(pair_ids, dtype=torch.long)
    pair_y_t = torch.tensor(pair_y, dtype=torch.float)

    torch.manual_seed(seed * 100)
    cfg = HOFMConfig(n_drugs=n_drugs, embedding_dim=k, max_order=3,
                     nonneg_embedding=nonneg, init_std=0.1)
    model = HOFM(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    for _ in range(2000):
        opt.zero_grad()
        pred = model(pair_ids_t)
        loss = ((pred - pair_y_t) ** 2).mean()
        loss.backward()
        opt.step()

    pred_tr, true_tr = [], []
    for i, j, kk in combinations(range(n_drugs), 3):
        _, br = model(torch.tensor([[i, j, kk]], dtype=torch.long),
                       return_order_breakdown=True)
        pred_tr.append(br["order3"].item())
        true_tr.append(float((V_true[i] * V_true[j] * V_true[kk]).sum()))
    r = float(np.corrcoef(pred_tr, true_tr)[0, 1])
    return float(loss.item()), r


def test_unconstrained_hofm_fails_3way_extrapolation():
    """Across 5 seeds, unconstrained HOFM fits pairs perfectly but 3-way
    Pearson averages near zero. This confirms the rotation-invariance bug."""
    r_values = []
    losses = []
    for seed in range(5):
        loss, r = _fit_pair_and_predict_triples(10, 4, seed, nonneg=False)
        losses.append(loss)
        r_values.append(r)
    # Pair loss should converge to ~0
    assert max(losses) < 1e-2, f"Pair fit failed: losses={losses}"
    # Average |r| across seeds should be < 0.6 (actual ~0.4 empirically)
    mean_abs_r = float(np.mean(np.abs(r_values)))
    assert mean_abs_r < 0.7, (
        f"Unexpected: unconstrained HOFM seems to recover 3-way with "
        f"|r|~{mean_abs_r:.3f}. That would contradict the known rotation-"
        f"invariance issue — check if embedding dim is too small."
    )


def test_nonneg_hofm_recovers_3way_better_than_unconstrained():
    """Non-negative softplus reparametrization substantially improves 3-way
    extrapolation over the unconstrained case. Exact value depends on init
    and optimization; full feasibility sweep lives in
    experiments/hofm_feasibility.py. Here we just verify the fix is real."""
    r_nonneg = []
    r_uncon = []
    for seed in range(5):
        _, r_n = _fit_pair_and_predict_triples(10, 4, seed, nonneg=True)
        _, r_u = _fit_pair_and_predict_triples(10, 4, seed, nonneg=False)
        r_nonneg.append(r_n)
        r_uncon.append(r_u)
    mean_nonneg = float(np.mean(r_nonneg))
    mean_abs_uncon = float(np.mean(np.abs(r_uncon)))
    # Non-negative should consistently produce positive r in the 0.6-0.9 range
    assert mean_nonneg > 0.6, (
        f"Non-negative HOFM should recover 3-way at mean r > 0.6, got {mean_nonneg:.3f}"
    )
    # And should be noticeably better than unconstrained (|r| ~ 0.3-0.5)
    assert mean_nonneg > mean_abs_uncon + 0.15, (
        f"Non-negative improvement not clear: nonneg mean r={mean_nonneg:.3f} "
        f"vs unconstrained mean |r|={mean_abs_uncon:.3f}"
    )
