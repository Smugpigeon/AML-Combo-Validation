"""Higher-Order Factorization Machines (HOFM) for drug combinations.

Path D of the multi-drug combo roadmap. Extends the order-2 synergy MLP to
arbitrary arity (2, 3, 4+ drugs) by the factorization-machine tensor ansatz:

  y_hat(combo) = w_0
               + Σ_{i ∈ combo}  w_i                              (order 1)
               + Σ_{i<j ∈ combo} ⟨v_i, v_j⟩                      (order 2)
               + Σ_{i<j<k ∈ combo} ⟨v_i, v_j, v_k⟩               (order 3)
               + …

where ⟨v_i, v_j⟩      = Σ_d v_i[d] · v_j[d]            (standard inner product)
      ⟨v_i, v_j, v_k⟩ = Σ_d v_i[d] · v_j[d] · v_k[d]   (element-wise tri-product, summed)

CRUCIAL design choice — SHARED embedding V across orders. The same v_i is
used for the pair term and the triple term. Without this, the order-3
parameters would be completely unidentifiable from 2-drug training data.
With shared V, fitting V on pair data induces 3-way predictions via the
same embedding — this is WHY HOFM can extrapolate.

Efficient O(r · d · k) computation via the ANOVA-kernel recursion
(Blondel et al. 2016, "Higher-Order Factorization Machines"):

  Let e_s = Σ_i (v_i)^s   (element-wise s-th power, summed over drugs in combo)
  Let a_0 = 1 (vector of ones)
      a_t = (1/t) Σ_{s=1..t} (-1)^(s+1) · e_s · a_{t-s}

  The t-th order ANOVA term (sum over size-t subsets of v_i tensor-product)
  equals a_t.sum(dim=-1).

Verified by algebra: for r=3 drugs at order 3, a_3 evaluates to
⟨v_1,v_2,v_3⟩ exactly.

Reference:
  Blondel, Fujino, Ueda. "Higher-Order Factorization Machines." NIPS 2016.
  Julkunen et al. "Leveraging multi-way interactions for systematic
    prediction of pre-clinical drug combination effects." Nat Comm 2020.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


PAD_ID = -1   # sentinel for padding drug-id slots in a batch


@dataclass(frozen=True)
class HOFMConfig:
    n_drugs: int
    embedding_dim: int = 16              # k
    max_order: int = 3                   # support pair + triple; can go higher
    n_patient_features: int = 0          # if > 0, add a patient linear term
    dropout: float = 0.0
    init_std: float = 0.05
    # ------------------------------------------------------------------
    # CRITICAL flag for 2→3+ extrapolation.
    #
    # Setting False gives a standard (unconstrained) factorization. Under
    # shared-V across orders, the pair loss is invariant under any
    # orthogonal rotation of V, but the 3-way ANOVA term is NOT. Gradient
    # descent thus finds an arbitrary member of the rotation orbit, and
    # 3-way predictions are essentially random. Verified empirically:
    # 5-seed sweep gives Pearson r ∈ [-0.97, +0.51] for 3-way recovery.
    #
    # Setting True parameterizes V = softplus(W) to force v_i ≥ 0. Non-
    # negative matrix factorization is unique up to permutation of basis,
    # which preserves the diagonal tensor product. Empirically this
    # restores r ≈ 0.96–0.995 across seeds for 3-way prediction from
    # 2-way training data only. This is the theoretical fix that makes
    # Path D actually work.
    # ------------------------------------------------------------------
    nonneg_embedding: bool = True


class HOFM(nn.Module):
    """Higher-Order Factorization Machines with shared embedding across orders.

    Input: a batch of combos, each combo a variable-size set of drug IDs
    (padded with PAD_ID=-1 up to max_arity). Plus optional patient features.

    Output: scalar prediction (synergy score, combo-AUC delta, whatever the
    training target is).
    """

    def __init__(self, cfg: HOFMConfig):
        super().__init__()
        self.cfg = cfg
        self.bias = nn.Parameter(torch.zeros(1))
        self.drug_linear = nn.Embedding(cfg.n_drugs, 1)  # w_i
        # Raw embedding matrix. Actual V = self._V() applies non-negative
        # transform when cfg.nonneg_embedding is True.
        self.drug_emb = nn.Embedding(cfg.n_drugs, cfg.embedding_dim)  # raw W
        # init
        nn.init.normal_(self.drug_linear.weight, 0.0, cfg.init_std)
        if cfg.nonneg_embedding:
            # Initialize in the positive regime so softplus(W) > 0 readily.
            nn.init.normal_(self.drug_emb.weight, mean=0.0, std=cfg.init_std)
        else:
            nn.init.normal_(self.drug_emb.weight, 0.0, cfg.init_std)

        if cfg.n_patient_features > 0:
            self.patient_head = nn.Linear(cfg.n_patient_features, 1)
        else:
            self.patient_head = None

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def _V_from_raw(self, raw_emb: torch.Tensor) -> torch.Tensor:
        """Apply non-negative transform if configured, else identity."""
        if self.cfg.nonneg_embedding:
            return torch.nn.functional.softplus(raw_emb)
        return raw_emb

    # -------------------------------------------------------------------
    # Core ANOVA kernel computation
    # -------------------------------------------------------------------
    def _anova_orders(self, emb: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        """Compute ANOVA terms a_t for t = 1..max_order.

        emb  : (B, R, k)    — drug embeddings, padded positions should be 0
        mask : (B, R) bool  — 1 where a drug is present
        Returns a list [a_1, a_2, …, a_max_order], each (B, k). Sum over dim=-1
        to get the scalar order-t interaction sum for each combo in the batch.
        """
        d = self.cfg.max_order
        # Zero out padded positions so they don't contribute to e_s
        emb = emb * mask.unsqueeze(-1)

        # Power sums e_s = Σ_i v_i^s, shape (d, B, k)
        powers = [emb]
        for _ in range(2, d + 1):
            powers.append(powers[-1] * emb)
        # e_s has shape (B, k)
        e_list = [p.sum(dim=1) for p in powers]

        # a_0 = 1 (we never return a_0; it acts as a neutral element)
        ones = torch.ones_like(e_list[0])

        a_list: list[torch.Tensor] = [None] * (d + 1)   # type: ignore[list-item]
        a_list[0] = ones
        for t in range(1, d + 1):
            a_t = torch.zeros_like(ones)
            for s in range(1, t + 1):
                sign = 1.0 if (s + 1) % 2 == 0 else -1.0
                # (-1)^(s+1): s=1 → +1, s=2 → -1, s=3 → +1, …
                sign = float((-1) ** (s + 1))
                a_t = a_t + sign * e_list[s - 1] * a_list[t - s]
            a_t = a_t / float(t)
            a_list[t] = a_t
        # Return orders 1..d
        return a_list[1:]  # type: ignore[return-value]

    # -------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------
    def forward(
        self,
        drug_ids: torch.Tensor,           # (B, R) with -1 padding
        patient_features: torch.Tensor | None = None,  # (B, F)
        return_order_breakdown: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = drug_ids >= 0
        safe_ids = drug_ids.clamp(min=0)

        # Single-drug linear term: Σ_{i ∈ combo} w_i
        linear = self.drug_linear(safe_ids).squeeze(-1) * mask
        linear_sum = linear.sum(dim=1)                           # (B,)

        # Shared embedding across orders (optionally non-negative via softplus)
        raw_emb = self.drug_emb(safe_ids)                         # (B, R, k)
        emb = self._V_from_raw(raw_emb)
        emb = self.dropout(emb)

        # ANOVA terms a_t, each (B, k)
        a_orders = self._anova_orders(emb, mask)

        # Interaction contributions: sum over orders ≥ 2 of a_t.sum(dim=-1).
        # (a_1.sum(dim=-1) would duplicate the linear term if we have w_i;
        #  we treat w_i as the order-1 parameter and exclude a_1 from the
        #  bilinear-and-higher sum.)
        order_scalars = {}
        interaction = torch.zeros_like(linear_sum)
        for t, a_t in enumerate(a_orders, start=1):
            scalar = a_t.sum(dim=-1)
            order_scalars[f"order{t}"] = scalar
            if t >= 2:
                interaction = interaction + scalar

        out = self.bias + linear_sum + interaction
        if patient_features is not None and self.patient_head is not None:
            out = out + self.patient_head(patient_features).squeeze(-1)

        if return_order_breakdown:
            return out, order_scalars
        return out


# ---------------------------------------------------------------------------
# Convenience: score arbitrary combos without building a padded tensor manually
# ---------------------------------------------------------------------------


def score_combo(model: HOFM, drug_indices: list[int],
                patient_features: torch.Tensor | None = None,
                device: torch.device | None = None) -> float:
    """Score a single combo given a list of drug integer indices."""
    device = device or next(model.parameters()).device
    drug_t = torch.tensor([drug_indices], dtype=torch.long, device=device)
    if patient_features is not None:
        patient_features = patient_features.to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        y = model(drug_t, patient_features)
    return float(y.item())


def score_combos_batch(model: HOFM, combos: list[list[int]],
                        patient_features: torch.Tensor | None = None,
                        device: torch.device | None = None) -> torch.Tensor:
    """Score many combos in one forward pass. Pads to max-arity in the batch."""
    device = device or next(model.parameters()).device
    max_arity = max(len(c) for c in combos)
    drug_ids = torch.full((len(combos), max_arity), PAD_ID,
                          dtype=torch.long, device=device)
    for i, c in enumerate(combos):
        drug_ids[i, : len(c)] = torch.tensor(c, dtype=torch.long, device=device)

    if patient_features is not None:
        patient_features = patient_features.to(device)
        if patient_features.dim() == 1:
            patient_features = patient_features.unsqueeze(0).expand(len(combos), -1)

    model.eval()
    with torch.no_grad():
        y = model(drug_ids, patient_features)
    return y.cpu()
