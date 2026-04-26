"""SMILES → PyTorch-Geometric Data graph featurizer.

Per-atom node features (one-hot + numeric concat, total 39 dim):
  - Element one-hot (16): C, N, O, F, P, S, Cl, Br, I, B, Si, Se, As, Te, H, other
  - Degree one-hot (6): 0-5
  - Formal charge (numeric, 1)
  - Hybridization one-hot (5): SP, SP2, SP3, SP3D, SP3D2
  - Aromatic flag (1)
  - In-ring flag (1)
  - Total H count (numeric, 1)
  - Chirality one-hot (4): None, R, S, other
  - is_donor / is_acceptor heuristic (2): O/N with H attached / O/N without H
  - Mass (numeric, normalized to /12, 1)

Per-bond edge features (one-hot, total 7 dim):
  - Bond type one-hot (4): SINGLE, DOUBLE, TRIPLE, AROMATIC
  - Is conjugated (1)
  - In ring (1)
  - Stereo: only NONE / E / Z used → simplified to "any-stereo" flag (1)

This featurization is a standard "MPNN baseline" — see e.g. Coley et al.
2017 "Convolutional Embedding of Attributed Molecular Graphs", Open
Graph Benchmark (Hu et al. NeurIPS 2020) molhiv featurizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Lazy import — module imports without rdkit available; functions raise.
try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False


_ELEMENT_LIST = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
                  "B", "Si", "Se", "As", "Te", "H"]   # +1 "other" → 16
_DEGREE_LIST = [0, 1, 2, 3, 4, 5]
_HYBRIDIZATION_NAMES = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
_CHIRAL_TAGS = ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW",
                 "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"]


def _one_hot(value, options: list, with_other: bool = True) -> list[float]:
    if value in options:
        idx = options.index(value)
        n = len(options) + (1 if with_other else 0)
        return [1.0 if i == idx else 0.0 for i in range(n)]
    if with_other:
        return [0.0] * len(options) + [1.0]
    return [0.0] * len(options)


def atom_features(atom) -> list[float]:
    """39-dim per-atom feature vector."""
    sym = atom.GetSymbol()
    feats = []
    feats += _one_hot(sym, _ELEMENT_LIST, with_other=True)            # 16
    feats += _one_hot(atom.GetDegree(), _DEGREE_LIST, with_other=False)  # 6
    feats.append(float(atom.GetFormalCharge()))                        # 1
    feats += _one_hot(str(atom.GetHybridization()).split(".")[-1],
                       _HYBRIDIZATION_NAMES, with_other=False)         # 5
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)                 # 1
    feats.append(1.0 if atom.IsInRing() else 0.0)                      # 1
    feats.append(float(atom.GetTotalNumHs()))                          # 1
    feats += _one_hot(str(atom.GetChiralTag()).split(".")[-1],
                       _CHIRAL_TAGS, with_other=False)                 # 4
    n_h = atom.GetTotalNumHs()
    is_donor = (sym in ("N", "O") and n_h > 0)
    is_acceptor = (sym in ("N", "O") and n_h == 0)
    feats.append(1.0 if is_donor else 0.0)                             # 1
    feats.append(1.0 if is_acceptor else 0.0)                          # 1
    feats.append(atom.GetMass() / 12.0)                                # 1 normalized
    return feats


def bond_features(bond) -> list[float]:
    """7-dim per-bond feature vector."""
    bt = str(bond.GetBondType()).split(".")[-1]
    feats = []
    feats += _one_hot(bt, ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"], with_other=False)  # 4
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)               # 1
    feats.append(1.0 if bond.IsInRing() else 0.0)                      # 1
    has_stereo = str(bond.GetStereo()).split(".")[-1] != "STEREONONE"
    feats.append(1.0 if has_stereo else 0.0)                           # 1
    return feats


ATOM_FEATURE_DIM = (
    len(_ELEMENT_LIST) + 1                # 16 element (incl 'other')
    + len(_DEGREE_LIST)                   # 6
    + 1                                   # charge
    + len(_HYBRIDIZATION_NAMES)           # 5
    + 2                                   # aromatic + in-ring
    + 1                                   # H count
    + len(_CHIRAL_TAGS)                   # 4
    + 2                                   # donor + acceptor
    + 1                                   # mass
)
BOND_FEATURE_DIM = 4 + 1 + 1 + 1


@dataclass
class MolGraph:
    """Lightweight container — keeps numpy arrays, converted to torch
    Tensor inside the encoder forward pass. Decouples featurizer from
    torch import."""
    x: np.ndarray         # (n_atoms, ATOM_FEATURE_DIM)
    edge_index: np.ndarray  # (2, n_edges) — directed both ways
    edge_attr: np.ndarray   # (n_edges, BOND_FEATURE_DIM)
    n_atoms: int


def smiles_to_molgraph(smiles: str) -> Optional[MolGraph]:
    """Parse SMILES → MolGraph. Returns None on parse failure."""
    if not _RDKIT_OK:
        raise ImportError("rdkit not installed; pip install rdkit")
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atoms = list(mol.GetAtoms())
    n = len(atoms)
    if n == 0:
        return None
    x = np.array([atom_features(a) for a in atoms], dtype=np.float32)

    edges_src, edges_dst, edge_attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ef = bond_features(bond)
        # Add both directions for undirected message-passing
        edges_src += [i, j]
        edges_dst += [j, i]
        edge_attrs += [ef, ef]
    if not edges_src:
        # single-atom molecule — add self-loop so the GIN doesn't get
        # an empty edge_index
        edges_src = [0]
        edges_dst = [0]
        edge_attrs = [[0.0] * BOND_FEATURE_DIM]
    edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
    edge_attr = np.array(edge_attrs, dtype=np.float32)
    return MolGraph(x=x, edge_index=edge_index, edge_attr=edge_attr, n_atoms=n)


def to_pyg_data(g: MolGraph):
    """Convert MolGraph → torch_geometric.data.Data (lazy torch import)."""
    import torch
    from torch_geometric.data import Data
    return Data(
        x=torch.from_numpy(g.x),
        edge_index=torch.from_numpy(g.edge_index),
        edge_attr=torch.from_numpy(g.edge_attr),
    )
