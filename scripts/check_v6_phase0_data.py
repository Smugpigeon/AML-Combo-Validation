#!/usr/bin/env python3
"""Phase 0.6 — Verify all v0.6 Phase 0 deliverables are coherent.

Checks:
  1. drug_smiles.csv has 165 rows, one per BeatAML drug_id
  2. SMILES coverage rate ≥ 80% (target — manual fixes in Phase 0.3 cover the rest)
  3. RDKit can parse every non-empty SMILES (no malformed)
  4. gene_ppi_subgraph.json has 25 nodes matching BeatAML mut_* features
  5. PPI edges have valid endpoints (both in node list)
  6. (Optional) drugcomb_pairs_pretrain.csv if present has SMILES join coverage

Exit code 0 if all checks pass (with WARN allowed); non-zero if blocking
errors.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def _emit(level: str, msg: str) -> None:
    """Print a check result. level: PASS / WARN / FAIL"""
    print(f"  [{level}] {msg}", flush=True)


def check_smiles(repo_root: Path) -> tuple[int, int]:
    """Returns (n_warn, n_fail)."""
    print(f"\n=== Check 1-3: drug_smiles.csv ===", flush=True)
    n_warn = n_fail = 0
    smiles_path = repo_root / "data" / "canonical" / "drug_smiles.csv"
    if not smiles_path.exists():
        _emit("FAIL", f"{smiles_path} not present — run scripts/fetch_drug_smiles_pubchem.py")
        return 0, 1

    with open(smiles_path) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 165:
        _emit("FAIL", f"Expected 165 rows, got {len(rows)}")
        n_fail += 1
    else:
        _emit("PASS", f"165 rows present")

    n_with_smiles = sum(
        1 for r in rows
        if r.get("canonical_smiles") or r.get("isomeric_smiles")
    )
    coverage = n_with_smiles / max(1, len(rows))
    if coverage >= 0.95:
        _emit("PASS", f"SMILES coverage: {n_with_smiles}/{len(rows)} ({100*coverage:.1f}%)")
    elif coverage >= 0.80:
        _emit("WARN", f"SMILES coverage: {n_with_smiles}/{len(rows)} ({100*coverage:.1f}%) "
                       "— Phase 0.3 manual fix recommended")
        n_warn += 1
    else:
        _emit("FAIL", f"SMILES coverage too low: {100*coverage:.1f}% (target ≥ 80%)")
        n_fail += 1

    # Check RDKit parseability
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        _emit("WARN", "rdkit not installed — skipping SMILES validity check. "
                      "`pip install rdkit` for v0.6 GIN training")
        n_warn += 1
    else:
        n_invalid = 0
        for r in rows:
            smi = r.get("isomeric_smiles") or r.get("canonical_smiles") or ""
            if smi and Chem.MolFromSmiles(smi) is None:
                _emit("WARN", f"Unparseable SMILES for {r['beataml_drug_id']!r}: {smi[:60]}")
                n_invalid += 1
        if n_invalid == 0:
            _emit("PASS", f"All {n_with_smiles} non-empty SMILES are RDKit-parseable")
        else:
            _emit("WARN", f"{n_invalid} SMILES failed RDKit parsing — fix in Phase 0.3")
            n_warn += 1

    return n_warn, n_fail


def check_ppi(repo_root: Path) -> tuple[int, int]:
    """Returns (n_warn, n_fail)."""
    print(f"\n=== Check 4-5: gene_ppi_subgraph.json ===", flush=True)
    n_warn = n_fail = 0
    ppi_path = repo_root / "data" / "canonical" / "gene_ppi_subgraph.json"
    if not ppi_path.exists():
        _emit("FAIL", f"{ppi_path} not present — run scripts/fetch_string_ppi.py")
        return 0, 1

    with open(ppi_path) as f:
        g = json.load(f)

    nodes = g.get("nodes", [])
    edges = g.get("edges", [])

    if len(nodes) == 25:
        _emit("PASS", f"25 nodes in PPI subgraph")
    else:
        _emit("FAIL", f"Expected 25 nodes, got {len(nodes)}")
        n_fail += 1

    # Verify nodes match BeatAML mut_* columns
    features_csv = repo_root / "data" / "canonical" / "beataml_patient_features.csv"
    with open(features_csv) as f:
        header = f.readline().strip().split(",")
    panel_genes = [h[len("mut_"):] for h in header if h.startswith("mut_")]
    if set(nodes) == set(panel_genes):
        _emit("PASS", "PPI nodes match BeatAML mut_* feature columns 1:1")
    else:
        missing = set(panel_genes) - set(nodes)
        extra = set(nodes) - set(panel_genes)
        _emit("FAIL", f"PPI vs panel mismatch: missing {missing}, extra {extra}")
        n_fail += 1

    node_set = set(nodes)
    n_bad_edges = 0
    for e in edges:
        if e["a"] not in node_set or e["b"] not in node_set:
            _emit("WARN", f"Edge {e['a']}-{e['b']} has unknown endpoint")
            n_bad_edges += 1
    if n_bad_edges == 0:
        _emit("PASS", f"All {len(edges)} edges have valid endpoints in node set")
    else:
        n_warn += 1

    density = len(edges) / max(1, len(nodes) * (len(nodes) - 1) / 2)
    if 0.01 <= density <= 0.30:
        _emit("PASS", f"Edge density {density:.3f} within expected range [0.01, 0.30]")
    else:
        _emit("WARN", f"Edge density {density:.3f} unusual — investigate threshold")
        n_warn += 1

    return n_warn, n_fail


def check_drugcomb(repo_root: Path) -> tuple[int, int]:
    """Optional — only checked if file present. Returns (n_warn, n_fail)."""
    print(f"\n=== Check 6: drugcomb_pairs_pretrain.csv (optional) ===", flush=True)
    drugcomb_path = repo_root / "data" / "canonical" / "drugcomb_pairs_pretrain.csv"
    if not drugcomb_path.exists():
        _emit("WARN", f"{drugcomb_path.name} not present — DrugComb pretrain (Phase 4) deferred")
        return 1, 0

    n = sum(1 for _ in open(drugcomb_path)) - 1
    _emit("PASS", f"{n} DrugComb pretraining records present")
    return 0, 0


def main():
    repo_root = Path(__file__).parents[1]
    print(f"v0.6 Phase 0 data integration check  ({repo_root})", flush=True)

    total_warn = total_fail = 0
    for fn in (check_smiles, check_ppi, check_drugcomb):
        w, f = fn(repo_root)
        total_warn += w
        total_fail += f

    print(f"\n=== SUMMARY ===")
    print(f"  WARN: {total_warn}")
    print(f"  FAIL: {total_fail}")
    if total_fail > 0:
        print(f"\n[STATUS] BLOCKING — fix FAILs before proceeding to Phase 1")
        sys.exit(1)
    elif total_warn > 0:
        print(f"\n[STATUS] OK with warnings — Phase 1 may proceed")
        sys.exit(0)
    else:
        print(f"\n[STATUS] CLEAN — Phase 0 deliverables ready for Phase 1")
        sys.exit(0)


if __name__ == "__main__":
    main()
