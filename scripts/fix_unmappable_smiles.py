#!/usr/bin/env python3
"""Phase 0.3 — Manually fix the 6 unmappable BeatAML drugs.

After Phase 0.2 PubChem fetch, 6 drugs failed name resolution:
  - AT-101              → known synonym "Gossypol" (BCL2/MCL1 inhibitor)
  - Baiclein            → typo of "Baicalein" (flavone)
  - MEK1/2 Inhibitor    → generic placeholder; assign canonical U0126 (judgment)
  - SB-202190           → PubChem name lookup misses dash variant; try "SB202190"
  - SB-203580           → same; try "SB203580"
  - Vargetef            → synonym of "Nintedanib" (BIBF-1120) — anti-angiogenic

For each, hit PubChem with the corrected name. If still missing, leave
the row blank with source='manual-unresolved' so the GIN encoder
zero-vectors that drug at inference time.

Updates `data/canonical/drug_smiles.csv` IN PLACE; backs up to
`drug_smiles.csv.bak` first.
"""

from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_drug_smiles_pubchem import name_to_cid, cid_to_smiles


MANUAL_NAME_FIXES = {
    "AT-101": "Gossypol",
    "Baiclein": "Baicalein",
    "MEK1/2 Inhibitor": "U0126",   # judgment call — most common MEK1/2i
    "SB-202190": "SB202190",
    "SB-203580": "SB203580",
    "Vargetef": "Nintedanib",
}


def main():
    repo_root = Path(__file__).parents[1]
    csv_path = repo_root / "data" / "canonical" / "drug_smiles.csv"
    bak_path = csv_path.with_suffix(".csv.bak")

    print(f"[backup] {csv_path} → {bak_path}")
    shutil.copy(csv_path, bak_path)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    n_fixed = 0
    n_unresolved = 0
    for row in rows:
        beataml_id = row["beataml_drug_id"]
        if beataml_id not in MANUAL_NAME_FIXES:
            continue
        new_name = MANUAL_NAME_FIXES[beataml_id]
        print(f"[fix] {beataml_id!r}: try '{new_name}' ...", flush=True)
        cid = name_to_cid(new_name)
        time.sleep(0.5)
        if cid is None:
            print(f"  ✗ STILL UNMAPPED — manual SMILES required", flush=True)
            row["source"] = "manual-unresolved"
            n_unresolved += 1
            continue
        smiles = cid_to_smiles(cid)
        time.sleep(0.5)
        if not (smiles.get("isomeric_smiles") or smiles.get("canonical_smiles")):
            print(f"  ✗ CID found ({cid}) but no SMILES", flush=True)
            row["source"] = "manual-unresolved"
            n_unresolved += 1
            continue

        row["pubchem_cid"] = str(cid)
        row["isomeric_smiles"] = smiles.get("isomeric_smiles", "")
        row["canonical_smiles"] = smiles.get("canonical_smiles", "")
        row["iupac_name"] = smiles.get("iupac_name", "")
        row["source"] = f"manual-fix (alias={new_name!r}, cid={cid})"
        smi_short = (row["isomeric_smiles"][:60] +
                      ("..." if len(row["isomeric_smiles"]) > 60 else ""))
        print(f"  ✓ {smi_short}", flush=True)
        n_fixed += 1

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_with_smiles = sum(1 for r in rows
                          if r.get("canonical_smiles") or r.get("isomeric_smiles"))
    print(f"\n=== SUMMARY ===")
    print(f"  Fixed:      {n_fixed}/6")
    print(f"  Unresolved: {n_unresolved}/6")
    print(f"  Total with SMILES: {n_with_smiles}/{len(rows)}  "
           f"({100 * n_with_smiles / len(rows):.1f}%)")


if __name__ == "__main__":
    main()
