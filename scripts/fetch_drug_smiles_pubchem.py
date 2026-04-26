#!/usr/bin/env python3
"""Phase 0.2 — Fetch SMILES for all 165 BeatAML drugs from PubChem.

BeatAML drug_id naming convention is "GenericName (synonym/code)", e.g.:
  "17-AAG (Tanespimycin)"
  "Afatinib (BIBW-2992)"
  "Venetoclax"

Strategy:
  1. Parse drug_id → (primary_name, parenthetical_synonym)
  2. Try PubChem name → CID lookup with primary_name
  3. If miss, try parenthetical synonym
  4. From CID, fetch CanonicalSMILES + IsomericSMILES + IUPACName
  5. Cache response so re-runs are idempotent
  6. Output:
     data/canonical/drug_smiles.csv with columns
       beataml_drug_id, primary_name, parenthetical, pubchem_cid,
       canonical_smiles, isomeric_smiles, iupac_name, source

Drugs that fail (e.g., "AKT Inhibitor IV" — no canonical name in PubChem)
are written with empty smiles + source="missing" so the manual-fix step
(Phase 0.3) can target them.

API: PubChem PUG REST is free, no key, but rate-limited at ~5 req/sec.
We sleep 0.25s between calls to stay under the limit.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote


PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
USER_AGENT = "amlcombo-org-research/0.6 (eric@amlcombo.org)"


def _http_get(url: str, timeout: float = 30.0, max_retries: int = 3) -> Optional[bytes]:
    """GET via curl subprocess — uses system proxy that urllib can't reach.

    Returns body bytes on success, None on 404 / non-recoverable failure.
    Retries on transient errors with exponential backoff.
    """
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            cp = subprocess.run(
                ["curl", "-sSL", "--max-time", str(timeout),
                 "-w", "%{http_code}", "-A", USER_AGENT, url],
                capture_output=True, timeout=timeout + 5,
            )
        except subprocess.TimeoutExpired:
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            return None

        out = cp.stdout
        if not out:
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            return None

        # curl writes body then http_code at end; split off the trailing 3 digits
        if len(out) >= 3 and out[-3:].decode("ascii", errors="ignore").isdigit():
            body, code = out[:-3], int(out[-3:])
        else:
            body, code = out, 0

        if code == 200:
            return body
        if code == 404:
            return None
        if code in (429, 503, 0):
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            return None
        return None
    return None


def parse_drug_id(drug_id: str) -> tuple[str, Optional[str]]:
    """Parse 'Name (synonym)' → (Name, synonym). 'Name' → (Name, None)."""
    m = re.match(r"^(.*?)\s*\((.+?)\)\s*$", drug_id)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return drug_id.strip(), None


def name_to_cid(name: str) -> Optional[int]:
    """Lookup PubChem CID by compound name (uses /name endpoint)."""
    url = f"{PUBCHEM_BASE}/compound/name/{quote(name)}/cids/JSON"
    body = _http_get(url)
    if body is None:
        return None
    try:
        d = json.loads(body)
        cids = d.get("IdentifierList", {}).get("CID", [])
        return int(cids[0]) if cids else None
    except (json.JSONDecodeError, KeyError, IndexError, ValueError):
        return None


def cid_to_smiles(cid: int) -> dict:
    """Given a CID, fetch SMILES + ConnectivitySMILES + IUPACName.

    PubChem renamed properties (Q1 2025):
      old IsomericSMILES → SMILES (with stereochemistry)
      old CanonicalSMILES → ConnectivitySMILES (without stereo)
    """
    url = (f"{PUBCHEM_BASE}/compound/cid/{cid}/property/"
           "SMILES,ConnectivitySMILES,IUPACName/JSON")
    body = _http_get(url)
    if body is None:
        return {}
    try:
        d = json.loads(body)
        props = d.get("PropertyTable", {}).get("Properties", [{}])[0]
        return {
            "isomeric_smiles": props.get("SMILES", ""),
            "canonical_smiles": props.get("ConnectivitySMILES", ""),
            "iupac_name": props.get("IUPACName", ""),
        }
    except (json.JSONDecodeError, KeyError, IndexError):
        return {}


def fetch_one(drug_id: str, cache: dict) -> dict:
    """Try (primary, then synonym) → CID → SMILES. Returns row dict."""
    if drug_id in cache:
        return cache[drug_id]

    primary, synonym = parse_drug_id(drug_id)
    row = {
        "beataml_drug_id": drug_id,
        "primary_name": primary,
        "parenthetical": synonym or "",
        "pubchem_cid": "",
        "canonical_smiles": "",
        "isomeric_smiles": "",
        "iupac_name": "",
        "source": "missing",
    }

    cid = name_to_cid(primary)
    used_name = primary
    time.sleep(0.5)
    if cid is None and synonym:
        cid = name_to_cid(synonym)
        used_name = synonym
        time.sleep(0.5)

    if cid is None:
        cache[drug_id] = row
        return row

    row["pubchem_cid"] = str(cid)
    smiles = cid_to_smiles(cid)
    time.sleep(0.5)
    row.update(smiles)
    if row["canonical_smiles"] or row["isomeric_smiles"]:
        row["source"] = f"pubchem (matched on {'primary' if used_name == primary else 'synonym'!r}={used_name!r})"

    cache[drug_id] = row
    return row


def main():
    repo_root = Path(__file__).parents[1]
    drugs_path = repo_root / "data" / "canonical" / "beataml_drug_response_long.csv"
    out_path = repo_root / "data" / "canonical" / "drug_smiles.csv"
    cache_path = repo_root / "data" / "canonical" / "_drug_smiles_cache.json"

    # Load drug list
    print(f"[load] {drugs_path}", flush=True)
    drug_ids = []
    with open(drugs_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row["drug_id"]
            if d not in drug_ids:
                drug_ids.append(d)
    print(f"[load] {len(drug_ids)} unique drugs", flush=True)

    # Load cache (idempotent re-runs)
    cache: dict = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"[cache] {len(cache)} cached entries from {cache_path}", flush=True)

    # Fetch
    rows = []
    n_hit, n_miss = 0, 0
    for i, did in enumerate(sorted(drug_ids)):
        row = fetch_one(did, cache)
        if row["source"] != "missing":
            n_hit += 1
            tag = "✓"
        else:
            n_miss += 1
            tag = "✗"
        smi = row["canonical_smiles"][:40] + ("..." if len(row["canonical_smiles"]) > 40 else "")
        print(f"  [{i+1:3d}/{len(drug_ids)}] {tag} {did:<35s} → {smi}", flush=True)
        rows.append(row)

        # Save cache every 10 drugs (resilience to interruption)
        if (i + 1) % 10 == 0:
            with open(cache_path, "w") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

    # Final cache write
    with open(cache_path, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # Write output CSV
    fieldnames = [
        "beataml_drug_id", "primary_name", "parenthetical",
        "pubchem_cid", "canonical_smiles", "isomeric_smiles",
        "iupac_name", "source",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[write] {out_path}", flush=True)
    print(f"\n=== SUMMARY ===")
    print(f"  Hit:  {n_hit}/{len(drug_ids)}  ({100*n_hit/len(drug_ids):.1f}%)")
    print(f"  Miss: {n_miss}/{len(drug_ids)}")
    if n_miss > 0:
        print(f"\n[next] Phase 0.3: manually fix {n_miss} unmappable drugs in {out_path}")


if __name__ == "__main__":
    main()
