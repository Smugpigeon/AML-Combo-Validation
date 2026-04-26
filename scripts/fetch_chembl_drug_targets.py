#!/usr/bin/env python3
"""Phase B.1 — Fetch ChEMBL drug-target binding data for BeatAML 165 + custom drugs.

For each drug name:
  1. Search ChEMBL by molecule synonym → CHEMBL_ID
  2. Pull all bioactivities for that CHEMBL_ID
  3. Filter: standard_type ∈ {IC50, Ki, Kd}, standard_units = nM,
             target.target_type = SINGLE PROTEIN, organism = Homo sapiens
  4. Convert standard_value (nM) → pIC50 = 9 - log10(nM)
  5. Save (drug_name, chembl_id, target_uniprot, target_name, pIC50,
            assay_type, n_assays_supporting)

Output: data/canonical/chembl_drug_targets.csv

Then a separate step (B.2) maps target_uniprot → 18 taxonomy axes via
chembl_target_to_taxonomy.yaml + computes per-drug 18-dim coverage vector.

Usage:
  python scripts/fetch_chembl_drug_targets.py
    [--drug-list data/canonical/drug_smiles.csv]
    [--out data/canonical/chembl_drug_targets.csv]
    [--limit-bioactivities 200]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlencode


CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
USER_AGENT = "amlcombo-research/0.6 (eric@amlcombo.org)"


def _http_get(url: str, timeout: float = 30.0, max_retries: int = 3) -> bytes | None:
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            cp = subprocess.run(
                ["curl", "-sSL", "--max-time", str(timeout),
                 "-H", "Accept: application/json",
                 "-A", USER_AGENT, url],
                capture_output=True, timeout=timeout + 5,
            )
        except subprocess.TimeoutExpired:
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            return None
        if cp.returncode == 0 and cp.stdout:
            return cp.stdout
        if attempt < max_retries:
            time.sleep(delay)
            delay *= 2
    return None


def search_chembl_id(drug_name: str) -> str | None:
    """Return first CHEMBL_ID matching drug_name as a synonym."""
    url = (f"{CHEMBL_BASE}/molecule.json"
            f"?molecule_synonyms__synonyms__iexact={quote(drug_name)}&limit=1")
    body = _http_get(url)
    if body is None:
        return None
    try:
        d = json.loads(body)
        molecules = d.get("molecules", [])
        return molecules[0]["molecule_chembl_id"] if molecules else None
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def fetch_bioactivities(chembl_id: str, max_records: int = 200) -> list[dict]:
    """Fetch bioactivities for a CHEMBL_ID (single-protein human IC50/Ki/Kd
    in nM). Returns list of dicts with target + value normalized to pIC50.
    """
    url = (f"{CHEMBL_BASE}/activity.json"
            f"?molecule_chembl_id={chembl_id}"
            f"&standard_type__in=IC50,Ki,Kd"
            f"&standard_units=nM"
            f"&target_organism=Homo+sapiens"
            f"&limit={max_records}")
    body = _http_get(url, timeout=60.0)
    if body is None:
        return []
    try:
        d = json.loads(body)
    except json.JSONDecodeError:
        return []

    out = []
    for act in d.get("activities", []):
        val = act.get("standard_value")
        if val is None:
            continue
        try:
            nM = float(val)
        except (TypeError, ValueError):
            continue
        if nM <= 0 or nM > 1e6:    # skip absurd values
            continue
        pIC50 = 9.0 - math.log10(nM)
        out.append({
            "target_chembl_id": act.get("target_chembl_id", ""),
            "target_name": act.get("target_pref_name", ""),
            "standard_type": act.get("standard_type", ""),
            "standard_value_nM": nM,
            "pIC50": round(pIC50, 2),
            "assay_chembl_id": act.get("assay_chembl_id", ""),
        })
    return out


def fetch_target_uniprot(target_chembl_id: str) -> str | None:
    """For SINGLE PROTEIN targets, fetch UniProt accession."""
    url = f"{CHEMBL_BASE}/target/{target_chembl_id}.json"
    body = _http_get(url)
    if body is None:
        return None
    try:
        d = json.loads(body)
        if d.get("target_type") != "SINGLE PROTEIN":
            return None
        comps = d.get("target_components", [])
        if not comps:
            return None
        return comps[0].get("accession")
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug-list", type=Path,
                     default=Path("data/canonical/drug_smiles.csv"))
    ap.add_argument("--out", type=Path,
                     default=Path("data/canonical/chembl_drug_targets.csv"))
    ap.add_argument("--limit-bioactivities", type=int, default=200)
    ap.add_argument("--max-drugs", type=int, default=None,
                     help="Process only first N drugs (debugging).")
    args = ap.parse_args()

    repo_root = Path(__file__).parents[1]
    drug_csv = args.drug_list if args.drug_list.is_absolute() else repo_root / args.drug_list
    out_csv = args.out if args.out.is_absolute() else repo_root / args.out
    cache_path = out_csv.with_suffix(".cache.json")

    # Load drug list (BeatAML drug ID + primary_name + parenthetical synonym)
    drugs = []
    with open(drug_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            drugs.append({
                "id": r["beataml_drug_id"],
                "names": [n for n in [r.get("primary_name"),
                                        r.get("parenthetical")] if n],
            })
    if args.max_drugs:
        drugs = drugs[:args.max_drugs]
    print(f"[load] {len(drugs)} drugs from {drug_csv}", flush=True)

    # Resume cache
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"[cache] {len(cache)} drugs already processed", flush=True)

    # UniProt lookup cache (target_chembl_id → uniprot)
    uniprot_cache = cache.get("_uniprot_cache", {})

    out_rows = []
    for i, drug in enumerate(drugs):
        did = drug["id"]
        if did in cache and "bioactivities" in cache[did]:
            # Already fetched
            for b in cache[did]["bioactivities"]:
                out_rows.append({**b, "beataml_drug_id": did,
                                 "chembl_id": cache[did].get("chembl_id", "")})
            continue

        chembl_id = None
        for name in drug["names"]:
            chembl_id = search_chembl_id(name)
            if chembl_id:
                break
            time.sleep(0.3)

        if chembl_id is None:
            cache[did] = {"chembl_id": None, "bioactivities": []}
            tag = "✗ no chembl_id"
            print(f"  [{i+1:3d}/{len(drugs)}] {tag:<24s} {did}", flush=True)
            continue

        time.sleep(0.3)
        bios = fetch_bioactivities(chembl_id, args.limit_bioactivities)
        # Resolve UniProt for unique target_chembl_ids
        for b in bios:
            tcid = b["target_chembl_id"]
            if tcid not in uniprot_cache:
                uniprot_cache[tcid] = fetch_target_uniprot(tcid)
                time.sleep(0.3)
            b["target_uniprot"] = uniprot_cache[tcid]

        cache[did] = {"chembl_id": chembl_id, "bioactivities": bios}
        cache["_uniprot_cache"] = uniprot_cache

        n_with_uniprot = sum(1 for b in bios if b.get("target_uniprot"))
        print(f"  [{i+1:3d}/{len(drugs)}] ✓ {chembl_id:<14s} "
               f"{len(bios)} bioactivities ({n_with_uniprot} w/ UniProt)  {did}", flush=True)

        for b in bios:
            out_rows.append({**b, "beataml_drug_id": did, "chembl_id": chembl_id})

        # Save cache every 5 drugs
        if (i + 1) % 5 == 0:
            with open(cache_path, "w") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

    # Final cache + output write
    with open(cache_path, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    fieldnames = ["beataml_drug_id", "chembl_id", "target_chembl_id",
                   "target_uniprot", "target_name",
                   "standard_type", "standard_value_nM", "pIC50",
                   "assay_chembl_id"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n[write] {out_csv}  ({len(out_rows)} (drug, target) records)")
    n_drugs_with_data = sum(1 for d in cache.values()
                              if isinstance(d, dict) and d.get("bioactivities"))
    print(f"[stats] {n_drugs_with_data}/{len(drugs)} drugs have ChEMBL data")


if __name__ == "__main__":
    main()
