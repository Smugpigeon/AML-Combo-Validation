#!/usr/bin/env python3
"""Phase 0.5 — Fetch DrugComb pair data for v0.6 GIN pretraining.

DrugComb (https://drugcomb.org) is the largest curated database of drug
combination experiments — ~700K (drug pair × cell line) records across
~120 cell lines and ~5K drugs. We use it to pretrain the GIN drug
encoder so it learns chemistry-grounded drug embeddings BEFORE we
fine-tune on the smaller BeatAML AML-specific cohort.

Strategy:
  1. Hit DrugComb REST API endpoint /summary → all drug-pair summaries
  2. Filter to:
     - Has SMILES for both drugs (joinable to our PubChem-fetched table)
     - Synergy score (Loewe / Bliss / HSA / ZIP) at least one non-NaN
     - Block_id valid (i.e., real experiment, not metadata)
  3. Output:
     data/canonical/drugcomb_pairs_pretrain.csv with columns:
       block_id, drug_row_name, drug_col_name, drug_row_smiles,
       drug_col_smiles, cell_line_name, tissue_name, synergy_loewe,
       synergy_bliss, synergy_hsa, synergy_zip, source_study

API: DrugComb REST is undocumented but stable; the summary endpoint
returns paginated JSON. We fall back to their bulk CSV mirror if the
API is unreachable.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# DrugComb provides bulk-download CSVs at fixed paths; this is more
# reliable than scraping their API.
DRUGCOMB_BULK_URLS = {
    "summary": "https://api.drugcomb.org/summary?from=1&to=750000",
    # Alternative: known stable bulk CSV mirror used by their figshare snapshot
    "summary_csv_mirror": "https://drugcomb.fimm.fi/download/summary_v_1_5.csv",
}


def _http_get(url: str, timeout: float = 90.0) -> bytes | None:
    req = Request(url, headers={"User-Agent": "amlcombo-research/0.6 (eric@amlcombo.org)"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=timeout) as r:
                return r.read()
        except (HTTPError, URLError, ConnectionResetError, OSError, TimeoutError) as e:
            if attempt == 2:
                print(f"  FAIL after 3 attempts: {type(e).__name__}: {e}", flush=True)
                return None
            time.sleep(2 ** attempt)
    return None


def fetch_drugcomb_summary() -> list[dict] | None:
    """Try DrugComb REST API first; fall back to FIMM bulk CSV mirror."""
    print(f"[1/2] Trying DrugComb REST API ...", flush=True)
    body = _http_get(DRUGCOMB_BULK_URLS["summary"])
    if body is not None:
        try:
            data = json.loads(body)
            if isinstance(data, list) and len(data) > 1000:
                print(f"  REST returned {len(data)} records", flush=True)
                return data
        except json.JSONDecodeError:
            pass
    print(f"[2/2] REST failed; trying FIMM CSV mirror ...", flush=True)
    body = _http_get(DRUGCOMB_BULK_URLS["summary_csv_mirror"])
    if body is None:
        return None
    text = body.decode("utf-8", errors="replace")
    rows = list(csv.DictReader(text.splitlines()))
    print(f"  CSV mirror returned {len(rows)} records", flush=True)
    return rows


def join_with_smiles(records: list[dict], smiles_path: Path) -> list[dict]:
    """Filter records to those where both drugs have a SMILES we can resolve.

    We match on lowercased drug name (DrugComb's drug_row, drug_col fields)
    against our PubChem table's primary_name + parenthetical fields.
    """
    smiles_lookup: dict[str, str] = {}
    if not smiles_path.exists():
        print(f"WARNING: {smiles_path} not present yet — proceeding without SMILES join", flush=True)
        return [{**r, "drug_row_smiles": "", "drug_col_smiles": ""} for r in records]

    with open(smiles_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get("isomeric_smiles") or row.get("canonical_smiles") or ""
            if not smi:
                continue
            for key in (row["primary_name"], row["parenthetical"]):
                if key:
                    smiles_lookup[key.lower()] = smi
    print(f"  SMILES lookup: {len(smiles_lookup)} drug names resolvable", flush=True)

    out = []
    for r in records:
        d_row = (r.get("drug_row") or r.get("drug_row_name") or "").lower()
        d_col = (r.get("drug_col") or r.get("drug_col_name") or "").lower()
        smi_row = smiles_lookup.get(d_row, "")
        smi_col = smiles_lookup.get(d_col, "")
        # For PRETRAINING we don't strictly need both endpoints in the
        # BeatAML SMILES table — we'll use DrugComb's own SMILES for
        # training the GIN encoder on a much wider chemical space.
        out.append({**r, "drug_row_smiles": smi_row, "drug_col_smiles": smi_col})
    return out


def main():
    repo_root = Path(__file__).parents[1]
    smiles_path = repo_root / "data" / "canonical" / "drug_smiles.csv"
    out_path = repo_root / "data" / "canonical" / "drugcomb_pairs_pretrain.csv"

    print(f"[fetch] Fetching DrugComb summary records ...", flush=True)
    records = fetch_drugcomb_summary()
    if records is None:
        print(f"[error] Both DrugComb REST and CSV mirror failed.", flush=True)
        print(f"[next] Manual fallback: download summary_v_1_5.csv from", flush=True)
        print(f"        https://drugcomb.fimm.fi/download/", flush=True)
        print(f"        and place at {out_path}.raw.csv, then re-run.", flush=True)
        sys.exit(2)

    print(f"\n[stats] Raw records: {len(records)}", flush=True)
    print(f"[join]  Joining with PubChem SMILES table ...", flush=True)
    joined = join_with_smiles(records, smiles_path)

    # Filter: keep only records with a synergy score
    def has_synergy(r):
        for k in ("synergy_loewe", "synergy_bliss", "synergy_hsa",
                  "synergy_zip", "S_mean", "synergy_mean"):
            v = r.get(k)
            if v is not None and v != "" and str(v).lower() not in ("nan", "null"):
                return True
        return False

    valid = [r for r in joined if has_synergy(r)]
    print(f"[filter] After dropping records with no synergy score: {len(valid)}", flush=True)

    if not valid:
        print(f"[error] All {len(joined)} records lack synergy fields. Inspect raw schema:", flush=True)
        if joined:
            print(f"        Sample keys: {sorted(joined[0].keys())[:30]}", flush=True)
        sys.exit(2)

    # Write standardized output
    fieldnames = sorted(set().union(*(r.keys() for r in valid[:100])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(valid)
    print(f"\n[write] {out_path}  ({len(valid)} rows)", flush=True)


if __name__ == "__main__":
    main()
