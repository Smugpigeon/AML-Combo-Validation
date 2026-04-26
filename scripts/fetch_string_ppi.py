#!/usr/bin/env python3
"""Phase 0.4 — Fetch STRING PPI subgraph for the 25 driver genes.

The 25-gene panel comes from `data/canonical/beataml_patient_features.csv`
columns matching `mut_<GENE>`. We use STRING v12 REST API to pull all
high-confidence (combined_score >= 700) protein-protein interactions
where BOTH endpoints are in our panel.

Output: data/canonical/gene_ppi_subgraph.json with format:
  {
    "schema_version": "1.0",
    "string_version": "12.0",
    "score_threshold": 700,
    "nodes": ["FLT3", "NPM1", ...],         (25 genes, in stable order)
    "edges": [
      {"a": "FLT3", "b": "NPM1", "combined_score": 850, "experimental_score": 600, ...},
      ...
    ],
    "node_features": {                       (optional STRING-derived)
      "FLT3": {"protein_size": 993, ...},
      ...
    }
  }

API: STRING REST (https://string-db.org/api/) — no key required.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


STRING_BASE = "https://string-db.org/api"
USER_AGENT = "amlcombo-org-research/0.6 (eric@amlcombo.org)"
SPECIES_HUMAN = 9606


def _http_get(url: str, timeout: float = 30.0) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def get_panel_genes(features_csv: Path) -> list[str]:
    with open(features_csv) as f:
        header = f.readline().strip().split(",")
    genes = [h[len("mut_"):] for h in header if h.startswith("mut_")]
    return genes


def map_to_string_ids(symbols: list[str]) -> dict[str, dict]:
    """STRING /get_string_ids — official identifier resolution."""
    params = {
        "identifiers": "%0d".join(symbols),
        "species": SPECIES_HUMAN,
        "limit": 1,
        "echo_query": 1,
    }
    url = f"{STRING_BASE}/json/get_string_ids?{urlencode(params, safe='%')}"
    body = _http_get(url)
    arr = json.loads(body)
    out = {}
    for entry in arr:
        symbol = entry.get("queryItem")
        if symbol:
            out[symbol] = {
                "string_id": entry.get("stringId"),
                "preferred_name": entry.get("preferredName"),
                "ncbi_taxon_id": entry.get("ncbiTaxonId"),
                "annotation": entry.get("annotation", "")[:200],
            }
    return out


def fetch_network(string_ids: list[str], score_threshold: int = 700) -> list[dict]:
    """STRING /network — pull all interactions among the input set."""
    params = {
        "identifiers": "%0d".join(string_ids),
        "species": SPECIES_HUMAN,
        "required_score": score_threshold,
        "network_type": "physical",   # protein-protein binding/interaction (excludes co-expression-only links)
    }
    url = f"{STRING_BASE}/json/network?{urlencode(params, safe='%')}"
    body = _http_get(url)
    return json.loads(body)


def main():
    repo_root = Path(__file__).parents[1]
    features_csv = repo_root / "data" / "canonical" / "beataml_patient_features.csv"
    out_path = repo_root / "data" / "canonical" / "gene_ppi_subgraph.json"

    genes = get_panel_genes(features_csv)
    print(f"[panel] {len(genes)} driver genes: {genes}", flush=True)

    print(f"[string] Resolving {len(genes)} symbols → STRING IDs ...", flush=True)
    id_map = map_to_string_ids(genes)
    print(f"[string]   resolved {len(id_map)}/{len(genes)}", flush=True)
    for sym in genes:
        if sym not in id_map:
            print(f"  WARNING: STRING could not resolve {sym!r}", flush=True)

    string_ids = [m["string_id"] for m in id_map.values() if m.get("string_id")]
    print(f"[string] Fetching network for {len(string_ids)} resolved IDs ...", flush=True)
    raw_edges = fetch_network(string_ids, score_threshold=700)
    print(f"[string]   got {len(raw_edges)} edges", flush=True)

    # Build symmetric edge list keyed by HUGO symbol
    string_to_symbol = {m["string_id"]: sym for sym, m in id_map.items() if m.get("string_id")}
    seen = set()
    edges = []
    for e in raw_edges:
        a_sid = e.get("stringId_A")
        b_sid = e.get("stringId_B")
        a_sym = string_to_symbol.get(a_sid)
        b_sym = string_to_symbol.get(b_sid)
        if not (a_sym and b_sym) or a_sym == b_sym:
            continue
        key = tuple(sorted([a_sym, b_sym]))
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "a": key[0], "b": key[1],
            "combined_score": int(round(float(e.get("score", 0)) * 1000)),
            "experimental": float(e.get("escore", 0)),
            "database": float(e.get("dscore", 0)),
            "textmining": float(e.get("tscore", 0)),
            "coexpression": float(e.get("ascore", 0)),
        })
    print(f"[edges] {len(edges)} unique within-panel edges", flush=True)

    out = {
        "schema_version": "1.0",
        "string_version": "12.0",
        "species_taxon_id": SPECIES_HUMAN,
        "score_threshold": 700,
        "network_type": "physical",
        "nodes": genes,
        "node_features": {sym: id_map.get(sym, {}) for sym in genes},
        "edges": edges,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[write] {out_path}", flush=True)
    print(f"\n=== SUMMARY ===")
    print(f"  Genes: {len(genes)}")
    print(f"  Resolved: {len(id_map)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Density: {len(edges) / (len(genes) * (len(genes) - 1) / 2):.3f}")


if __name__ == "__main__":
    main()
