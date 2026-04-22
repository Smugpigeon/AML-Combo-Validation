"""DrugComb AML subset ETL — filter the 1.4GB public dump to AML cell lines,
align drug names to BeatAML vocabulary, output a single canonical CSV.

Inputs:
  - data/raw/drugcomb/summary_v_1_5.csv   (1.4 GB, downloaded via Zenodo)
  - data/canonical/beataml_drug_response_long.csv  (for drug vocabulary)

Outputs:
  - data/canonical/drugcomb_aml_pairs.csv
      columns: block_id, cell_line_id, drug1_id, drug2_id,
               synergy_loewe, synergy_bliss, synergy_zip, synergy_hsa,
               drug1_match_source, drug2_match_source, source_study
  - data/canonical/drugcomb_drug_alignment.csv
      (per DrugComb drug name → BeatAML id + match source; transparency)
  - data/canonical/drugcomb_filter_manifest.json
      (cell-line filter stats, drug-match stats, reproducibility)

Design:
  1. Stream the 1.4GB CSV in ~100K-row chunks to keep memory under 1 GB.
  2. Filter each chunk to AML cell lines (normalized name match).
  3. Accumulate the AML subset (~1-2M rows typically → ~30-60 MB).
  4. Drug-align both columns via DrugNameAligner.
  5. Keep rows where BOTH drugs map to BeatAML vocab (stricter path). Also
     save "one-sided match" count for transparency.

If the full file isn't available yet, use `--smoke` to just peek at the
first N rows and validate schema + cell line matching logic.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from combo_val.data.drug_name_normalizer import DrugNameAligner, normalize_drug_name


# ---------------------------------------------------------------------------
# Cell-line panel (authoritative AML model lines)
# ---------------------------------------------------------------------------

# Canonical + common aliases seen in DrugComb / CCLE / ATCC. Normalizer will
# compare against normalize_drug_name(...) of these entries.
AML_CELL_LINES_ALIASES: dict[str, list[str]] = {
    # canonical key       : [aliases]
    "HL-60":       ["HL-60", "HL60", "HL 60", "HL-60(TB)"],
    "MV4-11":      ["MV4-11", "MV-4-11", "MV411", "MV 4 11", "MV4;11"],
    "MOLM-13":     ["MOLM-13", "MOLM13", "MOLM 13"],
    "MOLM-14":     ["MOLM-14", "MOLM14", "MOLM 14"],
    "KASUMI-1":    ["KASUMI-1", "KASUMI1", "KASUMI 1"],
    "NB4":         ["NB4", "NB-4", "NB 4"],
    "OCI-AML2":    ["OCI-AML2", "OCIAML2", "OCI-AML-2", "OCI AML 2"],
    "OCI-AML3":    ["OCI-AML3", "OCIAML3", "OCI-AML-3", "OCI AML 3"],
    "OCI-AML5":    ["OCI-AML5", "OCIAML5", "OCI-AML-5"],
    "THP-1":       ["THP-1", "THP1", "THP 1"],
    "U-937":       ["U-937", "U937"],
    "KG-1":        ["KG-1", "KG1", "KG 1", "KG-1A", "KG1A"],
    "ME-1":        ["ME-1", "ME1"],
    "MUTZ-3":      ["MUTZ-3", "MUTZ3"],
    "SET-2":       ["SET-2", "SET2"],
    "HEL":         ["HEL", "HEL 92.1.7"],   # erythroleukemia, borderline AML but sometimes used
}


def _build_cell_line_lookup() -> dict[str, str]:
    """Normalized alias → canonical name."""
    out = {}
    for canonical, aliases in AML_CELL_LINES_ALIASES.items():
        for a in aliases:
            key = re.sub(r"[^a-z0-9]+", "", a.lower())
            out[key] = canonical
    return out


def _normalize_cell_line(raw: str) -> str | None:
    """Normalize a raw cell line name, return canonical if it matches AML panel."""
    if not raw:
        return None
    key = re.sub(r"[^a-z0-9]+", "", str(raw).lower())
    return _CELL_LINE_LOOKUP.get(key)


_CELL_LINE_LOOKUP = _build_cell_line_lookup()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrugCombConfig:
    input_path: Path = Path("data/raw/drugcomb/summary_v_1_5.csv")
    beataml_long_path: Path = Path("data/canonical/beataml_drug_response_long.csv")
    out_dir: Path = Path("data/canonical")
    chunk_size: int = 100_000
    fuzzy_cutoff: float = 0.87
    # Expected DrugComb columns (v1.5 standard). Missing → fall back to alternatives.
    drug1_col_aliases: tuple[str, ...] = ("drug_row", "drug1", "drug_row_cid", "drug_row_name")
    drug2_col_aliases: tuple[str, ...] = ("drug_col", "drug2", "drug_col_cid", "drug_col_name")
    cell_line_col_aliases: tuple[str, ...] = ("cell_line_name", "cell_line", "cell")
    synergy_col_aliases: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "synergy_loewe": ("synergy_loewe", "S_loewe", "loewe"),
        "synergy_bliss": ("synergy_bliss", "S_bliss", "bliss"),
        "synergy_zip":   ("synergy_zip", "S_zip", "zip"),
        "synergy_hsa":   ("synergy_hsa", "S_hsa", "hsa"),
    })
    block_id_col_aliases: tuple[str, ...] = ("block_id", "BlockID", "block")
    study_col_aliases: tuple[str, ...] = ("study_name", "study", "source")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_present(cols: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols_set = set(cols)
    for c in candidates:
        if c in cols_set:
            return c
    return None


def _detect_schema(sample_df: pd.DataFrame, cfg: DrugCombConfig) -> dict:
    """From the first chunk, figure out which columns to use."""
    cols = sample_df.columns.tolist()
    schema = {
        "drug1": _first_present(cols, cfg.drug1_col_aliases),
        "drug2": _first_present(cols, cfg.drug2_col_aliases),
        "cell_line": _first_present(cols, cfg.cell_line_col_aliases),
        "block_id": _first_present(cols, cfg.block_id_col_aliases),
        "study": _first_present(cols, cfg.study_col_aliases),
    }
    for metric_key, aliases in cfg.synergy_col_aliases.items():
        schema[metric_key] = _first_present(cols, aliases)

    missing_required = [k for k in ("drug1", "drug2", "cell_line") if schema[k] is None]
    if missing_required:
        raise ValueError(
            f"DrugComb schema detection failed: missing required columns {missing_required}. "
            f"Available columns: {cols[:30]}..."
        )
    return schema


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_drugcomb_etl(
    cfg: DrugCombConfig | None = None,
    smoke_limit: int | None = None,
) -> dict:
    """Stream-read DrugComb, filter to AML, align drug names, write canonical.

    Args:
      smoke_limit: if set, stop after reading this many TOTAL rows (for testing).
    """
    cfg = cfg or DrugCombConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.input_path.exists():
        raise FileNotFoundError(
            f"DrugComb file not found at {cfg.input_path}. "
            f"Download from https://zenodo.org/records/15235991 first."
        )

    # Load BeatAML drug vocabulary
    beataml_long = pd.read_csv(cfg.beataml_long_path, usecols=["drug_id"])
    beataml_drug_vocab = sorted(beataml_long["drug_id"].unique().tolist())
    print(f"[drugcomb] BeatAML vocab: {len(beataml_drug_vocab)} drugs")

    aligner = DrugNameAligner(
        beataml_drug_names=beataml_drug_vocab, fuzzy_cutoff=cfg.fuzzy_cutoff
    )

    # --- Streaming read ---
    total_rows = 0
    aml_rows: list[pd.DataFrame] = []
    aml_cell_line_hits = {name: 0 for name in AML_CELL_LINES_ALIASES}

    t0 = time.time()
    schema = None
    reader = pd.read_csv(
        cfg.input_path,
        chunksize=cfg.chunk_size,
        low_memory=False,
    )
    for chunk_i, chunk in enumerate(reader):
        if schema is None:
            schema = _detect_schema(chunk, cfg)
            print(f"[drugcomb] detected schema: {schema}")

        total_rows += len(chunk)

        # Cell line filter
        cell_raw = chunk[schema["cell_line"]].astype(str)
        canonical = cell_raw.map(lambda x: _normalize_cell_line(x))
        keep = canonical.notna()
        if keep.sum() > 0:
            aml_chunk = chunk[keep].copy()
            aml_chunk["cell_line_id"] = canonical[keep].values
            # Update cell-line hit counter
            for cl, n in aml_chunk["cell_line_id"].value_counts().items():
                aml_cell_line_hits[cl] = aml_cell_line_hits.get(cl, 0) + int(n)
            aml_rows.append(aml_chunk)

        if (chunk_i + 1) % 5 == 0:
            print(
                f"[drugcomb] chunk {chunk_i + 1}: total_rows={total_rows:,} "
                f"aml_hit={sum(len(d) for d in aml_rows):,}  "
                f"[{time.time() - t0:.1f}s]"
            )

        if smoke_limit is not None and total_rows >= smoke_limit:
            print(f"[drugcomb] SMOKE MODE: stopping at total_rows={total_rows}")
            break

    if not aml_rows:
        raise RuntimeError(
            "No rows matched AML cell line aliases. Check AML_CELL_LINES_ALIASES "
            "vs actual DrugComb cell line names."
        )

    aml_df = pd.concat(aml_rows, ignore_index=True)
    print(
        f"[drugcomb] AML subset: {len(aml_df):,} rows  "
        f"across {aml_df['cell_line_id'].nunique()} cell lines  "
        f"[elapsed {time.time() - t0:.1f}s]"
    )
    print("[drugcomb] cell line hit counts:")
    for cl, n in sorted(aml_cell_line_hits.items(), key=lambda x: -x[1])[:15]:
        if n > 0:
            print(f"    {cl:15s}: {n:,}")

    # --- Drug name alignment ---
    unique_drugs_1 = aml_df[schema["drug1"]].dropna().astype(str).unique()
    unique_drugs_2 = aml_df[schema["drug2"]].dropna().astype(str).unique()
    all_unique_drugs = sorted(set(unique_drugs_1) | set(unique_drugs_2))
    print(f"[drugcomb] aligning {len(all_unique_drugs)} unique drug names to BeatAML vocab ...")

    alignment = aligner.align_many(all_unique_drugs)

    alignment_df = pd.DataFrame([
        {"drugcomb_name": k, "beataml_name": v[0], "match_source": v[1]}
        for k, v in alignment.items()
    ]).sort_values(["match_source", "drugcomb_name"])
    alignment_out = cfg.out_dir / "drugcomb_drug_alignment.csv"
    alignment_df.to_csv(alignment_out, index=False)
    print(f"[drugcomb] drug alignment saved: {alignment_out}")

    match_source_counts = alignment_df["match_source"].value_counts().to_dict()
    print(f"[drugcomb] alignment breakdown: {match_source_counts}")

    # Apply alignment to the big table
    aml_df["drug1_id"] = aml_df[schema["drug1"]].map(
        lambda x: alignment.get(str(x), (None,))[0]
    )
    aml_df["drug2_id"] = aml_df[schema["drug2"]].map(
        lambda x: alignment.get(str(x), (None,))[0]
    )
    aml_df["drug1_match_source"] = aml_df[schema["drug1"]].map(
        lambda x: alignment.get(str(x), (None, "no_match"))[1]
    )
    aml_df["drug2_match_source"] = aml_df[schema["drug2"]].map(
        lambda x: alignment.get(str(x), (None, "no_match"))[1]
    )

    # DrugComb v1.5 is a unified table: monotherapy screens (CTRPv2, GDSC, FIMM,
    # CCLE, gCSI) have drug2 = nan. True combination rows (ALMANAC, ONEIL, etc.)
    # have both non-null. Partition explicitly.
    raw_d1 = aml_df[schema["drug1"]]
    raw_d2 = aml_df[schema["drug2"]]
    is_combo_row = raw_d1.notna() & raw_d2.notna() & (raw_d1.astype(str) != raw_d2.astype(str))
    is_mono_row = ~is_combo_row
    print(
        f"[drugcomb] assay type: combination rows {is_combo_row.sum():,} | "
        f"monotherapy rows {is_mono_row.sum():,}"
    )

    # Kept rows (both drugs mapped AND actually a combination row)
    both_mapped = aml_df["drug1_id"].notna() & aml_df["drug2_id"].notna() & is_combo_row
    one_mapped = (aml_df["drug1_id"].notna() ^ aml_df["drug2_id"].notna()) & is_combo_row
    print(
        f"[drugcomb] combo drug mapping: "
        f"both {both_mapped.sum():,} | one-side {one_mapped.sum():,} | "
        f"neither (among combo rows) {(is_combo_row & ~(both_mapped | one_mapped)).sum():,}"
    )

    # --- Build output table ---
    out_cols = {
        "block_id": schema.get("block_id"),
        "cell_line_id": "cell_line_id",
        "drug1_original": schema["drug1"],
        "drug1_id": "drug1_id",
        "drug1_match_source": "drug1_match_source",
        "drug2_original": schema["drug2"],
        "drug2_id": "drug2_id",
        "drug2_match_source": "drug2_match_source",
        "synergy_loewe": schema.get("synergy_loewe"),
        "synergy_bliss": schema.get("synergy_bliss"),
        "synergy_zip": schema.get("synergy_zip"),
        "synergy_hsa": schema.get("synergy_hsa"),
        "source_study": schema.get("study"),
    }
    out_rows = []
    for out_name, src_name in out_cols.items():
        if src_name and src_name in aml_df.columns:
            aml_df[out_name] = aml_df[src_name]
    keep_cols = [c for c in out_cols.keys() if c in aml_df.columns]

    final = aml_df[keep_cols + (["drug1_id", "drug2_id"] if False else [])].copy()
    # Dedup columns if any doubled
    final = final.loc[:, ~final.columns.duplicated()]

    # Keep only rows where both drugs mapped AND row is a combination (strict)
    final_strict = final[both_mapped.values].copy()
    final_out = cfg.out_dir / "drugcomb_aml_pairs.csv"
    final_strict.to_csv(final_out, index=False)

    # Also save the looser version (one-side), combo rows only
    final_loose_out = cfg.out_dir / "drugcomb_aml_pairs_any_match.csv"
    final[(both_mapped | one_mapped).values].to_csv(final_loose_out, index=False)

    # Monotherapy subset — supplementary single-drug AML cell-line screens
    mono_with_match = is_mono_row & aml_df["drug1_id"].notna()
    final_mono_out = cfg.out_dir / "drugcomb_aml_monotherapy.csv"
    final[mono_with_match.values].to_csv(final_mono_out, index=False)
    print(
        f"[drugcomb] monotherapy rows with drug1 mapped to BeatAML: "
        f"{int(mono_with_match.sum()):,} → {final_mono_out}"
    )

    # Manifest
    manifest = {
        "cfg": asdict(cfg),
        "elapsed_s": round(time.time() - t0, 1),
        "total_rows_scanned": int(total_rows),
        "aml_rows_kept": int(len(aml_df)),
        "aml_cell_line_hits": {k: v for k, v in aml_cell_line_hits.items() if v > 0},
        "n_unique_drugs_drugcomb": int(len(all_unique_drugs)),
        "drug_alignment_breakdown": {k: int(v) for k, v in match_source_counts.items()},
        "n_combination_rows": int(is_combo_row.sum()),
        "n_monotherapy_rows": int(is_mono_row.sum()),
        "n_pairs_strict_both_mapped": int(both_mapped.sum()),
        "n_pairs_any_mapped": int((both_mapped | one_mapped).sum()),
        "n_monotherapy_with_beataml_match": int(mono_with_match.sum()),
        "schema": schema,
        "outputs": {
            "strict_pairs": str(final_out),
            "any_match_pairs": str(final_loose_out),
            "monotherapy": str(final_mono_out),
            "alignment": str(alignment_out),
        },
    }
    manifest_path = cfg.out_dir / "drugcomb_filter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(
        f"[drugcomb] DONE. strict pairs: {len(final_strict):,}  "
        f"any-match pairs: {int((both_mapped | one_mapped).sum()):,}  "
        f"manifest: {manifest_path}"
    )
    return manifest


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", default="data/raw/drugcomb/summary_v_1_5.csv")
    ap.add_argument("--beataml-long", default="data/canonical/beataml_drug_response_long.csv")
    ap.add_argument("--out", default="data/canonical")
    ap.add_argument("--chunk-size", type=int, default=100_000)
    ap.add_argument(
        "--smoke-limit", type=int, default=None,
        help="If set, stop after this many total rows — for quick validation.",
    )
    args = ap.parse_args()

    cfg = DrugCombConfig(
        input_path=Path(args.input),
        beataml_long_path=Path(args.beataml_long),
        out_dir=Path(args.out),
        chunk_size=args.chunk_size,
    )
    run_drugcomb_etl(cfg, smoke_limit=args.smoke_limit)


if __name__ == "__main__":
    main()
