"""One-time: compute BeatAML per-gene reference stats for the 25-gene core panel.

Output:
  data/canonical/driver_gene_ref_stats.json
    {
      "reference_cohort": "BeatAML 2.0",
      "n_samples": <int>,
      "log1p_scale": true,
      "genes": {
        "FLT3": {"mean": <float>, "std": <float>, "n": <int>, "median": <float>,
                 "p05": <float>, "p95": <float>},
        ...
      },
      "missing_genes": [...]  # genes not found in BeatAML symbol list
    }

These stats power `expression_outlier.build_rnaseq_outlier_table()` which
computes per-gene z-scores on log1p-normalized counts. We pick log1p because:
(1) counts are right-skewed, (2) log1p handles zeros, (3) it's the scale the
kit's preprocessor uses internally.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# Core 25-gene AML driver panel (kept in sync with combo_val.clinical.dna_report)
CORE_DRIVER_GENES: tuple[str, ...] = (
    "FLT3", "IDH1", "IDH2", "KMT2A",
    "NPM1", "TP53", "RUNX1", "ASXL1", "CEBPA",
    "DNMT3A", "TET2", "KIT",
    "NRAS", "KRAS", "PTPN11",
    "WT1", "BCOR",
    "STAG2", "PHF6",
    "SRSF2", "SF3B1", "U2AF1", "EZH2",
    "MECOM", "CBFB",
)

# Extra genes with clinical expression-outlier relevance for AML
# (used as hints for cryptic events the 25-gene mutation panel might miss)
EXTRA_EXPRESSION_HINT_GENES: tuple[str, ...] = (
    "HOXA9", "MEIS1",        # high → KMT2A-r / NPM1-mut signature
    "BCL2", "MCL1",          # high → venetoclax / MCL1-inhibitor rationale
    "EVI1",                  # alt symbol for MECOM
    "BAALC",                 # high → adverse
    "MN1",                   # high → adverse (inv(16)-like)
    "CD33",                  # high → GO (Mylotarg) rationale
    "CD123", "IL3RA",        # high → Tagraxofusp / CD123-CAR
)


def main():
    src = Path("/Users/ericktom/AML-combo-validation/data/raw/BeatAML2.0/beatAML.xlsx")
    out = Path("/Users/ericktom/AML-combo-validation/data/canonical/driver_gene_ref_stats.json")

    print(f"Loading BeatAML expression matrix from {src.name}...")
    df = pd.read_excel(src, sheet_name="Sheet1")
    print(f"  Shape: {df.shape}  (cols[0]='{df.columns[0]}', cols[-1]='{df.columns[-1]}')")

    # First column is the gene symbol; the rest are patient IDs
    symbol_col = df.columns[0]
    sample_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
    print(f"  Numeric sample columns: {len(sample_cols)}")

    df = df.set_index(symbol_col)
    # Handle duplicates (some HUGO symbols have multiple rows — sum across them)
    if df.index.duplicated().any():
        n_dup = df.index.duplicated().sum()
        print(f"  Collapsing {n_dup} duplicate gene symbols via mean")
        df = df.groupby(level=0).mean(numeric_only=True)

    # BeatAML's xlsx sheet distributes values on a log2-CPM-ish scale already
    # (range [-5, +11], median ~4.4). Use values as-is for the reference
    # distribution — the kit's preprocessor output is on the same scale after QN.
    vals = df[sample_cols].astype(float).values
    print(f"  Expression value stats: min={vals.min():.2f}  median={np.median(vals):.2f}  "
          f"max={vals.max():.2f}  (already log-scaled, no further transform)")
    expr = vals
    gene_index = df.index

    genes_of_interest = list(CORE_DRIVER_GENES) + list(EXTRA_EXPRESSION_HINT_GENES)
    gene_stats = {}
    missing = []
    for g in genes_of_interest:
        if g not in gene_index:
            missing.append(g)
            continue
        row = expr[gene_index.get_loc(g)]
        # If there are multiple rows with same symbol (shouldn't happen after groupby),
        # pandas returns a 2-D slice; handle defensively.
        if row.ndim == 2:
            row = row.sum(axis=0)
        gene_stats[g] = {
            "mean": float(np.mean(row)),
            "std": float(np.std(row, ddof=1)),
            "n": int(len(row)),
            "median": float(np.median(row)),
            "p05": float(np.quantile(row, 0.05)),
            "p95": float(np.quantile(row, 0.95)),
        }

    print(f"  Stats computed for {len(gene_stats)} genes; missing: {missing}")

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_cohort": "BeatAML 2.0",
        "n_samples": len(sample_cols),
        "log1p_scale": False,
        "transform_note": ("Values are BeatAML 2.0 Sheet1 as-delivered — already "
                            "on a log2-CPM-like scale (range [-5,+11], median ~4.4). "
                            "The kit's live input should be QN-aligned to this "
                            "distribution before computing z-scores."),
        "core_driver_genes": list(CORE_DRIVER_GENES),
        "extra_hint_genes": list(EXTRA_EXPRESSION_HINT_GENES),
        "genes": gene_stats,
        "missing_genes": missing,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out} ({out.stat().st_size} bytes)")

    # Also print a quick sanity table
    print("\nQuick check (top-5 highest-expressed drivers):")
    ranked = sorted(gene_stats.items(), key=lambda kv: -kv[1]["mean"])[:5]
    for g, s in ranked:
        print(f"  {g:8s}  log1p mean={s['mean']:.2f}  std={s['std']:.2f}  n={s['n']}")


if __name__ == "__main__":
    main()
