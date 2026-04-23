"""One-time: compute BeatAML per-gene mean/std/median for the FULL
transcriptome (~22k genes), for use by the Route-B expression-outlier
module's "top-N transcriptome scan" feature.

This complements driver_gene_ref_stats.json (25+8 curated genes, ~7 KB)
by providing per-gene stats for ALL genes in the BeatAML expression
matrix, so we can scan a new patient's full transcriptome for any gene
that's dramatically up/down-regulated (not just the curated list).

Output: data/canonical/full_transcriptome_ref_stats.npz  (~1 MB)
  genes   : (n_genes,) U-string array of HUGO symbols
  means   : (n_genes,) float64
  stds    : (n_genes,) float64
  medians : (n_genes,) float64
  p05     : (n_genes,) float64  (5th percentile)
  p95     : (n_genes,) float64  (95th percentile)
  meta    : dict — cohort name, n_samples, scale note
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    src = Path("/Users/ericktom/AML-combo-validation/data/raw/BeatAML2.0/beatAML.xlsx")
    out = Path("/Users/ericktom/AML-combo-validation/data/canonical/full_transcriptome_ref_stats.npz")

    print(f"Loading BeatAML expression matrix from {src.name}...")
    df = pd.read_excel(src, sheet_name="Sheet1")
    print(f"  Shape: {df.shape}")

    symbol_col = df.columns[0]
    sample_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]

    df = df.set_index(symbol_col)
    if df.index.duplicated().any():
        n_dup = df.index.duplicated().sum()
        print(f"  Collapsing {n_dup} duplicate gene symbols via mean")
        df = df.groupby(level=0).mean(numeric_only=True)

    vals = df[sample_cols].astype(float).values   # (n_genes, n_samples)
    print(f"  Computing stats over {vals.shape[0]} genes × {vals.shape[1]} samples...")

    means = np.mean(vals, axis=1)
    stds = np.std(vals, axis=1, ddof=1)
    medians = np.median(vals, axis=1)
    p05 = np.quantile(vals, 0.05, axis=1)
    p95 = np.quantile(vals, 0.95, axis=1)

    genes = df.index.to_numpy(dtype=str)

    meta = {
        "reference_cohort": "BeatAML 2.0",
        "n_samples": len(sample_cols),
        "n_genes": len(genes),
        "scale_note": ("log2-CPM-like (values taken as delivered from "
                        "BeatAML Sheet1); range ~[-5, +11], median ~4.4"),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, genes=genes, means=means, stds=stds, medians=medians,
        p05=p05, p95=p95, meta=json.dumps(meta),
    )
    print(f"Wrote {out} ({out.stat().st_size / 1024:.1f} KB)")

    # Spot check
    print("\nHighest-variance genes (top 5):")
    ranked = np.argsort(-stds)[:5]
    for i in ranked:
        print(f"  {genes[i]:12s}  mean={means[i]:+.2f}  std={stds[i]:.2f}")
    print("\nHighest-mean genes (top 5):")
    ranked = np.argsort(-means)[:5]
    for i in ranked:
        print(f"  {genes[i]:12s}  mean={means[i]:+.2f}  std={stds[i]:.2f}")


if __name__ == "__main__":
    main()
