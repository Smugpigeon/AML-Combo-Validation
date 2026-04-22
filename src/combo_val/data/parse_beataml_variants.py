#!/usr/bin/env python3
"""Build a canonical mutation calls file from BeatAML clinical summary.

BeatAML 2.0's official `WES-targeted Sequencing Mutation Calls.txt` lives on dbGaP
and requires controlled access. In the meantime, the `beataml_clinical.xlsx`
file (sheet: summary) contains enough mutation information to build a reasonable
approximation, by combining:

  1. `variantSummary` — free-text per-sample list of variants like
     "NPM1 (p.W288fs*12, MAF 50%)|FLT3-ITD (MAF 40%)|DNMT3A (p.R882H, MAF 48%)"

  2. Dedicated gene flag columns — "positive"/"negative" for:
     FLT3-ITD, NPM1, RUNX1, ASXL1, TP53.

  3. `allelic_ratio` — FLT3-ITD variant allele frequency proxy.

  4. `consensusAMLFusions` — recurrent fusion calls
     (e.g. "RUNX1-RUNX1T1", "PML-RARA").

  5. CEBPA_Biallelic — 1/0 flag for biallelic CEBPA.

Output (tab-separated, matches `_load_mutation_calls` schema):

    dbgap_dnaseq_sample  gene  variant_classification  t_vaf  hgvsp_short

Placed at `platform/data/raw/BeatAML2.0/WES-targeted Sequencing Mutation Calls.txt`
so that `etl-beataml` picks it up automatically.

Known limitations
-----------------
- VAF granularity is limited by whatever BeatAML chose to summarise
  (typically bins of 5%).
- HGVSp strings may be truncated (e.g. `p.W288fs*12` missing full notation).
- No VAF for gene-flag-column rows without matching variantSummary tokens.
- No chromosomal position / reference / alternate allele — these are not
  recoverable from the clinical summary.
- Silent / synonymous variants are almost never reported in variantSummary,
  so this file is biased toward oncogenic / reportable events.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Gene-flag columns in clinical summary that behave as binary mutation markers.
GENE_FLAG_COLUMNS: dict[str, dict[str, str]] = {
    "FLT3-ITD": {"gene": "FLT3", "variant_classification": "ITD"},
    "NPM1": {"gene": "NPM1", "variant_classification": "SNV"},
    "RUNX1": {"gene": "RUNX1", "variant_classification": "SNV"},
    "ASXL1": {"gene": "ASXL1", "variant_classification": "SNV"},
    "TP53": {"gene": "TP53", "variant_classification": "SNV"},
}

# Tokens inside variantSummary that look like a VAF/MAF specification.
# Matches "MAF 50%", "VAF 0.48", "maf=45", etc.
_VAF_PATTERN = re.compile(r"(?:MAF|VAF)\s*[:=]?\s*([\d.]+)\s*%?", re.IGNORECASE)
# Tokens inside variantSummary that look like an HGVSp amino-acid change.
_HGVSP_PATTERN = re.compile(r"(p\.[A-Za-z0-9*_>]+)")
# Token splitter — variantSummary uses '|' between events and rarely ';'.
# We intentionally do NOT split on ',' because comma appears inside parens.
_TOKEN_SPLIT = re.compile(r"[|;]")
# Gene-head pattern: the FIRST contiguous run of [A-Z0-9_-] before space or paren.
_GENE_HEAD = re.compile(r"^\s*([A-Z][A-Z0-9_-]*?)(?:-(ITD|TKD|PTD))?\s*(?:\(|$)", re.IGNORECASE)


@dataclass(frozen=True)
class VariantRecord:
    sample_id: str
    gene: str
    variant_classification: str
    t_vaf: Optional[float]
    hgvsp_short: str
    source: str  # 'variantSummary' | 'gene_flag' | 'fusion' | 'cebpa_biallelic'


def _norm_float(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if pd.isna(f):
        return None
    return f


def _parse_vaf_string(value: Optional[float]) -> Optional[float]:
    """Normalize a VAF/MAF numeric to fraction in [0, 1]."""
    if value is None:
        return None
    if value > 1.5:  # percentage form, e.g. 50 -> 0.5
        return float(value) / 100.0
    if value < 0.0:
        return None
    return float(value)


def parse_variant_token(token: str) -> Optional[dict]:
    """Parse one token from variantSummary.

    Examples:
        'CEBPA' -> gene=CEBPA, SNV
        'FLT3-ITD' -> gene=FLT3, ITD
        'NPM1 (p.W288fs*12, MAF 50%)' -> gene=NPM1, SNV, vaf=0.5, hgvsp='p.W288fs*12'
        'IDH2 (p.R140Q, MAF 46%)' -> gene=IDH2, SNV, vaf=0.46, hgvsp='p.R140Q'
    """
    text = (token or "").strip()
    if not text:
        return None

    head = _GENE_HEAD.match(text)
    if not head:
        return None
    gene = head.group(1).upper()
    special = head.group(2)  # ITD / TKD / PTD or None

    # Skip garbage tokens (e.g. a leftover "MAF 50%)" that never had a gene head)
    if gene in {"MAF", "VAF"} or len(gene) <= 1:
        return None
    # Skip things that look like percentages masquerading as gene names
    if gene.isdigit():
        return None

    variant_classification = special.upper() if special else "SNV"

    # Extract optional parenthesized details
    details_match = re.search(r"\((.*)\)\s*$", text)
    details = details_match.group(1) if details_match else ""

    # VAF
    vaf = None
    vm = _VAF_PATTERN.search(details) if details else None
    if vm:
        vaf = _parse_vaf_string(float(vm.group(1)))

    # HGVSp
    hgvsp = ""
    hm = _HGVSP_PATTERN.search(details) if details else None
    if hm:
        hgvsp = hm.group(1)

    return {
        "gene": gene,
        "variant_classification": variant_classification,
        "t_vaf": vaf,
        "hgvsp_short": hgvsp,
    }


def _bool_positive(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    s = str(value).strip().lower()
    return s in {"1", "1.0", "true", "t", "yes", "y", "mut", "mutated", "positive", "pos"}


def _parse_fusion_column(text: str) -> list[tuple[str, str]]:
    """Parse `consensusAMLFusions` content into (gene_partner_A, gene_partner_B) pairs.

    BeatAML formats: 'RUNX1-RUNX1T1', 'PML-RARA', sometimes '|'-separated multiples.
    """
    if not text or pd.isna(text):
        return []
    pairs: list[tuple[str, str]] = []
    for chunk in re.split(r"[|;,]", str(text)):
        chunk = chunk.strip()
        if not chunk or chunk.lower() == "nan":
            continue
        # Split on '-' or '::'; keep the two partner gene names.
        parts = re.split(r"[-:]{1,2}", chunk)
        parts = [p.strip().upper() for p in parts if p.strip()]
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))
        elif len(parts) == 1:
            # Single-gene fusion mention (unusual, keep it anyway)
            pairs.append((parts[0], ""))
    return pairs


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def extract_mutations(clinical: pd.DataFrame) -> pd.DataFrame:
    """Extract mutation calls from a BeatAML clinical summary DataFrame.

    Returns a DataFrame with columns:
        dbgap_dnaseq_sample, gene, variant_classification, t_vaf, hgvsp_short, source
    """
    records: list[VariantRecord] = []

    # Use DNA sample id as the key (ETL will map it to RNA sample id later).
    sample_col = "dbgap_dnaseq_sample"
    if sample_col not in clinical.columns:
        raise ValueError(f"Column {sample_col!r} not found in clinical summary")

    # --- Prime per-row accumulator ---
    for _, row in clinical.iterrows():
        sample_id = str(row.get(sample_col) or "").strip()
        if not sample_id or sample_id.lower() == "nan":
            continue

        # 1. variantSummary tokens
        vs = row.get("variantSummary")
        if isinstance(vs, str) and vs.strip():
            for tok in _TOKEN_SPLIT.split(vs):
                parsed = parse_variant_token(tok)
                if parsed is None:
                    continue
                records.append(
                    VariantRecord(
                        sample_id=sample_id,
                        gene=parsed["gene"],
                        variant_classification=parsed["variant_classification"],
                        t_vaf=parsed["t_vaf"],
                        hgvsp_short=parsed["hgvsp_short"],
                        source="variantSummary",
                    )
                )

        # 2. gene flag columns
        for flag_col, spec in GENE_FLAG_COLUMNS.items():
            if flag_col not in clinical.columns:
                continue
            if _bool_positive(row.get(flag_col)):
                # For FLT3-ITD, convert allelic_ratio (ITD/WT) to approximate VAF.
                # VAF = ratio / (1 + ratio) assuming ITD and WT are the two alleles.
                vaf = None
                if flag_col == "FLT3-ITD":
                    ar = _norm_float(row.get("allelic_ratio"))
                    if ar is not None and ar >= 0:
                        vaf = ar / (1.0 + ar)
                        # Clip defensively to [0, 1]
                        vaf = max(0.0, min(1.0, vaf))
                records.append(
                    VariantRecord(
                        sample_id=sample_id,
                        gene=spec["gene"],
                        variant_classification=spec["variant_classification"],
                        t_vaf=vaf,
                        hgvsp_short="",
                        source="gene_flag",
                    )
                )

        # 3. fusions from consensusAMLFusions
        fusions = _parse_fusion_column(row.get("consensusAMLFusions") or "")
        for gene_a, gene_b in fusions:
            fusion_str = f"{gene_a}::{gene_b}" if gene_b else gene_a
            for g in (gene_a, gene_b):
                if not g:
                    continue
                records.append(
                    VariantRecord(
                        sample_id=sample_id,
                        gene=g,
                        variant_classification="FUSION",
                        t_vaf=None,
                        hgvsp_short=fusion_str,
                        source="fusion",
                    )
                )

        # 4. CEBPA biallelic flag
        cebpa_bi = row.get("CEBPA_Biallelic")
        if _bool_positive(cebpa_bi):
            records.append(
                VariantRecord(
                    sample_id=sample_id,
                    gene="CEBPA",
                    variant_classification="BIALLELIC",
                    t_vaf=None,
                    hgvsp_short="",
                    source="cebpa_biallelic",
                )
            )

    if not records:
        return pd.DataFrame(
            columns=[sample_col, "gene", "variant_classification", "t_vaf", "hgvsp_short", "source"]
        )

    df = pd.DataFrame(
        {
            sample_col: [r.sample_id for r in records],
            "gene": [r.gene for r in records],
            "variant_classification": [r.variant_classification for r in records],
            "t_vaf": [r.t_vaf for r in records],
            "hgvsp_short": [r.hgvsp_short for r in records],
            "source": [r.source for r in records],
        }
    )

    # Three-stage dedup:
    # 1. Rescue: if a variantSummary row exists for (sample, gene, class) but
    #    has no VAF, and a gene_flag row for the same key has VAF (e.g. FLT3-ITD
    #    allelic_ratio transformed), copy the VAF across before dedup.
    # 2. If a variantSummary row covers (sample, gene, class), drop gene_flag
    #    rows for the same key (variantSummary strictly richer).
    # 3. Final dedup on (sample, gene, class, hgvsp_short) so two distinct
    #    p.XXX variants in the same sample are preserved.

    # --- stage 1: rescue VAF from gene_flag to variantSummary when missing ---
    key_cols = [sample_col, "gene", "variant_classification"]
    flag_vaf_map: dict[tuple, float] = {}
    for _, r in df[df["source"] == "gene_flag"].iterrows():
        if pd.notna(r["t_vaf"]):
            flag_vaf_map[tuple(r[c] for c in key_cols)] = float(r["t_vaf"])

    def _fill_vaf(row):
        if row["source"] == "variantSummary" and pd.isna(row["t_vaf"]):
            key = tuple(row[c] for c in key_cols)
            if key in flag_vaf_map:
                return flag_vaf_map[key]
        return row["t_vaf"]

    df["t_vaf"] = df.apply(_fill_vaf, axis=1)

    # --- stage 2: drop gene_flag rows covered by variantSummary ---
    rich_keys = set(
        map(tuple, df.loc[df["source"] == "variantSummary", key_cols].values.tolist())
    )
    is_redundant_flag = (
        (df["source"] == "gene_flag")
        & df[key_cols].apply(tuple, axis=1).isin(rich_keys)
    )
    df = df.loc[~is_redundant_flag].copy()

    # --- stage 3: final dedup ---
    priority = {"variantSummary": 0, "fusion": 1, "cebpa_biallelic": 2, "gene_flag": 3}
    df["_prio"] = df["source"].map(priority).fillna(99)
    df = (
        df.sort_values(["_prio"], kind="stable")
        .drop_duplicates(
            subset=[sample_col, "gene", "variant_classification", "hgvsp_short"],
            keep="first",
        )
        .drop(columns=["_prio"])
        .reset_index(drop=True)
    )
    return df


def build_report(mutation_df: pd.DataFrame, clinical: pd.DataFrame) -> str:
    """Generate a markdown coverage report."""
    n_samples_total = clinical["dbgap_dnaseq_sample"].nunique()
    n_samples_with_muts = mutation_df["dbgap_dnaseq_sample"].nunique()
    n_events = len(mutation_df)

    by_source = mutation_df["source"].value_counts().to_dict()
    top_genes = mutation_df.groupby("gene").size().sort_values(ascending=False).head(20)
    by_class = mutation_df["variant_classification"].value_counts().to_dict()
    with_vaf = mutation_df["t_vaf"].notna().sum()
    with_hgvsp = (mutation_df["hgvsp_short"].astype(str).str.len() > 0).sum()

    lines = [
        "# BeatAML variantSummary extraction report",
        "",
        "## Coverage",
        f"- Total samples (dbgap_dnaseq_sample): **{n_samples_total}**",
        f"- Samples with at least one extracted mutation: **{n_samples_with_muts}**  "
        f"({n_samples_with_muts/max(1,n_samples_total)*100:.1f}%)",
        f"- Total mutation events: **{n_events}**",
        f"- Events with VAF: {with_vaf} ({with_vaf/max(1,n_events)*100:.1f}%)",
        f"- Events with HGVSp: {with_hgvsp} ({with_hgvsp/max(1,n_events)*100:.1f}%)",
        "",
        "## Events by source",
        *[f"- `{k}`: {v}" for k, v in by_source.items()],
        "",
        "## Events by variant class",
        *[f"- `{k}`: {v}" for k, v in by_class.items()],
        "",
        "## Top 20 genes by event count",
        "| gene | events |",
        "|---|---|",
        *[f"| {g} | {n} |" for g, n in top_genes.items()],
        "",
        "## Caveats",
        "- This file is derived from clinical summary, not from official WES VCFs.",
        "- VAF values may be bin-rounded (5% bins are common in the source).",
        "- Silent/synonymous variants and non-reportable events are missing by design.",
        "- FLT3-ITD `t_vaf` is converted from `allelic_ratio` (ITD/WT) using "
        "VAF ≈ ratio / (1 + ratio); this is an approximation of true NGS-VAF.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/BeatAML2.0/beataml_clinical.xlsx"),
        help="Path to beataml_clinical.xlsx (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/BeatAML2.0/WES-targeted Sequencing Mutation Calls.txt"),
        help="Path to write the canonical mutation calls TSV.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/raw/BeatAML2.0/mutation_extraction_report.md"),
        help="Path to write the markdown coverage report.",
    )
    parser.add_argument(
        "--sheet",
        default="summary",
        help="Excel sheet name (default: summary)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"[ERROR] input file not found: {args.input}", file=sys.stderr)
        return 2

    print(f"[1/3] Loading clinical summary: {args.input}", file=sys.stderr)
    clinical = pd.read_excel(args.input, sheet_name=args.sheet)
    print(f"      Rows: {len(clinical)}  Columns: {len(clinical.columns)}", file=sys.stderr)

    print(f"[2/3] Extracting mutations ...", file=sys.stderr)
    mut = extract_mutations(clinical)
    print(
        f"      Samples with events: {mut['dbgap_dnaseq_sample'].nunique()}  "
        f"Total events: {len(mut)}",
        file=sys.stderr,
    )

    print(f"[3/3] Writing outputs ...", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Output columns in the order _load_mutation_calls expects; drop 'source'
    out_cols = ["dbgap_dnaseq_sample", "gene", "variant_classification", "t_vaf", "hgvsp_short"]
    mut[out_cols].to_csv(args.out, sep="\t", index=False)
    print(f"      Mutation calls: {args.out}", file=sys.stderr)

    report = build_report(mut, clinical)
    args.report.write_text(report, encoding="utf-8")
    print(f"      Report: {args.report}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
