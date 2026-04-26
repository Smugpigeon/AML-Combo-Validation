#!/usr/bin/env python3
"""CLI wrapper for the external-validation framework.

Each adopting institution runs this once on their local hold-out cohort,
reviews the resulting REPORT.html, and decides whether the kit is
calibrated enough for their patient population.

Usage:
  python scripts/run_external_validation.py \
      --cohort-csv /path/to/your_cohort.csv \
      --out-dir runs/external_validation_PUMC_2026 \
      --institution "PUMC Hematology, Beijing" \
      --description "150 newly-Dx AML treated 2022-2024, intent-to-treat"

See src/combo_val/validation/external_validation.py docstring for the
required cohort.csv schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from combo_val.validation.external_validation import (
    ExternalValidationConfig,
    run_external_validation,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort-csv", type=Path, required=True,
                    help="Path to cohort.csv (see schema in module docstring)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory for REPORT.html + PNGs + metrics.json")
    ap.add_argument("--institution", type=str, default="Unknown institution",
                    help="Institution name (appears in report)")
    ap.add_argument("--description", type=str, default="",
                    help="One-line cohort description (appears in report)")
    ap.add_argument("--bootstrap-n", type=int, default=2000,
                    help="Bootstrap iterations for CI (default 2000)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.cohort_csv.exists():
        print(f"ERROR: cohort.csv not found at {args.cohort_csv}", file=sys.stderr)
        sys.exit(1)

    cfg = ExternalValidationConfig(
        cohort_csv=args.cohort_csv,
        out_dir=args.out_dir,
        bootstrap_n=args.bootstrap_n,
        random_state=args.seed,
        institution_name=args.institution,
        cohort_description=args.description,
    )

    print(f"Running external validation:")
    print(f"  Cohort: {cfg.cohort_csv}")
    print(f"  Institution: {cfg.institution_name}")
    print(f"  Output: {cfg.out_dir}")
    print()

    metrics = run_external_validation(cfg)

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    eln = metrics["eln_concordance"]
    print(f"  N patients: {metrics['n_patients']}")
    print(f"  ELN accuracy: {eln['accuracy']:.3f}")
    print(f"  ELN kappa: {eln['cohens_kappa']:.3f} ({eln['interpretation_kappa']})")

    cal = metrics["calibration_via_eln_prior"]
    if "brier_score" in cal:
        lo, hi = cal["brier_score_ci_95"]
        print(f"  Brier score: {cal['brier_score']:.3f} (95% CI {lo:.3f}-{hi:.3f})")
        print(f"  ECE: {cal['expected_calibration_error']:.3f}")

    rm = metrics["regimen_top1_match"]
    if "match_rate" in rm and rm["match_rate"] is not None:
        print(f"  Top-1 regimen match: {rm['match_rate']:.3f} ({rm['interpretation']})")

    print()
    print(f"Full report: {cfg.out_dir / 'REPORT.html'}")
    print(f"Metrics JSON: {cfg.out_dir / 'metrics.json'}")
    print()
    print("Pass criteria (proposed for clinical deployment):")
    print("  ELN kappa >= 0.60   (substantial agreement)")
    print("  Brier <= 0.20")
    print("  Top-1 regimen match >= 0.50")
    print()


if __name__ == "__main__":
    main()
