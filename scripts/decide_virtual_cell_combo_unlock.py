#!/usr/bin/env python3
"""Unlock research-only combination work only after both prior gates pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from combo_val.virtual_cell.retrospective_validation import (  # noqa: E402
    combination_unlock_decision,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-summary", type=Path, required=True)
    parser.add_argument("--monotherapy-summary", type=Path, required=True)
    parser.add_argument("--readiness-summary", type=Path)
    parser.add_argument("--validation-summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    identity = json.loads(args.identity_summary.read_text(encoding="utf-8"))
    monotherapy = json.loads(args.monotherapy_summary.read_text(encoding="utf-8"))
    readiness = (
        json.loads(args.readiness_summary.read_text(encoding="utf-8"))
        if args.readiness_summary
        else None
    )
    validation = (
        json.loads(args.validation_summary.read_text(encoding="utf-8"))
        if args.validation_summary
        else None
    )
    decision = combination_unlock_decision(
        identity,
        monotherapy,
        readiness,
        validation,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
