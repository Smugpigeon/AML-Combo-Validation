#!/usr/bin/env python3
"""Generate an adversarial and neutral review for one evidence gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from combo_val.virtual_cell.stage_review import (  # noqa: E402
    build_stage_review,
    render_stage_review_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["identity", "monotherapy", "combination"], required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    review = build_stage_review(args.stage, summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.stage}_adversarial_review"
    (args.out_dir / f"{stem}.json").write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / f"{stem}.md").write_text(
        render_stage_review_markdown(review),
        encoding="utf-8",
    )
    print(json.dumps(review, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
