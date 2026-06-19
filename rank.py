#!/usr/bin/env python3
"""Produce the top-100 candidate ranking CSV for the Redrob JD.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Pure standard library. CPU only, no network. Handles .jsonl and .jsonl.gz.
"""

from __future__ import annotations

import argparse
import sys

from ranker.pipeline import rank_candidates, write_csv


def main(argv=None):
    ap = argparse.ArgumentParser(description="Redrob intelligent candidate ranker")
    ap.add_argument("--candidates", required=True,
                    help="Path to candidates.jsonl or candidates.jsonl.gz")
    ap.add_argument("--out", default="submission.csv", help="Output CSV path")
    ap.add_argument("--top", type=int, default=100, help="How many to rank")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress")
    args = ap.parse_args(argv)

    res = rank_candidates(args.candidates, top_n=args.top, progress=not args.quiet)
    write_csv(res.rows, args.out)

    if not args.quiet:
        flagged_in_top = sum(
            1 for (_, _, f, _) in res.top_records[:args.top]
            if f["integrity"]["penalty"] > 0
        )
        print(
            f"\nRanked {res.n_total:,} candidates in {res.elapsed_s:.1f}s "
            f"({res.n_integrity_flagged} integrity-flagged pool-wide; "
            f"{flagged_in_top} in top-{args.top}).",
            file=sys.stderr,
        )
        print(f"Wrote {len(res.rows)} rows -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
