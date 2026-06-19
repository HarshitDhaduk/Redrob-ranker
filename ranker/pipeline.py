"""End-to-end ranking pipeline: stream candidates -> features -> score -> top-100.

Single streaming pass, O(candidates), pure Python. Memory stays flat: we only
ever retain the running top-K (K=250) candidates, not the whole pool. Producing
the top-100 from 100k profiles runs in well under a minute on one CPU core.
"""

from __future__ import annotations

import gzip
import heapq
import io
import json
import time
from dataclasses import dataclass, field

from . import features as feat
from . import scoring
from . import reasoning

KEEP = 250          # keep a margin above 100 for stable tie-breaking / inspection


@dataclass
class RankResult:
    rows: list = field(default_factory=list)        # final top-100 (dicts)
    top_records: list = field(default_factory=list)  # (score, cid, f, br) kept
    n_total: int = 0
    n_scored: int = 0
    n_integrity_flagged: int = 0
    elapsed_s: float = 0.0


def _open(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def rank_candidates(path: str, top_n: int = 100, progress=False) -> RankResult:
    t0 = time.time()
    heap: list = []            # min-heap of (final, cid, f, br)
    counter = 0
    n_total = 0
    n_flagged = 0

    with _open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            n_total += 1
            c = json.loads(line)
            f = feat.extract(c)
            if f["integrity"]["penalty"] > 0:
                n_flagged += 1
            br = scoring.score(f)
            final = br["final"]
            cid = f["candidate_id"]
            item = (final, cid, f, br)
            if len(heap) < KEEP:
                heapq.heappush(heap, item)
            elif final > heap[0][0]:
                heapq.heapreplace(heap, item)
            if progress and n_total % 20000 == 0:
                print(f"  ...{n_total} scored ({time.time()-t0:.1f}s)", flush=True)

    # Order kept records: round score first so equal scores tie-break by
    # candidate_id ascending (exactly what the validator enforces).
    kept = [(round(final, 6), cid, f, br) for (final, cid, f, br) in heap]
    kept.sort(key=lambda x: (-x[0], x[1]))

    rows = []
    for i, (sc, cid, f, br) in enumerate(kept[:top_n]):
        rank = i + 1
        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": f"{sc:.6f}",
            "reasoning": reasoning.build(f, br, rank),
        })

    return RankResult(
        rows=rows,
        top_records=kept[:KEEP],
        n_total=n_total,
        n_scored=n_total,
        n_integrity_flagged=n_flagged,
        elapsed_s=time.time() - t0,
    )


def write_csv(rows: list, out_path: str) -> None:
    import csv
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            w.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])
