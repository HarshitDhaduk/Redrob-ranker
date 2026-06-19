#!/usr/bin/env python3
"""Diagnostic harness: run the ranker and explain the result.

    python debug_explain.py <candidates.jsonl> [N]

Prints the top-N with a score breakdown, checks known honeypots / strong
candidates, and summarises the composition of the shortlist.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from ranker.pipeline import rank_candidates

KNOWN_HONEYPOTS = {  # relevant-titled honeypots found during data exploration
    "CAND_0010770", "CAND_0013536", "CAND_0090900", "CAND_0019480",
    "CAND_0037000", "CAND_0038431", "CAND_0039754", "CAND_0055992",
    "CAND_0071115", "CAND_0091534", "CAND_0093331", "CAND_0036299",
}
KNOWN_STRONG = {"CAND_0002025", "CAND_0005538", "CAND_0006418"}

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
res = rank_candidates(path, top_n=100, progress=True)

print(f"\nTotal {res.n_total:,} | integrity-flagged pool-wide: {res.n_integrity_flagged} "
      f"| elapsed {res.elapsed_s:.1f}s")

top100_ids = [r["candidate_id"] for r in res.rows]
hp_in_top = [c for c in top100_ids if c in KNOWN_HONEYPOTS]
print(f"\nKNOWN honeypots in top-100: {len(hp_in_top)} -> {hp_in_top}")
rank_of = {r['candidate_id']: r['rank'] for r in res.rows}
for s in sorted(KNOWN_STRONG):
    print(f"  strong {s}: rank {rank_of.get(s, '>100')}")

print(f"\n===== TOP {N} =====")
for (sc, cid, f, br) in res.top_records[:N]:
    p = f["profile"]
    rank = top100_ids.index(cid) + 1 if cid in top100_ids else "-"
    print(f"\n#{rank} {cid} score={sc:.4f} | {p['current_title']} @ {p['current_company']} "
          f"| {p['yoe']:.1f}y | {p['location']},{p['country']} [{f['location_tier']}/{f['title_tier']}]")
    print(f"    core={br['core']:.2f} plus={br['plus']:.2f} sup={br['support']:.2f} "
          f"title={br['title_prior']:.2f} exp={br['experience']:.2f} co={br['company']:.2f} "
          f"| base={br['base']:.3f} neg={br['neg_mult']:.2f} behav={br['behav_mult']:.2f} "
          f"loc={br['loc_mult']:.2f} integ={br['integrity_penalty']:.1f}")
    top_ev = sorted(br["ev"].items(), key=lambda x: -x[1])[:6]
    print("    ev: " + ", ".join(f"{k}={v:.2f}" for k, v in top_ev if v > 0.2))
    print("    R: " + res.rows[rank-1]["reasoning"] if rank != "-" else "    (outside top100)")

print("\n===== top-100 composition =====")
from collections import Counter
top100 = res.top_records[:100]
tc = Counter(f["profile"]["current_title"] for (_, _, f, _) in top100)
lc = Counter(f["location_tier"] for (_, _, f, _) in top100)
cc = Counter(f["profile"]["country"] for (_, _, f, _) in top100)
print("titles:", dict(tc.most_common(15)))
print("loc_tier:", dict(lc))
print("country:", dict(cc))

# Quality guards: nothing off-domain or integrity-flagged should be here.
suspicious = [(i + 1, cid, f["title_tier"], f["integrity"]["penalty"])
              for i, (_, cid, f, _) in enumerate(top100)
              if f["title_tier"] in ("off", "low") or f["integrity"]["penalty"] > 0]
print(f"\noff/low-title or integrity-flagged in top-100: {len(suspicious)}")
for r, cid, tt, ip in suspicious:
    print(f"  rank {r} {cid} title_tier={tt} integ={ip}")

print("\n===== ranks 90-100 (the cutoff) =====")
for i, (sc, cid, f, _) in enumerate(top100[89:100], start=90):
    p = f["profile"]
    print(f"#{i} {cid} {sc:.4f} | {p['current_title']} @ {p['current_company']} "
          f"| {p['yoe']:.1f}y | {f['location_tier']}/{f['title_tier']}")
