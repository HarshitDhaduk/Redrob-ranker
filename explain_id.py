#!/usr/bin/env python3
"""Explain the score for specific candidate IDs (full breakdown + evidence)."""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from ranker import features as feat, scoring

path = sys.argv[1]
want = set(sys.argv[2:])
with open(path, encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        c = json.loads(line)
        if c["candidate_id"] not in want:
            continue
        f = feat.extract(c); br = scoring.score(f); p = f["profile"]
        print(f"\n########## {c['candidate_id']} :: {p['current_title']} @ {p['current_company']} "
              f":: {p['yoe']}y :: {p['location']},{p['country']} [{f['location_tier']}/{f['title_tier']}]")
        print("HEADLINE:", p["headline"])
        print("SUMMARY:", c["profile"]["summary"][:500])
        print("CAREER DESCS:")
        for r in c["career_history"]:
            print(f"  - {r['title']} @ {r['company']} [{r['industry']}]: {r['description'][:170]}")
        print("FINAL=%.4f | core=%.2f plus=%.2f sup=%.2f title=%.2f exp=%.2f co=%.2f base=%.3f "
              "neg=%.2f sen=%.2f behav=%.2f loc=%.2f integ=%.1f" % (
              br["final"], br["core"], br["plus"], br["support"], br["title_prior"],
              br["experience"], br["company"], br["base"], br["neg_mult"], br["sen_mult"],
              br["behav_mult"], br["loc_mult"], br["integrity_penalty"]))
        print("EV:", ", ".join(f"{k}={v:.2f}" for k, v in sorted(br["ev"].items(), key=lambda x:-x[1]) if v > 0.15))
        print("MATCHED TERMS:", {k: v for k, v in f["matched_terms"].items() if k in br["ev"] and br["ev"][k] > 0.3})
        print("NOTES:", br.get("exp_notes"), br["neg_notes"], br["behav_notes"])
        want.discard(c["candidate_id"])
        if not want:
            break
