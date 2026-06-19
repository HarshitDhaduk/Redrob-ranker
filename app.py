"""Sandbox demo for the Redrob ranker (Streamlit).

Satisfies the submission-spec sandbox requirement: a hosted environment that
accepts a small candidate sample (<=100), runs the ranking end-to-end on CPU
within the compute budget, and shows the ranked output with reasoning.

    pip install streamlit
    streamlit run app.py

Upload a .jsonl/.json(.gz) candidate file, or use the bundled sample. The exact
same `ranker` package powers the full 100k run via rank.py — this is only a UI.
"""

from __future__ import annotations

import json
import os
import tempfile

import streamlit as st

from ranker.pipeline import rank_candidates

st.set_page_config(page_title="Redrob Intelligent Candidate Ranker", layout="wide")
st.title("Redrob — Intelligent Candidate Discovery & Ranking")
st.caption(
    "Senior AI Engineer (Founding Team) JD · evidence-based ranking · "
    "pure-Python, CPU-only, no network. Reads profiles instead of keyword-matching them."
)

with st.sidebar:
    st.header("Input")
    up = st.file_uploader("Candidate file (.jsonl / .json / .gz)",
                          type=["jsonl", "json", "gz"])
    top_n = st.slider("How many to rank", 5, 100, 25)
    sample_path = "sample_candidates.jsonl"
    use_sample = st.checkbox("Use bundled 50-candidate sample",
                             value=not up and os.path.exists(sample_path))
    st.markdown(
        "**What it does**\n"
        "- Deep JD understanding (must-haves, disqualifiers)\n"
        "- Semantic fit from career *descriptions*, not buzzwords\n"
        "- Honeypot / impossible-profile filter\n"
        "- Behavioural availability modifier"
    )


def _resolve_input() -> str | None:
    if up is not None:
        suffix = "." + up.name.split(".")[-1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(up.getvalue())
        tmp.close()
        # a .json array -> normalise to jsonl
        if up.name.endswith(".json"):
            data = json.load(open(tmp.name, encoding="utf-8"))
            jl = tmp.name + ".jsonl"
            with open(jl, "w", encoding="utf-8") as f:
                for c in data:
                    f.write(json.dumps(c) + "\n")
            return jl
        return tmp.name
    if use_sample and os.path.exists(sample_path):
        return sample_path
    return None


path = _resolve_input()
if not path:
    st.info("Upload a candidate file or tick the bundled sample to begin.")
    st.stop()

with st.spinner("Ranking…"):
    res = rank_candidates(path, top_n=top_n, progress=False)

c1, c2, c3 = st.columns(3)
c1.metric("Candidates scored", f"{res.n_total:,}")
c2.metric("Ranking time", f"{res.elapsed_s:.2f}s")
c3.metric("Integrity-flagged (pool)", res.n_integrity_flagged)

st.subheader(f"Top {len(res.rows)}")
st.dataframe(
    [{"rank": r["rank"], "candidate_id": r["candidate_id"],
      "score": r["score"], "reasoning": r["reasoning"]} for r in res.rows],
    use_container_width=True, hide_index=True,
)

st.subheader("Score breakdown (why)")
recs = {cid: (f, br) for (_, cid, f, br) in res.top_records}
for r in res.rows:
    f, br = recs[r["candidate_id"]]
    p = f["profile"]
    with st.expander(f"#{r['rank']}  {p['current_title']} @ {p['current_company']} "
                     f"— {p['yoe']:.1f}y — score {r['score']}"):
        st.write(r["reasoning"])
        st.json({
            "must_have_coverage(core)": round(br["core"], 3),
            "nice_to_haves(plus)": round(br["plus"], 3),
            "support(prod/nlp/mlops)": round(br["support"], 3),
            "title_prior": round(br["title_prior"], 3),
            "experience_fit": round(br["experience"], 3),
            "company_fit": round(br["company"], 3),
            "penalty_multiplier": round(br["neg_mult"] * br["sen_mult"], 3),
            "behavioural_multiplier": round(br["behav_mult"], 3),
            "location_multiplier": round(br["loc_mult"], 3),
            "integrity_penalty": br["integrity_penalty"],
            "top_evidence": {k: round(v, 2) for k, v in
                             sorted(br["ev"].items(), key=lambda x: -x[1])[:6] if v > 0.2},
        })
