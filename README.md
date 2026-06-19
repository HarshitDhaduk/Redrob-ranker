# Redrob — Intelligent Candidate Discovery & Ranking

An AI-recruiter that **reads** 100,000 candidate profiles against one nuanced job
description and returns an expertly-ranked top-100 — fast, on a CPU, with no
network and no hosted-LLM calls.

The job is *Senior AI Engineer — Founding Team*. The decisive insight of this
challenge is baked into the JD itself:

> *"The right answer is not 'find candidates whose skills section contains the
> most AI keywords.' … A candidate who has all the AI keywords listed as skills
> but whose title is 'Marketing Manager' is not a fit. A Tier-5 candidate may
> not use the words 'RAG' or 'Pinecone', but if their career history shows they
> built a recommendation system at a product company, they're a fit."*

So this system is built around one principle: **score the evidence of what a
candidate actually did, not the buzzwords they listed.**

---

## TL;DR / reproduce

```bash
# zero third-party dependencies — Python 3.9+ standard library only
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
# ~90s for 100k candidates on one CPU core, <1 GB RAM. Handles .jsonl and .jsonl.gz.

python validate_submission.py submission.csv   # -> "Submission is valid."
```

Optional hosted sandbox (UI only, not on the ranking path):

```bash
pip install streamlit && streamlit run app.py
```

---

## Why this design

A naive embedding/keyword ranker fails this dataset on purpose — the organisers
seeded four traps. Each maps to a specific defence in this system:

| Trap in the dataset | Why naive rankers fall for it | Defence here |
|---|---|---|
| **Keyword stuffers** — "HR Manager" / "Marketing Manager" with a skills list full of `FAISS, RAG, LoRA…` | Embeds/keyword-matches as a perfect hit | Concept evidence is read from **career *descriptions*** (weighted highest) and **titles**; the skills array is weighted low and **trust-adjusted**. A stuffer's descriptions are about marketing, so their must-have coverage is ~0. |
| **Plain-language Tier-5s** — a real retrieval engineer who writes "systems that decide what to show" instead of "RAG" | Keyword match misses them entirely | A hand-built **ontology of concept clusters** includes the plain-language paraphrases ("surface relevant", "what to show", "connect them to the most relevant matches"), so a hidden gem still lights up *ranking & retrieval*. |
| **Behavioural twins** — two near-identical profiles, one dormant & unresponsive | Static-profile rankers can't tell them apart | A **behavioural modifier** (availability, recruiter-response rate, recency, notice period, demand, verification) multiplies the fit score by ~0.6–1.1. |
| **Honeypots** (~80, forced to tier 0; >10% in top-100 = disqualified) — *subtly impossible* profiles (15y experience over a 7y career; 10 "expert" skills with 0 months used) | Looks elite to an embedder | An **integrity gate** detects the internal contradictions and crushes the score to ~0. |

This is exactly the system the JD says it wants: *"people who think about systems,
not frameworks."* No GPU, no API, no 100k LLM calls — a small, fast, **auditable**
ranker over engineered features, which is what scales to a 200k production pool.

---

## Architecture

```
candidates.jsonl ─► [ feature extraction ] ─► [ scoring ] ─► top-100 + reasoning ─► submission.csv
                          │                        │
            evidence-based concept hits     fit × penalties × behaviour
            integrity / honeypot check      × location × (1 − integrity)
```

A single streaming pass, O(candidates), pure Python. Memory is flat (only the
running top-250 is retained). Modules:

| File | Role |
|---|---|
| `ranker/jobspec.py` | **Deep JD understanding** — the JD compiled (offline) into weights, the 5-9y band, penalty multipliers, behavioural weights. The one place to tune priorities. |
| `ranker/ontology.py` | Weighted **concept clusters** (must-haves → nice-to-haves → negatives), title/company/location taxonomies. The "what the JD *means*" layer. |
| `ranker/text.py` | Dependency-free normaliser + n-gram indexer + concept matcher (multi-word phrases via set membership). |
| `ranker/features.py` | Per-candidate evidence: concept hits per source (descriptions ≫ titles ≫ skills), **trust-weighted** skills, career structure, behavioural signals. |
| `ranker/integrity.py` | **Honeypot filter** — experience-vs-career-span contradictions, phantom expertise, impossible dates. |
| `ranker/scoring.py` | Combines everything into a final score + a transparent breakdown. |
| `ranker/reasoning.py` | Fact-grounded, varied, honest 1–2 sentence justifications (no hallucination). |
| `ranker/pipeline.py` | Streaming orchestration + CSV writer (spec-compliant ordering & tie-breaks). |
| `rank.py` | CLI entry point. |
| `app.py` | Streamlit sandbox (optional). |
| `debug_explain.py`, `explain_id.py` | Dev tools used to calibrate the model (see git history). |

### How a score is built

```
base   = 0.52·core + 0.10·plus + 0.13·support + 0.10·title + 0.07·experience + 0.08·company
fit    = base × penalties(non-technical, CV-only, consulting-only, research-only, title-chaser, junior)
              × seniority(out-of-band damping for a *senior* role)
final  = fit × behavioural(0.6–1.1) × location(Pune/Noida > Tier-1 India > abroad) × (1 − integrity)
```

* **`core`** = coverage of the four must-haves: embeddings retrieval, vector/hybrid
  search, ranking & recommendation, ranking **evaluation** (NDCG/MRR/A-B). The
  role is defined by these, so `core` deliberately dominates.
* **Skills are trust-weighted**: a genuine `Weaviate (39 mo, 55 endorsements)`
  counts; a stuffer's `Weaviate (0 mo, 0 endorsements)` ≈ 0. Verified Redrob
  assessment scores nudge trust up or down. This is what lets real candidates
  whose tools live in the skills array score well *without* letting stuffers in.
* **Location** follows the JD: Pune/Noida preferred, Tier-1 India welcome,
  outside-India down-weighted (no visa sponsorship), `willing_to_relocate` helps.

---

## Results on the released pool (100,000 candidates)

* **Top-100 composition:** 100% relevant ML/AI titles (Applied ML / Recommendation
  Systems / AI / ML / NLP / Search Engineers, Senior Data Scientists), **100%
  India**, **100% within or adjacent to the 5-9y band**.
* **Honeypots in top-100: 0.** (68 impossible profiles flagged pool-wide; none ranked.)
* **No off-domain or integrity-flagged candidate** anywhere in the top-100.
* **Hidden gems recovered:** a deliberately plain-language Senior AI Engineer @ Adobe
  ("systems that decide what to show", no buzzwords) lands at rank 99 — included,
  not missed.
* **Runtime:** ~90s, single CPU core, <1 GB RAM, no network.

Example top rows (see `submission.csv`):

```
#1  Senior ML Engineer @ Zomato, 7.2y — ranking & recommendation (learning-to-rank),
    LLM/RAG work, production ML at scale. 61% recruiter response, sub-30-day notice, GitHub 95/100.
#41 Senior AI Engineer @ Apple, 5.9y — ranking & recommendation, production ML at scale,
    LLM fine-tuning. Strong availability: 80% response, sub-30-day notice, GitHub 97/100.
#99 Senior AI Engineer @ Adobe, 5.9y — ranking & recommendation (what to show),
    embeddings retrieval (sentence-transformers). [plain-language Tier-5 hidden gem]
```

---

## Compute & constraints (submission spec §3)

| Constraint | Limit | This system |
|---|---|---|
| Runtime | ≤ 5 min | ~90s |
| Memory | ≤ 16 GB | <1 GB |
| Compute | CPU only | CPU only |
| Network | off | none (no deps, no APIs) |
| Disk | ≤ 5 GB | a few MB |

No pre-computation step is required — the ranking command is fully self-contained.

## Honest limitations

* The JD model is hand-compiled; a new role means editing `jobspec.py` /
  `ontology.py` (by design — it keeps "deep JD understanding" inspectable and
  LLM-free at ranking time). An offline LLM could auto-compile a new `JobSpec`.
* Lexical-semantic matching (ontology + paraphrases) is intentionally chosen over
  dense embeddings because the honeypot warning shows pure embedding *fails* here;
  the trade-off is that a paraphrase outside the ontology can be missed. Precomputed
  embeddings could be added as a re-rank signal without breaking the compute budget.
