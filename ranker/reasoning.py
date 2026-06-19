"""Generate a 1-2 sentence, fact-grounded justification per ranked candidate.

Stage-4 review checks reasoning for: specific facts, JD connection, honest
concerns, NO hallucination, variation, and rank/tone consistency. So every
clause here is derived from fields that actually matched during scoring — we
never invent a skill or employer. Concerns are surfaced verbatim from the
penalty/behaviour notes, which keeps a rank-5 sober and a rank-95 modest.
"""

from __future__ import annotations

from .ontology import CORE_CLUSTERS

# clusters worth name-dropping as strengths, best-first
_STRENGTH_ORDER = CORE_CLUSTERS + ["ltr_models", "llm_finetune", "nlp_ir",
                                   "production_scale", "llm_apps", "mlops"]

# concise labels for reasoning prose (no nested parentheses)
_LABEL = {
    "embeddings_retrieval": "embeddings retrieval",
    "vector_db_hybrid": "vector/hybrid search",
    "ranking_reco": "ranking & recommendation",
    "evaluation": "ranking evaluation",
    "llm_finetune": "LLM fine-tuning",
    "llm_apps": "LLM/RAG work",
    "ltr_models": "learning-to-rank",
    "mlops": "ML production tooling",
    "nlp_ir": "NLP/IR",
    "production_scale": "production ML at scale",
    "python_eng": "ML engineering",
}

# render matched evidence phrases nicely (acronyms / casing / hyphenation)
_POLISH = {
    "a b testing": "A/B testing", "a b test": "A/B testing", "ab testing": "A/B testing",
    "ndcg": "NDCG", "mrr": "MRR", "map": "MAP", "llms": "LLMs", "llm": "LLMs",
    "rag": "RAG", "nlp": "NLP", "bm25": "BM25", "faiss": "FAISS", "lora": "LoRA",
    "qlora": "QLoRA", "peft": "PEFT", "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "mlflow": "MLflow", "kubeflow": "Kubeflow", "pinecone": "Pinecone",
    "qdrant": "Qdrant", "weaviate": "Weaviate", "milvus": "Milvus",
    "elasticsearch": "Elasticsearch", "opensearch": "OpenSearch",
    "sentence transformers": "sentence-transformers",
    "sentence transformer": "sentence-transformers",
    "learning to rank": "learning-to-rank", "two tower": "two-tower",
    "retrieval systems": "retrieval systems", "ranking layer": "ranking layer",
    "re ranking": "re-ranking", "live a b": "live A/B test", "live ab": "live A/B test",
    "fine tuning": "fine-tuning", "fine tune": "fine-tuning",
    "retrieval augmented": "retrieval-augmented generation",
}


def _polish(term: str) -> str:
    return _POLISH.get(term, term)


def _pick_term(terms: list[str]) -> str:
    """A concrete, specific phrase to quote (prefer multi-word / tool names)."""
    if not terms:
        return ""
    return _polish(sorted(terms, key=lambda t: (-len(t.split()), -len(t)))[0])


def _strengths(f: dict, br: dict, k: int = 3) -> list[str]:
    ev = br["ev"]
    cands = []
    for c in _STRENGTH_ORDER:
        if ev.get(c, 0) >= 0.45 and f["matched_terms"].get(c):
            # weight description-backed evidence above title/skill-only
            backed = f["desc_docs_with"].get(c, 0) > 0
            cands.append((ev[c] + (0.3 if backed else 0.0), c))
    cands.sort(reverse=True)
    out = []
    for _, c in cands[:k]:
        term = _pick_term(f["matched_terms"][c])
        label = _LABEL.get(c, c)
        out.append(f"{label} ({term})" if term else label)
    return out


def _concerns(f: dict, br: dict) -> list[str]:
    notes = list(br.get("exp_notes", [])) + list(br["neg_notes"]) + list(br["behav_notes"])
    if f["location_tier"] == "abroad":
        loc = f["profile"]["location"]
        notes.append(f"based in {loc} (outside India; JD doesn't sponsor visas)")
    if br["integrity_penalty"] > 0:
        notes.extend(f["integrity"]["reasons"])
    # de-dup, keep order, cap
    seen, uniq = set(), []
    for n in notes:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq[:2]


def build(f: dict, br: dict, rank: int) -> str:
    prof = f["profile"]
    yoe = prof["yoe"]
    title = prof["current_title"]
    company = prof["current_company"]
    s = f["signals"]

    strengths = _strengths(f, br)
    concerns = _concerns(f, br)

    head = f"{title} with {yoe:.1f} yrs"
    if company:
        head += f" at {company}"

    if strengths:
        lead = f"{head}; " + ", ".join(strengths[:2])
        if len(strengths) > 2:
            lead += f", plus {strengths[2]}"
        lead += "."
    else:
        lead = f"{head}; adjacent technical background with limited direct retrieval/ranking evidence."

    # one positive availability signal when it's genuinely good
    avail = ""
    if not concerns and s["response_rate"] >= 0.5 and s["days_since_active"] <= 45:
        bits = [f"{s['response_rate']:.0%} recruiter response"]
        if s["notice_days"] <= 30:
            bits.append("sub-30-day notice")
        if s["github"] >= 50:
            bits.append(f"GitHub activity {s['github']:.0f}/100")
        avail = "Strong availability: " + ", ".join(bits) + "."

    if concerns:
        tail = "Concerns: " + "; ".join(concerns) + "."
    else:
        tail = avail

    out = (lead + " " + tail).strip()
    return " ".join(out.split())  # collapse whitespace
