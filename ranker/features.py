"""Per-candidate feature extraction.

The central idea: build the candidate's *evidence* from what they describe doing
(career-history descriptions + summary, weighted highest), reinforced by their
titles, and only lightly by their self-declared skills (trust-weighted, because
the skills array is the keyword-stuffer's weapon). Concept evidence is computed
per JD cluster; behavioural raw signals and career structure are summarised for
the scorer.
"""

from __future__ import annotations

from datetime import date

from . import ontology as ont
from . import integrity
from .text import tokens, ngram_set, concept_hits

_TODAY = date(2026, 6, 19)

_PROF_BASE = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}


def _skill_trust(skill: dict, assessment: dict) -> float:
    """How much to believe a self-declared skill: proficiency tempered by how
    long it was actually used, endorsements, and any verified assessment score.
    A flashy 'expert' skill with 0 months of use and no endorsements ~= 0."""
    prof = _PROF_BASE.get(skill.get("proficiency", "intermediate"), 0.5)
    dur = skill.get("duration_months") or 0
    dur_factor = 0.15 if dur == 0 else min(1.0, 0.4 + dur / 36.0)
    endo = skill.get("endorsements") or 0
    e_factor = min(1.0, 0.6 + endo / 40.0)
    trust = prof * dur_factor * e_factor
    # Verified assessment is hard to fake -> strong adjustment up or down.
    score = assessment.get(skill.get("name"))
    if score is not None:
        trust *= max(0.5, min(1.35, 0.55 + score / 100.0))
    return max(0.0, min(1.0, trust))


def _days_since(d: str) -> float:
    try:
        return max(0.0, (_TODAY - date.fromisoformat(d[:10])).days)
    except (ValueError, TypeError):
        return 9999.0


def extract(candidate: dict) -> dict:
    p = candidate.get("profile", {})
    ch = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    sig = candidate.get("redrob_signals", {}) or {}
    assessment = sig.get("skill_assessment_scores", {}) or {}

    # ---- concept evidence from each source -------------------------------
    # descriptions (summary + per-role descriptions): count DOCS mentioning a
    # cluster, so breadth across a career counts more than one keyword.
    desc_docs_with: dict[str, int] = {}
    matched_terms: dict[str, set] = {}
    docs = [p.get("summary", "")] + [r.get("description", "") for r in ch]
    for d in docs:
        hits = concept_hits(ngram_set(tokens(d)))
        for cluster, terms in hits.items():
            desc_docs_with[cluster] = desc_docs_with.get(cluster, 0) + 1
            matched_terms.setdefault(cluster, set()).update(terms)

    # titles (current + career) and headline, as binary presence per cluster
    title_text = " ".join([p.get("current_title", "")] + [r.get("title", "") for r in ch])
    title_hits = concept_hits(ngram_set(tokens(title_text)))
    head_hits = concept_hits(ngram_set(tokens(p.get("headline", ""))))
    for cluster, terms in list(title_hits.items()) + list(head_hits.items()):
        matched_terms.setdefault(cluster, set()).update(terms)

    # skills -> cluster, weighted by trust (capped so skills can't dominate)
    skill_cluster_trust: dict[str, float] = {}
    matched_skills: dict[str, list] = {}
    for s in skills:
        name = s.get("name", "")
        hits = concept_hits(ngram_set(tokens(name)))
        if not hits:
            continue
        tr = _skill_trust(s, assessment)
        for cluster in hits:
            skill_cluster_trust[cluster] = skill_cluster_trust.get(cluster, 0.0) + tr
            matched_skills.setdefault(cluster, []).append((name, round(tr, 2)))

    return {
        "candidate_id": candidate.get("candidate_id"),
        "profile": {
            "name": p.get("anonymized_name", ""),
            "headline": p.get("headline", ""),
            "current_title": p.get("current_title", ""),
            "current_company": p.get("current_company", ""),
            "current_industry": p.get("current_industry", ""),
            "yoe": float(p.get("years_of_experience") or 0.0),
            "location": p.get("location", ""),
            "country": p.get("country", ""),
        },
        "title_tier": ont.title_tier(p.get("current_title", "")),
        "location_tier": ont.location_tier(p.get("location", ""), p.get("country", "")),
        # evidence
        "desc_docs_with": desc_docs_with,
        "title_hits": set(title_hits.keys()),
        "head_hits": set(head_hits.keys()),
        "skill_cluster_trust": skill_cluster_trust,
        "matched_terms": {k: sorted(v) for k, v in matched_terms.items()},
        "matched_skills": matched_skills,
        # career structure
        "career": _career_summary(ch),
        # behavioural raw
        "signals": _signal_summary(sig),
        # integrity / honeypot
        "integrity": integrity.assess(candidate),
    }


def _career_summary(ch: list) -> dict:
    n = len(ch)
    durations = [r.get("duration_months") or 0 for r in ch]
    avg_tenure = (sum(durations) / n) if n else 0.0
    product_roles = sum(1 for r in ch if ont.is_product(r.get("company", ""), r.get("industry", "")))
    consulting_roles = sum(1 for r in ch if ont.is_consulting(r.get("company", ""), r.get("industry", "")))
    # current/most-recent role: is it ML-relevant by title?
    cur = ch[0] if ch else {}
    cur_tier = ont.title_tier(cur.get("title", "")) if cur else "off"
    return {
        "num_roles": n,
        "avg_tenure_months": avg_tenure,
        "product_roles": product_roles,
        "consulting_roles": consulting_roles,
        "current_tier": cur_tier,
        "current_company": cur.get("company", ""),
    }


def _signal_summary(sig: dict) -> dict:
    return {
        "open_to_work": bool(sig.get("open_to_work_flag")),
        "days_since_active": _days_since(sig.get("last_active_date", "")),
        "response_rate": float(sig.get("recruiter_response_rate") or 0.0),
        "notice_days": int(sig.get("notice_period_days") or 90),
        "saved": int(sig.get("saved_by_recruiters_30d") or 0),
        "search_appearance": int(sig.get("search_appearance_30d") or 0),
        "profile_views": int(sig.get("profile_views_received_30d") or 0),
        "interview_rate": float(sig.get("interview_completion_rate") or 0.0),
        "offer_rate": float(sig.get("offer_acceptance_rate")
                            if sig.get("offer_acceptance_rate") is not None else -1.0),
        "completeness": float(sig.get("profile_completeness_score") or 0.0),
        "verified_email": bool(sig.get("verified_email")),
        "verified_phone": bool(sig.get("verified_phone")),
        "linkedin": bool(sig.get("linkedin_connected")),
        "github": float(sig.get("github_activity_score")
                        if sig.get("github_activity_score") is not None else -1.0),
        "willing_to_relocate": bool(sig.get("willing_to_relocate")),
        "preferred_work_mode": sig.get("preferred_work_mode", ""),
        "expected_salary": sig.get("expected_salary_range_inr_lpa", {}) or {},
    }
