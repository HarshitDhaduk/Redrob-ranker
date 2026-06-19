"""Scoring: turn extracted features into a single fit score + a breakdown.

Pipeline of a score:

    base fit  = weighted blend of
                  core concept coverage (retrieval / vector-db / ranking / eval)
                + nice-to-haves + production/NLP support
                + title prior + experience fit + company background
    fit_adj   = base fit x JD "do-NOT-want" penalties
                  (non-technical, CV-only, consulting-only, research-only,
                   title-chaser, junior)
    final     = fit_adj
                x behavioural modifier  (availability / credibility / demand)
                x location modifier      (Pune-Noida > Tier-1 India > abroad)
                x (1 - integrity penalty)  (honeypots -> ~0)

Every intermediate lands in `breakdown` so the reasoning layer can cite real
numbers and the design stays auditable.
"""

from __future__ import annotations

import math

from . import ontology as ont
from .jobspec import JD


def _saturate(raw: float) -> float:
    # diminishing returns: 1 evidence ~0.59, 2 ~0.83, 3 ~0.93
    return 1.0 - math.exp(-0.9 * raw)


def cluster_evidence(f: dict) -> dict[str, float]:
    """Evidence strength in [0,1] per concept cluster, blending sources with
    descriptions weighted highest and skills (trust-weighted) lowest."""
    ev: dict[str, float] = {}
    for cluster in ont.CONCEPTS:
        desc = f["desc_docs_with"].get(cluster, 0)              # # of career docs
        title = 1.0 if cluster in f["title_hits"] else 0.0
        head = 1.0 if cluster in f["head_hits"] else 0.0
        # `skill_cluster_trust` is already trust-weighted per skill, so a genuine
        # tool (e.g. Weaviate, 39 mo, 55 endorsements) contributes while a
        # stuffer's 0-month / 0-endorsement skill is ~0. Description evidence
        # still dominates (each career doc adds 1.0 vs skills capped well below).
        skill = min(1.3, f["skill_cluster_trust"].get(cluster, 0.0))
        raw = 1.0 * desc + 0.7 * title + 0.4 * head + 0.6 * skill
        ev[cluster] = _saturate(raw)
    return ev


def _weighted(ev, clusters):
    num = sum(ev[c] * ont.CONCEPTS[c]["weight"] for c in clusters)
    den = sum(ont.CONCEPTS[c]["weight"] for c in clusters)
    return num / den if den else 0.0


def experience_fit(yoe: float, junior: bool) -> float:
    """Additive within-/near-band differentiator (ideal 6-8 gets the edge)."""
    if junior:
        return 0.40
    y = yoe
    if JD.ideal_lo <= y <= JD.ideal_hi:
        f = 1.0
    elif JD.ok_lo <= y <= JD.ok_hi:
        f = 0.92
    elif 4 <= y < 10.5:
        f = 0.80
    elif 10.5 <= y < 12:
        f = 0.62
    elif 3 <= y < 4:
        f = 0.58
    elif 12 <= y < 14:
        f = 0.45
    else:                       # <3 (too junior) or >=14 (too senior for IC)
        f = 0.30
    return f


def seniority_multiplier(yoe: float) -> float:
    """Multiplicative damping for a *senior* role: a brilliant skillset with
    3 years is still not a senior hire. Out-of-band candidates are not excluded
    (JD: "range, not a requirement") — just down-weighted versus in-band peers."""
    y = yoe
    if 5.0 <= y <= 9.5:
        return 1.00
    if 4.0 <= y < 5.0 or 9.5 < y <= 11.0:
        return 0.92
    if 3.0 <= y < 4.0:
        return 0.80
    if 11.0 < y <= 13.0:
        return 0.85
    if 2.0 <= y < 3.0:
        return 0.68
    if y > 13.0:
        return 0.70
    return 0.55                 # < 2 years


def behavioural_score(s: dict) -> tuple[float, list[str]]:
    """Availability + credibility + demand, each normalised to [0,1]."""
    w = JD.behav_weights
    notes = []

    resp = s["response_rate"]
    if resp < 0.15:
        notes.append(f"low recruiter response rate ({resp:.0%})")
    dsa = s["days_since_active"]
    recency = 1.0 if dsa <= 30 else max(0.0, 1.0 - (dsa - 30) / 240.0)
    if dsa > 120:
        notes.append(f"inactive for ~{int(dsa)} days")
    otw = 1.0 if s["open_to_work"] else 0.40
    nd = s["notice_days"]
    notice = 1.0 if nd <= 30 else 0.82 if nd <= 60 else 0.55 if nd <= 90 else 0.30
    if nd > 90:
        notes.append(f"{nd}-day notice period")
    saved = min(1.0, s["saved"] / 15.0)
    interview = s["interview_rate"]
    search = min(1.0, s["search_appearance"] / 200.0)
    offer = 0.6 if s["offer_rate"] < 0 else s["offer_rate"]
    completeness = s["completeness"] / 100.0
    verified = (int(s["verified_email"]) + int(s["verified_phone"]) + int(s["linkedin"])) / 3.0
    views = min(1.0, s["profile_views"] / 40.0)
    github = 0.45 if s["github"] < 0 else s["github"] / 100.0

    score = (
        w["response"] * resp + w["recency"] * recency + w["open_to_work"] * otw
        + w["notice"] * notice + w["saved"] * saved + w["interview"] * interview
        + w["search"] * search + w["offer"] * offer + w["completeness"] * completeness
        + w["verified"] * verified + w["views"] * views + w["github"] * github
    )
    return max(0.0, min(1.0, score)), notes


def location_modifier(f: dict) -> float:
    tier = f["location_tier"]
    m = JD.loc_mult[tier]
    if tier in ("other_india", "tier1", "abroad") and f["signals"]["willing_to_relocate"]:
        m = min(1.0, m + JD.loc_relocate_bonus)
    return m


def _negatives(f: dict, ev: dict) -> tuple[float, list[str]]:
    """Multiplicative penalty for the JD's explicit 'do NOT want' signals."""
    mult = 1.0
    notes = []
    prof = f["profile"]
    career = f["career"]

    # share of positive ML signal that is CV/speech vs NLP/IR & retrieval
    cv = ev["cv_speech"]
    nlp_ir = max(ev["nlp_ir"], ev["embeddings_retrieval"], ev["ranking_reco"])
    if cv >= 0.6 and cv > nlp_ir + 0.15:
        mult *= JD.pen_cv_dominant
        notes.append("background is primarily computer-vision/speech, not NLP/IR")

    # non-technical content dominates the descriptions (keyword-stuffer host)
    nt = f["desc_docs_with"].get("non_tech", 0)
    total_docs = 1 + career["num_roles"]
    nt_frac = nt / total_docs if total_docs else 0.0
    core_sig = _weighted(ev, ont.CORE_CLUSTERS)
    if nt_frac > 0 and core_sig < 0.5:
        mult *= (1.0 - JD.pen_non_tech_max * min(1.0, nt_frac))
        if nt_frac >= 0.4:
            notes.append("career is largely non-technical despite listed skills")

    # entire career at IT-services / consulting houses
    if career["consulting_roles"] >= 2 and career["product_roles"] == 0:
        mult *= JD.pen_consulting_only
        notes.append("career entirely at IT-services/consulting firms")

    # pure research with no production deployment
    research = ev["research_only"]
    production = ev["production_scale"]
    if research >= 0.6 and production < 0.4:
        mult *= JD.pen_research_only
        notes.append("research-heavy with little production-deployment evidence")

    # title-chaser: a genuine hopping pattern (JD: "switching every ~1.5y").
    # Kept strict so strong candidates with a few solid stints aren't penalised.
    at = career["avg_tenure_months"]
    if (career["num_roles"] >= 4 and 0 < at < 15) or (career["num_roles"] >= 5 and 0 < at < 21):
        mult *= JD.pen_title_chaser
        notes.append(f"frequent job changes (~{at:.0f}-mo avg tenure across {career['num_roles']} roles)")

    return mult, notes


def score(f: dict) -> dict:
    ev = cluster_evidence(f)
    prof = f["profile"]
    junior = "junior" in prof["current_title"].lower()

    core = _weighted(ev, ont.CORE_CLUSTERS)
    plus = _weighted(ev, ont.PLUS_CLUSTERS)
    support = _weighted(ev, ont.SUPPORT_CLUSTERS)
    title_prior = ont.TITLE_PRIOR[f["title_tier"]]
    exp = experience_fit(prof["yoe"], junior)
    company = min(1.0, 0.5 + 0.25 * f["career"]["product_roles"])

    base = (JD.w_core * core + JD.w_plus * plus + JD.w_support * support
            + JD.w_title * title_prior + JD.w_experience * exp + JD.w_company * company)

    neg_mult, neg_notes = _negatives(f, ev)
    if junior:
        neg_mult *= JD.pen_junior_title
        neg_notes.append("explicitly junior-titled for a senior role")
    sen_mult = seniority_multiplier(prof["yoe"])
    fit_adj = base * neg_mult * sen_mult

    behav, behav_notes = behavioural_score(f["signals"])
    behav_mult = JD.behav_floor + (JD.behav_ceiling - JD.behav_floor) * behav
    loc_mult = location_modifier(f)
    integ = f["integrity"]["penalty"]

    final = fit_adj * behav_mult * loc_mult * (1.0 - integ)

    # surface an experience concern for the reasoning layer when off-band
    exp_notes = []
    if not junior:
        if prof["yoe"] < JD.ok_lo:
            exp_notes.append(f"{prof['yoe']:.1f}y experience is below the 5-9y band")
        elif prof["yoe"] > JD.ok_hi + 2:
            exp_notes.append(f"{prof['yoe']:.1f}y is senior-heavy for an IC role")

    return {
        "final": final,
        "ev": ev,
        "core": core, "plus": plus, "support": support,
        "title_prior": title_prior, "experience": exp, "company": company,
        "base": base, "neg_mult": neg_mult, "sen_mult": sen_mult, "fit_adj": fit_adj,
        "behavioural": behav, "behav_mult": behav_mult, "loc_mult": loc_mult,
        "integrity_penalty": integ,
        "neg_notes": neg_notes, "behav_notes": behav_notes, "exp_notes": exp_notes,
    }
