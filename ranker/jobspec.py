"""Structured representation of the released job description.

This is the compiled output of "deep job understanding": a close reading of
job_description.docx turned into machine-usable weights and rules. It is
authored offline (no LLM at ranking time) and is the single place to tune how
the JD's priorities translate into scoring.

The JD is a Senior AI Engineer (founding team) role whose decisive requirements
are: production embeddings-retrieval, vector/hybrid-search infra, ranking &
recommendation systems, and rigorous ranking *evaluation* — plus strong Python,
a product (not services/research) background, 5-9y experience, India
(Pune/Noida-preferred) location, and genuine availability.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class JobSpec:
    title: str = "Senior AI Engineer — Founding Team (Redrob AI)"

    # --- how the fit components combine into the base fit score (sum ~= 1) ---
    # The role is defined by its must-haves (embeddings retrieval, vector/hybrid
    # search, ranking & evaluation), so `core` deliberately dominates: a
    # candidate strong on generic LLM/MLOps but weak on retrieval/ranking must
    # not outrank one who has the actual must-haves.
    w_core: float = 0.52         # must-have concept coverage (retrieval/rank/eval)
    w_plus: float = 0.10         # nice-to-haves (fine-tuning, LTR, LLM apps)
    w_support: float = 0.13      # production / NLP / python / mlops evidence
    w_title: float = 0.10        # current+career title prior
    w_experience: float = 0.07   # closeness to the 5-9 (ideal 6-8) band
    w_company: float = 0.08      # product- vs services-company background

    # --- experience band (JD: "5-9, ideal 6-8, range not a hard rule") ------
    ideal_lo: float = 6.0
    ideal_hi: float = 8.0
    ok_lo: float = 5.0
    ok_hi: float = 9.0

    # --- penalty multipliers for the JD's explicit "do NOT want" list -------
    # (multiplicative on the base fit; 1.0 = no penalty)
    pen_non_tech_max: float = 0.85    # max share of fit removed when work is non-technical
    pen_cv_dominant: float = 0.70     # CV/speech-primary without NLP/IR exposure
    pen_consulting_only: float = 0.80 # entire career at IT-services/consulting
    pen_research_only: float = 0.78   # pure-research, no production deployment
    pen_title_chaser: float = 0.90    # job-hops every ~1.5y chasing titles
    pen_junior_title: float = 0.55    # explicit "Junior" seniority mismatch

    # --- behavioural modifier range (multiplier on fit) ---------------------
    # JD: "a perfect-on-paper candidate who hasn't logged in for 6 months and
    # has a 5% recruiter response rate is, for hiring purposes, not actually
    # available. Down-weight them appropriately."
    behav_floor: float = 0.60
    behav_ceiling: float = 1.12

    # --- location modifier (JD: Pune/Noida preferred; Tier-1 India welcome;
    #     outside India case-by-case, no visa sponsorship) -------------------
    loc_mult: dict = field(default_factory=lambda: {
        "preferred": 1.00, "tier1": 0.97, "other_india": 0.90, "abroad": 0.62,
    })
    loc_relocate_bonus: float = 0.08  # added (capped at 1.0) if willing_to_relocate

    # behavioural sub-signal weights (sum ~= 1) — see scoring.behavioural_score
    behav_weights: dict = field(default_factory=lambda: {
        "response": 0.22, "recency": 0.16, "open_to_work": 0.10, "notice": 0.10,
        "saved": 0.08, "interview": 0.06, "search": 0.05, "offer": 0.05,
        "completeness": 0.05, "verified": 0.05, "views": 0.04, "github": 0.04,
    })


JD = JobSpec()
