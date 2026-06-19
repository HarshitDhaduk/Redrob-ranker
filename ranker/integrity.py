"""Profile-integrity checks — the honeypot / "subtly impossible profile" filter.

The dataset hides ~80 honeypots that are forced to relevance tier 0 in the
ground truth, and a submission with >10% honeypots in its top-100 is
disqualified. They are *designed to look perfect to a keyword/embedding ranker*
(e.g. a "Recommendation Systems Engineer" whose profile name-drops every
retrieval tool) but contain an internal contradiction that profile-reading
catches.

Observed honeypot signatures in this dataset (verified by inspection):

1. INFLATED EXPERIENCE — `profile.years_of_experience` (e.g. 15.2) contradicts
   both the self-summary ("...with 7.2 years...") and the actual career span
   computed from start/end dates (~7.2 yrs). This is the most common and most
   dangerous family, because it lands on otherwise-perfect ML profiles.
2. PHANTOM EXPERTISE — several skills at "expert" proficiency with
   `duration_months == 0` ("expert in 10 skills with 0 years used").
3. DATE CONTRADICTIONS — a role's `duration_months` disagrees with its own
   start/end dates; a single role longer than the whole career; end before
   start; or a start date in the future.

We return a 0..1 `penalty` (1.0 = certainly impossible) plus the reasons, so the
scorer can crush the score and the reasoning string can stay honest. We do NOT
special-case individual IDs — only general consistency, exactly as the spec
recommends ("we expect a good ranking system to naturally avoid them").
"""

from __future__ import annotations

import re
from datetime import date

_TODAY = date(2026, 6, 19)          # dataset reference "now" (latest activity)
_SUMMARY_YEARS = re.compile(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*years?", re.I)


def _pdate(s):
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def career_span_years(career_history) -> float:
    start = end = None
    for r in career_history:
        sd = _pdate(r.get("start_date"))
        ed = _pdate(r.get("end_date")) or _TODAY
        if sd and (start is None or sd < start):
            start = sd
        if end is None or ed > end:
            end = ed
    if not start:
        return 0.0
    return max(0.0, (end - start).days / 365.25)


def summary_years(summary: str):
    m = _SUMMARY_YEARS.search(summary or "")
    return float(m.group(1)) if m else None


def assess(candidate) -> dict:
    """Return {'penalty': float 0..1, 'reasons': [str], 'span_years': float}."""
    p = candidate.get("profile", {})
    ch = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    yoe = float(p.get("years_of_experience") or 0.0)
    span = career_span_years(ch)
    reasons: list[str] = []
    penalty = 0.0

    # 1) Experience inflation: yoe far exceeds the demonstrable career span.
    #    (span is an *upper bound* on real experience — roles may overlap — so
    #     yoe materially above span is physically impossible.)
    if span > 0 and yoe - span > 2.0:
        penalty = max(penalty, 1.0)
        reasons.append(
            f"stated {yoe:.0f}y experience but career history spans only ~{span:.0f}y"
        )

    # 1b) Cross-check against the self-written summary's stated years.
    sy = summary_years(p.get("summary", ""))
    if sy is not None and abs(sy - yoe) > 2.5:
        penalty = max(penalty, 1.0)
        reasons.append(f"summary says ~{sy:.0f}y but profile claims {yoe:.0f}y")

    # 2) Phantom expertise: many "expert" skills never actually used.
    expert_zero = [s for s in skills
                   if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0]
    if len(expert_zero) >= 3:
        penalty = max(penalty, 1.0)
        reasons.append(f"{len(expert_zero)} 'expert' skills with 0 months of use")
    else:
        adv_zero = [s for s in skills
                    if s.get("proficiency") in ("expert", "advanced")
                    and (s.get("duration_months") or 0) == 0]
        if len(adv_zero) >= 5:
            penalty = max(penalty, 0.9)
            reasons.append(f"{len(adv_zero)} advanced/expert skills with 0 months of use")

    # 3) Date contradictions inside career history.
    for r in ch:
        sd = _pdate(r.get("start_date"))
        ed = _pdate(r.get("end_date"))
        dur = r.get("duration_months") or 0
        if sd and sd > _TODAY:
            penalty = max(penalty, 1.0)
            reasons.append("a role starts in the future")
        if sd and ed and ed < sd:
            penalty = max(penalty, 1.0)
            reasons.append("a role ends before it starts")
        if sd:
            end_ref = ed or _TODAY
            months = (end_ref.year - sd.year) * 12 + (end_ref.month - sd.month)
            if abs(months - dur) > 9:
                penalty = max(penalty, 0.95)
                reasons.append("a role's stated duration contradicts its dates")
        if yoe > 0 and dur > yoe * 12 + 18:
            penalty = max(penalty, 1.0)
            reasons.append("a single role is longer than the whole stated career")

    # de-duplicate reasons, keep order
    seen = set()
    uniq = [r for r in reasons if not (r in seen or seen.add(r))]
    return {"penalty": penalty, "reasons": uniq, "span_years": span}
