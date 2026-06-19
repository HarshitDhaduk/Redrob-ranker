"""Behavioural tests for the ranker's anti-trap defences.

Run:  python -m unittest discover -s tests   (from the repo root)

These build synthetic profiles that isolate each trap and assert the system
responds correctly: honeypots are crushed, keyword-stuffers stay low, genuine
candidates (including buzzword-free hidden gems) score well, and a strong
candidate beats a stuffer with the same flashy skills list.
"""

import unittest

from ranker import features as feat
from ranker import scoring
from ranker import integrity


def _sig(**over):
    base = dict(
        profile_completeness_score=80, signup_date="2024-01-01",
        last_active_date="2026-06-01", open_to_work_flag=True,
        profile_views_received_30d=20, applications_submitted_30d=3,
        recruiter_response_rate=0.7, avg_response_time_hours=20,
        skill_assessment_scores={}, connection_count=300,
        endorsements_received=40, notice_period_days=30,
        expected_salary_range_inr_lpa={"min": 20, "max": 40},
        preferred_work_mode="hybrid", willing_to_relocate=True,
        github_activity_score=60, search_appearance_30d=120,
        saved_by_recruiters_30d=8, interview_completion_rate=0.8,
        offer_acceptance_rate=0.6, verified_email=True, verified_phone=True,
        linkedin_connected=True,
    )
    base.update(over)
    return base


def _cand(cid, title, yoe, summary, roles, skills, sig=None):
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Test", "headline": title, "summary": summary,
            "location": "Pune, Maharashtra", "country": "India",
            "years_of_experience": yoe, "current_title": title,
            "current_company": roles[0]["company"], "current_company_size": "201-500",
            "current_industry": roles[0]["industry"],
        },
        "career_history": roles, "education": [], "skills": skills,
        "redrob_signals": sig or _sig(),
    }


def _months_before(anchor_year, anchor_month, months):
    idx = anchor_year * 12 + (anchor_month - 1) - months
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}-01"


def _role(company, title, industry, months, desc, current=False):
    # Generate start/end consistent with `months` (anchored at the dataset's
    # 2026-06 "now") so the integrity date-checks don't fire on valid fixtures.
    if current:
        start = _months_before(2026, 6, months)
        end = None
    else:
        end = "2025-01-01"
        start = _months_before(2025, 1, months)
    return {"company": company, "title": title, "start_date": start,
            "end_date": end, "duration_months": months,
            "is_current": current, "industry": industry, "company_size": "201-500",
            "description": desc}


def _final(c):
    return scoring.score(feat.extract(c))["final"]


STRONG_DESC = ("Built a hybrid retrieval system combining BM25 with dense vector "
               "recall over millions of documents; owned the learning-to-rank model "
               "and the offline evaluation framework (NDCG, A/B testing) in production.")
# A genuine "Tier-5" hidden gem: describes the WORK (ranking, retrieval,
# evaluation) in plain language, with no tool buzzwords (no RAG/Pinecone/embeddings).
HIDDEN_DESC = ("Owned the ranking and retrieval systems that decide what to show users — "
               "connecting them to the most relevant matches across a large dataset. "
               "Evolved a hand-tuned scoring function into a learning-to-rank model, "
               "shipped to production, and measured relevance with offline metrics and "
               "A/B testing.")
AI_SKILLS = [
    {"name": "FAISS", "proficiency": "expert", "endorsements": 50, "duration_months": 40},
    {"name": "Pinecone", "proficiency": "expert", "endorsements": 45, "duration_months": 36},
    {"name": "Learning to Rank", "proficiency": "advanced", "endorsements": 30, "duration_months": 30},
]
AI_SKILLS_FAKE = [  # same names, but never used and never endorsed (stuffer signature)
    {"name": "FAISS", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
    {"name": "Pinecone", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
    {"name": "Learning to Rank", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
]


class TestIntegrity(unittest.TestCase):
    def test_experience_inflation_is_caught(self):
        c = _cand("CAND_0000001", "Recommendation Systems Engineer", 15.0,
                  "ML engineer with 7 years of experience.",
                  [_role("Swiggy", "Recommendation Systems Engineer", "Food Delivery", 36,
                         STRONG_DESC, current=True)], AI_SKILLS)
        a = integrity.assess(c)
        self.assertEqual(a["penalty"], 1.0)
        self.assertTrue(_final(c) < 0.05)

    def test_phantom_expertise_is_caught(self):
        c = _cand("CAND_0000002", "ML Engineer", 6.0, "ML engineer with 6 years.",
                  [_role("Acme", "ML Engineer", "Software", 72, STRONG_DESC, current=True)],
                  AI_SKILLS_FAKE + [{"name": "Docker", "proficiency": "expert",
                                     "endorsements": 0, "duration_months": 0}])
        self.assertGreaterEqual(integrity.assess(c)["penalty"], 0.9)

    def test_clean_profile_not_flagged(self):
        c = _cand("CAND_0000003", "ML Engineer", 6.0, "ML engineer with 6 years.",
                  [_role("Swiggy", "ML Engineer", "Food Delivery", 72, STRONG_DESC, current=True)],
                  AI_SKILLS)
        self.assertEqual(integrity.assess(c)["penalty"], 0.0)


class TestTraps(unittest.TestCase):
    def test_strong_beats_keyword_stuffer_with_same_skills(self):
        strong = _cand("CAND_0000010", "Senior AI Engineer", 6.5, "AI engineer, 6 years.",
                       [_role("Swiggy", "Senior AI Engineer", "Food Delivery", 78,
                              STRONG_DESC, current=True)], AI_SKILLS)
        stuffer = _cand("CAND_0000011", "Marketing Manager", 6.5,
                        "Marketing manager who runs campaigns and owns brand KPIs.",
                        [_role("Acme", "Marketing Manager", "Manufacturing", 78,
                               "Owned brand campaigns, SEO and social media growth.", current=True)],
                        AI_SKILLS)  # identical flashy skills
        self.assertGreater(_final(strong), _final(stuffer))
        self.assertLess(scoring.score(feat.extract(stuffer))["core"], 0.4)

    def test_hidden_gem_scores_well_without_buzzwords(self):
        gem = _cand("CAND_0000020", "Software Engineer", 6.0, "Engineer, 6 years.",
                    [_role("Flipkart", "Software Engineer", "E-commerce", 72,
                           HIDDEN_DESC, current=True)], [])  # NO skills listed at all
        self.assertGreater(scoring.score(feat.extract(gem))["core"], 0.4)

    def test_trust_weighting_real_vs_fake_skills(self):
        real = feat.extract(_cand("CAND_0000030", "ML Engineer", 6.0, "6 years.",
                            [_role("Swiggy", "ML Engineer", "Food Delivery", 72,
                                   "Worked on backend services.", current=True)], AI_SKILLS))
        fake = feat.extract(_cand("CAND_0000031", "ML Engineer", 6.0, "6 years.",
                            [_role("Swiggy", "ML Engineer", "Food Delivery", 72,
                                   "Worked on backend services.", current=True)], AI_SKILLS_FAKE))
        self.assertGreater(real["skill_cluster_trust"].get("vector_db_hybrid", 0),
                           3 * fake["skill_cluster_trust"].get("vector_db_hybrid", 0.001))


class TestBehaviour(unittest.TestCase):
    def test_dormant_unresponsive_is_downweighted(self):
        roles = [_role("Swiggy", "Senior AI Engineer", "Food Delivery", 78, STRONG_DESC, current=True)]
        available = _cand("CAND_0000040", "Senior AI Engineer", 6.5, "6 years.", roles, AI_SKILLS,
                          _sig(recruiter_response_rate=0.9, last_active_date="2026-06-15"))
        dormant = _cand("CAND_0000041", "Senior AI Engineer", 6.5, "6 years.", roles, AI_SKILLS,
                        _sig(recruiter_response_rate=0.05, last_active_date="2025-10-01",
                             open_to_work_flag=False))
        self.assertGreater(_final(available), _final(dormant))


if __name__ == "__main__":
    unittest.main(verbosity=2)
