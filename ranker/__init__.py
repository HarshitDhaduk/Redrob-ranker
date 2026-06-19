"""Redrob Intelligent Candidate Discovery & Ranking engine.

A pure-Python (zero third-party dependency) ranker that scores 100k candidate
profiles against a single, nuanced job description and emits the top-100.

Design goals
------------
* Read profiles, do not keyword-match them. The decisive signal is what a
  candidate's *career history describes them doing*, not the buzzwords stuffed
  into their skills list.
* Integrate behavioural / availability signals as a modifier on fit.
* Refuse to be fooled by the dataset's deliberate traps: keyword stuffers,
  plain-language hidden gems ("Tier 5s"), behavioural twins, and honeypots
  with subtly impossible profiles.
* Run end-to-end on CPU, no network, well within 5 minutes / 16 GB.
"""

from .pipeline import rank_candidates, RankResult  # noqa: F401

__all__ = ["rank_candidates", "RankResult"]
