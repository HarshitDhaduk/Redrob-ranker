"""Lightweight text processing: normalisation, n-gram indexing, concept hits.

No third-party libraries — just a fast regex tokenizer and set operations.
The trick that lets us match multi-word concepts ("learning to rank",
"vector search") cheaply across 100k docs is to expand each document into a set
of 1/2/3-grams once, then test concept phrases by set membership.
"""

from __future__ import annotations

import re
from .ontology import CONCEPTS

# Keep '+' and '#' so c++ / c# survive; everything else becomes a separator.
_NON_TOKEN = re.compile(r"[^a-z0-9+#]+")

# Words so common they carry no concept signal; dropped before n-gram building
# only matters for the optional TF-IDF similarity, not concept matching.
STOPWORDS = frozenset(
    """a an the and or of to for in on with at by from as is are was were be been
    being this that these those i we our my me you your they their it its he she
    his her our ours we've i've i'm we're using used use built build building
    designed design developed develop worked work working across most some few
    more than over also into out up down here there about which who whom whose
    what when where why how all any both each""".split()
)


def normalize(text: str) -> str:
    if not text:
        return ""
    return _NON_TOKEN.sub(" ", text.lower()).strip()


def tokens(text: str) -> list[str]:
    norm = normalize(text)
    return norm.split() if norm else []


def ngram_set(toks: list[str], n_max: int = 3) -> set[str]:
    """All 1..n_max grams as space-joined strings."""
    grams: set[str] = set(toks)
    L = len(toks)
    for n in (2, 3):
        if n > n_max:
            break
        for i in range(L - n + 1):
            grams.add(" ".join(toks[i:i + n]))
    return grams


# Pre-normalise every concept phrase once at import time.
_CONCEPT_PHRASES: dict[str, list[str]] = {
    name: [normalize(p) for p in spec["terms"]] for name, spec in CONCEPTS.items()
}


def concept_hits(grams: set[str]) -> dict[str, list[str]]:
    """Return {cluster: [matched phrase, ...]} for phrases present in `grams`."""
    out: dict[str, list[str]] = {}
    for cluster, phrases in _CONCEPT_PHRASES.items():
        hits = [p for p in phrases if p in grams]
        if hits:
            out[cluster] = hits
    return out
