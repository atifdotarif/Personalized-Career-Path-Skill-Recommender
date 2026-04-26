"""
TF-IDF + cosine similarity over role skill bags.

Each role is represented as a "document" of canonical skill tokens, where each
skill is repeated proportionally to how often it appears across the role's
postings (weight 1–10). TF-IDF then learns which skills are *characteristic*
of each role — common skills like "Excel" get down-weighted, distinctive
skills like "Kubernetes" or "PyTorch" get up-weighted.

The user's skill set is encoded as a query document with each skill appearing
once. Cosine similarity gives the Match Score (0–1) the proposal describes.

We deliberately use a custom analyzer that emits canonical skill tokens
verbatim. Stopword removal and stemming are skipped — our vocabulary is
already curated.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# How many times to repeat a skill in a role's document, given its frequency
# across postings. Cap at 10 so a universally-mentioned skill doesn't dominate.
def _skill_doc(skill_freq: dict[str, float]) -> list[str]:
    tokens: list[str] = []
    for skill, freq in skill_freq.items():
        repeats = max(1, min(10, int(round(freq * 10))))
        tokens.extend([skill] * repeats)
    return tokens


def _identity_analyzer(tokens: list[str]) -> list[str]:
    return tokens


class RoleTFIDFModel:
    """TF-IDF index of roles. Fit once, query many times."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer=_identity_analyzer,
            lowercase=False,  # canonical names are case-sensitive ("R" ≠ "r")
            norm="l2",
            sublinear_tf=True,
        )
        self.role_matrix = None
        self.roles: pd.DataFrame | None = None

    def fit(self, roles: pd.DataFrame) -> "RoleTFIDFModel":
        logger.info("Fitting TF-IDF over %d roles", len(roles))
        documents = [_skill_doc(sf) for sf in roles["skill_freq"]]
        self.role_matrix = self.vectorizer.fit_transform(documents)
        self.roles = roles.reset_index(drop=True)
        return self

    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_)

    def query(self, user_skills: set[str], top_k: int = 10) -> pd.DataFrame:
        """Return top-k roles ranked by cosine similarity to user skills."""
        if self.role_matrix is None or self.roles is None:
            raise RuntimeError("Call fit() first")
        if not user_skills:
            return self.roles.head(0).assign(match_score=[])

        # Drop user skills outside the vocabulary so the vectorizer doesn't
        # silently swallow them (those skills will still surface in the gap
        # analyzer).
        known = [s for s in user_skills if s in self.vectorizer.vocabulary_]
        if not known:
            return self.roles.head(0).assign(match_score=[])

        query_vec = self.vectorizer.transform([known])
        sims = cosine_similarity(query_vec, self.role_matrix).ravel()
        # Argpartition for top-k then sort that small slice — cheaper than full sort.
        k = min(top_k, len(sims))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        result = self.roles.iloc[top_idx].copy()
        result["match_score"] = sims[top_idx]
        return result.reset_index(drop=True)
