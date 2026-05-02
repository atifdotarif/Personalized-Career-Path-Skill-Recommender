"""
Top-level recommender — wires together preprocessing, role aggregation,
TF-IDF, skill-gap analysis, and the bridge-path graph.

Usage
-----
    >>> from src.recommender import CareerRecommender
    >>> rec = CareerRecommender.build()        # builds from cached parquet
    >>> rec.recommend("python, sql, react", top_k=5)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from .data_loader import load_processed
from .role_aggregator import aggregate_roles
from .skill_extractor import extract_skills_from_user_input
from .skill_gap import SkillGap, analyze_gap
from .skill_graph import build_role_graph, build_skill_graph, find_bridge_path
from .tfidf_model import RoleTFIDFModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("data/processed/recommender.pkl")
DEFAULT_SLIM_MODEL_PATH = Path("data/processed/recommender_slim.pkl")


@dataclass
class Recommendation:
    role_id: str
    title: str
    match_score: float
    n_postings: int
    median_salary: float
    top_companies: str
    gap: SkillGap
    sample_job_ids: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "role_id": self.role_id,
            "title": self.title,
            "match_score": self.match_score,
            "n_postings": self.n_postings,
            "median_salary": self.median_salary,
            "top_companies": self.top_companies,
            "gap": self.gap.to_dict(),
            "sample_job_ids": self.sample_job_ids,
        }


class CareerRecommender:
    def __init__(
        self,
        postings: pd.DataFrame | None,
        roles: pd.DataFrame,
        tfidf: RoleTFIDFModel,
        skill_graph: nx.Graph,
        role_graph: nx.DiGraph,
        n_postings_total: int | None = None,
    ) -> None:
        # `postings` is optional: not needed for inference, only for rebuilding.
        # The slim deployment bundle drops it to stay under the GitHub 100 MB
        # file size limit.
        self.postings = postings
        self.roles = roles
        self.tfidf = tfidf
        self.skill_graph = skill_graph
        self.role_graph = role_graph
        self.n_postings_total = (
            n_postings_total
            if n_postings_total is not None
            else (len(postings) if postings is not None else 0)
        )

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, force_rebuild_data: bool = False) -> "CareerRecommender":
        postings = load_processed(force=force_rebuild_data)
        roles = aggregate_roles(postings)
        tfidf = RoleTFIDFModel().fit(roles)
        skill_graph = build_skill_graph(postings)
        role_graph = build_role_graph(roles)
        return cls(postings, roles, tfidf, skill_graph, role_graph)

    # ----------------------------------------------------------------- save/load
    def save(self, path: Path | str = DEFAULT_MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved recommender bundle to %s", path)

    def save_slim(self, path: Path | str = DEFAULT_SLIM_MODEL_PATH) -> None:
        """Pickle the inference-only fields — drops the postings DataFrame
        (the bulk of the bundle) so the result is small enough to commit and
        deploy to Streamlit Cloud."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Defensive: support unpickling from older bundles where
        # n_postings_total wasn't an attribute.
        n_total = getattr(self, "n_postings_total", None)
        if not n_total and self.postings is not None:
            n_total = len(self.postings)
        slim = CareerRecommender(
            postings=None,
            roles=self.roles,
            tfidf=self.tfidf,
            skill_graph=self.skill_graph,
            role_graph=self.role_graph,
            n_postings_total=int(n_total or 0),
        )
        with open(path, "wb") as f:
            pickle.dump(slim, f)
        logger.info("Saved slim recommender bundle to %s (size: %.1f MB)", path, path.stat().st_size / 1e6)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_MODEL_PATH) -> "CareerRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def load_or_build(cls, path: Path | str = DEFAULT_MODEL_PATH) -> "CareerRecommender":
        """Load the bundle at `path`. If absent, fall back to the slim bundle
        next to it. If both are absent, build from raw data."""
        path = Path(path)
        if path.exists():
            logger.info("Loading recommender bundle from %s", path)
            return cls.load(path)
        slim_path = path.with_name("recommender_slim.pkl")
        if slim_path.exists():
            logger.info("Loading slim recommender bundle from %s", slim_path)
            return cls.load(slim_path)
        rec = cls.build()
        rec.save(path)
        return rec

    # --------------------------------------------------------------- inference
    def parse_user_skills(self, raw: str | set[str]) -> set[str]:
        if isinstance(raw, set):
            return raw
        return extract_skills_from_user_input(raw)

    def recommend(
        self,
        user_input: str | set[str],
        top_k: int = 10,
    ) -> tuple[set[str], list[Recommendation]]:
        user_skills = self.parse_user_skills(user_input)
        ranked = self.tfidf.query(user_skills, top_k=top_k)
        recommendations: list[Recommendation] = []
        for _, row in ranked.iterrows():
            gap = analyze_gap(user_skills, row["skill_freq"])
            recommendations.append(
                Recommendation(
                    role_id=row["role_id"],
                    title=row["title"],
                    match_score=float(row["match_score"]),
                    n_postings=int(row["n_postings"]),
                    median_salary=float(row["median_salary"]) if pd.notna(row["median_salary"]) else float("nan"),
                    top_companies=row["top_companies"],
                    gap=gap,
                    sample_job_ids=row["sample_job_ids"],
                )
            )
        return user_skills, recommendations

    # ------------------------------------------------------------ bridge paths
    def closest_role_for_user(
        self,
        user_skills: set[str],
        min_role_skills: int = 4,
    ) -> str | None:
        """Find the role most similar to the user's skill set — i.e. their
        implicit "current role". Used as the source for bridge pathing.

        We rank candidates by `match_score * weighted_coverage` to balance
        two failure modes of plain cosine:
          - generalist role descriptions that just mention many skills
          - tiny roles with only 1–2 vocab skills that trivially "cover" the user

        Roles with fewer than `min_role_skills` distinct skills are excluded
        from the source-role search to avoid the second failure mode.
        """
        if not user_skills:
            return None
        ranked = self.tfidf.query(user_skills, top_k=50)
        if ranked.empty:
            return None
        scores: list[tuple[float, str]] = []
        for _, row in ranked.iterrows():
            if len(row["skills"]) < min_role_skills:
                continue
            gap = analyze_gap(user_skills, row["skill_freq"])
            combined = float(row["match_score"]) * gap.weighted_coverage
            scores.append((combined, row["role_id"]))
        if not scores:
            # No role passes the min-skills filter — fall back to top cosine.
            return ranked.iloc[0]["role_id"]
        scores.sort(key=lambda t: -t[0])
        return scores[0][1]

    def bridge_path(
        self,
        user_skills: set[str],
        target_role_id: str,
        max_hops: int = 3,
    ) -> list[dict] | None:
        source = self.closest_role_for_user(user_skills)
        if source is None or source == target_role_id:
            return None
        return find_bridge_path(self.role_graph, source, target_role_id, max_hops=max_hops)

    # ------------------------------------------------------------ adjacent skills
    def adjacent_skills_for(self, skills: set[str], top_k: int = 5) -> dict[str, list[tuple[str, float]]]:
        """For each user skill, return the strongest co-occurring skills the
        user does *not* already have. Useful as 'what to learn next' hints."""
        out: dict[str, list[tuple[str, float]]] = {}
        for s in sorted(skills):
            if s not in self.skill_graph:
                continue
            neighbors = sorted(
                ((n, self.skill_graph[s][n]["weight"]) for n in self.skill_graph.neighbors(s) if n not in skills),
                key=lambda kv: -kv[1],
            )[:top_k]
            if neighbors:
                out[s] = neighbors
        return out
