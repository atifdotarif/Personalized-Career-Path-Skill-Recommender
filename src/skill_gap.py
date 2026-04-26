"""
Skill-gap analysis — pure set operations over canonical skill names.

Given a user's skills U and a role's required skills R (with frequencies),
return:
  - matched   = U ∩ R         (what the user already brings)
  - missing   = R \\ U         (skills the user lacks for this role)
  - extras    = U \\ R         (skills the user has that the role doesn't ask for)
  - coverage  = |matched| / |R|        (proportion of required skills covered)
  - weighted_coverage                  (same, weighted by per-skill frequency)

Weighted coverage answers "what % of this role's typical skill mentions does
the user cover?" — better signal than raw set overlap because it discounts
skills only mentioned in 1 of 200 postings for the role.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SkillGap:
    matched: list[str]
    missing: list[str]
    extras: list[str]
    coverage: float
    weighted_coverage: float
    missing_ranked: list[tuple[str, float]]  # missing skills sorted by importance to role

    def to_dict(self) -> dict:
        return {
            "matched": self.matched,
            "missing": self.missing,
            "extras": self.extras,
            "coverage": self.coverage,
            "weighted_coverage": self.weighted_coverage,
            "missing_ranked": self.missing_ranked,
        }


def analyze_gap(
    user_skills: set[str],
    role_skill_freq: dict[str, float],
) -> SkillGap:
    role_skills = set(role_skill_freq.keys())

    matched_set = user_skills & role_skills
    missing_set = role_skills - user_skills
    extras_set = user_skills - role_skills

    # Sort each list by frequency in the role (most important first), then
    # alphabetically as tiebreaker — gives a deterministic, useful order.
    def by_role_freq(skill: str) -> tuple[float, str]:
        return (-role_skill_freq.get(skill, 0.0), skill)

    matched = sorted(matched_set, key=by_role_freq)
    missing = sorted(missing_set, key=by_role_freq)
    extras = sorted(extras_set)

    # Plain coverage.
    coverage = len(matched_set) / len(role_skills) if role_skills else 0.0

    # Weighted coverage = sum of frequencies of matched skills / sum of all role skill frequencies.
    total_weight = float(np.sum(list(role_skill_freq.values()))) or 1.0
    matched_weight = float(np.sum([role_skill_freq[s] for s in matched_set]))
    weighted_coverage = matched_weight / total_weight

    missing_ranked = [(s, role_skill_freq[s]) for s in missing]

    return SkillGap(
        matched=matched,
        missing=missing,
        extras=extras,
        coverage=coverage,
        weighted_coverage=weighted_coverage,
        missing_ranked=missing_ranked,
    )
