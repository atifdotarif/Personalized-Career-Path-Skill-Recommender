"""
Aggregate posting-level rows into role-level rows.

A *role* is a normalized job title. Each role aggregates:
  - the union of skills across its postings (with per-skill frequency)
  - posting count, median salary, work-type/experience distributions
  - sample postings (for UI)

Aggregating at the role level is the right unit for a recommender that
suggests "career paths" — the user does not want 300 copies of "Software
Engineer", they want one Software Engineer recommendation backed by
representative data.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


MIN_POSTINGS_PER_ROLE = 5  # roles backed by fewer postings are too noisy

# LinkedIn job URLs are deterministic from job_id, so we don't need to store
# the URL field — we synthesize on demand to keep the slim bundle small.
LINKEDIN_JOB_URL = "https://www.linkedin.com/jobs/view/{job_id}/"


def _sample_postings(group: pd.DataFrame, k: int = 5) -> list[dict]:
    """Pick k representative postings for a role and return UI-ready dicts."""
    # Prefer postings with the most extracted skills — they're typically the
    # most informative example of what the role looks like.
    head = group.sort_values("skill_count", ascending=False).head(k)
    samples: list[dict] = []
    for _, r in head.iterrows():
        samples.append({
            "job_id": int(r["job_id"]),
            "title": r["title"],
            "company": r.get("company_name") if pd.notna(r.get("company_name")) else "",
            "location": r.get("location") if pd.notna(r.get("location")) else "",
            "url": LINKEDIN_JOB_URL.format(job_id=int(r["job_id"])),
        })
    return samples


def aggregate_roles(postings: pd.DataFrame) -> pd.DataFrame:
    """Build a role-level dataframe from preprocessed postings."""
    logger.info("Aggregating %d postings into roles", len(postings))

    rows: list[dict] = []
    for title_norm, group in postings.groupby("title_norm"):
        n = len(group)
        if n < MIN_POSTINGS_PER_ROLE:
            continue

        skill_counter: Counter[str] = Counter()
        for s in group["skills"]:
            skill_counter.update(s)
        # Frequency = fraction of postings for this role that mention the skill.
        skill_freq = {k: v / n for k, v in skill_counter.items()}
        # Display title: most common original casing for this normalized title.
        display_title = group["title"].mode().iloc[0]

        salaries = group["normalized_salary"].dropna()
        rows.append(
            {
                "role_id": title_norm,
                "title": display_title,
                "n_postings": n,
                "skills": set(skill_counter.keys()),
                "skill_freq": skill_freq,
                "median_salary": float(salaries.median()) if len(salaries) else np.nan,
                "top_companies": ", ".join(
                    group["company_name"].dropna().value_counts().head(3).index.tolist()
                ),
                "experience_levels": dict(
                    group["formatted_experience_level"].dropna().value_counts()
                ),
                "work_types": dict(group["formatted_work_type"].dropna().value_counts()),
                "samples": _sample_postings(group, k=5),
            }
        )

    roles = pd.DataFrame(rows).sort_values("n_postings", ascending=False).reset_index(drop=True)
    logger.info("Built %d roles (≥%d postings each)", len(roles), MIN_POSTINGS_PER_ROLE)
    return roles
