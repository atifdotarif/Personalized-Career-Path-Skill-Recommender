"""
Quantitative evaluation of the Career Path Recommender.

Run from the project root:

    .venv/Scripts/python.exe scripts/evaluate.py

Outputs a JSON of metrics + a console report. The report numbers in
docs/REPORT.md are pulled from this script's output.

Metrics
-------
1. self_recommendation_precision_at_k
   For each role-aggregated test posting, encode its extracted skills as a
   user profile and check whether the posting's true role appears in the
   recommender's top-k. Precision@1 / @5 / @10.

2. held_out_skill_recovery
   For each role with >= 6 distinct skills, hide each of the top-3 skills in
   turn from the role's profile, treat the remainder as the user's skills,
   and check whether the recommender surfaces the hidden skill in the
   missing-skill ranking of the same role. Recall@k for the missing skill.

3. coverage
   Across N synthetic user profiles, what fraction of the 1060 roles ever
   appear in any user's top-10?

4. diversity
   Mean intra-list pairwise Jaccard distance between role skill sets in the
   top-5 recommendations, averaged across test profiles. Higher = the
   recommender shows more varied roles per query.

5. latency
   Mean / p95 milliseconds per recommend() call after warm-up.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommender import CareerRecommender


# Reproducible sampling.
RNG_SEED = 42


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - (len(a & b) / max(1, len(a | b)))


# ------------------------------------------------------------- self-rec precision
def self_recommendation_precision(rec: CareerRecommender, n_samples: int = 2000) -> dict:
    """Pick N postings at random; treat each posting's skills as a user query
    and check whether the posting's true role is in the top-k recommendations.
    """
    rng = random.Random(RNG_SEED)
    if rec.postings is None:
        # Slim bundle path — fall back to using the role profiles themselves.
        # In that case we can only measure self-recommendation at the role
        # level (each role's skill set queries itself).
        rows = rec.roles.sample(min(n_samples, len(rec.roles)), random_state=RNG_SEED)
        skills_col = "skills"
        target_col = "role_id"
        items = list(rows[[skills_col, target_col]].itertuples(index=False, name=None))
    else:
        # Full bundle: use real postings (more rigorous).
        candidates = rec.postings[rec.postings["skill_count"] >= 2]
        sample = candidates.sample(
            min(n_samples, len(candidates)), random_state=RNG_SEED
        )
        items = [(set(r["skills"]), r["title_norm"]) for _, r in sample.iterrows()]

    role_ids = set(rec.roles["role_id"].tolist())
    items = [(s, t) for s, t in items if t in role_ids]

    hits_at_1 = hits_at_5 = hits_at_10 = 0
    for skills, true_role in items:
        ranked = rec.tfidf.query(skills, top_k=10)
        if ranked.empty:
            continue
        ids = ranked["role_id"].tolist()
        if ids[0] == true_role:
            hits_at_1 += 1
        if true_role in ids[:5]:
            hits_at_5 += 1
        if true_role in ids[:10]:
            hits_at_10 += 1

    n = len(items) or 1
    return {
        "n": len(items),
        "precision_at_1": hits_at_1 / n,
        "precision_at_5": hits_at_5 / n,
        "precision_at_10": hits_at_10 / n,
    }


# --------------------------------------------------- skill-extractor agreement
def skill_extractor_agreement(rec: CareerRecommender, n_samples: int = 1500) -> dict:
    """Use the human-curated `skills_desc` field (populated for ~2% of
    postings) as a soft ground truth and measure how well our regex extractor
    over `description` recovers the same skills.

    `skills_desc` is itself free-text, so we run *both* the regex extractor
    over `skills_desc` (as the gold label) and over `description` (as the
    prediction). This isn't a perfect benchmark, but it's the closest signal
    of "does our extractor agree with what a recruiter said the skills are?"
    """
    if rec.postings is None:
        return {"skipped": "postings DataFrame not available in slim bundle"}

    from src.skill_extractor import extract_skills

    # Re-load raw to get the original skills_desc text — the cached parquet
    # has a normalized version that's still usable.
    candidates = rec.postings[
        rec.postings["skills_desc"].fillna("").str.len() > 30
    ]
    sample = candidates.sample(min(n_samples, len(candidates)), random_state=RNG_SEED)

    precisions: list[float] = []
    recalls: list[float] = []
    jaccards: list[float] = []
    n_with_overlap = 0

    for _, r in sample.iterrows():
        gold = extract_skills(r["skills_desc"])
        pred = extract_skills(r["description"])
        if not gold:
            continue
        tp = len(gold & pred)
        if tp == 0 and not pred:
            precisions.append(0.0)
            recalls.append(0.0)
            jaccards.append(0.0)
            continue
        precisions.append(tp / max(1, len(pred)))
        recalls.append(tp / len(gold))
        jaccards.append(len(gold & pred) / len(gold | pred))
        if tp > 0:
            n_with_overlap += 1

    return {
        "n_evaluated": len(precisions),
        "n_with_overlap": n_with_overlap,
        "mean_precision": mean(precisions) if precisions else 0.0,
        "mean_recall": mean(recalls) if recalls else 0.0,
        "mean_jaccard": mean(jaccards) if jaccards else 0.0,
        "f1": (
            2 * mean(precisions) * mean(recalls) / (mean(precisions) + mean(recalls))
            if precisions and recalls and (mean(precisions) + mean(recalls)) > 0
            else 0.0
        ),
    }


# ------------------------------------------------------------- next-skill prediction
def next_skill_prediction(rec: CareerRecommender, n_samples: int = 1000) -> dict:
    """A meaningful held-out-skill metric: sample postings with >=4 skills,
    hide one random skill, query with the rest, and check whether the hidden
    skill appears among the union of top-N missing skills across the top-5
    recommended roles. This tests the recommender's ability to suggest the
    *right next skill to learn* given a partial skill profile.
    """
    if rec.postings is None:
        return {"skipped": "postings DataFrame not available in slim bundle"}
    rng = random.Random(RNG_SEED)

    candidates = rec.postings[rec.postings["skill_count"] >= 4]
    sample = candidates.sample(min(n_samples, len(candidates)), random_state=RNG_SEED)

    n_eval = 0
    hits_at_3 = hits_at_5 = hits_at_10 = 0

    from src.skill_gap import analyze_gap

    for _, r in sample.iterrows():
        skills = list(r["skills"])
        held = rng.choice(skills)
        user_skills = set(skills) - {held}
        ranked = rec.tfidf.query(user_skills, top_k=5)
        if ranked.empty:
            continue

        # Aggregate missing skills across top-5 roles, ranked by max
        # importance (frequency) across any of them.
        agg: dict[str, float] = {}
        for _, role in ranked.iterrows():
            gap = analyze_gap(user_skills, role["skill_freq"])
            for skill, w in gap.missing_ranked:
                agg[skill] = max(agg.get(skill, 0.0), w)
        ordered = [s for s, _ in sorted(agg.items(), key=lambda kv: -kv[1])]

        n_eval += 1
        if held in ordered[:3]:
            hits_at_3 += 1
        if held in ordered[:5]:
            hits_at_5 += 1
        if held in ordered[:10]:
            hits_at_10 += 1

    return {
        "n_evaluated": n_eval,
        "next_skill_at_3": hits_at_3 / n_eval if n_eval else 0.0,
        "next_skill_at_5": hits_at_5 / n_eval if n_eval else 0.0,
        "next_skill_at_10": hits_at_10 / n_eval if n_eval else 0.0,
    }


# ------------------------------------------------------------- coverage & diversity
def coverage_and_diversity(rec: CareerRecommender, n_profiles: int = 500) -> dict:
    """Synthesize random user profiles by drawing 3-7 skills uniformly from the
    vocabulary, then measure:
      - coverage: # unique roles ever appearing in top-10 / total roles
      - diversity: mean intra-list Jaccard distance among top-5 role skill sets
    """
    rng = random.Random(RNG_SEED)
    vocab = list(rec.tfidf.vectorizer.vocabulary_.keys())
    role_skills = {r["role_id"]: r["skills"] for _, r in rec.roles.iterrows()}

    unique_roles_seen: set[str] = set()
    diversities: list[float] = []

    for _ in range(n_profiles):
        n_skills = rng.randint(3, 7)
        skills = set(rng.sample(vocab, k=n_skills))
        ranked = rec.tfidf.query(skills, top_k=10)
        if ranked.empty:
            continue
        unique_roles_seen.update(ranked["role_id"].tolist())

        top5 = ranked.head(5)["role_id"].tolist()
        if len(top5) >= 2:
            pair_distances = []
            for i in range(len(top5)):
                for j in range(i + 1, len(top5)):
                    pair_distances.append(jaccard(role_skills[top5[i]], role_skills[top5[j]]))
            diversities.append(mean(pair_distances))

    return {
        "n_profiles": n_profiles,
        "coverage": len(unique_roles_seen) / len(rec.roles),
        "unique_roles_seen": len(unique_roles_seen),
        "total_roles": len(rec.roles),
        "mean_diversity": mean(diversities) if diversities else 0.0,
    }


# ------------------------------------------------------------- latency
def latency(rec: CareerRecommender, n_calls: int = 200) -> dict:
    rng = random.Random(RNG_SEED)
    vocab = list(rec.tfidf.vectorizer.vocabulary_.keys())
    # Warm-up
    rec.recommend(set(rng.sample(vocab, k=5)), top_k=10)

    timings_ms: list[float] = []
    for _ in range(n_calls):
        skills = set(rng.sample(vocab, k=rng.randint(3, 7)))
        t0 = time.perf_counter()
        rec.recommend(skills, top_k=10)
        timings_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "n_calls": n_calls,
        "mean_ms": float(np.mean(timings_ms)),
        "p50_ms": float(np.percentile(timings_ms, 50)),
        "p95_ms": float(np.percentile(timings_ms, 95)),
        "p99_ms": float(np.percentile(timings_ms, 99)),
    }


# ------------------------------------------------------------- main
def main() -> None:
    print("Loading recommender...")
    t0 = time.time()
    rec = CareerRecommender.load_or_build()
    print(f"  loaded in {time.time() - t0:.2f}s, has_postings={rec.postings is not None}\n")

    print("=" * 60)
    print("1. Self-recommendation precision (random posting -> its role)")
    print("=" * 60)
    sr = self_recommendation_precision(rec, n_samples=2000)
    print(json.dumps(sr, indent=2))

    print("\n" + "=" * 60)
    print("2a. Skill-extractor agreement vs. human skills_desc labels")
    print("=" * 60)
    sea = skill_extractor_agreement(rec)
    print(json.dumps(sea, indent=2))

    print("\n" + "=" * 60)
    print("2b. Next-skill prediction (hide a skill, recover from missing list)")
    print("=" * 60)
    nsp = next_skill_prediction(rec)
    print(json.dumps(nsp, indent=2))

    print("\n" + "=" * 60)
    print("3. Coverage & diversity (500 synthetic profiles)")
    print("=" * 60)
    cov = coverage_and_diversity(rec, n_profiles=500)
    print(json.dumps(cov, indent=2))

    print("\n" + "=" * 60)
    print("4. Latency (200 random queries)")
    print("=" * 60)
    lat = latency(rec, n_calls=200)
    print(json.dumps(lat, indent=2))

    out = {
        "self_recommendation": sr,
        "skill_extractor_agreement": sea,
        "next_skill_prediction": nsp,
        "coverage_and_diversity": cov,
        "latency": lat,
    }
    out_path = ROOT / "docs" / "evaluation_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote metrics to {out_path}")


if __name__ == "__main__":
    main()
