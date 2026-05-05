"""
Refresh the recommender bundles from the cached `data/processed/postings.parquet`
without re-running the slow skill extraction. Use this after schema changes
to `role_aggregator.py` / `tfidf_model.py` / `skill_graph.py` / `recommender.py`.

Run from the project root:

    .venv/Scripts/python.exe scripts/refresh_bundle.py

If the parquet cache doesn't exist, run `scripts/build_model.py` first.
"""

from __future__ import annotations

import logging
import multiprocessing
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import DEFAULT_PROCESSED_PATH, load_processed
from src.recommender import CareerRecommender
from src.role_aggregator import aggregate_roles
from src.skill_graph import build_role_graph, build_skill_graph
from src.tfidf_model import RoleTFIDFModel


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )
    if not Path(DEFAULT_PROCESSED_PATH).exists():
        sys.exit(
            f"ERROR: {DEFAULT_PROCESSED_PATH} not found. "
            "Run scripts/build_model.py first to populate the parquet cache."
        )

    t0 = time.time()
    postings = load_processed()
    roles = aggregate_roles(postings)
    tfidf = RoleTFIDFModel().fit(roles)
    skill_graph = build_skill_graph(postings)
    role_graph = build_role_graph(roles)

    rec = CareerRecommender(
        postings=postings,
        roles=roles,
        tfidf=tfidf,
        skill_graph=skill_graph,
        role_graph=role_graph,
    )
    rec.save()
    rec.save_slim()
    print(f"\nRefreshed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
