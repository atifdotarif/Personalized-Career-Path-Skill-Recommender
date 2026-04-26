"""
Build the recommender bundle from the raw LinkedIn dataset and pickle it to
`data/processed/recommender.pkl`.

Run from the project root:

    .venv/Scripts/python.exe scripts/build_model.py

This is a one-time setup step. The Streamlit app will lazily build on first
launch if you skip this, but running it explicitly is faster (you can watch
the progress logs) and avoids the Streamlit cache spinner timing out.
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

from src.recommender import CareerRecommender


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )
    t0 = time.time()
    rec = CareerRecommender.build(force_rebuild_data=True)
    rec.save()
    print(f"\nBuild complete in {time.time() - t0:.1f}s")
    print(f"  Roles indexed:    {len(rec.roles)}")
    print(f"  Postings indexed: {len(rec.postings):,}")
    print(f"  TF-IDF vocab:     {rec.tfidf.vocabulary_size()}")
    print(
        f"  Skill graph:      {rec.skill_graph.number_of_nodes()} nodes / "
        f"{rec.skill_graph.number_of_edges()} edges"
    )
    print(
        f"  Role graph:       {rec.role_graph.number_of_nodes()} nodes / "
        f"{rec.role_graph.number_of_edges()} edges"
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
