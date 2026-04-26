"""
Load and preprocess the LinkedIn job postings dataset.

The raw CSV is large (~500 MB, 123k rows). We:
  1. Read only the columns the recommender actually uses.
  2. Normalize text (lowercase, collapse whitespace).
  3. Extract a structured `skills` set per posting via `skill_extractor`.
  4. Drop rows that have no description or no extractable skills (they cannot
     contribute to either the TF-IDF index or the skill graph).
  5. Cache the processed result as Parquet so the slow extraction step only
     runs once.
"""

from __future__ import annotations

import logging
import os
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd

from .skill_extractor import extract_skills

logger = logging.getLogger(__name__)

DEFAULT_RAW_PATH = Path("postings.csv/postings.csv")
DEFAULT_PROCESSED_PATH = Path("data/processed/postings.parquet")

USED_COLUMNS = [
    "job_id",
    "company_name",
    "title",
    "description",
    "location",
    "formatted_work_type",
    "formatted_experience_level",
    "skills_desc",
    "normalized_salary",
    "remote_allowed",
]


_WS_RE = re.compile(r"\s+")


def _normalize_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return _WS_RE.sub(" ", text).strip()


def _normalize_title(title: str) -> str:
    """Cheap title normalization: collapse case + whitespace, strip punctuation
    that varies between identical-meaning postings (e.g. trailing dashes)."""
    t = _normalize_text(title).lower()
    t = re.sub(r"[^a-z0-9 +/#.&-]", " ", t)
    return _WS_RE.sub(" ", t).strip()


def _parallel_extract(texts: list[str]) -> list[set[str]]:
    """Run skill extraction across all available CPU cores.

    Falls back to single-process when texts is small or only one core is
    available — the multiprocessing overhead isn't worth it for small inputs.
    """
    n = len(texts)
    workers = max(1, cpu_count() - 1)
    if n < 2000 or workers == 1:
        return [extract_skills(t) for t in texts]
    chunksize = max(200, n // (workers * 8))
    logger.info("Parallel extract: %d texts, %d workers, chunksize=%d", n, workers, chunksize)
    with Pool(processes=workers) as pool:
        return pool.map(extract_skills, texts, chunksize=chunksize)


def load_raw(path: Path | str = DEFAULT_RAW_PATH, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw postings CSV not found at {path}. "
            "Place the Kaggle file there or pass an explicit path."
        )
    logger.info("Reading raw postings from %s", path)
    df = pd.read_csv(path, usecols=USED_COLUMNS, nrows=nrows, low_memory=False)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text fields, extract skills, and drop unusable rows."""
    df = df.copy()
    df["title"] = df["title"].map(_normalize_text)
    df["description"] = df["description"].map(_normalize_text)
    df["skills_desc"] = df["skills_desc"].map(_normalize_text)
    df["title_norm"] = df["title"].map(_normalize_title)

    # Extract skills from description ∪ skills_desc — skills_desc is sparse but
    # often more precise when present, so concatenating maximizes recall.
    # Skill extraction is the slowest step (~65 rows/sec single-threaded over a
    # 150-alternation regex), so parallelize across CPU cores.
    logger.info("Extracting skills from %d postings", len(df))
    combined = (df["description"].fillna("") + " " + df["skills_desc"].fillna("")).str.strip()
    df["skills"] = _parallel_extract(combined.tolist())
    df["skill_count"] = df["skills"].map(len)

    before = len(df)
    df = df[(df["description"].str.len() > 50) & (df["skill_count"] > 0)].reset_index(drop=True)
    logger.info("Dropped %d unusable rows; %d remain", before - len(df), len(df))
    return df


def load_processed(
    raw_path: Path | str = DEFAULT_RAW_PATH,
    processed_path: Path | str = DEFAULT_PROCESSED_PATH,
    nrows: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Load a cached processed parquet, or build it from the raw CSV."""
    processed_path = Path(processed_path)
    if processed_path.exists() and not force and nrows is None:
        logger.info("Loading cached processed parquet from %s", processed_path)
        df = pd.read_parquet(processed_path)
        # Parquet stores skills as a list (Arrow has no set type) — restore.
        df["skills"] = df["skills"].map(lambda xs: set(xs) if xs is not None else set())
        return df

    raw = load_raw(raw_path, nrows=nrows)
    processed = preprocess(raw)

    if nrows is None:  # only cache full builds
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert sets -> lists for parquet serialization.
        to_write = processed.copy()
        to_write["skills"] = to_write["skills"].map(lambda s: sorted(s))
        to_write.to_parquet(processed_path, index=False)
        logger.info("Cached processed parquet to %s", processed_path)
    return processed
