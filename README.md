# Personalized Career Path & Skill Recommendation System

> An explainable, content-based recommender that matches a user's current
> technical skills against real job-market demand, surfaces specific skill
> gaps, and suggests bridge roles to a target career.

**Course**: Recommender Systems — Spring 2026
**Institute**: National University of Computer and Emerging Sciences, Karachi
**Instructor**: Syed Zain Ul Hassan
**Team**: Atif Arif (22K-4358), Unzila Javed (22K-4168), Mishkaat Yousuf (22K-4624), Rao Abdul Hassi (22K-4202)

---

## What it does

Given a user's current skills (free text, comma-separated, or picked from a vocabulary), the system:

1. **Recommends roles** ranked by a 0–1 *Match Score* — TF-IDF + cosine similarity between the user's skill vector and each role's prerequisite-skill vector.
2. **Explains every recommendation** by listing the matched skills, missing skills (ranked by importance to the role), and weighted skill coverage.
3. **Suggests a bridge path** — the shortest sequence of role transitions through a co-occurrence graph that takes the user from their closest current role to a chosen target role, surfacing the intermediate roles and the new skills needed at each hop.
4. **Hints at adjacent skills** — for each user skill, the top co-occurring skills the user does *not* yet have, weighted by normalized pointwise mutual information (NPMI) across the corpus.

Every output is deterministic and traceable — there is no opaque embedding model, no neural network. The recommender is fully explainable end-to-end, in line with the project proposal's stated requirements.

## Architecture

```
postings.csv (123k LinkedIn postings)
        │
        ▼
data_loader.py    ── normalize text, extract skills via curated vocab → cached parquet
        │
        ▼
role_aggregator.py ── group by normalized title, compute per-role skill frequencies
        │
        ├──► tfidf_model.py    ── TF-IDF over role skill bags + cosine similarity
        ├──► skill_gap.py      ── set ops: matched / missing / extras + weighted coverage
        └──► skill_graph.py    ── NetworkX co-occurrence graph + role-transition graph
                                       │
                                       ▼
                              recommender.py (top-level API)
                                       │
                                       ▼
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
              app/streamlit_app.py            notebooks/01_eda.ipynb
```

## Techniques (as committed in the proposal)

| Module | Technique | Library |
|---|---|---|
| `tfidf_model.py` | TF-IDF + cosine similarity over canonical-skill bags | `sklearn.feature_extraction.text.TfidfVectorizer`, `sklearn.metrics.pairwise.cosine_similarity` |
| `skill_gap.py` | Pure set operations (∩, ∖) for matched/missing/extras + weighted coverage | NumPy |
| `skill_graph.py` | Skill co-occurrence graph (NPMI) + role-transition graph + Dijkstra shortest path | NetworkX |
| `skill_extractor.py` | Regex matcher over a curated alias-indexed vocabulary, with context-cue filtering for ambiguous short tokens (R, Go, C#, C++) | Pure Python |

## Project structure

```
RecommenderSystemsProject/
├── postings.csv/postings.csv          # raw Kaggle data (gitignored, you supply)
├── data/processed/                    # cached parquet + pickled model bundle
├── src/
│   ├── skills_vocab.py                # curated skill list (~110 skills, 8 categories)
│   ├── skill_extractor.py             # extract canonical skills from free text
│   ├── data_loader.py                 # ingest + preprocess LinkedIn postings
│   ├── role_aggregator.py             # postings → roles (per-role skill frequency)
│   ├── tfidf_model.py                 # TF-IDF + cosine recommender
│   ├── skill_gap.py                   # set-based gap analysis
│   ├── skill_graph.py                 # NetworkX skill + role graphs
│   └── recommender.py                 # top-level API
├── app/streamlit_app.py               # interactive UI
├── notebooks/01_eda.ipynb             # corpus + skill EDA
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create venv and install dependencies
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt          # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

# 2. Place the raw dataset
#    Download the Kaggle "LinkedIn Job Postings (2023-2024)" dataset and put
#    postings.csv at:  postings.csv/postings.csv
```

## Run the demo

```bash
.venv/Scripts/streamlit run app/streamlit_app.py
```

The first run preprocesses the full 123k-row corpus (~5 minutes); after that, the cached parquet + pickled model load in seconds.

## Run the EDA notebook

```bash
.venv/Scripts/jupyter notebook notebooks/01_eda.ipynb
```

## Programmatic use

```python
from src.recommender import CareerRecommender

rec = CareerRecommender.load_or_build()
parsed, recs = rec.recommend("Python, SQL, Pandas, Statistics", top_k=5)

for r in recs:
    print(f"{r.title} | match={r.match_score:.2f} | missing={r.gap.missing[:5]}")

# Bridge path from user's closest role to a target
path = rec.bridge_path(parsed, target_role_id="machine learning engineer")
```

## Notes on data quality

- Only ~2% of LinkedIn postings populate the structured `skills_desc` column. We extract skills from `description` (free text) using a curated vocabulary of ~110 technical skills with alias resolution. Extending `src/skills_vocab.py` is the cleanest way to broaden coverage.
- Roles with fewer than 5 postings are excluded from the index to keep recommendations statistically meaningful (configurable in `role_aggregator.py`).
- O\*NET integration is mentioned in the proposal as supplementary. Code is structured so an O\*NET skill taxonomy can be plugged into `skills_vocab.py` later without changing any downstream module.

## Honesty undertaking

Per the proposal, the team commits to original work. All code in this repository was written for this project; no copy-pasted implementations from third-party recommender libraries are used. Algorithms (TF-IDF, cosine, set ops, Dijkstra) come from `scikit-learn` and `networkx` as the proposal explicitly allows.
