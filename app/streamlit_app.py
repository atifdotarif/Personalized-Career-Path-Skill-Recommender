"""
Streamlit demo for the Career Path & Skill Recommendation System.

Run from the project root:

    streamlit run app/streamlit_app.py

The app loads (or rebuilds) the pickled recommender bundle on startup, so the
first launch may take a few minutes while the full LinkedIn dataset is
processed. Subsequent launches load instantly from cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src.*` importable when running `streamlit run app/streamlit_app.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.recommender import CareerRecommender
from src.skills_vocab import canonical_names

st.set_page_config(
    page_title="Career Path Recommender",
    page_icon=":briefcase:",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading recommender (first run builds from raw data — may take ~2 min)…")
def get_recommender() -> CareerRecommender:
    return CareerRecommender.load_or_build()


def render_match_score(score: float) -> str:
    pct = int(round(score * 100))
    if pct >= 70:
        color = "#16a34a"
    elif pct >= 40:
        color = "#d97706"
    else:
        color = "#dc2626"
    return f"<span style='color:{color};font-weight:600'>{pct}%</span>"


def skill_badges(skills: list[str], color: str) -> str:
    if not skills:
        return "<em>none</em>"
    style = (
        f"display:inline-block;padding:2px 8px;margin:2px;"
        f"border-radius:12px;background-color:{color};color:white;font-size:12px"
    )
    return " ".join(f"<span style='{style}'>{s}</span>" for s in skills)


# ---------------------------------------------------------------- header
st.title("Personalized Career Path & Skill Recommender")
st.caption(
    "An explainable, content-based recommender built on 123k LinkedIn job "
    "postings. TF-IDF + cosine similarity for ranking, set operations for "
    "skill-gap explanation, NetworkX for bridge-role pathing."
)

rec = get_recommender()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Your skills")
    st.write(
        "Type a comma-separated list, paste from your CV, or pick from the "
        "vocabulary below."
    )

    user_input = st.text_area(
        "Skills (free text or comma-separated)",
        value="Python, SQL, Pandas, Statistics, Machine Learning",
        height=120,
    )
    picked = st.multiselect("Or pick from vocabulary", options=canonical_names())

    top_k = st.slider("Number of recommendations", min_value=3, max_value=20, value=8)

    st.divider()
    st.metric("Roles indexed", len(rec.roles))
    st.metric("Postings indexed", f"{rec.n_postings_total:,}")
    st.metric("Skill vocabulary", rec.tfidf.vocabulary_size())

# Combine free text + picked into one effective skill set.
parsed_skills = rec.parse_user_skills(user_input)
combined = parsed_skills | set(picked)

if not combined:
    st.warning("Add at least one recognized skill to see recommendations.")
    st.stop()

st.subheader("Recognized skills from your input")
st.markdown(skill_badges(sorted(combined), "#2563eb"), unsafe_allow_html=True)

_, recommendations = rec.recommend(combined, top_k=top_k)

# ---------------------------------------------------------------- recommendations
st.subheader("Top role matches")
for r in recommendations:
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        c1.markdown(f"### {r.title}")
        c2.markdown(
            f"**Match**<br>{render_match_score(r.match_score)}",
            unsafe_allow_html=True,
        )
        c3.metric("Postings", f"{r.n_postings:,}")
        if not pd.isna(r.median_salary):
            c4.metric("Median salary", f"${r.median_salary:,.0f}")
        else:
            c4.metric("Median salary", "—")

        cov_pct = int(round(r.gap.weighted_coverage * 100))
        st.progress(r.gap.weighted_coverage, text=f"Weighted skill coverage: {cov_pct}%")

        cm, cmiss = st.columns(2)
        with cm:
            st.markdown("**You already have**")
            st.markdown(
                skill_badges(r.gap.matched, "#16a34a"), unsafe_allow_html=True
            )
        with cmiss:
            st.markdown("**Skills to learn (ranked by importance)**")
            top_missing = [s for s, _ in r.gap.missing_ranked[:8]]
            st.markdown(
                skill_badges(top_missing, "#dc2626"), unsafe_allow_html=True
            )

        if r.top_companies:
            st.caption(f"Top hiring companies: {r.top_companies}")

        if r.samples:
            with st.expander(f"View {len(r.samples)} sample LinkedIn postings for this role"):
                for s in r.samples:
                    company = s.get("company") or "—"
                    loc = s.get("location") or ""
                    loc_str = f" · {loc}" if loc else ""
                    st.markdown(
                        f"- [{s['title']}]({s['url']}) — *{company}*{loc_str}",
                    )

# ---------------------------------------------------------------- bridge path
st.subheader("Bridge path: how to get to a target role")
st.write(
    "Pick the role you currently identify with, then a target role you want to move "
    "into. The system finds the shortest sequence of role transitions through the "
    "role-graph and shows the new skills required at each step."
)

# Build a (label, role_id) list ordered by posting count for usability.
role_options = rec.roles[["role_id", "title", "n_postings"]].sort_values("n_postings", ascending=False)
role_options["label"] = role_options["title"] + "  (" + role_options["n_postings"].astype(str) + " postings)"
role_id_to_label = dict(zip(role_options["role_id"], role_options["label"]))

# Default source = our heuristic guess.
default_source = rec.closest_role_for_user(combined)
default_source_idx = (
    role_options["role_id"].tolist().index(default_source) if default_source in role_id_to_label else 0
)

c_src, c_tgt = st.columns(2)
with c_src:
    source = st.selectbox(
        "Source role (your current role)",
        options=role_options["role_id"].tolist(),
        format_func=lambda rid: role_id_to_label[rid],
        index=default_source_idx,
        help="Pre-filled with the role most aligned to your current skills, but you can override.",
    )
with c_tgt:
    target = st.selectbox(
        "Target role",
        options=role_options["role_id"].tolist(),
        format_func=lambda rid: role_id_to_label[rid],
        index=0,
    )

if source == target:
    st.info("Source and target are the same role — pick a different target to compute a bridge.")
else:
    from src.skill_graph import find_bridge_path
    path = find_bridge_path(rec.role_graph, source, target, max_hops=3)
    if path is None:
        st.warning(
            "No bridge path found within 3 hops. The skill jump between these "
            "roles is too large for the role graph to bridge automatically."
        )
    else:
        st.success(f"Found a {len(path)}-hop bridge:")
        for i, step in enumerate(path, 1):
            new_skills_str = ", ".join(step["new_skills"]) if step["new_skills"] else "—"
            st.markdown(
                f"**Step {i}.** `{step['from_title']}` → `{step['to_title']}`  "
                f"&nbsp;&nbsp;_(learn: {new_skills_str})_",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------- adjacent skills
st.subheader("Skills adjacent to yours (what to learn next)")
adj = rec.adjacent_skills_for(combined, top_k=4)
if not adj:
    st.write("No adjacency data — your skills aren't strongly connected in the co-occurrence graph.")
else:
    rows: list[dict] = []
    for skill, neighbors in adj.items():
        for n, w in neighbors:
            rows.append({"From your skill": skill, "Suggested next": n, "NPMI": round(w, 3)})
    df_adj = pd.DataFrame(rows)
    st.dataframe(df_adj, width="stretch", hide_index=True)

    # Top suggested-next skills aggregated.
    agg = (
        df_adj.groupby("Suggested next")["NPMI"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig = px.bar(
        agg,
        x="NPMI",
        y="Suggested next",
        orientation="h",
        title="Top suggested next skills (cumulative NPMI)",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
    st.plotly_chart(fig, width="stretch")
