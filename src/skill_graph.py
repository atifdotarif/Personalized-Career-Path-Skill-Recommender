"""
Skill co-occurrence graph + bridge-role pathing.

We build two graphs:

1. **Skill graph** — nodes are canonical skills, edges weighted by co-occurrence
   strength across postings (PMI-style normalization). This captures
   "skills that go together": Python ↔ Pandas ↔ scikit-learn cluster, AWS ↔
   Docker ↔ Kubernetes cluster, etc.

2. **Role transition graph** — nodes are roles, an edge from role A to role B
   exists when B requires only a small extension of A's skills (configurable
   threshold). Edge weight = the size of the skill jump (smaller = easier
   transition). Shortest path from the user's "current role" to a "target role"
   surfaces the best *bridge roles* — the proposal's novel contribution.

The graphs are stored as NetworkX objects so we can lean on its shortest-path
algorithms instead of reimplementing Dijkstra.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------- skill graph -----------------------------------


def build_skill_graph(postings: pd.DataFrame, min_cooccurrence: int = 5) -> nx.Graph:
    """Co-occurrence graph weighted by normalized pointwise mutual information."""
    skill_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    n_docs = len(postings)

    for skills in postings["skills"]:
        skill_counts.update(skills)
        skill_list = sorted(skills)
        for i in range(len(skill_list)):
            for j in range(i + 1, len(skill_list)):
                pair_counts[(skill_list[i], skill_list[j])] += 1

    g = nx.Graph()
    for skill, count in skill_counts.items():
        g.add_node(skill, posting_count=count)

    for (a, b), c in pair_counts.items():
        if c < min_cooccurrence:
            continue
        # NPMI in [-1, 1] — we keep only positive associations.
        p_ab = c / n_docs
        p_a = skill_counts[a] / n_docs
        p_b = skill_counts[b] / n_docs
        pmi = math.log(p_ab / (p_a * p_b))
        npmi = pmi / (-math.log(p_ab))
        if npmi > 0:
            g.add_edge(a, b, weight=npmi, cooccurrence=c)

    logger.info("Skill graph: %d nodes, %d edges", g.number_of_nodes(), g.number_of_edges())
    return g


def adjacent_skills(graph: nx.Graph, skill: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Return up to top_k strongest neighbors of `skill` by NPMI."""
    if skill not in graph:
        return []
    neighbors = [(n, graph[skill][n]["weight"]) for n in graph.neighbors(skill)]
    neighbors.sort(key=lambda kv: -kv[1])
    return neighbors[:top_k]


# --------------------------- role transition graph ----------------------------


def build_role_graph(roles: pd.DataFrame, max_jump: int = 4) -> nx.DiGraph:
    """Directed graph of role transitions. Edge A → B exists if moving from A
    to B requires learning ≤ max_jump new skills (relative to A's skill set).
    Edge weight = number of new skills (smaller = easier transition).
    """
    g = nx.DiGraph()
    role_skills = {row["role_id"]: row["skills"] for _, row in roles.iterrows()}
    role_titles = {row["role_id"]: row["title"] for _, row in roles.iterrows()}

    for rid, title in role_titles.items():
        g.add_node(rid, title=title, n_skills=len(role_skills[rid]))

    role_ids = list(role_skills.keys())
    for i, a in enumerate(role_ids):
        a_skills = role_skills[a]
        if not a_skills:
            continue
        for b in role_ids:
            if a == b:
                continue
            b_skills = role_skills[b]
            if not b_skills:
                continue
            new_skills = b_skills - a_skills
            if 0 < len(new_skills) <= max_jump:
                # Also require meaningful overlap so unrelated roles aren't bridged.
                overlap = len(a_skills & b_skills) / len(b_skills)
                if overlap >= 0.3:
                    g.add_edge(
                        a, b,
                        weight=len(new_skills),
                        new_skills=sorted(new_skills),
                        overlap=overlap,
                    )

    logger.info(
        "Role graph: %d nodes, %d transitions (max_jump=%d)",
        g.number_of_nodes(), g.number_of_edges(), max_jump,
    )
    return g


def find_bridge_path(
    role_graph: nx.DiGraph,
    source_role_id: str,
    target_role_id: str,
    max_hops: int = 3,
) -> list[dict] | None:
    """Shortest path (by total skill jumps) from source role to target role.

    Returns a list of {from, to, new_skills} dicts, or None if no path exists
    within max_hops. Direct edges (source → target) collapse to a single hop.
    """
    if source_role_id not in role_graph or target_role_id not in role_graph:
        return None
    try:
        path = nx.shortest_path(
            role_graph, source_role_id, target_role_id, weight="weight"
        )
    except nx.NetworkXNoPath:
        return None
    if len(path) - 1 > max_hops:
        return None

    steps: list[dict] = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        edge = role_graph[a][b]
        steps.append({
            "from_id": a,
            "to_id": b,
            "from_title": role_graph.nodes[a]["title"],
            "to_title": role_graph.nodes[b]["title"],
            "new_skills": edge["new_skills"],
            "skill_jump": edge["weight"],
        })
    return steps
