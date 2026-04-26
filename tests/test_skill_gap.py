"""Tests for the skill-gap analyzer."""

import math

from src.skill_gap import analyze_gap


def test_full_match_gives_complete_coverage():
    user = {"Python", "SQL"}
    role = {"Python": 1.0, "SQL": 1.0}
    gap = analyze_gap(user, role)
    assert gap.matched == ["Python", "SQL"]
    assert gap.missing == []
    assert math.isclose(gap.coverage, 1.0)
    assert math.isclose(gap.weighted_coverage, 1.0)


def test_no_overlap_gives_zero_coverage():
    user = {"PHP"}
    role = {"Python": 0.8, "SQL": 0.5}
    gap = analyze_gap(user, role)
    assert gap.matched == []
    assert set(gap.missing) == {"Python", "SQL"}
    assert gap.extras == ["PHP"]
    assert math.isclose(gap.coverage, 0.0)
    assert math.isclose(gap.weighted_coverage, 0.0)


def test_weighted_coverage_favors_frequent_skills():
    # User has the rare skill (low freq) but is missing the dominant one.
    user = {"Rust"}
    role = {"Python": 0.9, "Rust": 0.1}
    gap = analyze_gap(user, role)
    # Plain coverage = 1/2 = 0.5
    assert math.isclose(gap.coverage, 0.5)
    # Weighted coverage = 0.1 / (0.9 + 0.1) = 0.1
    assert math.isclose(gap.weighted_coverage, 0.1)


def test_missing_ranked_by_role_importance():
    user: set[str] = set()
    role = {"Python": 0.9, "Docker": 0.7, "Kubernetes": 0.3}
    gap = analyze_gap(user, role)
    ranked_skills = [s for s, _ in gap.missing_ranked]
    assert ranked_skills == ["Python", "Docker", "Kubernetes"]
