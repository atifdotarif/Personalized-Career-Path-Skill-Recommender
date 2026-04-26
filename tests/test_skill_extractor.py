"""Tests for the skill extractor — focus on precision of ambiguous tokens."""

from src.skill_extractor import extract_skills, extract_skills_from_user_input


def test_extracts_common_tech_stack():
    text = (
        "Looking for a Python developer with experience in Django and "
        "PostgreSQL. AWS and Docker required."
    )
    assert extract_skills(text) >= {"Python", "Django", "PostgreSQL", "AWS", "Docker"}


def test_handles_aliases():
    text = "We use Postgres and k8s in production with golang microservices."
    skills = extract_skills(text)
    assert "PostgreSQL" in skills
    assert "Kubernetes" in skills
    assert "Go" in skills


def test_multiword_skills():
    text = "Strong background in machine learning and natural language processing."
    skills = extract_skills(text)
    assert "Machine Learning" in skills
    assert "NLP" in skills


def test_ambiguous_R_in_english_is_rejected():
    assert "R" not in extract_skills("R is for retire — start saving early.")


def test_ambiguous_R_in_programming_context_is_kept():
    assert "R" in extract_skills("Strong R programming skills required.")


def test_ambiguous_go_in_english_is_rejected():
    assert "Go" not in extract_skills("Please go to the store and buy milk.")


def test_ambiguous_go_in_programming_context_is_kept():
    assert "Go" in extract_skills("Backend development experience in Go preferred.")


def test_user_input_skips_context_check():
    # User typing "R" or "Go" in their skill list clearly means the language.
    skills = extract_skills_from_user_input("python, R, Go, sql")
    assert skills == {"Python", "R", "Go", "SQL"}


def test_empty_and_none():
    assert extract_skills("") == set()
    assert extract_skills(None) == set()
