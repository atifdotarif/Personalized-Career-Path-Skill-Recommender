"""
Extract a structured set of canonical skills from free-text job descriptions
(or user resumes) by matching against the curated vocabulary.

Single-pass regex finds candidate matches. For ambiguous short tokens (R, Go,
C#, C++) we then verify a programming-context cue is nearby — otherwise we
discard the match. This trades a tiny bit of recall for big precision wins on
"go to the store" and "R is for retire" style false positives.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .skills_vocab import build_alias_index


_AMBIGUOUS_SHORT = {"r", "go", "c#", "c++"}
_CONTEXT_WINDOW = 60  # chars on either side of an ambiguous token
_CONTEXT_CUE = re.compile(
    r"programm|languag|develop|code|coding|script|experien|skill|profic|"
    r"knowledg|familiar|engineer|stack|backend|frontend|fullstack",
    flags=re.IGNORECASE,
)


def _build_pattern(forms: list[str]) -> re.Pattern[str]:
    # Sort by length desc so multi-word forms ("machine learning") win over
    # shorter substrings during scanning.
    forms_sorted = sorted(set(forms), key=len, reverse=True)
    parts: list[str] = []
    for form in forms_sorted:
        escaped = re.escape(form)
        if form in _AMBIGUOUS_SHORT:
            # Custom boundary that respects '+' and '#' on the right edge.
            parts.append(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_+#])")
        else:
            parts.append(rf"\b{escaped}\b")
    return re.compile("|".join(parts), flags=re.IGNORECASE)


@lru_cache(maxsize=1)
def _index_and_pattern() -> tuple[dict[str, str], re.Pattern[str]]:
    alias_index = build_alias_index()
    pattern = _build_pattern(list(alias_index.keys()))
    return alias_index, pattern


def _has_context_cue(text: str, start: int, end: int) -> bool:
    window_start = max(0, start - _CONTEXT_WINDOW)
    window_end = min(len(text), end + _CONTEXT_WINDOW)
    return bool(_CONTEXT_CUE.search(text[window_start:window_end]))


def extract_skills(text: str | None) -> set[str]:
    """Return the set of canonical skill names mentioned in `text`."""
    if not text or not isinstance(text, str):
        return set()
    alias_index, pattern = _index_and_pattern()
    found: set[str] = set()
    for match in pattern.finditer(text):
        surface = match.group(0).lower()
        canonical = alias_index.get(surface)
        if not canonical:
            continue
        if surface in _AMBIGUOUS_SHORT and not _has_context_cue(
            text, match.start(), match.end()
        ):
            continue
        found.add(canonical)
    return found


def extract_skills_from_user_input(user_text: str) -> set[str]:
    """Same as `extract_skills` but tolerates comma/newline separated lists.

    Skips the ambiguous-short context check because user input is short and
    the user typing 'R' or 'Go' clearly means the language.
    """
    if not user_text:
        return set()
    alias_index, pattern = _index_and_pattern()
    normalized = re.sub(r"[,;/\n\r\t]+", " ", user_text)
    found: set[str] = set()
    for match in pattern.finditer(normalized):
        surface = match.group(0).lower()
        canonical = alias_index.get(surface)
        if canonical:
            found.add(canonical)
    return found
