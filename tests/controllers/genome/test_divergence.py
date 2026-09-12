"""
Focused tests for Jensen-Shannon divergence math (§3.6) using the shipped test baseline.

Complements ``test_engine.py``'s broader engine coverage with the divergence formula's edge
cases and its interaction with a real (fixture) :class:`Baseline`.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

from music_assistant.controllers.genome import engine
from music_assistant.controllers.genome.baseline.loader import load_baseline

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "genome" / "baseline_test.json"


def test_js_divergence_is_symmetric() -> None:
    p = {"rock": 0.6, "pop": 0.4}
    q = {"rock": 0.1, "pop": 0.2, "jazz": 0.7}
    assert abs(engine.js_divergence(p, q) - engine.js_divergence(q, p)) < 1e-12


def test_js_divergence_bounded_zero_to_one() -> None:
    p = {"rock": 0.9, "pop": 0.05, "jazz": 0.05}
    q = {"rock": 0.1, "pop": 0.1, "jazz": 0.8}
    jsd = engine.js_divergence(p, q)
    assert 0.0 <= jsd <= 1.0


def test_js_divergence_empty_household_vector_is_zero_not_maximal() -> None:
    # An empty genome (no genre data yet) must not read as "100% divergent from the baseline".
    assert engine.js_divergence({}, {"rock": 1.0}) == 0.0
    assert engine.js_divergence({"rock": 1.0}, {}) == 0.0


def test_js_contributions_all_zero_when_one_side_empty() -> None:
    contributions = engine.js_contributions({}, {"rock": 0.5, "pop": 0.5})
    assert set(contributions) == {"rock", "pop"}
    assert all(value == 0.0 for value in contributions.values())


def test_divergence_facts_ratio_is_clamped() -> None:
    p = {"niche": 0.5, "rock": 0.5}
    q = {"niche": 1e-9, "rock": 0.5}
    facts = engine.divergence_facts(p, q, {"niche": "Niche", "rock": "Rock"})
    niche = next(s for s in facts["top_over"] if s["key"] == "niche")
    assert niche["ratio"] == 99.0


def test_divergence_facts_percent_matches_score() -> None:
    p = {"rock": 1.0}
    q = {"pop": 1.0}
    facts = engine.divergence_facts(p, q, {"rock": "Rock", "pop": "Pop"})
    assert facts["score"] == 1.0
    assert facts["percent"] == 100


def test_divergence_against_shipped_test_baseline() -> None:
    """A genome vector identical to the test baseline diverges by exactly 0."""
    baseline = asyncio.run(load_baseline(str(FIXTURE_PATH)))
    household = dict(baseline.genre_shares)
    jsd = engine.js_divergence(household, dict(baseline.genre_shares))
    assert jsd == 0.0
    assert baseline.version == "test-v1"
    assert len(baseline.genre_shares) == 8
    assert abs(sum(baseline.genre_shares.values()) - 1.0) < 1e-6


def test_divergence_against_shipped_test_baseline_when_household_differs() -> None:
    baseline = asyncio.run(load_baseline(str(FIXTURE_PATH)))
    # household listens to nothing but the baseline's smallest genre
    smallest_key = min(baseline.genre_shares, key=lambda key: baseline.genre_shares[key])
    household = {smallest_key: 1.0}
    jsd = engine.js_divergence(household, dict(baseline.genre_shares))
    assert 0.0 < jsd <= 1.0
    assert not math.isnan(jsd)
