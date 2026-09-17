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
    """Test js divergence is symmetric."""
    p = {"rock": 0.6, "pop": 0.4}
    q = {"rock": 0.1, "pop": 0.2, "jazz": 0.7}
    assert abs(engine.js_divergence(p, q) - engine.js_divergence(q, p)) < 1e-12


def test_js_divergence_bounded_zero_to_one() -> None:
    """Test js divergence bounded zero to one."""
    p = {"rock": 0.9, "pop": 0.05, "jazz": 0.05}
    q = {"rock": 0.1, "pop": 0.1, "jazz": 0.8}
    jsd = engine.js_divergence(p, q)
    assert 0.0 <= jsd <= 1.0


def test_js_divergence_empty_household_vector_is_zero_not_maximal() -> None:
    # An empty genome (no genre data yet) must not read as "100% divergent from the baseline".
    """Test js divergence empty household vector is zero not maximal."""
    assert engine.js_divergence({}, {"rock": 1.0}) == 0.0
    assert engine.js_divergence({"rock": 1.0}, {}) == 0.0


def test_js_contributions_all_zero_when_one_side_empty() -> None:
    """Test js contributions all zero when one side empty."""
    contributions = engine.js_contributions({}, {"rock": 0.5, "pop": 0.5})
    assert set(contributions) == {"rock", "pop"}
    assert all(value == 0.0 for value in contributions.values())


def test_divergence_facts_ratio_is_clamped() -> None:
    """A comparable genre the household plays far more than average still clamps at 99x."""
    p = {"niche": 0.5, "rock": 0.5}
    q = {"niche": 0.0012, "rock": 0.5}
    facts = engine.divergence_facts(p, q, {"niche": "Niche", "rock": "Rock"})
    niche = next(s for s in facts["top_over"] if s["key"] == "niche")
    assert niche["baseline_known"] is True
    assert niche["ratio"] == 99.0


def test_divergence_facts_drops_a_genre_the_baseline_never_saw() -> None:
    """
    A baseline share of ~0 is an absence of reference data, so the genre earns no chip.

    The previous behaviour reported ratio 99.0 here, which read as a finding about the
    household rather than what it was: no reference data for the genre at all.
    """
    p = {"niche": 0.5, "rock": 0.5}
    q = {"niche": 1e-9, "rock": 0.5}
    facts = engine.divergence_facts(p, q, {"niche": "Niche", "rock": "Rock"})
    assert "niche" not in {s["key"] for s in facts["top_over"]}
    assert "niche" not in {s["key"] for s in facts["top_under"]}


def test_divergence_facts_percent_matches_score() -> None:
    """Test divergence facts percent matches score."""
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
    """Test divergence against shipped test baseline when household differs."""
    baseline = asyncio.run(load_baseline(str(FIXTURE_PATH)))
    # household listens to nothing but the baseline's smallest genre
    smallest_key = min(baseline.genre_shares, key=lambda key: baseline.genre_shares[key])
    household = {smallest_key: 1.0}
    jsd = engine.js_divergence(household, dict(baseline.genre_shares))
    assert 0.0 < jsd <= 1.0
    assert not math.isnan(jsd)
