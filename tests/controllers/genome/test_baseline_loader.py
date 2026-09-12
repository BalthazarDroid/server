"""Tests for ``music_assistant/controllers/genome/baseline/loader.py`` (§3.7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from music_assistant.controllers.genome.baseline.loader import (
    BASELINE_V1_PATH,
    load_baseline,
    uniform_baseline,
)

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "genome" / "baseline_test.json"


async def test_load_shipped_baseline_v1() -> None:
    """Test load shipped baseline v1."""
    baseline = await load_baseline(BASELINE_V1_PATH)
    assert baseline.version == "v1-2026-09"
    assert len(baseline.genre_shares) == 59
    assert abs(sum(baseline.genre_shares.values()) - 1.0) < 1e-3
    assert abs(sum(baseline.era_shares.values()) - 1.0) < 1e-3
    for percentile in (5, 10, 25, 50, 75, 90):
        assert percentile in baseline.listener_percentiles


async def test_load_test_fixture_baseline() -> None:
    """Test load test fixture baseline."""
    baseline = await load_baseline(str(FIXTURE_PATH))
    assert baseline.version == "test-v1"
    assert len(baseline.genre_shares) == 8


async def test_missing_file_falls_back_to_uniform(tmp_path: Path) -> None:
    """Test missing file falls back to uniform."""
    baseline = await load_baseline(str(tmp_path / "does_not_exist.json"))
    assert baseline.version == "uniform-fallback"
    assert len(baseline.genre_shares) == 59
    assert all(pytest.approx(1 / 59, rel=1e-6) == share for share in baseline.genre_shares.values())


async def test_genre_shares_not_summing_to_one_falls_back(tmp_path: Path) -> None:
    """Test genre shares not summing to one falls back."""
    path = tmp_path / "bad_baseline.json"
    path.write_text(
        json.dumps(
            {
                "version": "bad",
                "genre_shares": {"rock": 0.1, "pop": 0.1},
                "era_shares": {"1990": 1.0},
                "listener_percentiles": {"5": 1, "10": 2, "25": 3, "50": 4, "75": 5, "90": 6},
                "concentration": 0.1,
            }
        )
    )
    baseline = await load_baseline(str(path))
    assert baseline.version == "uniform-fallback"


async def test_unknown_genre_key_falls_back(tmp_path: Path) -> None:
    """Test unknown genre key falls back."""
    path = tmp_path / "bad_baseline.json"
    path.write_text(
        json.dumps(
            {
                "version": "bad",
                "genre_shares": {"not_a_real_genre": 1.0},
                "era_shares": {"1990": 1.0},
                "listener_percentiles": {"5": 1, "10": 2, "25": 3, "50": 4, "75": 5, "90": 6},
                "concentration": 0.1,
            }
        )
    )
    baseline = await load_baseline(str(path))
    assert baseline.version == "uniform-fallback"


async def test_missing_required_percentile_falls_back(tmp_path: Path) -> None:
    """Test missing required percentile falls back."""
    path = tmp_path / "bad_baseline.json"
    path.write_text(
        json.dumps(
            {
                "version": "bad",
                "genre_shares": {"rock": 1.0},
                "era_shares": {"1990": 1.0},
                "listener_percentiles": {"5": 1, "10": 2, "25": 3, "50": 4},
                "concentration": 0.1,
            }
        )
    )
    baseline = await load_baseline(str(path))
    assert baseline.version == "uniform-fallback"


async def test_malformed_json_falls_back(tmp_path: Path) -> None:
    """Test malformed json falls back."""
    path = tmp_path / "not_json.json"
    path.write_text("{not valid json")
    baseline = await load_baseline(str(path))
    assert baseline.version == "uniform-fallback"


def test_uniform_baseline_shape() -> None:
    """Test uniform baseline shape."""
    baseline = uniform_baseline()
    assert baseline.version == "uniform-fallback"
    assert abs(sum(baseline.genre_shares.values()) - 1.0) < 1e-6
    assert abs(sum(baseline.era_shares.values()) - 1.0) < 1e-6
    for percentile in (5, 10, 25, 50, 75, 90):
        assert percentile in baseline.listener_percentiles
