"""
Tests for the Apple Music "Play History Daily Tracks" importer.

The export folder holds fifteen CSVs and the obviously-named one is the wrong file: the current
``Apple Music Play Activity.csv`` has no artist column anywhere in its 145, so a real 306,140-row
import produced nothing but "no artist name in the export (190539)". This file is the one with
the artist, at the cost of keeping time to the hour rather than the second.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from music_assistant.controllers.genome.importers.apple_csv import ApplePlayActivityStats
from music_assistant.controllers.genome.importers.apple_daily_tracks import (
    is_daily_tracks_header,
    parse_daily_tracks,
)

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "genome"
FIXTURE = FIXTURES_DIR / "apple_daily_tracks.csv"


def _hour(played_at: int) -> int:
    return datetime.fromtimestamp(played_at, tz=UTC).hour


async def _parse(stats: ApplePlayActivityStats, *, min_seconds: int = 30):
    return [
        listen
        async for listen in parse_daily_tracks(str(FIXTURE), min_seconds=min_seconds, stats=stats)
    ]


async def test_reads_the_artist_out_of_track_description() -> None:
    """The whole reason this file exists for us: it is the only one carrying an artist."""
    rows = await _parse(ApplePlayActivityStats())
    by_track = {r.track_name: r.artist_name for r in rows}
    assert by_track["Black Sheep"] == "Metric"
    assert by_track["Teenage Dream"] == "T. Rex"


async def test_splits_on_the_first_separator_only() -> None:
    """
    An artist's name rarely contains " - "; a title very often does.

    Splitting on the last separator, or on every one, would turn "Anthems for a Seventeen Year
    Old Girl" into a different track from itself depending on the row.
    """
    rows = await _parse(ApplePlayActivityStats())
    scene = next(r for r in rows if r.artist_name == "Broken Social Scene")
    assert scene.track_name == "Anthems for a Seventeen Year Old Girl"


async def test_play_count_becomes_that_many_plays_spread_over_the_listed_hours() -> None:
    """
    ``Play Count 2`` across ``Hours "1, 4"`` is one play in each hour, not two at 01:00.

    Collapsing the count would understate every play total in the profile, which is the number
    the whole genome is keyed on.
    """
    rows = await _parse(ApplePlayActivityStats())
    metric = sorted(r.played_at for r in rows if r.artist_name == "Metric")
    assert len(metric) == 2
    assert [_hour(t) for t in metric] == [1, 4]


async def test_repeat_plays_in_one_hour_land_on_distinct_minutes() -> None:
    """
    The store dedupes on ``(listener, source, artist, track, minute)``.

    Three plays of one track in one hour would otherwise be stored as one, silently, and the
    reconstruction would undercount exactly the tracks played most.
    """
    rows = await _parse(ApplePlayActivityStats())
    beck = sorted(r.played_at for r in rows if r.artist_name == "Beck")
    assert len(beck) == 3
    assert len({t // 60 for t in beck}) == 3
    assert {_hour(t) for t in beck} == {9}


async def test_duration_is_divided_across_the_plays_it_covers() -> None:
    """``Play Duration Milliseconds`` is the day's total for that track, not one play's."""
    rows = await _parse(ApplePlayActivityStats())
    beck = next(r for r in rows if r.artist_name == "Beck")
    assert beck.played_ms == 300_000


async def test_confidence_reflects_an_invented_minute() -> None:
    """The hour is Apple's; the minute is ours. The row should not claim otherwise."""
    rows = await _parse(ApplePlayActivityStats())
    assert all(r.confidence < 1.0 for r in rows)


async def test_skips_video_short_plays_and_unsplittable_descriptions() -> None:
    """Each filtered row records why, so an import that adds little can explain itself."""
    stats = ApplePlayActivityStats()
    rows = await _parse(stats)
    assert "Some Band" not in {r.artist_name for r in rows}
    assert stats.skip_reasons["not audio"] == 1
    assert stats.skip_reasons["played too briefly"] == 1
    assert stats.skip_reasons["track description has no 'Artist - Song' split"] == 1
    assert stats.skip_reasons["no date played"] == 1


async def test_a_natural_end_is_kept_however_brief_the_average() -> None:
    """A finished track is a finished track; min_seconds only guards partial plays."""
    stats = ApplePlayActivityStats()
    rows = await _parse(stats, min_seconds=600)
    assert {r.artist_name for r in rows} >= {"Metric", "Beck"}


async def test_header_sniffing_tells_the_two_exports_apart() -> None:
    """One upload button, two possible files - the header has to decide which parser runs."""
    daily = FIXTURE.read_text(encoding="utf-8").splitlines()[0].replace('"', "").split(",")
    activity = (
        (FIXTURES_DIR / "apple_play_activity_real_header.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
        .replace('"', "")
        .split(",")
    )
    assert is_daily_tracks_header(daily)
    assert not is_daily_tracks_header(activity)


async def test_a_wrong_file_says_so_instead_of_reporting_every_row_skipped() -> None:
    """The failure mode this whole episode was: a readable file, silently yielding nothing."""
    stats = ApplePlayActivityStats()
    rows = [
        listen
        async for listen in parse_daily_tracks(
            str(FIXTURES_DIR / "apple_play_activity_real_header.csv"),
            min_seconds=30,
            stats=stats,
        )
    ]
    assert rows == []
    assert any("not a Play History Daily Tracks export" in w for w in stats.warnings)
