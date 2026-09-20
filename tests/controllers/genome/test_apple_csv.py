"""Tests for the Apple Music Play Activity CSV importer (§3.8, §3.9)."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

from music_assistant.controllers.genome.importers.apple_csv import (
    ApplePlayActivityStats,
    import_play_activity,
    parse_play_activity,
)
from music_assistant.controllers.genome.store import GenomeStore
from tests.controllers.genome.conftest import FIXTURES_DIR

if TYPE_CHECKING:
    from pathlib import Path

FIXTURE_CSV = str(FIXTURES_DIR / "apple_play_activity.csv")


async def _new_store(tmp_path: Path) -> GenomeStore:
    mass = types.SimpleNamespace(storage_path=str(tmp_path), players=None)
    store = GenomeStore(mass)
    await store.setup()
    return store


async def test_parse_play_activity_filters_and_counts() -> None:
    """The fixture's 20 rows resolve to exactly 14 listens, 6 filtered/skipped rows."""
    stats = ApplePlayActivityStats()
    listens = [
        listen async for listen in parse_play_activity(FIXTURE_CSV, min_seconds=30, stats=stats)
    ]
    assert stats.rows_read == 20
    assert stats.rows_skipped == 6
    # 14 NATURAL_END_OF_TRACK rows produce 14 listens (the dedupe pair is not collapsed at parse
    # time - that happens in GenomeStore.add_listens - so this level sees all of them)
    assert len(listens) == 14


async def test_parse_play_activity_handles_lucene_and_non_ascii_artists() -> None:
    """AC/DC (Lucene-special) and Sigur Rós (non-ASCII) both parse to sane keys."""
    listens = [listen async for listen in parse_play_activity(FIXTURE_CSV, min_seconds=30)]
    artist_names = {listen.artist_name for listen in listens}
    assert "AC/DC" in artist_names
    assert "Sigur Rós" in artist_names
    ac_dc = next(listen for listen in listens if listen.artist_name == "AC/DC")
    assert ac_dc.artist_key == "acdc"


async def test_parse_play_activity_filters_short_and_wrong_media_type() -> None:
    """Short MANUALLY_SELECTED plays, FAILED_TO_LOAD, and VIDEO rows never become listens."""
    listens = [listen async for listen in parse_play_activity(FIXTURE_CSV, min_seconds=30)]
    assert all(listen.fully_played for listen in listens)
    assert not any(
        name == "Karma Police (Video)" for name in (listen.track_name for listen in listens)
    )
    assert not any(listen.track_name == "Weird Fishes" for listen in listens)


async def test_import_play_activity_stores_deduped_listens(tmp_path: Path) -> None:
    """The end-to-end importer stores 13 rows (14 valid minus 1 same-minute duplicate)."""
    store = await _new_store(tmp_path)
    try:
        result = await import_play_activity(
            store, FIXTURE_CSV, listener="household", min_seconds=30
        )
        assert result["source"] == "apple_export"
        assert result["rows_read"] == 20
        assert result["rows_imported"] == 13
        assert result["rows_duplicate"] == 1
        assert result["rows_skipped"] == 6
        assert result["first_played_at"] is not None
        assert result["last_played_at"] is not None
        assert await store.count_listens("household") == 13
    finally:
        await store.close()


async def test_import_play_activity_is_idempotent(tmp_path: Path) -> None:
    """Running the import twice must not double-count rows."""
    store = await _new_store(tmp_path)
    try:
        await import_play_activity(store, FIXTURE_CSV, listener="household", min_seconds=30)
        second = await import_play_activity(
            store, FIXTURE_CSV, listener="household", min_seconds=30
        )
        assert second["rows_imported"] == 0
        assert await store.count_listens("household") == 13
    finally:
        await store.close()


async def test_an_unrecognised_header_says_so_instead_of_blaming_the_rows(
    tmp_path: Path,
) -> None:
    """
    The diagnostic this exists for.

    A real 306,140-row export imported as "0 added, 306140 skipped" with no other word. Every
    column lookup tolerates a missing column by returning an empty string, so an unrecognised
    header raises nothing - it just fails each row's "has a title and an artist" test, and the
    result reads as though the file were full of junk rather than as though the parser could
    not read its header.
    """
    csv_path = tmp_path / "unknown.csv"
    csv_path.write_text(
        "Some Column,Another Column\nvalue,other\n",
        encoding="utf-8",
    )
    stats = ApplePlayActivityStats()
    rows = [row async for row in parse_play_activity(str(csv_path), min_seconds=30, stats=stats)]

    assert rows == []
    joined = " ".join(stats.warnings)
    assert "header was not recognised" in joined
    # It must name what it wanted AND what it found, or the reader cannot act on it.
    assert "'Artist Name'" in joined
    assert "'Some Column'" in joined


async def test_skip_reasons_are_counted_and_ranked(tmp_path: Path) -> None:
    """A skipped row records why, so the dominant reason is visible in the total."""
    csv_path = tmp_path / "mixed.csv"
    csv_path.write_text(
        "Song Name,Artist Name,Event Start Timestamp,Play Duration Milliseconds,"
        "End Reason Type,Event Type,Media Type\n"
        # kept: a natural end needs no duration
        "Good,Artist,2024-01-01T00:00:00Z,1000,NATURAL_END_OF_TRACK,PLAY_END,AUDIO\n"
        # skipped: too brief, and not a natural end
        "Brief,Artist,2024-01-01T00:00:00Z,1000,STOPPED,PLAY_END,AUDIO\n"
        "Brief2,Artist,2024-01-01T00:00:00Z,900,STOPPED,PLAY_END,AUDIO\n"
        # skipped: video
        "Vid,Artist,2024-01-01T00:00:00Z,999999,NATURAL_END_OF_TRACK,PLAY_END,VIDEO\n",
        encoding="utf-8",
    )
    stats = ApplePlayActivityStats()
    rows = [row async for row in parse_play_activity(str(csv_path), min_seconds=30, stats=stats)]

    assert [r.track_name for r in rows] == ["Good"]
    assert stats.rows_skipped == 3
    assert stats.skip_reasons["played too briefly"] == 2
    assert stats.skip_reasons["not audio"] == 1
    assert not any("header was not recognised" in w for w in stats.warnings)
