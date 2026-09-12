"""Tests for the MA playlog live-capture and backfill importer (§1.4, §3.8, §3.9)."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import MediaType
from music_assistant_models.playlog_update import PlaylogUpdate

from music_assistant.controllers.genome.importers.ma_playlog import MaPlaylogImporter
from music_assistant.controllers.genome.store import GenomeStore
from music_assistant.helpers.database import DatabaseConnection
from music_assistant.helpers.json import json_dumps
from tests.controllers.genome.conftest import load_fixture

if TYPE_CHECKING:
    from pathlib import Path

_PLAYLOG_DDL = """CREATE TABLE IF NOT EXISTS playlog(
    [id] INTEGER PRIMARY KEY AUTOINCREMENT,
    [item_id] TEXT NOT NULL,
    [provider] TEXT NOT NULL,
    [media_type] TEXT NOT NULL,
    [name] TEXT NOT NULL,
    [image] json,
    [artists] json,
    [timestamp] INTEGER DEFAULT 0,
    [fully_played] BOOLEAN,
    [seconds_played] INTEGER,
    [userid] TEXT NOT NULL,
    [queue_id] TEXT,
    [user_initiated] BOOLEAN NOT NULL DEFAULT 1,
    [playback_speed] REAL NOT NULL DEFAULT 1.0,
    UNIQUE(item_id, provider, media_type, userid));"""

_TRACKS_DDL = """CREATE TABLE IF NOT EXISTS tracks(
    [item_id] INTEGER PRIMARY KEY AUTOINCREMENT,
    [name] TEXT NOT NULL,
    [sort_name] TEXT NOT NULL,
    [version] TEXT,
    [duration] INTEGER,
    [favorite] BOOLEAN NOT NULL DEFAULT 0,
    [metadata] json NOT NULL,
    [play_count] INTEGER DEFAULT 0,
    [last_played] INTEGER DEFAULT 0,
    [timestamp_added] INTEGER DEFAULT (cast(strftime('%s','now') as int)),
    [timestamp_modified] INTEGER NOT NULL DEFAULT 0,
    [search_name] TEXT NOT NULL,
    [search_sort_name] TEXT NOT NULL);"""


class FakeArtist:
    """Minimal stand-in for a MediaItem artist mapping."""

    def __init__(self, name: str) -> None:
        """Store the artist's name."""
        self.name = name


class FakeTrack:
    """Minimal stand-in for the Track object `mass.music.tracks.get` returns."""

    def __init__(self, name: str, artist: str, album: str | None, duration: int | None) -> None:
        """Build a fake track with the given fields."""
        self.name = name
        self.artists = [FakeArtist(artist)]
        self.duration = duration
        self.album = types.SimpleNamespace(name=album) if album else None


class FakeEvent:
    """Minimal stand-in for a `MassEvent`."""

    def __init__(self, data: Any) -> None:
        """Store the event payload."""
        self.data = data


async def _build_library_db(tmp_path: Path) -> DatabaseConnection:
    """Build a temp library.db pre-seeded with the ma_playlog/ma_tracks fixtures."""
    database = DatabaseConnection(str(tmp_path / "library.db"))
    await database.setup()
    await database.execute(_PLAYLOG_DDL)
    await database.execute(_TRACKS_DDL)
    await database.commit()

    for row in load_fixture("ma_playlog_rows"):
        values = dict(row)
        values["artists"] = json_dumps(values["artists"])
        await database.insert_or_replace("playlog", values)

    for row in load_fixture("ma_tracks_rows"):
        await database.insert(
            "tracks",
            {
                "item_id": row["item_id"],
                "name": row["name"],
                "sort_name": row["name"],
                "play_count": row["play_count"],
                "last_played": row["last_played"],
                "metadata": "{}",
                "search_name": row["name"].lower(),
                "search_sort_name": row["name"].lower(),
                "timestamp_modified": 0,
            },
        )
    return database


async def _new_store(tmp_path: Path) -> GenomeStore:
    genome_dir = tmp_path / "genome"
    genome_dir.mkdir(exist_ok=True)
    mass = types.SimpleNamespace(storage_path=str(genome_dir), players=None)
    store = GenomeStore(mass)
    await store.setup()
    return store


async def test_backfill_skips_rows_below_the_seconds_floor(tmp_path: Path) -> None:
    """The 2 fixture rows with seconds_played=12 (not fully played) are skipped."""
    library_db = await _build_library_db(tmp_path)
    store = await _new_store(tmp_path)
    try:
        mass = types.SimpleNamespace(
            music=types.SimpleNamespace(database=library_db),
            subscribe=lambda *_a, **_k: lambda: None,
        )
        importer = MaPlaylogImporter(mass, store)
        result = await importer.backfill(min_seconds_played=30)
        assert result["rows_read"] == 25
        assert result["rows_skipped"] == 2
        assert result["rows_imported"] > 0
        assert await store.count_listens("household") == result["rows_imported"]
    finally:
        await store.close()
        await library_db.close()


async def test_backfill_is_idempotent(tmp_path: Path) -> None:
    """Running backfill twice must not create duplicate rows (dedupe_key)."""
    library_db = await _build_library_db(tmp_path)
    store = await _new_store(tmp_path)
    try:
        mass = types.SimpleNamespace(
            music=types.SimpleNamespace(database=library_db),
            subscribe=lambda *_a, **_k: lambda: None,
        )
        importer = MaPlaylogImporter(mass, store)
        first = await importer.backfill()
        second = await importer.backfill()
        assert second["rows_imported"] == 0
        assert second["rows_duplicate"] == first["rows_imported"]
    finally:
        await store.close()
        await library_db.close()


async def test_backfill_caps_synthetic_plays_at_twenty(tmp_path: Path) -> None:
    """
    A track with play_count=30 contributes at most 21 rows (1 real + `min(n-1, 20)` synthetic).

    The fixture's 25 playlog rows cycle over only 12 distinct track names, so several rows share
    a `track_key`; this asserts the total import count matches hand-computed expectations for the
    exact `ma_tracks_rows.json` play_count distribution (7 tracks at play_count=5, 3 at
    play_count=30 capped to 21, 13 at play_count=1, 2 skipped below the floor).
    """
    library_db = await _build_library_db(tmp_path)
    store = await _new_store(tmp_path)
    try:
        mass = types.SimpleNamespace(
            music=types.SimpleNamespace(database=library_db),
            subscribe=lambda *_a, **_k: lambda: None,
        )
        importer = MaPlaylogImporter(mass, store)
        result = await importer.backfill()
        # 7 rows * 5 plays (play_count=5) + 3 rows * 21 plays (play_count=30, capped) +
        # 13 rows * 1 play (play_count=1) = 35 + 63 + 13 = 111
        assert result["rows_imported"] == 111
    finally:
        await store.close()
        await library_db.close()


async def test_live_capture_appends_a_listen(tmp_path: Path) -> None:
    """PLAYLOG_UPDATED for a track event appends exactly one genome_listens row."""
    store = await _new_store(tmp_path)
    try:

        async def fake_get(
            _item_id: str,
            _provider: str,
            recursive: bool = True,  # noqa: ARG001
        ) -> FakeTrack:
            # `recursive` is called by keyword (`mass.music.tracks.get(..., recursive=False)`),
            # so it cannot be underscore-prefixed like the two positional args above
            return FakeTrack("Svefn-g-englar", "Sigur Rós", "Ágætis byrjun", 600)

        mass = types.SimpleNamespace(
            music=types.SimpleNamespace(tracks=types.SimpleNamespace(get=fake_get)),
            subscribe=lambda *_a, **_k: lambda: None,
        )
        importer = MaPlaylogImporter(mass, store)
        update = PlaylogUpdate(
            uri="library://track/42",
            media_type=MediaType.TRACK,
            fully_played=True,
            seconds_played=300,
            userid="user-abc",
        )
        await importer._on_playlog(FakeEvent(update))
        assert await store.count_listens("household") == 1
        listen = await anext(store.iter_listens("household"))
        assert listen.source == "ma_playlog"
        assert listen.player_id is None  # PlaylogUpdate carries no queue/player id
        assert listen.artist_key == "sigur ros"
    finally:
        await store.close()


async def test_live_capture_ignores_non_track_events(tmp_path: Path) -> None:
    """A PLAYLOG_UPDATED for a non-track media type must be a no-op."""
    store = await _new_store(tmp_path)
    try:
        mass = types.SimpleNamespace(
            music=types.SimpleNamespace(tracks=None), subscribe=lambda *_a, **_k: lambda: None
        )
        importer = MaPlaylogImporter(mass, store)
        update = PlaylogUpdate(
            uri="library://podcast_episode/1",
            media_type=MediaType.PODCAST_EPISODE,
            fully_played=True,
            seconds_played=100,
            userid=None,
        )
        await importer._on_playlog(FakeEvent(update))
        assert await store.count_listens("household") == 0
    finally:
        await store.close()


def test_attach_subscribes_to_playlog_updated() -> None:
    """attach() subscribes exactly the handler to PLAYLOG_UPDATED and returns the unsubscriber."""
    calls: list[Any] = []

    def fake_subscribe(cb: Any, event_filter: Any = None, _id_filter: Any = None) -> Any:
        calls.append((cb, event_filter))
        return lambda: None

    mass = types.SimpleNamespace(subscribe=fake_subscribe)
    store = types.SimpleNamespace()
    importer = MaPlaylogImporter(mass, store)  # type: ignore[arg-type]
    unsubscribe = importer.attach()
    assert callable(unsubscribe)
    assert len(calls) == 1
    assert calls[0][0] == importer._on_playlog
