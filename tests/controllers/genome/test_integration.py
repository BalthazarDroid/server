"""
End-to-end integration tests: real ``GenomeController`` -> real ``GenomeStore`` -> real engine.

Unlike ``test_controller_api.py`` (which runs against the in-memory ``StubGenomeStore``), these
tests exercise the actual wiring done during integration (§Part 4 "Integration"): the real
``GenomeStore`` from ``store.py``, the real Apple CSV / MusicBrainz / ListenBrainz enrichment
code, and the live ``PLAYLOG_UPDATED`` capture + one-time backfill path. No network call is made
anywhere - MusicBrainz/ListenBrainz traffic goes through :class:`FixtureHttpClient`, matching
``docs/ARCHITECTURE.md`` BRIEF.md's "those hosts are unreachable here" constraint.
"""

from __future__ import annotations

import asyncio
import types
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import MediaType
from music_assistant_models.playlog_update import PlaylogUpdate

from music_assistant.controllers.genome.constants import CONF_APPLE_IMPORT_DIR, LISTENER_HOUSEHOLD
from music_assistant.controllers.genome.controller import GenomeController
from music_assistant.helpers.database import DatabaseConnection
from music_assistant.helpers.json import json_dumps
from tests.controllers.genome.conftest import FIXTURES_DIR, FixtureHttpClient, load_fixture

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

_APPLE_CSV_FILENAME = "apple_play_activity.csv"


class _FixtureBackedAiohttpClient:
    """Stands in for ``http.AiohttpClient`` everywhere it is constructed, fixture-backed."""

    def __init__(self, _mass: Any, *, rate_limit: int, period: float, delegate: Any) -> None:
        """Ignore rate-limit params (the fixture client has no real throttling to do)."""
        self._delegate = delegate

    async def get_json(self, url: str, *, params: Any = None, headers: Any = None) -> Any:
        """Delegate to the shared :class:`FixtureHttpClient`."""
        return await self._delegate.get_json(url, params=params, headers=headers)

    async def post_json(self, url: str, *, json: Any, headers: Any = None) -> Any:
        """Delegate to the shared :class:`FixtureHttpClient`."""
        return await self._delegate.post_json(url, json=json, headers=headers)


def _patch_aiohttp_client(monkeypatch: Any, fixture_client: FixtureHttpClient) -> None:
    """Make every ``AiohttpClient(...)`` construction anywhere in genome/* use fixtures."""

    def factory(mass: Any, *, rate_limit: int, period: float) -> _FixtureBackedAiohttpClient:
        return _FixtureBackedAiohttpClient(
            mass, rate_limit=rate_limit, period=period, delegate=fixture_client
        )

    monkeypatch.setattr("music_assistant.controllers.genome.http.AiohttpClient", factory)


def _build_mass(*, storage_path: str, database: DatabaseConnection | None = None) -> Any:
    """Build a minimal ``MusicAssistant`` double sufficient to drive ``GenomeController``."""
    subscriptions: list[tuple[Any, Any]] = []
    background_tasks: list[asyncio.Task[Any]] = []

    def subscribe(cb: Any, event_filter: Any = None, _id_filter: Any = None) -> Any:
        subscriptions.append((cb, event_filter))
        return lambda: subscriptions.remove((cb, event_filter))

    def create_task(coro: Any, *_a: Any, **_k: Any) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        background_tasks.append(task)
        return task

    mass = types.SimpleNamespace(
        storage_path=storage_path,
        players=None,
        subscribe=subscribe,
        create_task=create_task,
        get_provider=lambda _domain: None,
        tasks=types.SimpleNamespace(
            register_scheduled_task=lambda **_k: None,
            unregister_scheduled_task=lambda *_a, **_k: None,
        ),
        config=types.SimpleNamespace(get_raw_core_config_value=lambda *_a, **_k: "GLOBAL"),
        music=types.SimpleNamespace(database=database),
    )
    mass._subscriptions = subscriptions  # for assertions
    mass._background_tasks = background_tasks  # for awaiting
    return mass


def _defaulted_get_config_value(
    key: str,  # noqa: ARG001
    default: object = None,
    *,
    return_type: type | None = None,  # noqa: ARG001
) -> Any:
    """Return every config value's documented default, exactly like ``conftest.genome_controller``."""
    return default


def _get_config_value_with_apple_dir(
    key: str,
    default: object = None,
    *,
    return_type: type | None = None,  # noqa: ARG001
) -> Any:
    """Like :func:`_defaulted_get_config_value`, but points ``apple_import_dir`` at the fixtures."""
    if key == CONF_APPLE_IMPORT_DIR:
        return str(FIXTURES_DIR)
    return default


async def _build_library_db(tmp_path: Path) -> DatabaseConnection:
    """Build a temp ``library.db`` pre-seeded with the ma_playlog/ma_tracks fixtures."""
    tmp_path.mkdir(parents=True, exist_ok=True)
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


async def test_apple_import_then_rebuild_produces_enriched_genres(
    tmp_path: Path, fixture_http_client: FixtureHttpClient, monkeypatch: Any
) -> None:
    """Apple CSV -> real store -> MB/LB enrichment (fixtures) -> engine round-trips end to end."""
    _patch_aiohttp_client(monkeypatch, fixture_http_client)
    mass = _build_mass(storage_path=str(tmp_path))
    controller = GenomeController(mass)
    controller.get_config_value = _get_config_value_with_apple_dir  # type: ignore[method-assign]
    try:
        await controller.store.setup()
        import_result = await controller.import_apple(
            "", 0, "", final=True, filename=_APPLE_CSV_FILENAME
        )
        # 14 rows parse cleanly, but 2 share the same artist/track within the same minute (§3.9)
        # and collapse to one row via `dedupe_key`; the other 6 fixture rows are filtered out by
        # the parser itself before `add_listens` ever sees them, so they never reach this result
        assert import_result["rows_imported"] == 13
        assert import_result["rows_duplicate"] == 1

        rebuild_result = await controller.rebuild()
        assert rebuild_result["listens_scanned"] == 13
        assert rebuild_result["artists_enriched"] > 0

        genome = rebuild_result["genome"]
        assert genome["stats"]["total_listens"] == 13
        assert genome["stats"]["distinct_artists"] == 6
        # Sigur Ros resolves to rock/ambient genres via the MusicBrainz fixture (§3.9)
        artist_names = {a["name"] for a in genome["top_artists"]}
        assert "Sigur Rós" in artist_names
        sigur = next(a for a in genome["top_artists"] if a["name"] == "Sigur Rós")
        assert sigur["mbid"] == "f6f2326f-6b25-4170-b89d-e235b25508e8"
        assert sigur["genres"]  # resolved via the fixture MB tags

        # a second, cached read must not re-scan or re-enrich
        cached = await controller.get_genome()
        assert cached["stale"] is False
        assert cached["generated_at"] == genome["generated_at"]
    finally:
        await controller.close()


async def test_setup_attaches_live_capture_and_runs_backfill(tmp_path: Path) -> None:
    """
    setup() subscribes PLAYLOG_UPDATED and the one-time backfill seeds the real store.

    This is the feature's most important data path (§1.4): MA purges playlog rows after 90
    days, so live capture - not a one-off backfill - is what keeps history from being lost.
    """
    library_db = await _build_library_db(tmp_path / "library")
    genome_dir = tmp_path / "genome"
    genome_dir.mkdir()
    mass = _build_mass(storage_path=str(genome_dir), database=library_db)
    controller = GenomeController(mass)
    controller.get_config_value = _defaulted_get_config_value  # type: ignore[method-assign]
    try:
        await controller.setup(config=None)  # type: ignore[arg-type]
        # live capture: PLAYLOG_UPDATED must be subscribed, not just implemented-but-unused
        assert len(mass._subscriptions) == 1

        # the one-time backfill runs in the background; wait for it to finish
        await asyncio.gather(*mass._background_tasks)
        assert await controller.store.backfill_done() is True
        assert await controller.store.count_listens(LISTENER_HOUSEHOLD) > 0

        # running setup() again after a restart must not re-run the backfill or duplicate rows
        count_after_first_backfill = await controller.store.count_listens(LISTENER_HOUSEHOLD)
        mass2 = _build_mass(storage_path=str(genome_dir), database=library_db)
        controller2 = GenomeController(mass2, store=controller.store)
        controller2.get_config_value = _defaulted_get_config_value  # type: ignore[method-assign]
        await controller2.setup(config=None)  # type: ignore[arg-type]
        assert not mass2._background_tasks  # backfill_done() short-circuited the re-run
        assert (
            await controller.store.count_listens(LISTENER_HOUSEHOLD) == count_after_first_backfill
        )
    finally:
        await controller.close()
        await library_db.close()


async def test_live_playlog_capture_survives_the_90_day_purge(tmp_path: Path) -> None:
    """
    A live ``PLAYLOG_UPDATED`` event is captured into ``genome_listens`` immediately.

    Genome must not depend on the MA ``playlog`` row still existing later - it is captured at
    event time, independent of MA's own 90-day retention sweep (§1.4).
    """

    async def fake_get(_item_id: str, _provider: str, recursive: bool = True) -> Any:  # noqa: ARG001
        return types.SimpleNamespace(
            name="Svefn-g-englar",
            artists=[types.SimpleNamespace(name="Sigur Rós")],
            duration=600,
            album=types.SimpleNamespace(name="Ágætis byrjun"),
        )

    mass = _build_mass(storage_path=str(tmp_path))
    mass.music.tracks = types.SimpleNamespace(get=fake_get)
    controller = GenomeController(mass)
    controller.get_config_value = _defaulted_get_config_value  # type: ignore[method-assign]
    try:
        await controller.setup(config=None)  # type: ignore[arg-type]
        (playlog_cb, _event_filter) = mass._subscriptions[0]
        update = PlaylogUpdate(
            uri="library://track/42",
            media_type=MediaType.TRACK,
            fully_played=True,
            seconds_played=300,
            userid="user-abc",
        )
        await playlog_cb(types.SimpleNamespace(data=update))
        # even before any rebuild, the live-captured row is durably in genome_listens
        assert await controller.store.count_listens(LISTENER_HOUSEHOLD) >= 1
        counts = await controller.store.source_counts(LISTENER_HOUSEHOLD)
        assert counts.get("ma_playlog") == 1
    finally:
        await controller.close()
