"""Tests for :class:`GenomeStore` (§3.1)."""

from __future__ import annotations

import time
import types
from typing import TYPE_CHECKING

from music_assistant.controllers.genome.constants import (
    DB_TABLE_GENOME_ARTIST_META,
    GENOME_RESULT_SCHEMA_VERSION,
    RESOLVE_ERROR_COOLDOWN_HOURS,
    RESOLVE_STATE_ERROR,
)
from music_assistant.controllers.genome.models import Listen
from music_assistant.controllers.genome.store import ArtistMetaWrite, GenomeStore

if TYPE_CHECKING:
    from pathlib import Path


def _listen(**overrides: object) -> Listen:
    defaults: dict[str, object] = {
        "played_at": 1_700_000_000,
        "artist_key": "sigurros",
        "artist_name": "Sigur Rós",
        "track_key": "svefngenglar",
        "track_name": "Svefn-g-englar",
        "album_name": "Ágætis byrjun",
        "source": "apple_export",
        "player_id": None,
        "duration_ms": 600_000,
        "played_ms": 600_000,
        "fully_played": True,
        "confidence": 1.0,
    }
    defaults.update(overrides)
    return Listen(**defaults)  # type: ignore[arg-type]


async def _new_store(tmp_path: Path) -> GenomeStore:
    mass = types.SimpleNamespace(storage_path=str(tmp_path), players=None)
    store = GenomeStore(mass)
    await store.setup()
    return store


async def test_setup_is_idempotent(tmp_path: Path) -> None:
    """Calling setup twice against the same directory must not raise or duplicate schema."""
    store = await _new_store(tmp_path)
    await store.close()
    store2 = await _new_store(tmp_path)
    assert await store2.count_listens("household") == 0
    await store2.close()


async def test_add_listens_dedupes_within_same_minute(tmp_path: Path) -> None:
    """Two listens with the same dedupe_key must collapse to one stored row."""
    store = await _new_store(tmp_path)
    try:
        listens = [_listen(played_at=1_700_000_000), _listen(played_at=1_700_000_030)]
        result = await store.add_listens(listens, listener="household")
        assert result["rows_imported"] == 1
        assert result["rows_duplicate"] == 1
        assert await store.count_listens("household") == 1
    finally:
        await store.close()


async def test_add_listens_creates_pending_artist_stub(tmp_path: Path) -> None:
    """A newly-seen artist must appear in pending_artist_keys."""
    store = await _new_store(tmp_path)
    try:
        await store.add_listens([_listen()], listener="household")
        pending = await store.pending_artist_keys()
        assert pending == [("sigurros", "Sigur Rós")]
    finally:
        await store.close()


async def test_upsert_artist_meta_full_clears_pending_state(tmp_path: Path) -> None:
    """Resolving an artist must remove it from the pending queue and be readable back."""
    store = await _new_store(tmp_path)
    try:
        await store.add_listens([_listen()], listener="household")
        row: ArtistMetaWrite = {
            "artist_key": "sigurros",
            "artist_name": "Sigur Rós",
            "mbid": "f6f2326f-6b25-4170-b89d-e235b25508e8",
            "mb_tags": [{"name": "post-rock", "count": 8}],
            "genres": ["rock", "ambient"],
            "begin_year": 1994,
            "first_release_year": 1997,
            "country": "IS",
            "lb_listeners": 118422,
            "lb_listen_count": 4821334,
        }
        await store.upsert_artist_meta_full([row], state="ok")
        assert await store.pending_artist_keys() == []
        meta = await store.get_artist_meta(["sigurros"])
        assert meta["sigurros"].genres == ("rock", "ambient")
        assert meta["sigurros"].lb_listeners == 118422
    finally:
        await store.close()


async def test_source_counts_and_clear(tmp_path: Path) -> None:
    """source_counts groups by source; clear(listener=...) only drops that listener's rows."""
    store = await _new_store(tmp_path)
    try:
        await store.add_listens([_listen(source="apple_export")], listener="household")
        await store.add_listens(
            [
                _listen(
                    artist_key="acdc",
                    artist_name="AC/DC",
                    track_key="backinblack",
                    track_name="Back In Black",
                    played_at=1_700_100_000,
                    source="lastfm",
                )
            ],
            listener="household",
        )
        counts = await store.source_counts("household")
        assert counts == {"apple_export": 1, "lastfm": 1}

        await store.clear(listener="household")
        assert await store.count_listens("household") == 0
        # artist metadata (shared across listeners) is not cleared by a listener-scoped clear
        assert await store.pending_artist_keys(limit=10)
    finally:
        await store.close()


async def test_clear_all_drops_artist_meta_too(tmp_path: Path) -> None:
    """clear(listener=None) is a full reset, including artist metadata."""
    store = await _new_store(tmp_path)
    try:
        await store.add_listens([_listen()], listener="household")
        await store.clear()
        assert await store.count_listens("household") == 0
        assert await store.pending_artist_keys() == []
    finally:
        await store.close()


async def test_cached_genome_roundtrip(tmp_path: Path) -> None:
    """set_cached_genome/get_cached_genome round-trip an arbitrary JSON-shaped payload."""
    store = await _new_store(tmp_path)
    try:
        assert await store.get_cached_genome("household") is None
        payload = {
            "schema_version": GENOME_RESULT_SCHEMA_VERSION,
            "stats": {"total_listens": 0},
        }
        await store.set_cached_genome("household", payload)  # type: ignore[arg-type]
        assert await store.get_cached_genome("household") == payload
    finally:
        await store.close()


async def test_cached_genome_from_older_result_schema_is_discarded(tmp_path: Path) -> None:
    """
    A cached blob from an older result shape is dropped, not served.

    Regression: `bases`/`base_mix` were added to GenomeResult without the cache key
    changing, so `genome/get` kept serving pre-`bases` blobs. The frontend reads those
    fields unconditionally, and the missing key took the whole molecule card down with
    no server-side error to show for it.
    """
    store = await _new_store(tmp_path)
    try:
        stale = {
            "schema_version": GENOME_RESULT_SCHEMA_VERSION - 1,
            "stats": {"total_listens": 5},
        }
        await store.set_cached_genome("household", stale)  # type: ignore[arg-type]
        assert await store.get_cached_genome("household") is None
    finally:
        await store.close()


async def test_cached_genome_without_schema_version_is_discarded(tmp_path: Path) -> None:
    """A cache entry predating result versioning has no recorded shape, so it is dropped."""
    store = await _new_store(tmp_path)
    try:
        await store.set_cached_genome("household", {"stats": {}})  # type: ignore[arg-type]
        assert await store.get_cached_genome("household") is None
    finally:
        await store.close()


async def test_dedupe_window_drops_cross_source_duplicate(tmp_path: Path) -> None:
    """A Last.fm listen within 90s of an MA-sourced listen for the same track is dropped."""
    store = await _new_store(tmp_path)
    try:
        await store.add_listens(
            [_listen(source="ma_playlog", played_at=1_700_000_000)], listener="household"
        )
        await store.add_listens(
            [_listen(source="lastfm", played_at=1_700_000_050)], listener="household"
        )
        assert await store.count_listens("household") == 2
        deleted = await store.dedupe_window()
        assert deleted == 1
        assert await store.count_listens("household") == 1
    finally:
        await store.close()


async def test_iter_listens_respects_since(tmp_path: Path) -> None:
    """iter_listens only yields rows strictly newer than `since`."""
    store = await _new_store(tmp_path)
    try:
        await store.add_listens(
            [_listen(played_at=1_700_000_000), _listen(played_at=1_800_000_000, source="lastfm")],
            listener="household",
        )
        collected = [
            listen async for listen in store.iter_listens("household", since=1_750_000_000)
        ]
        assert len(collected) == 1
        assert collected[0].played_at == 1_800_000_000
    finally:
        await store.close()


async def test_lastfm_backfill_done_defaults_false_and_persists(tmp_path: Path) -> None:
    """The one-time Last.fm sweep flag (§3.1 `settings` table, P1) starts false and sticks."""
    store = await _new_store(tmp_path)
    try:
        assert await store.lastfm_backfill_done() is False
        await store.mark_lastfm_backfill_done()
        assert await store.lastfm_backfill_done() is True
        # independent of the (also settings-table-backed) MA playlog backfill flag
        assert await store.backfill_done() is False
    finally:
        await store.close()


async def test_artist_resolution_counts_groups_by_state(tmp_path: Path) -> None:
    """artist_resolution_counts (P3) reports a count per resolve_state across all artists."""
    store = await _new_store(tmp_path)
    try:
        assert await store.artist_resolution_counts() == {}
        await store.add_listens([_listen()], listener="household")  # -> one pending stub
        await store.upsert_artist_meta_full(
            [{"artist_key": "resolved-artist", "artist_name": "Resolved Artist"}], state="ok"
        )
        await store.upsert_artist_meta_full(
            [{"artist_key": "missing-artist", "artist_name": "Missing Artist"}],
            state="not_found",
        )
        counts = await store.artist_resolution_counts()
        assert counts == {"pending": 1, "ok": 1, "not_found": 1}
    finally:
        await store.close()


async def test_pending_popularity_keys_finds_resolved_artists_missing_listeners(
    tmp_path: Path,
) -> None:
    """An artist with an mbid but no lb_listeners is the popularity backlog, regardless of age."""
    store = await _new_store(tmp_path)
    try:
        await store.upsert_artist_meta_full(
            [
                {
                    "artist_key": "sigurros",
                    "artist_name": "Sigur Rós",
                    "mbid": "f6f2326f-6b25-4170-b89d-e235b25508e8",
                }
            ],
            state="ok",
        )
        assert await store.pending_popularity_keys() == [
            ("sigurros", "f6f2326f-6b25-4170-b89d-e235b25508e8")
        ]
    finally:
        await store.close()


async def test_pending_popularity_keys_excludes_unresolved_and_already_populated(
    tmp_path: Path,
) -> None:
    """No mbid (still pending) and an already-known lb_listeners must both be excluded."""
    store = await _new_store(tmp_path)
    try:
        await store.add_listens([_listen()], listener="household")  # pending, no mbid yet
        await store.upsert_artist_meta_full(
            [
                {
                    "artist_key": "known-popularity",
                    "artist_name": "Known Popularity",
                    "mbid": "c3ae7ee4-8b02-4c33-8ae9-3d15fcb9d4d0",
                    "lb_listeners": 61,
                    "lb_listen_count": 900,
                }
            ],
            state="ok",
        )
        assert await store.pending_popularity_keys() == []
    finally:
        await store.close()


async def test_pending_popularity_keys_respects_limit(tmp_path: Path) -> None:
    """The limit argument caps how many backlog rows come back in one pass."""
    store = await _new_store(tmp_path)
    try:
        for i in range(3):
            await store.upsert_artist_meta_full(
                [{"artist_key": f"artist-{i}", "artist_name": f"Artist {i}", "mbid": f"mbid-{i}"}],
                state="ok",
            )
        assert len(await store.pending_popularity_keys(limit=2)) == 2
    finally:
        await store.close()


async def test_mark_popularity_attempted_rotates_backlog_order(tmp_path: Path) -> None:
    """Marking a backlog artist as attempted moves it behind others in resolved_at order."""
    store = await _new_store(tmp_path)
    try:
        await store.upsert_artist_meta_full(
            [{"artist_key": "always-unknown", "artist_name": "Always Unknown", "mbid": "mbid-a"}],
            state="ok",
        )
        await store.upsert_artist_meta_full(
            [{"artist_key": "next-in-line", "artist_name": "Next In Line", "mbid": "mbid-b"}],
            state="ok",
        )
        # both rows land with the same real-clock resolved_at (same second) - pin them apart
        # explicitly so this test's ordering assertions don't depend on sqlite's tie-break.
        assert store.database is not None
        await store.database.execute(
            "UPDATE genome_artist_meta SET resolved_at = :t WHERE artist_key = :k",
            {"t": 1_000, "k": "always-unknown"},
        )
        await store.database.execute(
            "UPDATE genome_artist_meta SET resolved_at = :t WHERE artist_key = :k",
            {"t": 2_000, "k": "next-in-line"},
        )
        await store.database.commit()

        backlog = await store.pending_popularity_keys(limit=1)
        assert backlog == [("always-unknown", "mbid-a")]

        await store.mark_popularity_attempted(["always-unknown"])
        backlog = await store.pending_popularity_keys(limit=1)
        assert backlog == [("next-in-line", "mbid-b")]
    finally:
        await store.close()


async def test_mark_popularity_attempted_noop_on_empty_list(tmp_path: Path) -> None:
    """Calling with no keys must not raise or touch anything."""
    store = await _new_store(tmp_path)
    try:
        await store.mark_popularity_attempted([])
    finally:
        await store.close()


async def test_pending_artist_keys_holds_off_a_repeatedly_failing_artist(tmp_path: Path) -> None:
    """
    An artist whose lookup raised is not eligible again until its cooldown expires.

    Regression: `error` was always eligible, so three artists that failed every time were
    retried on every pass and sat in "still resolving" for days - a progress notice that
    could never finish.
    """
    store = await _new_store(tmp_path)
    try:
        await store.upsert_artist_meta_full(
            [{"artist_key": "a", "artist_name": "Broken"}], state=RESOLVE_STATE_ERROR
        )
        assert await store.pending_artist_keys() == []
    finally:
        await store.close()


async def test_failed_artist_keys_returns_newest_attempt_first(tmp_path: Path) -> None:
    """`failed_artist_keys` lists `error`-state artists, most recently attempted first."""
    store = await _new_store(tmp_path)
    try:
        await store.upsert_artist_meta_full(
            [{"artist_key": "a", "artist_name": "Artist A"}], state=RESOLVE_STATE_ERROR
        )
        await store.upsert_artist_meta_full(
            [{"artist_key": "b", "artist_name": "Artist B"}], state=RESOLVE_STATE_ERROR
        )
        assert store.database is not None
        # force distinct resolved_at values so ordering is unambiguous
        await store.database.execute(
            f"UPDATE {DB_TABLE_GENOME_ARTIST_META} SET resolved_at = 100 WHERE artist_key = 'a'"
        )
        await store.database.execute(
            f"UPDATE {DB_TABLE_GENOME_ARTIST_META} SET resolved_at = 200 WHERE artist_key = 'b'"
        )
        failed = await store.failed_artist_keys()
        assert [row["artist_key"] for row in failed] == ["b", "a"]
        assert failed[0]["artist_name"] == "Artist B"
        assert failed[0]["resolved_at"] == 200
    finally:
        await store.close()


async def test_failed_artist_keys_excludes_other_states(tmp_path: Path) -> None:
    """Only `error`-state rows are unresolved failures - pending/ok/not_found are not."""
    store = await _new_store(tmp_path)
    try:
        await store.add_listens([_listen()], listener="household")  # pending
        await store.upsert_artist_meta_full(
            [{"artist_key": "resolved-artist", "artist_name": "Resolved Artist"}], state="ok"
        )
        assert await store.failed_artist_keys() == []
    finally:
        await store.close()


async def test_failed_artist_keys_respects_limit(tmp_path: Path) -> None:
    """`limit` caps the number of rows returned."""
    store = await _new_store(tmp_path)
    try:
        for i in range(3):
            await store.upsert_artist_meta_full(
                [{"artist_key": f"artist-{i}", "artist_name": f"Artist {i}"}],
                state=RESOLVE_STATE_ERROR,
            )
        assert len(await store.failed_artist_keys(limit=2)) == 2
    finally:
        await store.close()


async def test_pending_artist_keys_retries_a_failure_once_cooled_off(tmp_path: Path) -> None:
    """A cooldown is a delay, not a grave: the artist comes back round eventually."""
    store = await _new_store(tmp_path)
    try:
        await store.upsert_artist_meta_full(
            [{"artist_key": "a", "artist_name": "Broken"}], state=RESOLVE_STATE_ERROR
        )
        stale = int(time.time()) - (RESOLVE_ERROR_COOLDOWN_HOURS + 1) * 3600
        assert store.database is not None
        await store.database.execute(
            f"UPDATE {DB_TABLE_GENOME_ARTIST_META} SET resolved_at = :t WHERE artist_key = 'a'",
            {"t": stale},
        )
        assert [key for key, _name in await store.pending_artist_keys()] == ["a"]
    finally:
        await store.close()
