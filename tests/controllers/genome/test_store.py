"""Tests for :class:`GenomeStore` (§3.1)."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

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
            [_listen(artist_key="acdc", artist_name="AC/DC", track_key="backinblack",
                     track_name="Back In Black", played_at=1_700_100_000, source="lastfm")],
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
        payload = {"schema_version": 1, "stats": {"total_listens": 0}}
        await store.set_cached_genome("household", payload)  # type: ignore[arg-type]
        assert await store.get_cached_genome("household") == payload
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
        collected = [listen async for listen in store.iter_listens("household", since=1_750_000_000)]
        assert len(collected) == 1
        assert collected[0].played_at == 1_800_000_000
    finally:
        await store.close()
