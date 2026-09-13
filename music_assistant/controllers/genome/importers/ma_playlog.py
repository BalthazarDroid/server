"""
MA playlog capture and one-time backfill (§1.4, §3.8, D-03).

Music Assistant's own ``playlog`` table is a last-played table, not a history (one row per
``(item_id, provider, media_type, userid)``, rows older than 90 days are swept). The only way to
build a real history is to capture ``EventType.PLAYLOG_UPDATED`` as it happens; a one-time sweep
of the existing ``playlog`` plus ``tracks.play_count``/``last_played`` gives a coarse, explicitly
low-confidence prior for everything that happened before Genome started listening.

Contract gap (see ``docs/STATUS.md`` "Contract gaps"): ``music_assistant_models.playlog_update
.PlaylogUpdate`` (the frozen, external event payload) carries no ``queue_id``/player identifier,
so a live-captured listen's ``player_id`` is always ``None`` — only the backfill path, which reads
the raw ``playlog`` table directly, can populate it.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import EventType, MediaType
from music_assistant_models.helpers import create_safe_string

from music_assistant.constants import DB_TABLE_PLAYLOG, DB_TABLE_TRACKS
from music_assistant.controllers.genome.constants import (
    LISTENER_HOUSEHOLD,
    LOGGER,
    SOURCE_MA_BACKFILL,
    SOURCE_MA_PLAYLOG,
)
from music_assistant.controllers.genome.models import GenomeImportResult, Listen
from music_assistant.helpers.json import json_loads
from music_assistant.helpers.uri import parse_uri
from music_assistant.helpers.util import parse_title_and_version

if TYPE_CHECKING:
    from collections.abc import Callable

    from music_assistant_models.event import MassEvent

    from music_assistant.controllers.genome.controller import _GenomeStoreProtocol
    from music_assistant.mass import MusicAssistant

# never synthesize more than this many backfilled plays per track, however high play_count is
_MAX_SYNTHETIC_PLAYS = 20
_MIN_SYNTHETIC_INTERVAL_SECONDS = 86400  # never space synthetic plays closer than a day apart


class MaPlaylogImporter:
    """Captures live MA plays and backfills a coarse prior from the existing playlog/tracks."""

    def __init__(self, mass: MusicAssistant, store: _GenomeStoreProtocol) -> None:
        """
        Initialize the importer.

        :param mass: The running :class:`MusicAssistant` instance.
        :param store: The ``GenomeStore``-shaped object to write captured listens into.
        """
        self.mass = mass
        self.store = store

    def attach(self) -> Callable[[], None]:
        """Subscribe to ``EventType.PLAYLOG_UPDATED`` and return the unsubscribe callable."""
        unsubscribe = self.mass.subscribe(self._on_playlog, EventType.PLAYLOG_UPDATED)
        LOGGER.info("Live playlog capture attached (EventType.PLAYLOG_UPDATED)")
        return unsubscribe

    async def backfill(self, *, min_seconds_played: int = 30) -> GenomeImportResult:
        """
        Sweep the existing MA ``playlog`` and ``tracks`` tables into a one-time, low-confidence prior.

        For each library track with ``play_count = n`` and at least one qualifying playlog row,
        one row is written at ``last_played`` with ``confidence = 1.0``; ``min(n - 1, 20)``
        additional rows are synthesized spaced backwards from ``last_played``, ``confidence =
        0.5``, ``source = "ma_backfill"``. No synthetic rows are emitted for ``n <= 1``. Safe to
        call more than once: every row is deduplicated through ``dedupe_key``.

        :param min_seconds_played: A playlog row must be fully played or clear this many seconds
            to count (mirrors §3.2 ``min_seconds_played``).
        """
        LOGGER.info("One-time MA playlog/tracks backfill starting")
        result: GenomeImportResult = {
            "source": SOURCE_MA_BACKFILL,
            "rows_read": 0,
            "rows_imported": 0,
            "rows_skipped": 0,
            "rows_duplicate": 0,
            "first_played_at": None,
            "last_played_at": None,
            "warnings": [],
        }
        database = self.mass.music.database
        playlog_rows = await database.get_rows(
            DB_TABLE_PLAYLOG, {"media_type": MediaType.TRACK.value}, limit=0
        )
        result["rows_read"] = len(playlog_rows)
        listens: list[Listen] = []
        now = int(time.time())
        for row in playlog_rows:
            if row["provider"] != "library":
                # tracks.play_count only exists for library items; a provider-only playlog row
                # has nothing to join against and is left to live capture going forward
                result["rows_skipped"] += 1
                continue
            seconds_played = int(row["seconds_played"] or 0)
            fully_played = bool(row["fully_played"])
            if not fully_played and seconds_played < min_seconds_played:
                result["rows_skipped"] += 1
                continue
            track_row = await database.get_row(DB_TABLE_TRACKS, {"item_id": row["item_id"]})
            play_count = int(track_row["play_count"]) if track_row else 1
            last_played = (
                int(track_row["last_played"])
                if track_row and track_row["last_played"]
                else int(row["timestamp"])
            )
            artist_name = _primary_artist_name(row["artists"])
            if not artist_name:
                result["rows_skipped"] += 1
                continue
            title, _version = parse_title_and_version(row["name"], strip_for_search=True)
            track_key = create_safe_string(title)
            artist_key = create_safe_string(artist_name)
            if not track_key or not artist_key:
                result["rows_skipped"] += 1
                continue
            listens.append(
                Listen(
                    played_at=last_played,
                    artist_key=artist_key,
                    artist_name=artist_name,
                    track_key=track_key,
                    track_name=row["name"],
                    album_name=None,
                    source=SOURCE_MA_BACKFILL,
                    player_id=row["queue_id"],
                    duration_ms=None,
                    played_ms=seconds_played * 1000 or None,
                    fully_played=fully_played,
                    confidence=1.0,
                )
            )
            synthetic_count = min(max(play_count - 1, 0), _MAX_SYNTHETIC_PLAYS)
            if synthetic_count:
                span = max(now - last_played, _MIN_SYNTHETIC_INTERVAL_SECONDS * synthetic_count)
                interval = max(span // max(play_count, 1), _MIN_SYNTHETIC_INTERVAL_SECONDS)
                for i in range(1, synthetic_count + 1):
                    listens.append(
                        Listen(
                            played_at=last_played - i * interval,
                            artist_key=artist_key,
                            artist_name=artist_name,
                            track_key=track_key,
                            track_name=row["name"],
                            album_name=None,
                            source=SOURCE_MA_BACKFILL,
                            player_id=row["queue_id"],
                            duration_ms=None,
                            played_ms=None,
                            fully_played=None,
                            confidence=0.5,
                        )
                    )
        if listens:
            store_result = await self.store.add_listens(listens, listener=LISTENER_HOUSEHOLD)
            result["rows_imported"] = store_result["rows_imported"]
            result["rows_duplicate"] = store_result["rows_duplicate"]
            result["first_played_at"] = store_result["first_played_at"]
            result["last_played_at"] = store_result["last_played_at"]
        LOGGER.info(
            "One-time MA playlog/tracks backfill finished: %d rows read, %d imported, "
            "%d skipped, %d duplicate",
            result["rows_read"],
            result["rows_imported"],
            result["rows_skipped"],
            result["rows_duplicate"],
        )
        return result

    async def _on_playlog(self, event: MassEvent) -> None:
        """Handle one ``PLAYLOG_UPDATED`` event by appending a live-captured listen."""
        update = event.data
        if update is None or update.media_type != MediaType.TRACK:
            return
        try:
            media_type, provider, item_id = await parse_uri(update.uri)
        except Exception as err:
            LOGGER.debug("Could not parse playlog uri %s: %s", update.uri, err)
            return
        if media_type != MediaType.TRACK:
            return
        try:
            track = await self.mass.music.tracks.get(item_id, provider, recursive=False)
        except Exception as err:
            LOGGER.debug("Could not resolve playlog track %s/%s: %s", provider, item_id, err)
            return
        artist_name = track.artists[0].name if track.artists else ""
        if not artist_name:
            return
        title, _version = parse_title_and_version(track.name, strip_for_search=True)
        track_key = create_safe_string(title)
        artist_key = create_safe_string(artist_name)
        if not track_key or not artist_key:
            return
        album_name = getattr(getattr(track, "album", None), "name", None)
        listen = Listen(
            played_at=int(time.time()),
            artist_key=artist_key,
            artist_name=artist_name,
            track_key=track_key,
            track_name=track.name,
            album_name=album_name,
            source=SOURCE_MA_PLAYLOG,
            # PlaylogUpdate carries no player/queue id (see the module docstring's contract gap)
            player_id=None,
            duration_ms=int(track.duration * 1000) if getattr(track, "duration", None) else None,
            played_ms=update.seconds_played * 1000,
            fully_played=update.fully_played,
            confidence=1.0,
        )
        await self.store.add_listens([listen], listener=LISTENER_HOUSEHOLD, ma_userid=update.userid)
        LOGGER.debug("Captured live listen: %s - %s", artist_name, track.name)


def _primary_artist_name(artists_json: Any) -> str:
    """Return the first artist's name from a playlog row's ``artists`` json column."""
    if not artists_json:
        return ""
    entries = artists_json
    if isinstance(entries, str):
        try:
            entries = json_loads(entries)
        except Exception:  # pragma: no cover - defensive, malformed json
            return ""
    if not isinstance(entries, list):
        return ""
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name"):
            return str(entry["name"])
    return ""


__all__ = ["MaPlaylogImporter"]
