"""
``genome.db`` access layer for the Listening Genome controller (§1.3, §3.1).

:class:`GenomeStore` owns the schema, the schema-version migration handshake (copied from
``controllers/cache/controller.py::_setup_database`` per §1.3), and every typed read/write the
rest of the Genome controller needs. Its public surface is frozen in
``docs/ARCHITECTURE.md`` Part 4 — WP-B's controller wiring calls it directly.

Contract gap (see ``docs/STATUS.md`` "Contract gaps"): the frozen :class:`ArtistMeta` dataclass
(``models.py``, the pure engine input) does not carry ``mb_tags``/``country``, but the
``genome_artist_meta`` schema in §3.1 has columns for both (``begin_year`` was added to
:class:`ArtistMeta` as ``era_facts``'s fallback proxy for a missing ``first_release_year``). The
enrichment path (:mod:`music_assistant.controllers.genome.enrich.musicbrainz`) needs somewhere to
put ``mb_tags``/``country``, so this module adds :meth:`GenomeStore.upsert_artist_meta_full`, a
superset write path, alongside the frozen :meth:`GenomeStore.upsert_artist_meta`. Both funnel
through the same private helper.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, TypedDict, cast

from music_assistant.controllers.genome.constants import (
    DB_SCHEMA_VERSION,
    DB_TABLE_GENOME_ARTIST_META,
    DB_TABLE_GENOME_CACHE,
    DB_TABLE_GENOME_LISTENS,
    DB_TABLE_SETTINGS,
    ENGINE_VERSION,
    GENOME_DISCOVERY_CACHE_VERSION,
    GENOME_RESULT_SCHEMA_VERSION,
    LOGGER,
    RESOLVE_ERROR_COOLDOWN_HOURS,
    RESOLVE_NOT_FOUND_COOLDOWN_DAYS,
    RESOLVE_OK_COOLDOWN_DAYS,
    RESOLVE_STATE_ERROR,
    RESOLVE_STATE_NOT_FOUND,
    RESOLVE_STATE_OK,
    RESOLVE_STATE_PENDING,
    SOURCE_MA_BACKFILL,
    SOURCE_MA_PLAYLOG,
)
from music_assistant.controllers.genome.models import (
    ArtistMeta,
    FailedArtist,
    GenomeImportResult,
    GenomeResult,
    Listen,
)
from music_assistant.helpers.database import UNSET, DatabaseConnection
from music_assistant.helpers.json import json_dumps, json_loads

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Mapping, Sequence

    from music_assistant.mass import MusicAssistant

# dedupe_key uses this instead of the real source name for both MA-derived sources (§3.1), so a
# live playlog capture and its own backfill collapse onto the same key.
_MA_SOURCE_CLASS = "ma"
_MA_SOURCES = (SOURCE_MA_PLAYLOG, SOURCE_MA_BACKFILL)

# a cross-source duplicate (§3.1 "dedupe_window") is dropped when a non-MA row lands within this
# many seconds of an MA row for the same (artist_key, track_key)
_CROSS_SOURCE_DEDUPE_WINDOW_SECONDS = 90

_GENOME_CACHE_TTL_SECONDS = 0  # 0 = no expiry; cache is invalidated explicitly on rebuild

# settings-table key recording whether the one-time MA playlog backfill has already run
_MA_BACKFILL_DONE_KEY = "ma_backfill_done"

# settings-table key recording whether the one-time Last.fm full-history sweep has completed
# (P1, 2026-09-13 real-hardware run). Reuses the existing `settings` table exactly like
# `_MA_BACKFILL_DONE_KEY` above - no schema change, no migration, no risk to imported rows.
_LASTFM_BACKFILL_DONE_KEY = "lastfm_backfill_done"

# settings-table key recording the fingerprint (sorted artist_key list, JSON-encoded) of the
# `error`-state artists that were current the last time the user dismissed the "could not be
# identified" notice. Same table, same pattern as `_LASTFM_BACKFILL_DONE_KEY` - a remembered
# fingerprint rather than a permanent mute, so a newly-failing artist brings the notice back
# (see GenomeStore.unresolved_dismissed_keys / all_failed_artist_keys).
_UNRESOLVED_DISMISSED_KEY = "unresolved_dismissed_keys"


class ArtistMetaWrite(TypedDict, total=False):
    """
    Full ``genome_artist_meta`` column set for a write.

    A superset of :class:`~music_assistant.controllers.genome.models.ArtistMeta` (the frozen
    engine input) that also carries the enrichment-only columns the schema has but the engine
    does not need: ``mb_tags``, ``country``. See the module docstring.
    """

    artist_key: str
    artist_name: str
    mbid: str | None
    mb_tags: list[dict[str, Any]]
    genres: list[str]
    begin_year: int | None
    first_release_year: int | None
    country: str | None
    lb_listeners: int | None
    lb_listen_count: int | None


class GenomeStore:
    """Owns ``genome.db``: schema, migrations, and typed reads/writes for the Genome controller."""

    @property
    def db_path(self) -> str:
        """Filesystem path of ``genome.db``."""
        return os.path.join(self.mass.storage_path, "genome.db")

    def __init__(self, mass: MusicAssistant) -> None:
        """
        Initialize the store.

        :param mass: The running :class:`MusicAssistant` instance.
        """
        self.mass = mass
        self.database: DatabaseConnection | None = None

    async def setup(self) -> None:
        """Open (creating if needed) ``genome.db`` and run the schema-version handshake."""
        db_path = self.db_path
        self.database = DatabaseConnection(db_path)
        await self.database.setup()

        await self.__create_database_tables()
        try:
            if row := await self.database.get_row(DB_TABLE_SETTINGS, {"key": "version"}):
                prev_version = int(row["value"])
            else:
                prev_version = 0
        except KeyError, ValueError:
            prev_version = 0

        if prev_version not in (0, DB_SCHEMA_VERSION):
            LOGGER.warning(
                "Performing genome database migration from %s to %s",
                prev_version,
                DB_SCHEMA_VERSION,
            )
            try:
                await self.__migrate_database(prev_version)
            except Exception as err:
                LOGGER.warning("Genome database migration failed: %s, resetting genome data", err)
                for table in (
                    DB_TABLE_GENOME_LISTENS,
                    DB_TABLE_GENOME_ARTIST_META,
                    DB_TABLE_GENOME_CACHE,
                ):
                    await self.database.execute(f"DROP TABLE IF EXISTS {table}")
                await self.database.commit()
                await self.__create_database_tables()

        await self.database.insert_or_replace(
            DB_TABLE_SETTINGS, {"key": "version", "value": str(DB_SCHEMA_VERSION), "type": "str"}
        )
        await self.__create_database_indexes()

    async def close(self) -> None:
        """Close the database connection."""
        if self.database is not None:
            await self.database.close()

    async def add_listens(
        self,
        listens: Sequence[Listen],
        *,
        listener: str,
        ma_userid: str | None = None,
    ) -> GenomeImportResult:
        """
        Insert a batch of listens, silently skipping duplicates by ``dedupe_key``.

        Also ensures a ``pending`` stub row exists in ``genome_artist_meta`` for every artist
        seen for the first time, so :meth:`pending_artist_keys` picks it up.

        :param listens: The normalized listens to store.
        :param listener: The listener partition these listens belong to (``"household"`` in v1).
        :param ma_userid: The MA ``user_id`` these listens are attributed to, when known.
        """
        assert self.database is not None
        source = listens[0].source if listens else ""
        result: GenomeImportResult = {
            "source": source,
            "rows_read": len(listens),
            "rows_imported": 0,
            "rows_skipped": 0,
            "rows_duplicate": 0,
            "first_played_at": None,
            "last_played_at": None,
            "warnings": [],
        }
        seen_artists: dict[str, str] = {}
        for listen in listens:
            dedupe_key = self._dedupe_key(listener, listen)
            values = {
                "listener": listener,
                "ma_userid": ma_userid,
                "played_at": listen.played_at,
                "artist_key": listen.artist_key,
                "artist_name": listen.artist_name,
                "track_key": listen.track_key,
                "track_name": listen.track_name,
                "album_name": listen.album_name,
                "source": listen.source,
                "player_id": listen.player_id,
                "duration_ms": listen.duration_ms,
                "played_ms": listen.played_ms,
                "fully_played": listen.fully_played,
                "confidence": listen.confidence,
                "dedupe_key": dedupe_key,
            }
            columns = list(values)
            cursor = await self.database.execute(
                f"INSERT OR IGNORE INTO {DB_TABLE_GENOME_LISTENS} "
                f"({', '.join(columns)}) VALUES ({', '.join(f':{c}' for c in columns)})",
                values,
            )
            if cursor.rowcount:
                result["rows_imported"] += 1
                seen_artists[listen.artist_key] = listen.artist_name
                if (
                    result["first_played_at"] is None
                    or listen.played_at < result["first_played_at"]
                ):
                    result["first_played_at"] = listen.played_at
                if result["last_played_at"] is None or listen.played_at > result["last_played_at"]:
                    result["last_played_at"] = listen.played_at
            else:
                result["rows_duplicate"] += 1
        await self.database.commit()
        for artist_key, artist_name in seen_artists.items():
            await self._ensure_artist_meta_stub(artist_key, artist_name)
        return result

    async def iter_listens(self, listener: str, *, since: int = 0) -> AsyncIterator[Listen]:
        """
        Stream every listen for ``listener`` newer than ``since``, oldest first.

        :param listener: The listener partition to read.
        :param since: Only return listens with ``played_at`` strictly greater than this.
        """
        assert self.database is not None
        query = (
            f"SELECT * FROM {DB_TABLE_GENOME_LISTENS} "
            "WHERE listener = :listener AND played_at > :since ORDER BY played_at ASC"
        )
        async for row in self.database.iter_rows_from_query(
            query, {"listener": listener, "since": since}
        ):
            yield self._row_to_listen(row)

    async def count_listens(self, listener: str) -> int:
        """Return the number of stored listens for ``listener``."""
        assert self.database is not None
        return await self.database.get_count_from_query(
            f"SELECT 1 FROM {DB_TABLE_GENOME_LISTENS} WHERE listener = :listener",
            {"listener": listener},
        )

    async def get_artist_meta(self, artist_keys: Sequence[str]) -> dict[str, ArtistMeta]:
        """Return known :class:`ArtistMeta` rows, keyed by ``artist_key`` (missing keys omitted)."""
        assert self.database is not None
        if not artist_keys:
            return {}
        rows = await self.database.get_rows_from_query(
            f"SELECT * FROM {DB_TABLE_GENOME_ARTIST_META} WHERE artist_key IN (:keys)",
            {"keys": list(artist_keys)},
            limit=0,
        )
        return {row["artist_key"]: self._row_to_artist_meta(row) for row in rows}

    async def upsert_artist_meta(self, rows: Sequence[ArtistMeta], *, state: str) -> None:
        """
        Upsert engine-relevant artist metadata (the frozen ``GenomeStore`` surface).

        Writes only the columns :class:`ArtistMeta` carries; ``mb_tags``/``country`` are left
        unset. Use :meth:`upsert_artist_meta_full` from enrichment code that has those values
        available.

        :param rows: The artist metadata to write.
        :param state: The ``resolve_state`` to stamp on every row.
        """
        await self._upsert_artist_meta_rows(
            (
                ArtistMetaWrite(
                    artist_key=row.artist_key,
                    artist_name=row.artist_name,
                    mbid=row.mbid,
                    genres=list(row.genres),
                    first_release_year=row.first_release_year,
                    begin_year=row.begin_year,
                    lb_listeners=row.lb_listeners,
                    lb_listen_count=row.lb_listen_count,
                )
                for row in rows
            ),
            state=state,
        )

    async def upsert_artist_meta_full(self, rows: Sequence[ArtistMetaWrite], *, state: str) -> None:
        """
        Upsert the full artist metadata column set, including enrichment-only columns.

        :param rows: The artist metadata rows to write.
        :param state: The ``resolve_state`` to stamp on every row.
        """
        await self._upsert_artist_meta_rows(rows, state=state)

    async def pending_artist_keys(self, limit: int = 200) -> list[tuple[str, str]]:
        """
        Return up to ``limit`` ``(artist_key, artist_name)`` pairs due for (re-)resolution.

        Eligible rows: ``pending`` (always), ``error`` older than
        :data:`RESOLVE_ERROR_COOLDOWN_HOURS`, ``ok`` older than
        :data:`RESOLVE_OK_COOLDOWN_DAYS`, or ``not_found`` older than
        :data:`RESOLVE_NOT_FOUND_COOLDOWN_DAYS` (§3.8).

        ``error`` carries a cooldown rather than being immediately eligible: without one, an
        artist whose lookup fails every time is retried on every pass forever and never leaves
        the "still resolving" count.
        """
        assert self.database is not None
        now = int(time.time())
        ok_cutoff = now - RESOLVE_OK_COOLDOWN_DAYS * 86400
        error_cutoff = now - RESOLVE_ERROR_COOLDOWN_HOURS * 3600
        not_found_cutoff = now - RESOLVE_NOT_FOUND_COOLDOWN_DAYS * 86400
        rows = await self.database.get_rows_from_query(
            f"SELECT artist_key, artist_name FROM {DB_TABLE_GENOME_ARTIST_META} "
            "WHERE resolve_state = :pending "
            "OR (resolve_state = :error AND resolved_at < :error_cutoff) "
            "OR (resolve_state = :ok AND resolved_at < :ok_cutoff) "
            "OR (resolve_state = :not_found AND resolved_at < :not_found_cutoff) "
            "ORDER BY resolved_at ASC",
            {
                "pending": RESOLVE_STATE_PENDING,
                "error": RESOLVE_STATE_ERROR,
                "error_cutoff": error_cutoff,
                "ok": RESOLVE_STATE_OK,
                "ok_cutoff": ok_cutoff,
                "not_found": RESOLVE_STATE_NOT_FOUND,
                "not_found_cutoff": not_found_cutoff,
            },
            limit=limit,
        )
        return [(row["artist_key"], row["artist_name"]) for row in rows]

    async def pending_popularity_keys(self, limit: int = 200) -> list[tuple[str, str]]:
        """
        Return up to ``limit`` ``(artist_key, mbid)`` pairs still missing ListenBrainz popularity.

        Modeled on :meth:`pending_artist_keys`: an artist lands here the moment it has an
        ``mbid`` and stays here until ``lb_listeners`` is set, regardless of whether it was
        resolved in this pass, an earlier pass, or before this backfill query existed at all.
        There is no schema column recording "ListenBrainz has no data for this artist" (a
        migration was avoided - see :meth:`mark_popularity_attempted`), so this is ordered
        oldest-``resolved_at``-first and the caller is expected to call
        :meth:`mark_popularity_attempted` for every key it looked up and still could not
        find popularity for; that bump is what rotates a permanently-unknown artist to the back
        of this same queue instead of it being retried every pass forever.

        :param limit: The maximum number of rows to return.
        """
        assert self.database is not None
        rows = await self.database.get_rows_from_query(
            f"SELECT artist_key, mbid FROM {DB_TABLE_GENOME_ARTIST_META} "
            "WHERE mbid IS NOT NULL AND lb_listeners IS NULL "
            "ORDER BY resolved_at ASC",
            limit=limit,
        )
        return [(row["artist_key"], row["mbid"]) for row in rows]

    async def failed_artist_keys(self, limit: int = 100) -> list[FailedArtist]:
        """
        Return up to ``limit`` artists currently sitting in the ``error`` resolve state.

        Newest attempt first, so a user checking "who failed" sees the most recent failures
        without having to scroll past ones that may already have been retried and moved on.
        Unlike :meth:`pending_artist_keys`, there is no cooldown filter here - this is a status
        listing, not a retry queue, so it reports every artist currently in ``error`` regardless
        of when it is next eligible for another lookup.

        :param limit: The maximum number of rows to return.
        """
        assert self.database is not None
        rows = await self.database.get_rows_from_query(
            f"SELECT artist_key, artist_name, resolved_at FROM {DB_TABLE_GENOME_ARTIST_META} "
            "WHERE resolve_state = :error "
            "ORDER BY resolved_at DESC",
            {"error": RESOLVE_STATE_ERROR},
            limit=limit,
        )
        return [
            {
                "artist_key": row["artist_key"],
                "artist_name": row["artist_name"],
                "resolved_at": row["resolved_at"],
            }
            for row in rows
        ]

    async def retry_failed_artists(self, artist_keys: Sequence[str] | None = None) -> int:
        """
        Move ``error`` artists back to ``pending`` immediately, ignoring the error cooldown.

        Reuses :meth:`upsert_artist_meta_full` with the same minimal row shape
        (``artist_key``/``artist_name`` only) that :mod:`enrich.musicbrainz` already writes for
        an ``error`` row - safe here for the same reason it is safe there: an ``error`` row
        never carries ``mbid``/``mb_tags``/``genres``/popularity data, so rewriting it with only
        those two columns clobbers nothing. This does no network work itself; the existing
        background/scheduled enrichment pass picks the row up on its next run because
        ``pending`` is always eligible, with no cooldown (see :meth:`pending_artist_keys`).

        :param artist_keys: The artist keys to retry; ``None`` retries every failed artist.
        :return: The number of rows moved back to ``pending``.
        """
        assert self.database is not None
        if artist_keys is not None and not artist_keys:
            return 0
        query = (
            f"SELECT artist_key, artist_name FROM {DB_TABLE_GENOME_ARTIST_META} "
            "WHERE resolve_state = :error"
        )
        params: dict[str, Any] = {"error": RESOLVE_STATE_ERROR}
        if artist_keys is not None:
            query += " AND artist_key IN (:keys)"
            params["keys"] = list(artist_keys)
        rows = await self.database.get_rows_from_query(query, params, limit=0)
        targets = [(row["artist_key"], row["artist_name"]) for row in rows]
        if not targets:
            return 0
        await self.upsert_artist_meta_full(
            [ArtistMetaWrite(artist_key=key, artist_name=name) for key, name in targets],
            state=RESOLVE_STATE_PENDING,
        )
        return len(targets)

    async def all_failed_artist_keys(self) -> frozenset[str]:
        """
        Return every ``artist_key`` currently in the ``error`` resolve state, unlimited.

        Unlike :meth:`failed_artist_keys` (a capped display listing), this exists only to be
        compared against a dismissed fingerprint (:meth:`unresolved_dismissed_keys`), so it
        must see the whole set regardless of how many artists have failed.
        """
        assert self.database is not None
        rows = await self.database.get_rows_from_query(
            f"SELECT artist_key FROM {DB_TABLE_GENOME_ARTIST_META} WHERE resolve_state = :error",
            {"error": RESOLVE_STATE_ERROR},
            limit=0,
        )
        return frozenset(row["artist_key"] for row in rows)

    async def dismiss_unresolved(self, artist_keys: Sequence[str]) -> None:
        """
        Record ``artist_keys`` as the dismissed "could not be identified" fingerprint.

        Not a permanent mute: :meth:`unresolved_dismissed_keys` lets the caller compare this
        fingerprint against the artists currently in ``error`` state, so a newly-failing (or
        newly-recovered) artist changes the current set and the notice returns.

        :param artist_keys: The artist keys currently in the ``error`` resolve state.
        """
        assert self.database is not None
        await self.database.insert_or_replace(
            DB_TABLE_SETTINGS,
            {
                "key": _UNRESOLVED_DISMISSED_KEY,
                "value": json_dumps(sorted(set(artist_keys))),
                "type": "json",
            },
        )
        await self.database.commit()

    async def unresolved_dismissed_keys(self) -> frozenset[str] | None:
        """Return the dismissed failed-artist fingerprint, or ``None`` if never dismissed."""
        assert self.database is not None
        row = await self.database.get_row(DB_TABLE_SETTINGS, {"key": _UNRESOLVED_DISMISSED_KEY})
        if row is None:
            return None
        try:
            keys = json_loads(row["value"])
        except Exception:  # pragma: no cover - defensive, malformed settings row
            return None
        if not isinstance(keys, list):  # pragma: no cover - defensive, malformed settings row
            return None
        return frozenset(keys)

    async def mark_popularity_attempted(self, artist_keys: Sequence[str]) -> None:
        """
        Bump ``resolved_at`` for artists a popularity backfill pass looked up but found nothing for.

        See :meth:`pending_popularity_keys` for why this is the cooldown mechanism for that
        backlog instead of a dedicated "not found" state.

        :param artist_keys: The artist keys to mark as attempted just now.
        """
        assert self.database is not None
        if not artist_keys:
            return
        now = int(time.time())
        for artist_key in artist_keys:
            await self.database.execute(
                f"UPDATE {DB_TABLE_GENOME_ARTIST_META} SET resolved_at = :now "
                "WHERE artist_key = :artist_key",
                {"now": now, "artist_key": artist_key},
            )
        await self.database.commit()

    async def get_cached_genome(self, listener: str) -> GenomeResult | None:
        """Return the cached :class:`GenomeResult` for ``listener``, or ``None`` if absent."""
        assert self.database is not None
        row = await self.database.get_row(DB_TABLE_GENOME_CACHE, {"key": self._cache_key(listener)})
        if row is None:
            return None
        try:
            data = json_loads(row["value"])
        except Exception as err:  # pragma: no cover - defensive, malformed cache row
            LOGGER.warning("Discarding malformed genome cache entry: %s", err)
            return None
        # A cached blob written by an older result shape is not merely stale, it is the
        # wrong shape: a field added since (`bases`, `base_mix`) is simply absent, and the
        # frontend reads it unconditionally. The cache key carries ENGINE_VERSION, but the
        # result shape can change without the engine's maths changing, so check the blob's
        # own recorded schema version rather than relying on someone remembering to bump
        # the key.
        if not isinstance(data, dict) or data.get("schema_version") != GENOME_RESULT_SCHEMA_VERSION:
            LOGGER.info(
                "Discarding genome cache entry from result schema %s (current: %s)",
                data.get("schema_version") if isinstance(data, dict) else "unknown",
                GENOME_RESULT_SCHEMA_VERSION,
            )
            return None
        return cast("GenomeResult", data)

    async def set_cached_genome(self, listener: str, genome: GenomeResult) -> None:
        """Store ``genome`` as the cached result for ``listener``."""
        assert self.database is not None
        now = int(time.time())
        await self.database.upsert(
            DB_TABLE_GENOME_CACHE,
            {
                "key": self._cache_key(listener),
                "value": json_dumps(genome),
                "created_at": now,
                "expires_at": now + _GENOME_CACHE_TTL_SECONDS if _GENOME_CACHE_TTL_SECONDS else 0,
            },
        )

    async def get_cached_discovery(self, listener: str) -> dict[str, Any] | None:
        """
        Return the stored discovery blob for ``listener``, or ``None`` if absent or stale.

        Stored in its own ``genome_cache`` row rather than inside the genome result, so a
        change to the discovery payload never discards the (far more expensive) genome blob,
        and neither version number has to move for the other's sake.

        :param listener: The listener id (``"household"`` in v1).
        """
        assert self.database is not None
        row = await self.database.get_row(
            DB_TABLE_GENOME_CACHE, {"key": self._discovery_cache_key(listener)}
        )
        if row is None:
            return None
        try:
            data = json_loads(row["value"])
        except Exception as err:  # pragma: no cover - defensive, malformed cache row
            LOGGER.warning("Discarding malformed genome discovery cache entry: %s", err)
            return None
        if not isinstance(data, dict) or data.get("version") != GENOME_DISCOVERY_CACHE_VERSION:
            LOGGER.info(
                "Discarding genome discovery cache entry from version %s (current: %s)",
                data.get("version") if isinstance(data, dict) else "unknown",
                GENOME_DISCOVERY_CACHE_VERSION,
            )
            return None
        return cast("dict[str, Any]", data)

    async def set_cached_discovery(self, listener: str, data: Mapping[str, Any]) -> None:
        """
        Store ``data`` as the discovery blob for ``listener``, stamped with the current version.

        :param listener: The listener id (``"household"`` in v1).
        :param data: The blob to store; the version stamp is applied here, not by the caller.
        """
        assert self.database is not None
        now = int(time.time())
        await self.database.upsert(
            DB_TABLE_GENOME_CACHE,
            {
                "key": self._discovery_cache_key(listener),
                "value": json_dumps({**data, "version": GENOME_DISCOVERY_CACHE_VERSION}),
                "created_at": now,
                "expires_at": 0,
            },
        )
        await self.database.commit()

    async def artist_play_counts(self, listener: str) -> dict[str, int]:
        """
        Return ``{artist_key: stored_listen_count}`` for ``listener``.

        The count Genome itself holds, across every source it has imported - which is not the
        same figure as the MA library's own ``play_count``, and is usually much larger on an
        install seeded from an Apple or Last.fm export.

        :param listener: The listener id (``"household"`` in v1).
        """
        assert self.database is not None
        rows = await self.database.get_rows_from_query(
            f"SELECT artist_key, COUNT(*) AS n FROM {DB_TABLE_GENOME_LISTENS} "
            "WHERE listener = :listener GROUP BY artist_key",
            {"listener": listener},
            limit=0,
        )
        return {row["artist_key"]: int(row["n"]) for row in rows}

    async def clear(self, listener: str | None = None) -> None:
        """
        Delete stored listens and cached genomes.

        :param listener: When given, only that listener's listens and cache entry are removed
            (artist metadata is shared across listeners and is left in place). When ``None``,
            every table is fully cleared.
        """
        assert self.database is not None
        if listener is None:
            for table in (
                DB_TABLE_GENOME_LISTENS,
                DB_TABLE_GENOME_ARTIST_META,
                DB_TABLE_GENOME_CACHE,
            ):
                await self.database.execute(f"DELETE FROM {table}")
        else:
            await self.database.delete(DB_TABLE_GENOME_LISTENS, {"listener": listener})
            await self.database.delete(DB_TABLE_GENOME_CACHE, {"key": self._cache_key(listener)})
            await self.database.delete(
                DB_TABLE_GENOME_CACHE, {"key": self._discovery_cache_key(listener)}
            )
        await self.database.commit()

    async def source_counts(self, listener: str) -> dict[str, int]:
        """Return ``{source: listen_count}`` for ``listener``."""
        assert self.database is not None
        rows = await self.database.get_rows_from_query(
            f"SELECT source, COUNT(*) AS n FROM {DB_TABLE_GENOME_LISTENS} "
            "WHERE listener = :listener GROUP BY source",
            {"listener": listener},
            limit=0,
        )
        return {row["source"]: int(row["n"]) for row in rows}

    async def player_names(self) -> dict[str, str]:
        """Return ``{player_id: display_name}`` for every player_id seen in stored listens."""
        assert self.database is not None
        rows = await self.database.get_rows_from_query(
            f"SELECT DISTINCT player_id FROM {DB_TABLE_GENOME_LISTENS} WHERE player_id IS NOT NULL",
            limit=0,
        )
        names: dict[str, str] = {}
        players = getattr(self.mass, "players", None)
        for row in rows:
            player_id = row["player_id"]
            display_name = player_id
            if players is not None:
                try:
                    if player := players.get_player(player_id):
                        display_name = player.display_name
                except Exception:  # pragma: no cover - defensive, player lookups must not fail this
                    LOGGER.debug("Could not resolve display name for player %s", player_id)
            names[player_id] = display_name
        return names

    async def update_lb_popularity(self, rows: Mapping[str, tuple[int, int]]) -> None:
        """
        Merge ListenBrainz popularity into existing artist rows without touching other columns.

        Unlike :meth:`upsert_artist_meta_full`, this only ever ``UPDATE``s ``lb_listeners``/
        ``lb_listen_count`` on a row that already exists (from MusicBrainz resolution or a prior
        listen) - it never inserts a new row, so it can never clobber ``genres``/``mb_tags``.

        :param rows: ``artist_key -> (lb_listeners, lb_listen_count)``.
        """
        assert self.database is not None
        for artist_key, (listeners, listen_count) in rows.items():
            await self.database.execute(
                f"UPDATE {DB_TABLE_GENOME_ARTIST_META} "
                "SET lb_listeners = :listeners, lb_listen_count = :listen_count "
                "WHERE artist_key = :artist_key",
                {
                    "listeners": listeners,
                    "listen_count": listen_count,
                    "artist_key": artist_key,
                },
            )
        if rows:
            await self.database.commit()

    async def backfill_done(self) -> bool:
        """Return whether the one-time MA playlog backfill has already run."""
        assert self.database is not None
        row = await self.database.get_row(DB_TABLE_SETTINGS, {"key": _MA_BACKFILL_DONE_KEY})
        return row is not None and row["value"] == "1"

    async def mark_backfill_done(self) -> None:
        """Record that the one-time MA playlog backfill has run, so it is not repeated."""
        assert self.database is not None
        await self.database.insert_or_replace(
            DB_TABLE_SETTINGS, {"key": _MA_BACKFILL_DONE_KEY, "value": "1", "type": "str"}
        )
        await self.database.commit()

    async def lastfm_backfill_done(self) -> bool:
        """Return whether the one-time Last.fm full-history sweep has already completed (P1)."""
        assert self.database is not None
        row = await self.database.get_row(DB_TABLE_SETTINGS, {"key": _LASTFM_BACKFILL_DONE_KEY})
        return row is not None and row["value"] == "1"

    async def mark_lastfm_backfill_done(self) -> None:
        """Record that the one-time Last.fm full-history sweep has completed (P1)."""
        assert self.database is not None
        await self.database.insert_or_replace(
            DB_TABLE_SETTINGS, {"key": _LASTFM_BACKFILL_DONE_KEY, "value": "1", "type": "str"}
        )
        await self.database.commit()

    async def artist_resolution_counts(self) -> dict[str, int]:
        """
        Return ``{resolve_state: count}`` across every known artist (P3).

        Lets the controller surface "N artists still resolving" in ``genome/get`` stats without
        the pure, I/O-free ``engine.py`` needing to know about ``resolve_state`` at all.
        """
        assert self.database is not None
        rows = await self.database.get_rows_from_query(
            f"SELECT resolve_state, COUNT(*) AS n FROM {DB_TABLE_GENOME_ARTIST_META} "
            "GROUP BY resolve_state",
            limit=0,
        )
        return {row["resolve_state"]: int(row["n"]) for row in rows}

    async def dedupe_window(self) -> int:
        """
        Drop non-MA listens that duplicate an MA-sourced listen within ±90s (§3.1).

        Second-pass cross-source dedup: ``dedupe_key`` alone only collapses same-source repeats
        within the same minute; this removes an external-source row (Last.fm, Apple) when an
        ``ma_playlog``/``ma_backfill`` row already covers the same ``(artist_key, track_key)``
        within :data:`_CROSS_SOURCE_DEDUPE_WINDOW_SECONDS`.

        :return: The number of rows deleted.
        """
        assert self.database is not None
        ma_placeholders = ", ".join(f"'{s}'" for s in _MA_SOURCES)
        candidates = await self.database.get_rows_from_query(
            f"SELECT id, listener, artist_key, track_key, played_at FROM {DB_TABLE_GENOME_LISTENS} "
            f"WHERE source NOT IN ({ma_placeholders})",
            limit=0,
        )
        if not candidates:
            return 0
        deleted = 0
        for row in candidates:
            match = await self.database.get_rows_from_query(
                f"SELECT id FROM {DB_TABLE_GENOME_LISTENS} "
                f"WHERE source IN ({ma_placeholders}) AND listener = :listener "
                "AND artist_key = :artist_key AND track_key = :track_key "
                "AND ABS(played_at - :played_at) <= :window",
                {
                    "listener": row["listener"],
                    "artist_key": row["artist_key"],
                    "track_key": row["track_key"],
                    "played_at": row["played_at"],
                    "window": _CROSS_SOURCE_DEDUPE_WINDOW_SECONDS,
                },
                limit=1,
            )
            if match:
                await self.database.delete(DB_TABLE_GENOME_LISTENS, {"id": row["id"]})
                deleted += 1
        if deleted:
            await self.database.commit()
        return deleted

    def _dedupe_key(self, listener: str, listen: Listen) -> str:
        """Build the ``dedupe_key`` for a listen per §3.1."""
        source_class = _MA_SOURCE_CLASS if listen.source in _MA_SOURCES else listen.source
        minute = listen.played_at // 60
        return f"{listener}|{source_class}|{listen.artist_key}|{listen.track_key}|{minute}"

    def _cache_key(self, listener: str) -> str:
        """Build the ``genome_cache`` key for a listener at the current engine version."""
        return f"genome:{listener}:{ENGINE_VERSION}"

    def _discovery_cache_key(self, listener: str) -> str:
        """Build the ``genome_cache`` key holding a listener's discovery blob."""
        return f"discovery:{listener}"

    def _row_to_listen(self, row: Mapping[str, Any]) -> Listen:
        """Convert a ``genome_listens`` row into a :class:`Listen`."""
        return Listen(
            played_at=row["played_at"],
            artist_key=row["artist_key"],
            artist_name=row["artist_name"],
            track_key=row["track_key"],
            track_name=row["track_name"],
            album_name=row["album_name"],
            source=row["source"],
            player_id=row["player_id"],
            duration_ms=row["duration_ms"],
            played_ms=row["played_ms"],
            fully_played=None if row["fully_played"] is None else bool(row["fully_played"]),
            confidence=row["confidence"],
        )

    def _row_to_artist_meta(self, row: Mapping[str, Any]) -> ArtistMeta:
        """Convert a ``genome_artist_meta`` row into an :class:`ArtistMeta`."""
        try:
            genres = tuple(json_loads(row["genres"]) or [])
        except Exception:  # pragma: no cover - defensive, malformed json
            genres = ()
        return ArtistMeta(
            artist_key=row["artist_key"],
            artist_name=row["artist_name"],
            mbid=row["mbid"],
            genres=genres,
            first_release_year=row["first_release_year"],
            begin_year=row["begin_year"],
            lb_listeners=row["lb_listeners"],
            lb_listen_count=row["lb_listen_count"],
        )

    async def _ensure_artist_meta_stub(self, artist_key: str, artist_name: str) -> None:
        """Insert a ``pending`` placeholder row for an artist if one does not already exist."""
        assert self.database is not None
        await self.database.execute(
            f"INSERT OR IGNORE INTO {DB_TABLE_GENOME_ARTIST_META} "
            "(artist_key, artist_name, mb_tags, genres, resolved_at, resolve_state) "
            "VALUES (:artist_key, :artist_name, '[]', '[]', 0, :state)",
            {"artist_key": artist_key, "artist_name": artist_name, "state": RESOLVE_STATE_PENDING},
        )
        await self.database.commit()

    async def _upsert_artist_meta_rows(
        self, rows: Iterable[ArtistMetaWrite], *, state: str
    ) -> None:
        """Write a batch of :class:`ArtistMetaWrite` rows with a single ``resolved_at``/state."""
        assert self.database is not None
        now = int(time.time())
        for row in rows:
            values = {
                "artist_key": row["artist_key"],
                "artist_name": row["artist_name"],
                "mbid": row.get("mbid"),
                "mb_tags": json_dumps(row.get("mb_tags") or []),
                "genres": json_dumps(row.get("genres") or []),
                "begin_year": row.get("begin_year"),
                # UNSET (not None) when absent: a caller with no fresh release-year data (e.g.
                # the MusicBrainz enrichment pass, which has no release-year source at all) must
                # not clobber a value written by another enrichment path - see musicbrainz.py.
                "first_release_year": row.get("first_release_year", UNSET),
                "country": row.get("country"),
                "lb_listeners": row.get("lb_listeners"),
                "lb_listen_count": row.get("lb_listen_count"),
                "resolved_at": now,
                "resolve_state": state,
            }
            await self.database.upsert(DB_TABLE_GENOME_ARTIST_META, values)
        await self.database.commit()

    async def __create_database_tables(self) -> None:
        """Create database tables (see §3.1)."""
        assert self.database is not None
        await self.database.execute(
            f"""CREATE TABLE IF NOT EXISTS {DB_TABLE_SETTINGS}(
                    [key] TEXT PRIMARY KEY,
                    [value] TEXT,
                    [type] TEXT
                );"""
        )
        await self.database.execute(
            f"""CREATE TABLE IF NOT EXISTS {DB_TABLE_GENOME_LISTENS}(
                    [id] INTEGER PRIMARY KEY AUTOINCREMENT,
                    [listener] TEXT NOT NULL,
                    [ma_userid] TEXT,
                    [played_at] INTEGER NOT NULL,
                    [artist_key] TEXT NOT NULL,
                    [artist_name] TEXT NOT NULL,
                    [track_key] TEXT NOT NULL,
                    [track_name] TEXT NOT NULL,
                    [album_name] TEXT,
                    [source] TEXT NOT NULL,
                    [player_id] TEXT,
                    [duration_ms] INTEGER,
                    [played_ms] INTEGER,
                    [fully_played] BOOLEAN,
                    [confidence] REAL NOT NULL DEFAULT 1.0,
                    [dedupe_key] TEXT NOT NULL,
                    UNIQUE(dedupe_key));"""
        )
        await self.database.execute(
            f"""CREATE TABLE IF NOT EXISTS {DB_TABLE_GENOME_ARTIST_META}(
                    [artist_key] TEXT PRIMARY KEY,
                    [artist_name] TEXT NOT NULL,
                    [mbid] TEXT,
                    [mb_tags] json NOT NULL DEFAULT '[]',
                    [genres] json NOT NULL DEFAULT '[]',
                    [begin_year] INTEGER,
                    [first_release_year] INTEGER,
                    [country] TEXT,
                    [lb_listeners] INTEGER,
                    [lb_listen_count] INTEGER,
                    [resolved_at] INTEGER NOT NULL DEFAULT 0,
                    [resolve_state] TEXT NOT NULL DEFAULT 'pending');"""
        )
        await self.database.execute(
            f"""CREATE TABLE IF NOT EXISTS {DB_TABLE_GENOME_CACHE}(
                    [key] TEXT PRIMARY KEY,
                    [value] json NOT NULL,
                    [created_at] INTEGER NOT NULL,
                    [expires_at] INTEGER NOT NULL DEFAULT 0);"""
        )
        await self.database.commit()

    async def __create_database_indexes(self) -> None:
        """Create database indexes (see §3.1)."""
        assert self.database is not None
        await self.database.execute(
            f"CREATE INDEX IF NOT EXISTS {DB_TABLE_GENOME_LISTENS}_played_at_idx "
            f"ON {DB_TABLE_GENOME_LISTENS}(played_at);"
        )
        await self.database.execute(
            f"CREATE INDEX IF NOT EXISTS {DB_TABLE_GENOME_LISTENS}_artist_idx "
            f"ON {DB_TABLE_GENOME_LISTENS}(artist_key);"
        )
        await self.database.execute(
            f"CREATE INDEX IF NOT EXISTS {DB_TABLE_GENOME_LISTENS}_listener_idx "
            f"ON {DB_TABLE_GENOME_LISTENS}(listener, played_at);"
        )
        await self.database.commit()

    async def __migrate_database(self, prev_version: int) -> None:
        """
        Perform a database migration.

        No prior schema versions exist yet (§3.1 ``DB_SCHEMA_VERSION`` starts at 1); this is a
        placeholder for future migration steps, following the ``cache`` controller's pattern of
        never raising and falling back to a reset on failure.

        :param prev_version: The schema version the on-disk database was last written at.
        """
        assert self.database is not None
        LOGGER.debug("No migration steps defined yet for version %s", prev_version)


__all__ = ["ArtistMetaWrite", "GenomeStore"]
