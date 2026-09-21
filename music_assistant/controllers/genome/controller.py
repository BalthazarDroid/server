"""
``GenomeController`` — the Listening Genome core controller (§3.2, §3.3, §1.1, §1.10).

Owns the API surface, config entries, chunked Apple-export upload protocol and the scheduled
rebuild. Talks to storage only through the frozen ``GenomeStore`` surface (Part 4) — modelled
here as a local :class:`_GenomeStoreProtocol` so this module (and its tests) never need
``store.py`` (owned by a separate work package) to exist on disk. The real ``GenomeStore`` is
imported lazily, at first use, so importing this module never fails while that file is still
being written elsewhere.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import re
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, Protocol, TypeVar, cast

from music_assistant_models.auth import Scope
from music_assistant_models.config_entries import ConfigActionResult, ConfigEntry
from music_assistant_models.enums import ConfigEntryType
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.helpers import create_safe_string

from music_assistant.helpers.api import api_command
from music_assistant.helpers.datetime import LOCAL_TIMEZONE, utc_timestamp
from music_assistant.models.core_controller import CoreController

from .baseline import load_baseline, uniform_baseline
from .constants import (
    CONF_ACTION_CLEAR_GENOME_DATA,
    CONF_ACTION_REBUILD_NOW,
    CONF_APPLE_IMPORT_DIR,
    CONF_ENRICH_ENABLED,
    CONF_LASTFM_API_KEY,
    CONF_LASTFM_POLL_ENABLED,
    CONF_LASTFM_POLL_INTERVAL_HOURS,
    CONF_LASTFM_USERNAME,
    CONF_MIN_SECONDS_PLAYED,
    CONF_OBSCURITY_PERCENTILE,
    CONF_REBUILD_SCHEDULE_HOUR,
    CONF_RECENCY_HALF_LIFE_DAYS,
    DEFAULT_APPLE_IMPORT_DIR,
    DEFAULT_ENRICH_ENABLED,
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_LASTFM_API_KEY,
    DEFAULT_LASTFM_POLL_ENABLED,
    DEFAULT_LASTFM_POLL_INTERVAL_HOURS,
    DEFAULT_LASTFM_USERNAME,
    DEFAULT_MIN_SECONDS_PLAYED,
    DEFAULT_NEW_ARTIST_WINDOW_DAYS,
    DEFAULT_OBSCURITY_PERCENTILE,
    DEFAULT_REBUILD_SCHEDULE_HOUR,
    DEFAULT_TOP_N,
    DISCOVERY_STATE_PENDING,
    DISCOVERY_STATE_READY,
    DISCOVERY_STATE_UNAVAILABLE,
    GENOME_DISCOVERY_COLD_LIMIT,
    GENOME_DISCOVERY_COLD_MAX_PLAYS,
    GENOME_DISCOVERY_LASTFM_MIN_INTERVAL_SECONDS,
    GENOME_DISCOVERY_REFRESH_INTERVAL_HOURS,
    GENOME_DISCOVERY_SEED_ERROR_COOLDOWN_HOURS,
    GENOME_DISCOVERY_SEED_LIMIT,
    GENOME_DISCOVERY_SIMILAR_PER_SEED,
    GENOME_DISCOVERY_SUGGESTED_LIMIT,
    GENOME_DISCOVERY_TASK_ID,
    GENOME_ENRICHMENT_BATCH_LIMIT,
    GENOME_ENRICHMENT_TASK_ID,
    GENOME_LASTFM_POLL_TASK_ID,
    GENOME_MB_ENRICHMENT_MIN_INTERVAL_SECONDS,
    GENOME_REBUILD_TASK_ID,
    GENOME_UPLOAD_CHUNK_MAX_B64_BYTES,
    GENOME_UPLOAD_MAX_TOTAL_BYTES,
    GENOME_UPLOAD_TTL_SECONDS,
    GENOME_UPLOADS_DIRNAME,
    LASTFM_API_KEY_PATTERN,
    LISTENER_HOUSEHOLD,
    LOGGER,
    RESOLVE_STATE_ERROR,
    RESOLVE_STATE_NOT_FOUND,
    RESOLVE_STATE_OK,
    RESOLVE_STATE_PENDING,
    SOURCE_APPLE_EXPORT,
)
from .discovery import (
    divergent_genres,
    rank_cold_corners,
    read_library_artists,
    select_seeds,
)
from .engine import build_genome
from .errors import LastfmNotConfiguredError
from .models import (
    ColdArtist,
    DiscoveryResult,
    EngineParams,
    FailedArtist,
    GenomeImportResult,
    GenomeInputs,
    GenomeRebuildResult,
    GenomeResult,
    GenomeSettings,
    # NOTE: every name used in an @api_command signature must be imported at RUNTIME, not
    # under TYPE_CHECKING. Music Assistant resolves handler annotations with get_type_hints()
    # when it registers the command, and its NameError fallback only searches
    # music_assistant_models — it cannot see this package's own types.
    GenomeSettingsPatch,
    SuggestedArtist,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence

    from music_assistant_models.config_entries import ConfigValueType, CoreConfig

    from music_assistant.mass import MusicAssistant

    from .importers.apple_csv import ApplePlayActivityStats
    from .models import ArtistMeta, Baseline, Listen


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _log_command_errors(
    command: str,
) -> Callable[[Callable[_P, Coroutine[Any, Any, _R]]], Callable[_P, Coroutine[Any, Any, _R]]]:
    """
    Ensure a user-facing API-command failure is always logged before it propagates.

    Every ``@api_command`` on this controller gets one of these: the bug that prompted it was
    an ``InvalidDataError`` guard (missing Last.fm credentials) raised with nothing logged
    first, so the failure reached the user as a popup while the server log stayed silent.
    A single decorator here covers every current and future raise site in the command's whole
    call graph, instead of a try/except repeated at each one.

    :param command: The API command name (e.g. ``"genome/import_lastfm"``), used only to label
        the log line.
    """

    def decorate(
        func: Callable[_P, Coroutine[Any, Any, _R]],
    ) -> Callable[_P, Coroutine[Any, Any, _R]]:
        @wraps(func)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return await func(*args, **kwargs)
            except InvalidDataError as err:
                # user-correctable (bad/missing input, missing config): the message already
                # says what to fix, so a traceback would only add noise
                LOGGER.warning("%s: %s", command, err)
                raise
            except Exception:
                # anything else is a bug or an infra problem nobody anticipated; the traceback
                # is the only way to diagnose it later, since the caller only ever sees str(exc)
                LOGGER.error("%s failed unexpectedly", command, exc_info=True)
                raise

        return wrapper

    return decorate


class _GenomeStoreProtocol(Protocol):
    """
    Structural mirror of ``GenomeStore``'s frozen public surface (Part 4).

    Duck-typed on purpose: ``store.py`` is owned by a different work package and may not exist
    on disk yet, so this controller (and its tests) never import the real class directly.
    """

    async def setup(self) -> None: ...
    async def close(self) -> None: ...
    async def add_listens(
        self, listens: Sequence[Listen], *, listener: str, ma_userid: str | None = None
    ) -> GenomeImportResult: ...
    def iter_listens(self, listener: str, *, since: int = 0) -> AsyncIterator[Listen]: ...
    async def count_listens(self, listener: str) -> int: ...
    async def get_artist_meta(self, artist_keys: Sequence[str]) -> dict[str, ArtistMeta]: ...
    async def upsert_artist_meta(self, rows: Sequence[ArtistMeta], *, state: str) -> None: ...
    async def pending_artist_keys(self, limit: int = 200) -> list[tuple[str, str]]: ...
    async def get_cached_genome(self, listener: str) -> GenomeResult | None: ...
    async def set_cached_genome(self, listener: str, genome: GenomeResult) -> None: ...
    async def clear(self, listener: str | None = None) -> None: ...
    async def source_counts(self, listener: str) -> dict[str, int]: ...
    async def player_names(self) -> dict[str, str]: ...
    # extensions beyond the frozen Part 4 surface, added during integration (see docs/STATUS.md):
    async def upsert_artist_meta_full(self, rows: Sequence[Any], *, state: str) -> None: ...
    async def update_lb_popularity(self, rows: dict[str, tuple[int, int]]) -> None: ...
    async def pending_popularity_keys(self, limit: int = 200) -> list[tuple[str, str]]: ...
    async def mark_popularity_attempted(self, artist_keys: Sequence[str]) -> None: ...
    async def backfill_done(self) -> bool: ...
    async def mark_backfill_done(self) -> None: ...
    async def lastfm_backfill_done(self) -> bool: ...
    async def mark_lastfm_backfill_done(self) -> None: ...
    async def artist_resolution_counts(self) -> dict[str, int]: ...
    async def failed_artist_keys(self, limit: int = 100) -> list[FailedArtist]: ...
    async def retry_failed_artists(self, artist_keys: Sequence[str] | None = None) -> int: ...
    async def all_failed_artist_keys(self) -> frozenset[str]: ...
    async def dismiss_unresolved(self, artist_keys: Sequence[str]) -> None: ...
    async def unresolved_dismissed_keys(self) -> frozenset[str] | None: ...
    async def get_cached_discovery(self, listener: str) -> dict[str, Any] | None: ...
    async def set_cached_discovery(self, listener: str, data: Mapping[str, Any]) -> None: ...
    async def artist_play_counts(self, listener: str) -> dict[str, int]: ...


class _UploadState:
    """In-progress chunked Apple-export upload (§3.3)."""

    __slots__ = ("created_at", "next_seq", "total_bytes")

    def __init__(self) -> None:
        """Initialize a fresh upload with no chunks received yet."""
        self.next_seq = 0
        self.total_bytes = 0
        self.created_at = time.monotonic()


def _create_default_store(mass: MusicAssistant) -> _GenomeStoreProtocol:
    """
    Construct the production ``GenomeStore``.

    Imported lazily so this module loads (and is testable) before ``store.py`` — owned by a
    separate work package — exists on disk. See ``docs/STATUS.md`` "Contract gaps".
    """
    from music_assistant.controllers.genome.store import GenomeStore  # noqa: PLC0415

    return GenomeStore(mass)


async def _default_apple_parser(
    path: str, *, min_seconds: int, stats: ApplePlayActivityStats | None = None
) -> AsyncIterator[Listen]:
    """Lazily delegate to the real Apple Music CSV parser (owned by a separate work package)."""
    from music_assistant.controllers.genome.importers.apple_csv import (  # noqa: PLC0415
        parse_play_activity,
    )

    async for listen in parse_play_activity(path, min_seconds=min_seconds, stats=stats):
        yield listen


def _default_lastfm_importer_factory(mass: MusicAssistant, *, username: str, api_key: str) -> Any:
    """Lazily construct the real Last.fm importer (owned by a separate work package)."""
    from music_assistant.controllers.genome.http import AiohttpClient  # noqa: PLC0415
    from music_assistant.controllers.genome.importers.lastfm import LastfmImporter  # noqa: PLC0415

    client = AiohttpClient(mass, rate_limit=5, period=1.0)
    return LastfmImporter(client, username, api_key)


def _default_lastfm_similar_client_factory(mass: MusicAssistant) -> Any:
    """Lazily construct the throttled HTTP client the background discovery pass fetches with."""
    from music_assistant.controllers.genome.http import AiohttpClient  # noqa: PLC0415

    return AiohttpClient(mass, rate_limit=5, period=1.0)


def _default_playlog_importer_factory(mass: MusicAssistant, store: _GenomeStoreProtocol) -> Any:
    """Lazily construct the real MA playlog importer (live capture + one-time backfill)."""
    from music_assistant.controllers.genome.importers.ma_playlog import (  # noqa: PLC0415
        MaPlaylogImporter,
    )

    return MaPlaylogImporter(mass, store)


class GenomeController(CoreController):
    """Core controller exposing the Listening Genome API (§3.2, §3.3)."""

    domain: str = "genome"

    def __init__(self, mass: MusicAssistant, *, store: _GenomeStoreProtocol | None = None) -> None:
        """
        Initialize the controller.

        :param mass: The running ``MusicAssistant`` instance.
        :param store: Injected ``GenomeStore``-shaped object; defaults to the real store,
            constructed lazily on first use. Tests pass an in-memory stub here.
        """
        super().__init__(mass)
        self.manifest.name = "Listening Genome"
        self.manifest.description = (
            "Music Assistant's core controller computing a household listening fingerprint."
        )
        self.manifest.icon = "dna"
        self.store: _GenomeStoreProtocol = (
            store if store is not None else _create_default_store(mass)
        )
        self.baseline: Baseline = uniform_baseline()
        self.last_rebuild_at: int | None = None
        self._uploads: dict[str, _UploadState] = {}
        self._apple_parser = _default_apple_parser
        self._lastfm_importer_factory = _default_lastfm_importer_factory
        self._playlog_importer_factory = _default_playlog_importer_factory
        self._unsubscribe_playlog: Callable[[], None] | None = None
        self._library_artists_reader = read_library_artists
        self._lastfm_similar_client_factory = _default_lastfm_similar_client_factory
        # serializes the daily rebuild's enrichment pass against the continuous background one
        # (§3.8, P3) so the two never issue MusicBrainz requests at the same time
        self._enrichment_lock = asyncio.Lock()
        # the discovery pass gets its own lock: a user pressing Refresh while the daily pass is
        # running must queue behind it rather than double the Last.fm request rate
        self._discovery_lock = asyncio.Lock()

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return all Config Entries for the genome core module (§3.2)."""
        return (
            ConfigEntry(
                key=CONF_RECENCY_HALF_LIFE_DAYS,
                type=ConfigEntryType.INTEGER,
                default_value=DEFAULT_HALF_LIFE_DAYS,
                required=False,
            ),
            ConfigEntry(
                key=CONF_LASTFM_USERNAME,
                type=ConfigEntryType.STRING,
                default_value=DEFAULT_LASTFM_USERNAME,
                required=False,
            ),
            ConfigEntry(
                key=CONF_LASTFM_API_KEY,
                type=ConfigEntryType.SECURE_STRING,
                default_value=DEFAULT_LASTFM_API_KEY,
                required=False,
            ),
            ConfigEntry(
                key=CONF_LASTFM_POLL_ENABLED,
                type=ConfigEntryType.BOOLEAN,
                default_value=DEFAULT_LASTFM_POLL_ENABLED,
            ),
            ConfigEntry(
                key=CONF_LASTFM_POLL_INTERVAL_HOURS,
                type=ConfigEntryType.INTEGER,
                default_value=DEFAULT_LASTFM_POLL_INTERVAL_HOURS,
                advanced=True,
            ),
            ConfigEntry(
                key=CONF_APPLE_IMPORT_DIR,
                type=ConfigEntryType.STRING,
                default_value=DEFAULT_APPLE_IMPORT_DIR,
                required=False,
                advanced=True,
            ),
            ConfigEntry(
                key=CONF_ENRICH_ENABLED,
                type=ConfigEntryType.BOOLEAN,
                default_value=DEFAULT_ENRICH_ENABLED,
            ),
            ConfigEntry(
                key=CONF_OBSCURITY_PERCENTILE,
                type=ConfigEntryType.INTEGER,
                default_value=DEFAULT_OBSCURITY_PERCENTILE,
                advanced=True,
            ),
            ConfigEntry(
                key=CONF_MIN_SECONDS_PLAYED,
                type=ConfigEntryType.INTEGER,
                default_value=DEFAULT_MIN_SECONDS_PLAYED,
                advanced=True,
            ),
            ConfigEntry(
                key=CONF_REBUILD_SCHEDULE_HOUR,
                type=ConfigEntryType.INTEGER,
                default_value=DEFAULT_REBUILD_SCHEDULE_HOUR,
                advanced=True,
            ),
            ConfigEntry(key=CONF_ACTION_REBUILD_NOW, type=ConfigEntryType.ACTION),
            ConfigEntry(key=CONF_ACTION_CLEAR_GENOME_DATA, type=ConfigEntryType.ACTION),
        )

    async def handle_config_action(
        self, action: str
    ) -> tuple[ConfigEntry, ...] | ConfigActionResult | None:
        """Handle a one-shot config action button press (§3.2)."""
        if action == CONF_ACTION_REBUILD_NOW:
            rebuild_result = await self.rebuild()
            listens_scanned = rebuild_result["listens_scanned"]
            if listens_scanned == 0:
                return ConfigActionResult(
                    translation_key=f"{CONF_ACTION_REBUILD_NOW}.result_empty",
                )
            return ConfigActionResult(
                translation_key=f"{CONF_ACTION_REBUILD_NOW}.result",
                translation_args=[str(listens_scanned)],
            )
        if action == CONF_ACTION_CLEAR_GENOME_DATA:
            await self.store.clear()
            return ConfigActionResult(translation_key=f"{CONF_ACTION_CLEAR_GENOME_DATA}.result")
        return await super().handle_config_action(action)

    async def setup(self, config: CoreConfig) -> None:
        """
        Load the baseline, set up storage, attach live playlog capture and schedule rebuilds.

        Live capture of ``EventType.PLAYLOG_UPDATED`` (§1.4) is the feature's primary ingestion
        path — MA's own ``playlog`` table purges rows older than 90 days, so subscribing here
        (rather than only backfilling once) is what makes a real listening history possible at
        all. A one-time backfill of the existing ``playlog``/``tracks`` tables runs afterwards,
        in the background, to seed a coarse prior for everything before Genome started listening.
        """
        self.baseline = await load_baseline()
        await self.store.setup()
        playlog_importer = self._playlog_importer_factory(self.mass, self.store)
        self._unsubscribe_playlog = playlog_importer.attach()
        if not await self.store.backfill_done():
            self.mass.create_task(self._run_ma_backfill(playlog_importer))
        self._register_rebuild_task()
        self._register_lastfm_poll_task()
        self._register_enrichment_task()
        self._register_discovery_task()
        listen_count = await self.store.count_listens(LISTENER_HOUSEHOLD)
        LOGGER.info(
            "Genome controller setup complete: baseline %s, %d listens stored",
            self.baseline.version,
            listen_count,
        )

    async def close(self) -> None:
        """Unsubscribe live playlog capture and close storage on server stop."""
        if self._unsubscribe_playlog is not None:
            self._unsubscribe_playlog()
            self._unsubscribe_playlog = None
        await self.store.close()

    async def update_config(self, config: CoreConfig, changed_keys: set[str]) -> None:
        """Re-register the scheduled tasks affected by the changed keys; defer the rest to base."""
        await super().update_config(config, changed_keys)
        if f"values/{CONF_REBUILD_SCHEDULE_HOUR}" in changed_keys:
            self._register_rebuild_task()
        if changed_keys & {
            f"values/{CONF_LASTFM_POLL_ENABLED}",
            f"values/{CONF_LASTFM_POLL_INTERVAL_HOURS}",
            f"values/{CONF_LASTFM_USERNAME}",
            f"values/{CONF_LASTFM_API_KEY}",
        }:
            self._register_lastfm_poll_task()

    @api_command("genome/get", required_scope=Scope.LIBRARY_READ)
    @_log_command_errors("genome/get")
    async def get_genome(
        self, listener: str = LISTENER_HOUSEHOLD, refresh: bool = False
    ) -> GenomeResult:
        """
        Return the household's listening genome, from cache unless ``refresh`` is set (§3.3).

        A cached result whose recorded listen count no longer matches the store is still
        returned (fast path), but with ``stale`` set so the caller can offer a rebuild.

        :param listener: The listener id (``"household"`` in v1).
        :param refresh: Force a rebuild instead of serving from cache.
        """
        if not refresh:
            cached = await self.store.get_cached_genome(listener)
            if cached is not None:
                current_count = await self.store.count_listens(listener)
                if current_count == cached["stats"]["total_listens"]:
                    return cached
                return cast("GenomeResult", {**cached, "stale": True})
        # Never enrich on a read: a page load must return what we already know. Resolving
        # artists means MusicBrainz, which is rate-limited in minutes, not milliseconds.
        return (await self._rebuild(listener))["genome"]

    @api_command("genome/unresolved_artists", required_scope=Scope.LIBRARY_READ)
    @_log_command_errors("genome/unresolved_artists")
    async def unresolved_artists(self, limit: int = 100) -> list[FailedArtist]:
        """
        Return the artists whose MusicBrainz lookup failed, so the UI can name them.

        These are exactly the artists counted in ``GenomeStats.artists_failed`` (see
        :meth:`GenomeStore.failed_artist_keys`).

        :param limit: The maximum number of rows to return.
        """
        return await self.store.failed_artist_keys(limit=limit)

    @api_command("genome/retry_artists", required_scope=Scope.LIBRARY_MANAGE)
    @_log_command_errors("genome/retry_artists")
    async def retry_artists(self, artist_keys: list[str] | None = None) -> int:
        """
        Put failed artists back in the resolution queue immediately, ignoring the error cooldown.

        Does no network work itself: it only resets stored state, so the existing background
        (hourly) or on-demand (rebuild) enrichment pass is what actually retries the
        MusicBrainz lookup, on its own schedule.

        :param artist_keys: The artist keys to retry; ``None`` retries every failed artist.
        :return: The number of artists moved back to the resolution queue.
        """
        return await self.store.retry_failed_artists(artist_keys)

    @api_command("genome/dismiss_unresolved", required_scope=Scope.LIBRARY_MANAGE)
    @_log_command_errors("genome/dismiss_unresolved")
    async def dismiss_unresolved(self) -> bool:
        """
        Dismiss the "could not be identified" notice for the artists failing right now.

        Records a fingerprint of the artists currently in the ``error`` resolve state rather
        than muting the notice outright: if a different (or additional) artist later ends up
        in ``error``, ``GenomeStats.unresolved_dismissed`` goes back to ``False`` and the
        notice returns, because it is never honest to promise "you will not hear about this
        again" about a problem that has not happened yet.

        :return: Whether the current failed-artist set is now fully dismissed.
        """
        failed = await self.store.all_failed_artist_keys()
        await self.store.dismiss_unresolved(sorted(failed))
        return await self._unresolved_dismissed()

    @api_command("genome/rebuild", required_scope=Scope.LIBRARY_MANAGE)
    @_log_command_errors("genome/rebuild")
    async def rebuild(
        self, listener: str = LISTENER_HOUSEHOLD, enrich: bool = True
    ) -> GenomeRebuildResult:
        """
        Recompute and cache the genome for ``listener`` (§3.3).

        Enrichment is dispatched to the background rather than awaited: MusicBrainz paces at
        roughly one artist per second and answers a burst with minute-long penalties, so a
        pass over a real backlog outlives any request. The recomputed genome comes back
        immediately from what is already stored, and ``stats.artists_pending`` tells the
        caller that more is still resolving.

        :param listener: The listener id (``"household"`` in v1).
        :param enrich: Kick off a background pass to resolve pending artist metadata.
        """
        result = await self._rebuild(listener)
        if enrich and self.get_config_value(
            CONF_ENRICH_ENABLED, DEFAULT_ENRICH_ENABLED, return_type=bool
        ):
            self.mass.create_task(self._background_enrichment())
        return result

    @api_command("genome/discovery", required_scope=Scope.LIBRARY_READ)
    @_log_command_errors("genome/discovery")
    async def discovery(self, listener: str = LISTENER_HOUSEHOLD) -> DiscoveryResult:
        """
        Return artists worth trying: cold library corners plus cached Last.fm suggestions (§3.3).

        Reads cache and database only, and deliberately takes no "refresh"/"enrich" flag: D-16
        exists because a websocket read that fell through to an inline MusicBrainz pass hung the
        page for an hour, and a flag someone can flip back is how that bug reaches a user again.
        The Last.fm half is produced solely by the background pass
        (:meth:`_background_discovery`); this method serves whatever that pass last stored, and
        says so through ``suggested_state``.

        The cold-corner half never falls through to a rebuild either: with no cached genome
        there are no divergent genres to rank against, which is an honest empty state rather
        than a reason to recompute the whole genome inside a page load.

        :param listener: The listener id (``"household"`` in v1).
        """
        genome = await self.store.get_cached_genome(listener)
        genres = divergent_genres(genome)
        in_library = await self._cold_corners(listener, genres)
        cached = await self.store.get_cached_discovery(listener)
        generated_at = _float_or_none(cached.get("generated_at")) if cached else None
        suggested = _suggested_from_cache(cached)
        api_key = self.get_config_value(
            CONF_LASTFM_API_KEY, DEFAULT_LASTFM_API_KEY, return_type=str
        )
        if not api_key:
            state = DISCOVERY_STATE_UNAVAILABLE
        elif generated_at is None:
            state = DISCOVERY_STATE_PENDING
        else:
            state = DISCOVERY_STATE_READY
        return DiscoveryResult(
            in_library=in_library,
            suggested=suggested,
            suggested_state=state,
            generated_at=generated_at,
        )

    @api_command("genome/discovery_refresh", required_scope=Scope.LIBRARY_MANAGE)
    @_log_command_errors("genome/discovery_refresh")
    async def discovery_refresh(self, listener: str = LISTENER_HOUSEHOLD) -> bool:
        """
        Dispatch a background Last.fm discovery pass and return immediately (§3.3, D-16).

        Returns whether a pass was dispatched at all: ``False`` means no Last.fm API key is
        configured, which is the same normal, non-error state ``genome/discovery`` reports as
        ``suggested_state="unavailable"``. The caller re-queries ``genome/discovery`` for the
        result; nothing is awaited here.

        :param listener: The listener id (``"household"`` in v1).
        """
        api_key = self.get_config_value(
            CONF_LASTFM_API_KEY, DEFAULT_LASTFM_API_KEY, return_type=str
        )
        if not api_key:
            LOGGER.info("Discovery refresh skipped: no Last.fm API key configured")
            return False
        self.mass.create_task(self._background_discovery(listener))
        return True

    @api_command("genome/import_apple", required_scope=Scope.LIBRARY_MANAGE)
    @_log_command_errors("genome/import_apple")
    async def import_apple(
        self,
        upload_id: str,
        seq: int,
        chunk_b64: str = "",
        final: bool = False,
        filename: str = "",
    ) -> GenomeImportResult:
        """
        Accept one chunk of a chunked Apple Music export upload, ingesting on ``final`` (§3.3).

        :param upload_id: Client-chosen upload id; empty selects the ``apple_import_dir`` path.
        :param seq: Zero-based, strictly sequential chunk index for this upload.
        :param chunk_b64: Base64-encoded CSV bytes for this chunk.
        :param final: Whether this is the last chunk; triggers parsing and ingestion.
        :param filename: Original filename; must end in ``.csv`` to select the parser.
        """
        if upload_id:
            await self._append_upload_chunk(upload_id, seq, chunk_b64)
        if not final:
            return _empty_import_result(SOURCE_APPLE_EXPORT)

        if upload_id:
            path = self._upload_path(upload_id)
        else:
            import_dir = self.get_config_value(
                CONF_APPLE_IMPORT_DIR, DEFAULT_APPLE_IMPORT_DIR, return_type=str
            )
            if not import_dir or not filename:
                msg = "No upload_id given and apple_import_dir/filename are not configured"
                raise InvalidDataError(msg)
            path = os.path.join(import_dir, filename)

        if not filename.lower().endswith(".csv"):
            if upload_id:
                self._uploads.pop(upload_id, None)
            msg = f"Unsupported Apple export file: {filename or '(no filename given)'}"
            raise InvalidDataError(msg)

        try:
            result = await self._ingest_apple_csv(path)
        finally:
            if upload_id:
                await self._cleanup_upload(upload_id)
        return result

    @api_command("genome/import_lastfm", required_scope=Scope.LIBRARY_MANAGE)
    @_log_command_errors("genome/import_lastfm")
    async def import_lastfm(self, username: str = "", max_pages: int = 0) -> GenomeImportResult:
        """
        Poll Last.fm's ``user.getRecentTracks`` and ingest the result (§3.3, §3.8).

        :param username: Overrides the configured ``lastfm_username`` when given.
        :param max_pages: Stop after this many pages; ``0`` means "until exhausted".
        """
        username = username or self.get_config_value(
            CONF_LASTFM_USERNAME, DEFAULT_LASTFM_USERNAME, return_type=str
        )
        api_key = self.get_config_value(
            CONF_LASTFM_API_KEY, DEFAULT_LASTFM_API_KEY, return_type=str
        )
        if not username or not api_key:
            missing = " and ".join(
                part
                for part, present in (("username", bool(username)), ("API key", bool(api_key)))
                if not present
            )
            msg = (
                f"Last.fm is not configured: the {missing} is missing. Add it on the "
                "Listening Genome import page, then try again."
            )
            raise LastfmNotConfiguredError(msg)
        if not re.match(LASTFM_API_KEY_PATTERN, api_key.strip()):
            # Catches a mis-pasted key (a URL, a truncated copy) here rather than letting it go
            # out as a query parameter and come back as an opaque 403. The key itself is never
            # echoed — only its length, which is enough to diagnose without leaking a secret.
            msg = (
                f"The configured Last.fm API key does not look like a key "
                f"(expected 32 hex characters, got {len(api_key.strip())}). "
                "Copy it from https://www.last.fm/api/accounts and save it again."
            )
            raise LastfmNotConfiguredError(msg)
        api_key = api_key.strip()
        importer = self._lastfm_importer_factory(self.mass, username=username, api_key=api_key)
        return cast(
            "GenomeImportResult",
            await importer.import_since(
                self.store, listener=LISTENER_HOUSEHOLD, max_pages=max_pages
            ),
        )

    @api_command("genome/settings", required_scope=Scope.LIBRARY_READ)
    @_log_command_errors("genome/settings")
    async def get_settings(self) -> GenomeSettings:
        """Return the current genome settings (§3.3). Never includes the Last.fm API key."""
        api_key = self.get_config_value(
            CONF_LASTFM_API_KEY, DEFAULT_LASTFM_API_KEY, return_type=str
        )
        return GenomeSettings(
            half_life_days=self.get_config_value(
                CONF_RECENCY_HALF_LIFE_DAYS, DEFAULT_HALF_LIFE_DAYS, return_type=int
            ),
            lastfm_username=self.get_config_value(
                CONF_LASTFM_USERNAME, DEFAULT_LASTFM_USERNAME, return_type=str
            ),
            lastfm_configured=bool(api_key),
            lastfm_poll_enabled=self.get_config_value(
                CONF_LASTFM_POLL_ENABLED, DEFAULT_LASTFM_POLL_ENABLED, return_type=bool
            ),
            enrich_enabled=self.get_config_value(
                CONF_ENRICH_ENABLED, DEFAULT_ENRICH_ENABLED, return_type=bool
            ),
            obscurity_percentile=self.get_config_value(
                CONF_OBSCURITY_PERCENTILE, DEFAULT_OBSCURITY_PERCENTILE, return_type=int
            ),
            min_seconds_played=self.get_config_value(
                CONF_MIN_SECONDS_PLAYED, DEFAULT_MIN_SECONDS_PLAYED, return_type=int
            ),
            apple_import_dir=self.get_config_value(
                CONF_APPLE_IMPORT_DIR, DEFAULT_APPLE_IMPORT_DIR, return_type=str
            ),
            baseline_version=self.baseline.version,
            last_rebuild_at=self.last_rebuild_at,
        )

    @api_command("genome/settings/set", required_scope=Scope.LIBRARY_MANAGE)
    @_log_command_errors("genome/settings/set")
    async def set_settings(self, settings: GenomeSettingsPatch) -> GenomeSettings:
        """
        Apply a partial settings update and return the resulting settings (§3.3).

        :param settings: Only the fields to change; every field is optional.
        """
        key_map: dict[str, str] = {
            "half_life_days": CONF_RECENCY_HALF_LIFE_DAYS,
            "lastfm_username": CONF_LASTFM_USERNAME,
            "lastfm_api_key": CONF_LASTFM_API_KEY,
            "lastfm_poll_enabled": CONF_LASTFM_POLL_ENABLED,
            "enrich_enabled": CONF_ENRICH_ENABLED,
            "obscurity_percentile": CONF_OBSCURITY_PERCENTILE,
            "min_seconds_played": CONF_MIN_SECONDS_PLAYED,
            "apple_import_dir": CONF_APPLE_IMPORT_DIR,
        }
        values: dict[str, ConfigValueType] = {}
        for field, conf_key in key_map.items():
            value = getattr(settings, field, None)
            # None means "not supplied" — only fields the caller actually set are written
            if value is not None:
                values[conf_key] = cast("ConfigValueType", value)
        if values:
            await self.mass.config.save_core_config(self.domain, values)
        return await self.get_settings()

    def _register_rebuild_task(self) -> None:
        """Register (or re-register) the scheduled daily rebuild task (§1.10)."""
        # imported here (not at module scope) to keep this file's happy-path imports light for
        # tests that never touch scheduling
        from music_assistant_models.background_task import TaskSchedule  # noqa: PLC0415

        from music_assistant.helpers.datetime import local_clock_time_to_utc  # noqa: PLC0415

        hour = self.get_config_value(
            CONF_REBUILD_SCHEDULE_HOUR, DEFAULT_REBUILD_SCHEDULE_HOUR, return_type=int
        )
        utc_hour, utc_minute = local_clock_time_to_utc(hour, 0)
        self.mass.tasks.register_scheduled_task(
            task_id=GENOME_REBUILD_TASK_ID,
            name="Genome rebuild",
            handler=self._scheduled_rebuild,
            schedule=TaskSchedule.daily(hour=utc_hour, minute=utc_minute),
            translation_key=GENOME_REBUILD_TASK_ID,
            translation_owner=self.translation_owner,
            metadata={"task_domain": GENOME_REBUILD_TASK_ID},
            allow_retry=True,
        )

    async def _scheduled_rebuild(self) -> None:
        """Scheduled-task entry point: rebuild the household genome."""
        await self.rebuild(LISTENER_HOUSEHOLD)

    def _register_lastfm_poll_task(self) -> None:
        """
        Register, re-register or unregister the recurring Last.fm poll (§3.2, §3.8).

        Music Assistant's scheduled-task API only offers hourly/daily/weekly cadences (no
        arbitrary interval-in-hours primitive), so ``lastfm_poll_interval_hours`` maps directly
        onto ``TaskSchedule.hourly(every=interval_hours)`` — the closest correct fit, and exactly
        what MA's other "poll every N hours" integrations use (see
        ``providers/lastfm_recommendations``). The task is unregistered outright whenever
        polling is turned off or Last.fm is not fully configured, so a stale task can never fire
        against a blank username/API key.
        """
        # imported here, not at module scope, to keep this file's happy-path imports light for
        # tests that never touch scheduling
        from music_assistant_models.background_task import TaskSchedule  # noqa: PLC0415

        enabled = self.get_config_value(
            CONF_LASTFM_POLL_ENABLED, DEFAULT_LASTFM_POLL_ENABLED, return_type=bool
        )
        username = self.get_config_value(
            CONF_LASTFM_USERNAME, DEFAULT_LASTFM_USERNAME, return_type=str
        )
        api_key = self.get_config_value(
            CONF_LASTFM_API_KEY, DEFAULT_LASTFM_API_KEY, return_type=str
        )
        if not enabled or not username or not api_key:
            self.mass.tasks.unregister_scheduled_task(GENOME_LASTFM_POLL_TASK_ID)
            LOGGER.info(
                "Last.fm polling disabled%s",
                "" if enabled else " (poll toggle is off)",
            )
            return
        interval_hours = self.get_config_value(
            CONF_LASTFM_POLL_INTERVAL_HOURS, DEFAULT_LASTFM_POLL_INTERVAL_HOURS, return_type=int
        )
        self.mass.tasks.register_scheduled_task(
            task_id=GENOME_LASTFM_POLL_TASK_ID,
            name="Genome Last.fm poll",
            handler=self._scheduled_lastfm_poll,
            schedule=TaskSchedule.hourly(every=max(interval_hours, 1)),
            translation_key=GENOME_LASTFM_POLL_TASK_ID,
            translation_owner=self.translation_owner,
            metadata={"task_domain": GENOME_LASTFM_POLL_TASK_ID},
            allow_retry=True,
        )
        LOGGER.info("Last.fm polling enabled: every %d hour(s) for %s", interval_hours, username)

    async def _scheduled_lastfm_poll(self) -> None:
        """Scheduled-task entry point: poll Last.fm for new scrobbles."""
        try:
            result = await self.import_lastfm()
        except LastfmNotConfiguredError:
            # config raced out from under an in-flight scheduled run; the next re-registration
            # (triggered by update_config) already handles unregistering the task itself
            LOGGER.debug("Skipping scheduled Last.fm poll: not configured")
            return
        except InvalidDataError:
            # a real Last.fm/network failure (bad key, unknown user, timeout, ...) - already
            # logged by _log_command_errors via import_lastfm; a background poll must not crash
            # the task loop over it, the next scheduled run tries again on its own
            return
        LOGGER.info(
            "Scheduled Last.fm poll: %d rows added, %d skipped, %d duplicate",
            result["rows_imported"],
            result["rows_skipped"],
            result["rows_duplicate"],
        )

    def _register_enrichment_task(self) -> None:
        """
        Register the continuous background MusicBrainz enrichment pass (§3.8, P3).

        A fixed 200-artist pass that only ran once a day at rebuild time left a real backlog
        (200 pending artists, hourly rebuild cadence measured in days) stuck behind MusicBrainz's
        own rate limiter. This keeps draining the pending queue hourly, paced comfortably under
        MusicBrainz's ~1 req/sec courtesy limit (:data:`GENOME_MB_ENRICHMENT_MIN_INTERVAL_SECONDS`)
        instead of relying on its 429/`Retry-After` path, and persists each artist's resolution
        immediately (:meth:`GenomeStore.upsert_artist_meta_full`), so a restart mid-pass loses at
        most the one artist in flight.
        """
        from music_assistant_models.background_task import TaskSchedule  # noqa: PLC0415

        self.mass.tasks.register_scheduled_task(
            task_id=GENOME_ENRICHMENT_TASK_ID,
            name="Genome MusicBrainz enrichment",
            handler=self._scheduled_enrichment,
            schedule=TaskSchedule.hourly(every=1),
            translation_key=GENOME_ENRICHMENT_TASK_ID,
            translation_owner=self.translation_owner,
            metadata={"task_domain": GENOME_ENRICHMENT_TASK_ID},
            allow_retry=True,
        )

    async def _scheduled_enrichment(self) -> None:
        """Scheduled-task entry point: drain the pending-artist queue, paced under MB's limit."""
        if not self.get_config_value(CONF_ENRICH_ENABLED, DEFAULT_ENRICH_ENABLED, return_type=bool):
            LOGGER.debug("Skipping scheduled MusicBrainz enrichment: enrichment is disabled")
            return
        try:
            resolved = await self._enrich_pending(
                limit=GENOME_ENRICHMENT_BATCH_LIMIT,
                min_interval_seconds=GENOME_MB_ENRICHMENT_MIN_INTERVAL_SECONDS,
            )
        except Exception:
            # a background pass must never crash the task loop over a transient MB problem -
            # the next hourly run tries again on its own
            LOGGER.warning("Scheduled MusicBrainz enrichment pass failed", exc_info=True)
            return
        if resolved:
            LOGGER.info("Scheduled MusicBrainz enrichment: %d artists resolved", resolved)

    async def _background_enrichment(self) -> None:
        """
        Run one enrichment pass detached from a request, paced under MusicBrainz's limit.

        Shares :attr:`_enrichment_lock` with the hourly pass, so a user pressing Rebuild
        while the scheduled pass is running queues behind it instead of doubling the request
        rate into MusicBrainz's rate limiter.
        """
        try:
            resolved = await self._enrich_pending(
                limit=GENOME_ENRICHMENT_BATCH_LIMIT,
                min_interval_seconds=GENOME_MB_ENRICHMENT_MIN_INTERVAL_SECONDS,
            )
        except Exception:
            LOGGER.warning("Background MusicBrainz enrichment pass failed", exc_info=True)
            return
        if resolved:
            LOGGER.info("Background MusicBrainz enrichment: %d artists resolved", resolved)

    def _register_discovery_task(self) -> None:
        """
        Register the background discovery pass (D-16).

        Mirrors :meth:`_register_enrichment_task`: the only place discovery is allowed to touch
        the network is a paced background pass, so that the read path can stay a pure
        cache/database lookup. The cadence is deliberately slow
        (:data:`GENOME_DISCOVERY_REFRESH_INTERVAL_HOURS`) - similar-artist graphs barely move,
        and the seeds only change when the household's own top artists do.
        """
        from music_assistant_models.background_task import TaskSchedule  # noqa: PLC0415

        self.mass.tasks.register_scheduled_task(
            task_id=GENOME_DISCOVERY_TASK_ID,
            name="Genome discovery",
            handler=self._scheduled_discovery,
            schedule=TaskSchedule.hourly(every=max(GENOME_DISCOVERY_REFRESH_INTERVAL_HOURS, 1)),
            translation_key=GENOME_DISCOVERY_TASK_ID,
            translation_owner=self.translation_owner,
            metadata={"task_domain": GENOME_DISCOVERY_TASK_ID},
            allow_retry=True,
        )

    async def _scheduled_discovery(self) -> None:
        """Scheduled-task entry point: refresh the Last.fm similar-artist suggestions."""
        await self._background_discovery(LISTENER_HOUSEHOLD)

    async def _background_discovery(self, listener: str = LISTENER_HOUSEHOLD) -> None:
        """
        Run one discovery pass detached from any request, never raising.

        A Last.fm problem must not crash the task loop or leave the stored result in a
        half-written state: the pass writes its blob once, at the end, and the next run tries
        again on its own.

        :param listener: The listener id (``"household"`` in v1).
        """
        try:
            async with self._discovery_lock:
                await self._run_discovery_pass(listener)
        except Exception:
            LOGGER.warning("Background discovery pass failed", exc_info=True)

    async def _run_discovery_pass(self, listener: str) -> None:
        """
        Fetch Last.fm similar artists for the household's divergent-genre seeds, then store them.

        Held to the same three rules as the MusicBrainz pass: per-seed pacing
        (:data:`GENOME_DISCOVERY_LASTFM_MIN_INTERVAL_SECONDS`), a persisted result so a restart
        does not lose the work, and a recorded cooldown on failure
        (:data:`GENOME_DISCOVERY_SEED_ERROR_COOLDOWN_HOURS`) so a seed Last.fm can never answer
        is not retried on every pass forever - the failure mode that once left a progress notice
        stuck for days.

        The blob is written even when the pass produced nothing, so ``suggested_state`` reports
        "ready with no suggestions" rather than sitting at "pending" indefinitely.

        :param listener: The listener id (``"household"`` in v1).
        """
        from .enrich.lastfm_similar import fetch_similar_artists  # noqa: PLC0415

        api_key = self.get_config_value(
            CONF_LASTFM_API_KEY, DEFAULT_LASTFM_API_KEY, return_type=str
        ).strip()
        if not api_key:
            LOGGER.debug("Skipping discovery pass: no Last.fm API key configured")
            return
        genome = await self.store.get_cached_genome(listener)
        genres = divergent_genres(genome)
        seeds = select_seeds(genome, genres, limit=GENOME_DISCOVERY_SEED_LIMIT)
        cached = await self.store.get_cached_discovery(listener) or {}
        failures = _seed_failures(cached)
        now = time.time()
        cooldown = GENOME_DISCOVERY_SEED_ERROR_COOLDOWN_HOURS * 3600
        excluded = await self._known_artist_keys(listener)
        client = self._lastfm_similar_client_factory(self.mass)
        best: dict[str, SuggestedArtist] = {}
        attempted = 0
        last_call = 0.0
        for seed in seeds:
            if now - failures.get(seed.artist_key, 0.0) < cooldown:
                LOGGER.debug("Discovery: seed %r still in error cooldown", seed.artist_name)
                continue
            if attempted:
                wait = GENOME_DISCOVERY_LASTFM_MIN_INTERVAL_SECONDS - (time.monotonic() - last_call)
                if wait > 0:
                    await asyncio.sleep(wait)
            last_call = time.monotonic()
            attempted += 1
            try:
                similar = await fetch_similar_artists(
                    seed.artist_name,
                    client=client,
                    api_key=api_key,
                    limit=GENOME_DISCOVERY_SIMILAR_PER_SEED,
                )
            except Exception as err:
                # never str(err) on the raw transport exception: the request URL carries the
                # API key as a query parameter (see importers/lastfm.py::describe_fetch_error)
                LOGGER.info("Discovery: seed %r failed: %s", seed.artist_name, err)
                failures[seed.artist_key] = now
                continue
            failures.pop(seed.artist_key, None)
            for entry in similar:
                key = create_safe_string(entry.name)
                if not key or key in excluded:
                    continue
                existing = best.get(key)
                if existing is not None and existing.match >= entry.match:
                    continue
                best[key] = SuggestedArtist(
                    artist_name=entry.name,
                    mbid=entry.mbid,
                    seed_artist=seed.artist_name,
                    genre_key=seed.genre.key,
                    genre_label=seed.genre.label,
                    match=entry.match,
                )
        suggested = sorted(best.values(), key=lambda row: (-row.match, row.artist_name.casefold()))
        suggested = suggested[:GENOME_DISCOVERY_SUGGESTED_LIMIT]
        # a seed that is gone from the genome entirely must not keep a cooldown row alive
        seed_keys = {seed.artist_key for seed in seeds}
        await self.store.set_cached_discovery(
            listener,
            {
                "generated_at": now,
                "suggested": [row.to_dict() for row in suggested],
                "seed_failures": {k: v for k, v in failures.items() if k in seed_keys},
            },
        )
        LOGGER.info(
            "Discovery pass finished: %d seed(s) queried, %d suggestion(s) stored, %d seed(s) failing",
            attempted,
            len(suggested),
            len(failures),
        )

    async def _cold_corners(self, listener: str, genres: Sequence[Any]) -> list[ColdArtist]:
        """
        Rank the library's barely-played artists against the household's divergent genres.

        Local reads only (the MA library database and ``genome.db``). A failure on either is
        logged and treated as "no cold corners": this feeds a discovery card, and an empty card
        is a better outcome than an API command that raises at a user.

        :param listener: The listener id (``"household"`` in v1).
        :param genres: The divergent genres to rank against.
        """
        if not genres:
            return []
        try:
            library_artists = await self._library_artists_reader(self.mass)
            extra_plays = await self.store.artist_play_counts(listener)
        except Exception:
            LOGGER.warning("Could not read the library for discovery cold corners", exc_info=True)
            return []
        return rank_cold_corners(
            library_artists,
            genres,
            max_plays=GENOME_DISCOVERY_COLD_MAX_PLAYS,
            limit=GENOME_DISCOVERY_COLD_LIMIT,
            extra_plays=extra_plays,
        )

    async def _known_artist_keys(self, listener: str) -> set[str]:
        """Return every artist key already in the library or in the stored listen history."""
        keys: set[str] = set()
        try:
            keys.update(
                artist.artist_key for artist in await self._library_artists_reader(self.mass)
            )
            keys.update(await self.store.artist_play_counts(listener))
        except Exception:
            # a suggestion list that is not filtered is worse than none at all (it would
            # recommend artists the household already owns), so fail the pass rather than
            # store an unfiltered result
            LOGGER.warning("Could not read known artists for discovery filtering", exc_info=True)
            raise
        return keys

    async def _run_ma_backfill(self, playlog_importer: Any) -> None:
        """
        Run the one-time MA playlog/tracks backfill in the background, then mark it done.

        Best-effort: a failure here must never crash server startup, and is not retried until
        the next restart (the backfill is a coarse prior only; live capture keeps working
        regardless).

        :param playlog_importer: The ``MaPlaylogImporter`` returned by the playlog importer
            factory that :meth:`setup` already attached.
        """
        min_seconds = self.get_config_value(
            CONF_MIN_SECONDS_PLAYED, DEFAULT_MIN_SECONDS_PLAYED, return_type=int
        )
        try:
            await playlog_importer.backfill(min_seconds_played=min_seconds)
        except Exception:
            LOGGER.warning("MA playlog backfill failed", exc_info=True)
        finally:
            await self.store.mark_backfill_done()

    async def _rebuild(self, listener: str) -> GenomeRebuildResult:
        """
        Do the actual rebuild work shared by :meth:`rebuild` and :meth:`get_genome`.

        Pure recomputation over what the store already holds: no network call happens here,
        by construction. Enrichment is a separate, background concern
        (:meth:`_background_enrichment`, :meth:`_scheduled_enrichment`) precisely so that
        nothing a user is waiting on can end up blocked behind MusicBrainz's rate limiter.
        """
        start = time.monotonic()
        listens = [listen async for listen in self.store.iter_listens(listener)]
        artist_keys = sorted({listen.artist_key for listen in listens})
        artist_meta = await self.store.get_artist_meta(artist_keys)
        player_names = await self.store.player_names()
        params = EngineParams(
            now=int(utc_timestamp()),
            half_life_days=self.get_config_value(
                CONF_RECENCY_HALF_LIFE_DAYS, DEFAULT_HALF_LIFE_DAYS, return_type=int
            ),
            obscurity_percentile=self.get_config_value(
                CONF_OBSCURITY_PERCENTILE, DEFAULT_OBSCURITY_PERCENTILE, return_type=int
            ),
            min_seconds_played=self.get_config_value(
                CONF_MIN_SECONDS_PLAYED, DEFAULT_MIN_SECONDS_PLAYED, return_type=int
            ),
            top_n=DEFAULT_TOP_N,
            new_artist_window_days=DEFAULT_NEW_ARTIST_WINDOW_DAYS,
        )
        inputs = GenomeInputs(
            listener=listener,
            listens=listens,
            artist_meta=artist_meta,
            baseline=self.baseline,
            params=params,
            player_names=player_names,
        )
        genome = build_genome(inputs, tz_offset_seconds=self._tz_offset_seconds())
        await self._apply_resolution_counts(genome)
        await self.store.set_cached_genome(listener, genome)
        self.last_rebuild_at = params.now
        duration_ms = int((time.monotonic() - start) * 1000)
        LOGGER.info(
            "Genome rebuilt for %s: %d listens, %d%% divergence, %dms",
            listener,
            len(listens),
            genome["divergence"]["percent"],
            duration_ms,
        )
        return GenomeRebuildResult(
            listener=listener,
            listens_scanned=len(listens),
            duration_ms=duration_ms,
            genome=genome,
        )

    async def _apply_resolution_counts(self, genome: GenomeResult) -> None:
        """Overwrite the engine's placeholder ``artists_pending``/``artists_resolved`` with real counts."""
        try:
            counts = await self.store.artist_resolution_counts()
        except Exception:  # pragma: no cover - defensive, must never fail a rebuild
            LOGGER.debug("Could not read artist resolution counts", exc_info=True)
            return
        genome["stats"]["artists_pending"] = counts.get(RESOLVE_STATE_PENDING, 0)
        genome["stats"]["artists_failed"] = counts.get(RESOLVE_STATE_ERROR, 0)
        genome["stats"]["artists_resolved"] = counts.get(RESOLVE_STATE_OK, 0) + counts.get(
            RESOLVE_STATE_NOT_FOUND, 0
        )
        genome["stats"]["unresolved_dismissed"] = await self._unresolved_dismissed()

    async def _unresolved_dismissed(self) -> bool:
        """Return whether the currently-failed artist set exactly matches the dismissed one."""
        try:
            dismissed = await self.store.unresolved_dismissed_keys()
            if dismissed is None:
                return False
            return dismissed == await self.store.all_failed_artist_keys()
        except Exception:  # pragma: no cover - defensive, must never fail a rebuild
            LOGGER.debug("Could not compute unresolved_dismissed", exc_info=True)
            return False

    async def _enrich_pending(self, *, limit: int = 200, min_interval_seconds: float = 0.0) -> int:
        """
        Resolve pending artist metadata via MusicBrainz, then backfill ListenBrainz popularity.

        Enrichment must never fail a rebuild: any problem (including the enrichment modules
        not being available yet in this checkout) is logged and treated as "0 enriched". The
        MusicBrainz pass and the popularity backfill (:meth:`_drain_popularity_backlog`) are
        independent: the backlog is every artist with an ``mbid`` but no ``lb_listeners`` yet,
        not just the ones this call's MusicBrainz pass happened to resolve, so a request whose
        ListenBrainz lookup previously failed - or that got its ``mbid`` before this backfill
        existed - still gets picked up here (§3.8, docs/STATUS.md "Contract gaps").

        Serialized against the continuous background enrichment pass (:attr:`_enrichment_lock`)
        so a daily rebuild and the hourly background pass never hammer MusicBrainz/ListenBrainz
        at once.

        :param limit: The maximum number of pending artists to resolve via MusicBrainz, and
            separately the maximum number of popularity-backlog artists to look up, in this pass.
        :param min_interval_seconds: Minimum spacing between per-artist MusicBrainz lookups
            (P3) - ``0`` (the default, used by an on-demand rebuild) leaves pacing entirely to
            the shared, already-throttled MusicBrainz client.
        """
        try:
            from music_assistant.controllers.genome.enrich.musicbrainz import (  # noqa: PLC0415
                enrich_pending_artists,
            )
            from music_assistant.controllers.genome.http import AiohttpClient  # noqa: PLC0415
        except ImportError:
            LOGGER.debug("Genome enrichment modules not available yet; skipping enrichment")
            return 0

        async with self._enrichment_lock:
            pending = await self.store.pending_artist_keys(limit=limit)
            resolved = 0
            if pending:
                mb_client = AiohttpClient(self.mass, rate_limit=10, period=10)
                resolved = await enrich_pending_artists(
                    self.store,
                    client=mb_client,
                    mass=self.mass,
                    limit=limit,
                    min_interval_seconds=min_interval_seconds,
                )
            await self._drain_popularity_backlog(limit=limit)
            return resolved

    async def _drain_popularity_backlog(self, *, limit: int) -> None:
        """
        Backfill ListenBrainz popularity for artists that have an ``mbid`` but no ``lb_listeners``.

        Must be called with :attr:`_enrichment_lock` already held. Never raises: a ListenBrainz
        problem is logged and leaves the backlog untouched for the next pass to retry. An artist
        this pass looks up but ListenBrainz has nothing for is not simply left in place -
        :meth:`_GenomeStoreProtocol.mark_popularity_attempted` bumps it to the back of the same
        queue (:meth:`_GenomeStoreProtocol.pending_popularity_keys`), so it does not get retried
        every single pass forever.

        :param limit: The maximum number of backlog artists to look up in this pass.
        """
        try:
            from music_assistant.controllers.genome.enrich.listenbrainz import (  # noqa: PLC0415
                artist_popularity,
            )
            from music_assistant.controllers.genome.http import AiohttpClient  # noqa: PLC0415
        except ImportError:
            LOGGER.debug(
                "Genome enrichment modules not available yet; skipping popularity backfill"
            )
            return

        backlog = await self.store.pending_popularity_keys(limit=limit)
        if not backlog:
            return
        mbid_by_key = dict(backlog)
        lb_client = AiohttpClient(self.mass, rate_limit=1, period=1.0)
        try:
            popularity = await artist_popularity(list({*mbid_by_key.values()}), client=lb_client)
        except Exception:
            LOGGER.warning("ListenBrainz popularity backfill failed", exc_info=True)
            return
        updates = {
            artist_key: (pop.listeners, pop.listen_count)
            for artist_key, mbid in mbid_by_key.items()
            if (pop := popularity.get(mbid)) is not None
        }
        if updates:
            await self.store.update_lb_popularity(updates)
        still_missing = [key for key in mbid_by_key if key not in updates]
        if still_missing:
            await self.store.mark_popularity_attempted(still_missing)
        LOGGER.info(
            "ListenBrainz popularity backfill: %d/%d backlog artists updated, %d still unknown",
            len(updates),
            len(mbid_by_key),
            len(still_missing),
        )

    async def _ingest_apple_csv(self, path: str) -> GenomeImportResult:
        """Parse an Apple Music export CSV and store the resulting listens."""
        from .importers.apple_csv import ApplePlayActivityStats  # noqa: PLC0415

        min_seconds = self.get_config_value(
            CONF_MIN_SECONDS_PLAYED, DEFAULT_MIN_SECONDS_PLAYED, return_type=int
        )
        LOGGER.info("Apple Music CSV import starting: %s", path)
        stats = ApplePlayActivityStats()
        listens = [
            listen
            async for listen in self._apple_parser(path, min_seconds=min_seconds, stats=stats)
        ]
        result = await self.store.add_listens(listens, listener=LISTENER_HOUSEHOLD)
        # the store only ever sees the rows that survived parsing (`listens`), so its own
        # rows_read/rows_skipped/warnings would silently hide anything the CSV parser itself
        # filtered out (bad rows, missing columns, below min_seconds); the parser's own
        # ApplePlayActivityStats sidecar is the source of truth for those (docs/STATUS.md).
        result["rows_read"] = stats.rows_read
        result["rows_skipped"] = stats.rows_skipped
        stats.summarise()
        result["warnings"] = list(stats.warnings)
        LOGGER.info(
            "Apple Music CSV import finished: %d rows read, %d imported, %d skipped, %d duplicate",
            result["rows_read"],
            result["rows_imported"],
            result["rows_skipped"],
            result["rows_duplicate"],
        )
        # The counts alone cannot distinguish "this export holds nothing we want" from "we did
        # not understand this file". The warnings carry that distinction, and the add-on log is
        # where anyone debugging an import actually looks - so they have to be written here and
        # not only handed back to the page that requested the import.
        for warning in result["warnings"]:
            LOGGER.warning("Apple Music CSV import: %s", warning)
        return result

    async def _append_upload_chunk(self, upload_id: str, seq: int, chunk_b64: str) -> None:
        """Validate and append one base64 chunk to an in-progress upload (§3.3)."""
        if len(chunk_b64) > GENOME_UPLOAD_CHUNK_MAX_B64_BYTES:
            self._uploads.pop(upload_id, None)
            msg = f"Upload chunk exceeds the {GENOME_UPLOAD_CHUNK_MAX_B64_BYTES} byte limit"
            raise InvalidDataError(msg)

        state = self._uploads.get(upload_id)
        if state is None:
            if seq != 0:
                msg = f"Unknown upload {upload_id!r}; expected the first chunk (seq=0), got {seq}"
                raise InvalidDataError(msg)
            state = _UploadState()
            self._uploads[upload_id] = state
            self.mass.call_later(
                GENOME_UPLOAD_TTL_SECONDS,
                self._sweep_stale_upload,
                upload_id,
                task_id=f"genome_upload_ttl_{upload_id}",
            )
        elif seq != state.next_seq:
            self._uploads.pop(upload_id, None)
            msg = f"Out-of-order chunk for upload {upload_id!r}: expected seq={state.next_seq}, got {seq}"
            raise InvalidDataError(msg)

        raw = base64.b64decode(chunk_b64) if chunk_b64 else b""
        state.total_bytes += len(raw)
        if state.total_bytes > GENOME_UPLOAD_MAX_TOTAL_BYTES:
            self._uploads.pop(upload_id, None)
            msg = f"Upload {upload_id!r} exceeds the {GENOME_UPLOAD_MAX_TOTAL_BYTES} byte limit"
            raise InvalidDataError(msg)

        path = self._upload_path(upload_id)
        await self._append_bytes(path, raw)
        state.next_seq += 1

    async def _append_bytes(self, path: str, raw: bytes) -> None:
        """Append ``raw`` bytes to ``path`` off the event loop, creating the directory first."""

        def _write() -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "ab") as handle:
                handle.write(raw)

        await asyncio.to_thread(_write)

    async def _sweep_stale_upload(self, upload_id: str) -> None:
        """Drop an upload's partial state/file if it never received a final chunk in time."""
        if upload_id not in self._uploads:
            return
        await self._cleanup_upload(upload_id)

    async def _cleanup_upload(self, upload_id: str) -> None:
        """Remove an upload's in-memory state and partial file, if any."""
        self._uploads.pop(upload_id, None)
        path = self._upload_path(upload_id)

        def _remove() -> None:
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)  # noqa: PTH107 - genuinely off-thread

        await asyncio.to_thread(_remove)

    def _upload_path(self, upload_id: str) -> str:
        """Return the on-disk path for an in-progress upload's partial file."""
        return os.path.join(self.mass.storage_path, GENOME_UPLOADS_DIRNAME, upload_id)

    def _tz_offset_seconds(self) -> int:
        """Return the server's current local UTC offset, in seconds."""
        offset = datetime.now(LOCAL_TIMEZONE).utcoffset()
        return int(offset.total_seconds()) if offset is not None else 0


def _float_or_none(value: Any) -> float | None:
    """Coerce a stored timestamp to a float, treating anything unusable as "never run"."""
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _seed_failures(cached: Mapping[str, Any]) -> dict[str, float]:
    """Read the ``{seed_artist_key: failed_at}`` cooldown map out of a stored discovery blob."""
    raw = cached.get("seed_failures")
    if not isinstance(raw, dict):
        return {}
    failures: dict[str, float] = {}
    for key, value in raw.items():
        failed_at = _float_or_none(value)
        if isinstance(key, str) and failed_at is not None:
            failures[key] = failed_at
    return failures


def _suggested_from_cache(cached: Mapping[str, Any] | None) -> list[SuggestedArtist]:
    """Rehydrate the stored suggestion rows, dropping any that no longer match the model."""
    if not cached:
        return []
    rows = cached.get("suggested")
    if not isinstance(rows, list):
        return []
    suggested: list[SuggestedArtist] = []
    for row in rows:
        try:
            suggested.append(SuggestedArtist.from_dict(row))
        except Exception:  # pragma: no cover - defensive, malformed stored row
            LOGGER.debug("Skipping unreadable stored discovery suggestion", exc_info=True)
    return suggested


def _empty_import_result(source: str) -> GenomeImportResult:
    """Return a zeroed :class:`GenomeImportResult` (e.g. a non-final upload chunk)."""
    return GenomeImportResult(
        source=source,
        rows_read=0,
        rows_imported=0,
        rows_skipped=0,
        rows_duplicate=0,
        first_played_at=None,
        last_played_at=None,
        warnings=[],
    )


__all__ = ["GenomeController"]
