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
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from music_assistant_models.auth import Scope
from music_assistant_models.config_entries import ConfigActionResult, ConfigEntry
from music_assistant_models.enums import ConfigEntryType
from music_assistant_models.errors import InvalidDataError

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
    GENOME_REBUILD_TASK_ID,
    GENOME_UPLOAD_CHUNK_MAX_B64_BYTES,
    GENOME_UPLOAD_MAX_TOTAL_BYTES,
    GENOME_UPLOAD_TTL_SECONDS,
    GENOME_UPLOADS_DIRNAME,
    LISTENER_HOUSEHOLD,
    LOGGER,
    SOURCE_APPLE_EXPORT,
)
from .engine import build_genome
from .models import (
    EngineParams,
    GenomeImportResult,
    GenomeInputs,
    GenomeRebuildResult,
    GenomeResult,
    GenomeSettings,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from music_assistant_models.config_entries import ConfigValueType, CoreConfig

    from music_assistant.mass import MusicAssistant

    from .models import ArtistMeta, Baseline, GenomeSettingsPatch, Listen


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


async def _default_apple_parser(path: str, *, min_seconds: int) -> AsyncIterator[Listen]:
    """Lazily delegate to the real Apple Music CSV parser (owned by a separate work package)."""
    from music_assistant.controllers.genome.importers.apple_csv import (  # noqa: PLC0415
        parse_play_activity,
    )

    async for listen in parse_play_activity(path, min_seconds=min_seconds):
        yield listen


def _default_lastfm_importer_factory(mass: MusicAssistant, *, username: str, api_key: str) -> Any:
    """Lazily construct the real Last.fm importer (owned by a separate work package)."""
    from music_assistant.controllers.genome.http import AiohttpClient  # noqa: PLC0415
    from music_assistant.controllers.genome.importers.lastfm import LastfmImporter  # noqa: PLC0415

    client = AiohttpClient(mass, rate_limit=5, period=1.0)
    return LastfmImporter(client, username, api_key)


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
        self.store: _GenomeStoreProtocol = store if store is not None else _create_default_store(mass)
        self.baseline: Baseline = uniform_baseline()
        self.last_rebuild_at: int | None = None
        self._uploads: dict[str, _UploadState] = {}
        self._apple_parser = _default_apple_parser
        self._lastfm_importer_factory = _default_lastfm_importer_factory

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
            await self.rebuild()
            return ConfigActionResult(translation_key=f"{CONF_ACTION_REBUILD_NOW}.result")
        if action == CONF_ACTION_CLEAR_GENOME_DATA:
            await self.store.clear()
            return ConfigActionResult(translation_key=f"{CONF_ACTION_CLEAR_GENOME_DATA}.result")
        return await super().handle_config_action(action)

    async def setup(self, config: CoreConfig) -> None:
        """Load the baseline, set up storage and register the scheduled rebuild (§1.10)."""
        self.baseline = await load_baseline()
        await self.store.setup()
        self._register_rebuild_task()

    async def close(self) -> None:
        """Close storage on server stop."""
        await self.store.close()

    async def update_config(self, config: CoreConfig, changed_keys: set[str]) -> None:
        """Re-register the scheduled rebuild when its hour changes; defer the rest to the base."""
        await super().update_config(config, changed_keys)
        if f"values/{CONF_REBUILD_SCHEDULE_HOUR}" in changed_keys:
            self._register_rebuild_task()

    @api_command("genome/get", required_scope=Scope.LIBRARY_READ)
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
        return (await self._rebuild(listener))["genome"]

    @api_command("genome/rebuild", required_scope=Scope.LIBRARY_MANAGE)
    async def rebuild(
        self, listener: str = LISTENER_HOUSEHOLD, enrich: bool = True
    ) -> GenomeRebuildResult:
        """
        Recompute and cache the genome for ``listener`` (§3.3).

        :param listener: The listener id (``"household"`` in v1).
        :param enrich: Resolve pending artist metadata (MusicBrainz/ListenBrainz) first.
        """
        return await self._rebuild(listener, enrich=enrich)

    @api_command("genome/import_apple", required_scope=Scope.LIBRARY_MANAGE)
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
    async def import_lastfm(self, username: str = "", max_pages: int = 0) -> GenomeImportResult:
        """
        Poll Last.fm's ``user.getRecentTracks`` and ingest the result (§3.3, §3.8).

        :param username: Overrides the configured ``lastfm_username`` when given.
        :param max_pages: Stop after this many pages; ``0`` means "until exhausted".
        """
        username = username or self.get_config_value(
            CONF_LASTFM_USERNAME, DEFAULT_LASTFM_USERNAME, return_type=str
        )
        api_key = self.get_config_value(CONF_LASTFM_API_KEY, DEFAULT_LASTFM_API_KEY, return_type=str)
        if not username or not api_key:
            msg = "Last.fm is not configured (username and API key are required)"
            raise InvalidDataError(msg)
        importer = self._lastfm_importer_factory(self.mass, username=username, api_key=api_key)

        listens: list[Listen] = []
        page = 1
        while True:
            lf_page = await importer.fetch_recent(page)
            listens.extend(getattr(lf_page, "listens", []))
            total_pages = getattr(lf_page, "total_pages", page)
            if (max_pages and page >= max_pages) or page >= total_pages:
                break
            page += 1
        return await self.store.add_listens(listens, listener=LISTENER_HOUSEHOLD)

    @api_command("genome/settings", required_scope=Scope.LIBRARY_READ)
    async def get_settings(self) -> GenomeSettings:
        """Return the current genome settings (§3.3). Never includes the Last.fm API key."""
        api_key = self.get_config_value(CONF_LASTFM_API_KEY, DEFAULT_LASTFM_API_KEY, return_type=str)
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
        values: dict[str, ConfigValueType] = {
            key_map[field]: value for field, value in settings.items() if field in key_map
        }
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

    async def _rebuild(self, listener: str, *, enrich: bool = True) -> GenomeRebuildResult:
        """Do the actual rebuild work shared by :meth:`rebuild` and :meth:`get_genome`."""
        start = time.monotonic()
        listens = [listen async for listen in self.store.iter_listens(listener)]
        artists_enriched = 0
        if enrich and self.get_config_value(CONF_ENRICH_ENABLED, DEFAULT_ENRICH_ENABLED, return_type=bool):
            artists_enriched = await self._enrich_pending()
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
        await self.store.set_cached_genome(listener, genome)
        self.last_rebuild_at = params.now
        duration_ms = int((time.monotonic() - start) * 1000)
        return GenomeRebuildResult(
            listener=listener,
            listens_scanned=len(listens),
            artists_enriched=artists_enriched,
            duration_ms=duration_ms,
            genome=genome,
        )

    async def _enrich_pending(self) -> int:
        """
        Resolve pending artist metadata via MusicBrainz/ListenBrainz, best-effort (§3.8).

        Enrichment must never fail a rebuild: any problem (including the enrichment modules
        not being available yet in this checkout) is logged and treated as "0 enriched".
        """
        try:
            from music_assistant.controllers.genome.enrich.listenbrainz import (  # noqa: PLC0415
                artist_popularity,
            )
            from music_assistant.controllers.genome.enrich.musicbrainz import (  # noqa: PLC0415
                resolve_artist,
            )
            from music_assistant.controllers.genome.http import AiohttpClient  # noqa: PLC0415
        except ImportError:
            LOGGER.debug("Genome enrichment modules not available yet; skipping enrichment")
            return 0

        pending = await self.store.pending_artist_keys()
        if not pending:
            return 0
        client = AiohttpClient(self.mass, rate_limit=10, period=10)
        resolved: list[ArtistMeta] = []
        for _artist_key, artist_name in pending:
            try:
                update = await resolve_artist(artist_name, client=client)
            except Exception:
                LOGGER.debug("MusicBrainz resolution failed for %s", artist_name, exc_info=True)
                continue
            if update is not None:
                resolved.append(update)
        if resolved:
            mbids = [meta.mbid for meta in resolved if meta.mbid]
            try:
                await artist_popularity(mbids, client=client)
            except Exception:
                LOGGER.debug("ListenBrainz popularity lookup failed", exc_info=True)
            await self.store.upsert_artist_meta(resolved, state="ok")
        return len(resolved)

    async def _ingest_apple_csv(self, path: str) -> GenomeImportResult:
        """Parse an Apple Music export CSV and store the resulting listens."""
        min_seconds = self.get_config_value(
            CONF_MIN_SECONDS_PLAYED, DEFAULT_MIN_SECONDS_PLAYED, return_type=int
        )
        listens = [listen async for listen in self._apple_parser(path, min_seconds=min_seconds)]
        return await self.store.add_listens(listens, listener=LISTENER_HOUSEHOLD)

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
            os.makedirs(os.path.dirname(path), exist_ok=True)
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
            try:
                os.remove(path)  # noqa: PTH107 - genuinely off-thread
            except FileNotFoundError:
                pass

        await asyncio.to_thread(_remove)

    def _upload_path(self, upload_id: str) -> str:
        """Return the on-disk path for an in-progress upload's partial file."""
        return os.path.join(self.mass.storage_path, GENOME_UPLOADS_DIRNAME, upload_id)

    def _tz_offset_seconds(self) -> int:
        """Return the server's current local UTC offset, in seconds."""
        offset = datetime.now(LOCAL_TIMEZONE).utcoffset()
        return int(offset.total_seconds()) if offset is not None else 0


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
