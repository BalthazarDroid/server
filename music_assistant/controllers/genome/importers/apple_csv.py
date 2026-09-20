"""
Apple Music Play Activity CSV importer (§3.8).

Parses the ``Apple Music Play Activity.csv`` file from a privacy.apple.com Apple Media Services
export. The file is streamed row by row (never read in full) through a background thread so the
event loop is never blocked on file IO (``check_blocking_io.py``), and every row is decoded
independently so a handful of malformed rows do not abort the whole import.
"""

from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import chardet
from music_assistant_models.helpers import create_safe_string

from music_assistant.controllers.genome.constants import SOURCE_APPLE_EXPORT
from music_assistant.controllers.genome.models import GenomeImportResult, Listen
from music_assistant.helpers.util import parse_title_and_version

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from music_assistant.controllers.genome.store import GenomeStore

# canonical column names this parser understands, exactly as the export spells them (§3.9 header)
_CANONICAL_COLUMNS: tuple[str, ...] = (
    "Event Start Timestamp",
    "Event End Timestamp",
    "Play Duration Milliseconds",
    "Media Duration In Milliseconds",
    "Song Name",
    "Artist Name",
    "Album Name",
    "Container Name",
    "End Reason Type",
    "Event Type",
    "Media Type",
    "Feature Name",
    "Event Reason Hint Type",
    "Source Type",
    "Play Count",
)

# alternate header spellings seen across export vintages, mapped to the canonical name above
_HEADER_ALIASES: dict[str, str] = {
    "track name": "Song Name",
    "content name": "Song Name",
    "event timestamp": "Event Start Timestamp",
    "play duration ms": "Play Duration Milliseconds",
}

# Without these three there is nothing to import: no title, no artist, no time. Their absence
# means the header was not understood, which is a different problem from a row being filtered
# and has to be reported differently - a bare "306140 skipped" says nothing at all.
_ESSENTIAL_COLUMNS: tuple[str, ...] = (
    "Song Name",
    "Artist Name",
    "Event Start Timestamp",
)

_NATURAL_END = "NATURAL_END_OF_TRACK"
_SKIPPED_END_REASONS = frozenset({"NOT_APPLICABLE", "FAILED_TO_LOAD"})
_ACCEPTED_EVENT_TYPES = frozenset({"PLAY_END", "play", ""})

# a producer/consumer queue bridges the blocking csv.DictReader (run in a thread) and the async
# generator this module exposes; a bounded size keeps memory flat on a very large export
_QUEUE_MAXSIZE = 256

_DONE = object()


@dataclass(slots=True, frozen=True)
class _Skipped:
    """Why one row was filtered out, so the totals can explain themselves."""

    reason: str


@dataclass(slots=True)
class ApplePlayActivityStats:
    """Mutable row-accounting sidecar for :func:`parse_play_activity`."""

    rows_read: int = 0
    rows_skipped: int = 0
    warnings: list[str] = field(default_factory=list)
    #: Skip reason -> count. An import that adds nothing is otherwise indistinguishable from
    #: an import that read a file it did not understand.
    skip_reasons: dict[str, int] = field(default_factory=dict)

    def note_skip(self, reason: str) -> None:
        """
        Record one skipped row under ``reason``.

        :param reason: Short, stable identifier for why the row was filtered.
        """
        self.rows_skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1


async def parse_play_activity(
    path: str,
    *,
    min_seconds: int,
    stats: ApplePlayActivityStats | None = None,
) -> AsyncIterator[Listen]:
    """
    Stream :class:`Listen` rows out of an Apple Music Play Activity CSV.

    :param path: Filesystem path to the CSV file.
    :param min_seconds: A row must clear this many seconds played (or be a natural end-of-track)
        to be imported.
    :param stats: Optional row-accounting sidecar, filled in as parsing proceeds.
    """
    stats = stats if stats is not None else ApplePlayActivityStats()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Listen | object] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)

    def producer() -> None:
        try:
            with _open_csv(path) as csv_file:
                reader = csv.DictReader(csv_file)
                headers = list(reader.fieldnames or ())
                field_map = _build_field_map(headers)
                # Checked once, up front. Every lookup tolerates a missing column by returning
                # an empty string, so an unrecognised header does not raise - it quietly fails
                # every row's "has a title and an artist" test and reports the whole file as
                # skipped, which looks like the data's fault rather than the parser's.
                missing = [c for c in _ESSENTIAL_COLUMNS if c not in field_map]
                if missing:
                    stats.warnings.append(
                        "This file's header was not recognised: no column matched "
                        + ", ".join(repr(c) for c in missing)
                        + ". Columns found: "
                        + ", ".join(repr(h) for h in headers[:40])
                        + ("..." if len(headers) > 40 else "")
                    )
                for row in reader:
                    stats.rows_read += 1
                    try:
                        listen = _parse_row(row, field_map, min_seconds=min_seconds)
                    except Exception as err:
                        stats.note_skip(f"row error: {type(err).__name__}")
                        stats.warnings.append(f"row {stats.rows_read}: {err}")
                        continue
                    if isinstance(listen, _Skipped):
                        stats.note_skip(listen.reason)
                        continue
                    loop.call_soon_threadsafe(queue.put_nowait, listen)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    producer_task = asyncio.create_task(asyncio.to_thread(producer))
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            yield item  # type: ignore[misc]
    finally:
        await producer_task


async def import_play_activity(
    store: GenomeStore,
    path: str,
    *,
    listener: str,
    min_seconds: int,
    batch_size: int = 500,
) -> GenomeImportResult:
    """
    Parse an Apple Music Play Activity CSV and store every valid listen.

    :param store: The :class:`GenomeStore` to write listens into.
    :param path: Filesystem path to the CSV file.
    :param listener: The listener partition to attribute these listens to.
    :param min_seconds: A row must clear this many seconds played to be imported (§3.2
        ``min_seconds_played``).
    :param batch_size: How many listens to buffer per :meth:`GenomeStore.add_listens` call.
    """
    stats = ApplePlayActivityStats()
    result: GenomeImportResult = {
        "source": SOURCE_APPLE_EXPORT,
        "rows_read": 0,
        "rows_imported": 0,
        "rows_skipped": 0,
        "rows_duplicate": 0,
        "first_played_at": None,
        "last_played_at": None,
        "warnings": [],
    }
    batch: list[Listen] = []

    async def flush() -> None:
        if not batch:
            return
        batch_result = await store.add_listens(batch, listener=listener)
        result["rows_imported"] += batch_result["rows_imported"]
        result["rows_duplicate"] += batch_result["rows_duplicate"]
        for key in ("first_played_at", "last_played_at"):
            value = batch_result[key]
            if value is None:
                continue
            current = result[key]
            if (
                current is None
                or (key == "first_played_at" and value < current)
                or (key == "last_played_at" and value > current)
            ):
                result[key] = value
        batch.clear()

    async for listen in parse_play_activity(path, min_seconds=min_seconds, stats=stats):
        batch.append(listen)
        if len(batch) >= batch_size:
            await flush()
    await flush()

    result["rows_read"] = stats.rows_read
    result["rows_skipped"] = stats.rows_skipped
    # An import that added nothing has to say why. The reasons are ranked because one usually
    # dominates, and that one is the answer.
    if stats.skip_reasons:
        ranked = sorted(stats.skip_reasons.items(), key=lambda kv: -kv[1])
        stats.warnings.append(
            "Skipped rows by reason: "
            + ", ".join(f"{reason} ({count})" for reason, count in ranked[:6])
        )
    result["warnings"] = stats.warnings
    return result


def _open_csv(path: str) -> Any:
    """
    Open the export for reading, tolerating an encoding other than UTF-8.

    Apple's export is UTF-8 with an optional BOM; a handful of very old exports are not.
    ``utf-8-sig`` is tried first (cheap, handles the common case including the BOM); on a
    decode error the file is re-opened at a ``chardet``-detected encoding.

    :param path: Filesystem path to the CSV file.
    """
    try:
        candidate = open(path, newline="", encoding="utf-8-sig")  # noqa: SIM115
        candidate.read(4096)
        candidate.seek(0)
    except UnicodeDecodeError:
        with open(path, "rb") as raw:
            sample = raw.read(65536)
        detected = chardet.detect(sample).get("encoding") or "utf-8"
        return open(path, newline="", encoding=detected, errors="replace")
    else:
        return candidate


def _build_field_map(fieldnames: Sequence[str]) -> dict[str, str]:
    """Map canonical column names to the actual header text present in this file's header row."""
    canonical_lookup = {name.lower(): name for name in _CANONICAL_COLUMNS}
    field_map: dict[str, str] = {}
    for header in fieldnames:
        key = header.strip().lower()
        canonical = _HEADER_ALIASES.get(key) or canonical_lookup.get(key)
        if canonical:
            field_map[canonical] = header
    return field_map


def _get(row: dict[str, str], field_map: dict[str, str], canonical: str) -> str:
    """Read a canonical column's raw value from a CSV row, tolerating a missing column."""
    header = field_map.get(canonical)
    if header is None:
        return ""
    return (row.get(header) or "").strip()


def _parse_row(
    row: dict[str, str], field_map: dict[str, str], *, min_seconds: int
) -> Listen | _Skipped:
    """Return a :class:`Listen` for one CSV row, or a :class:`_Skipped` saying why not."""
    media_type = _get(row, field_map, "Media Type").upper()
    if media_type and "AUDIO" not in media_type:
        return _Skipped("not audio")
    event_type = _get(row, field_map, "Event Type")
    if event_type not in _ACCEPTED_EVENT_TYPES:
        return _Skipped(f"event type {event_type!r}")
    end_reason = _get(row, field_map, "End Reason Type")
    if end_reason in _SKIPPED_END_REASONS:
        return _Skipped(f"end reason {end_reason!r}")
    song_name = _get(row, field_map, "Song Name")
    artist_name = _get(row, field_map, "Artist Name")
    if not song_name or not artist_name:
        return _Skipped("no song or artist name")
    play_duration_ms = _parse_int(_get(row, field_map, "Play Duration Milliseconds"))
    fully_played = end_reason == _NATURAL_END
    if not fully_played and (play_duration_ms is None or play_duration_ms < min_seconds * 1000):
        return _Skipped("played too briefly")
    timestamp_raw = _get(row, field_map, "Event Start Timestamp")
    if not timestamp_raw:
        return _Skipped("no timestamp")
    played_at = _parse_timestamp(timestamp_raw)
    title, _version = parse_title_and_version(song_name, strip_for_search=True)
    track_key = create_safe_string(title)
    artist_key = create_safe_string(artist_name)
    if not track_key or not artist_key:
        return _Skipped("name reduced to nothing")
    album_name = _get(row, field_map, "Album Name") or None
    return Listen(
        played_at=played_at,
        artist_key=artist_key,
        artist_name=artist_name,
        track_key=track_key,
        track_name=song_name,
        album_name=album_name,
        source=SOURCE_APPLE_EXPORT,
        player_id=None,
        duration_ms=_parse_int(_get(row, field_map, "Media Duration In Milliseconds")),
        played_ms=play_duration_ms,
        fully_played=fully_played,
        confidence=1.0,
    )


def _parse_timestamp(value: str) -> int:
    """Parse an Apple export ISO-8601 timestamp (``Z``-suffixed) to unix seconds."""
    dt = datetime.fromisoformat(value)
    return int(dt.timestamp())


def _parse_int(value: str) -> int | None:
    """Parse a numeric CSV field, tolerating blanks and non-numeric junk."""
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


__all__ = ["ApplePlayActivityStats", "import_play_activity", "parse_play_activity"]
