"""
Apple Music "Play History Daily Tracks" CSV importer.

This is the file the genome actually needs. ``Apple Music Play Activity.csv`` has precise
timestamps and, in the current export vintage, no per-track artist at all - every one of its
306,140 rows fails for want of a name. This file is the reverse: it carries the artist, in
``Track Description``, and keeps time to the hour rather than the second.

One row is one track on one day, with ``Play Count`` plays spread over the hours listed in
``Hours``. The plays are reconstructed from that rather than collapsed into one, because a play
count is the whole point of the profile - and they are spread across distinct minutes because
the store dedupes on ``(listener, source, artist, track, minute)`` and would otherwise keep one
of them.
"""

from __future__ import annotations

import asyncio
import csv
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from music_assistant_models.helpers import create_safe_string

from music_assistant.controllers.genome.constants import SOURCE_APPLE_EXPORT
from music_assistant.controllers.genome.importers.apple_csv import (
    ApplePlayActivityStats,
    _open_csv,
    _parse_int,
    _Skipped,
)
from music_assistant.controllers.genome.models import Listen
from music_assistant.helpers.util import parse_title_and_version

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# The columns this file is recognised by. "Track Description" is the one that matters: it is the
# only artist anywhere in the Apple Music export, and it is spelled "Artist - Song".
DAILY_TRACKS_COLUMNS: tuple[str, ...] = ("Date Played", "Hours", "Track Description")

_ARTIST_TITLE_SEPARATOR = " - "
_NATURAL_END = "NATURAL_END_OF_TRACK"

# A day's play count for one track, above which the row is treated as corrupt rather than as a
# day spent on one song. Apple has been seen to emit absurd counts for a stuck player.
_MAX_PLAYS_PER_DAY = 100

_QUEUE_MAXSIZE = 256
_MAX_ROW_WARNINGS = 8
_DONE = object()


def is_daily_tracks_header(headers: list[str]) -> bool:
    """
    Whether this header row is a Play History Daily Tracks export.

    :param headers: The header row exactly as the file spells it.
    """
    present = {h.strip().lower() for h in headers}
    return all(column.lower() in present for column in DAILY_TRACKS_COLUMNS)


async def parse_daily_tracks(
    path: str,
    *,
    min_seconds: int,
    stats: ApplePlayActivityStats | None = None,
) -> AsyncIterator[Listen]:
    """
    Stream :class:`Listen` rows out of an Apple Music Play History Daily Tracks CSV.

    :param path: Filesystem path to the CSV file.
    :param min_seconds: A play must clear this many seconds (or be a natural end-of-track) to be
        imported.
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
                if not is_daily_tracks_header(headers):
                    stats.warnings.append(
                        "This file is not a Play History Daily Tracks export: no column matched "
                        + ", ".join(repr(c) for c in DAILY_TRACKS_COLUMNS)
                        + ". Columns found: "
                        + ", ".join(repr(h) for h in headers[:40])
                        + ("..." if len(headers) > 40 else "")
                    )
                    return
                for row in reader:
                    stats.rows_read += 1
                    try:
                        listens = _parse_day(row, min_seconds=min_seconds)
                    except Exception as err:
                        stats.note_skip(f"row error: {type(err).__name__}")
                        if len(stats.warnings) < _MAX_ROW_WARNINGS:
                            stats.warnings.append(f"row {stats.rows_read}: {err}")
                        continue
                    if isinstance(listens, _Skipped):
                        stats.note_skip(listens.reason)
                        continue
                    for listen in listens:
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


def _get(row: dict[str, str], column: str) -> str:
    """Read a column's raw value, tolerating a spelling this file does not use."""
    return (row.get(column) or "").strip()


def _parse_day(row: dict[str, str], *, min_seconds: int) -> list[Listen] | _Skipped:
    """Expand one day-and-track row into the individual plays it stands for."""
    media_type = _get(row, "Media type").upper()
    if media_type and "AUDIO" not in media_type:
        return _Skipped("not audio")

    description = _get(row, "Track Description")
    if not description:
        return _Skipped("no track description")
    artist_name, _, track_name = description.partition(_ARTIST_TITLE_SEPARATOR)
    artist_name = artist_name.strip()
    track_name = track_name.strip()
    # Split on the FIRST separator only: an artist's name rarely contains " - " while a title
    # very often does ("Anthems - Live", "Reprise"), so everything after the first one is title.
    if not artist_name or not track_name:
        return _Skipped("track description has no 'Artist - Song' split")

    day = _parse_day_number(_get(row, "Date Played"))
    if day is None:
        return _Skipped("no date played")
    hours = _parse_hours(_get(row, "Hours"))
    if not hours:
        return _Skipped("no hours")

    play_count = _parse_int(_get(row, "Play Count")) or 1
    if play_count < 1:
        return _Skipped("no plays")
    if play_count > _MAX_PLAYS_PER_DAY:
        return _Skipped(f"implausible play count {play_count}")

    total_ms = _parse_int(_get(row, "Play Duration Milliseconds"))
    per_play_ms = total_ms // play_count if total_ms is not None else None
    fully_played = _get(row, "End Reason Type") == _NATURAL_END
    if not fully_played and (per_play_ms is None or per_play_ms < min_seconds * 1000):
        return _Skipped("played too briefly")

    title, _version = parse_title_and_version(track_name, strip_for_search=True)
    track_key = create_safe_string(title)
    artist_key = create_safe_string(artist_name)
    if not track_key or not artist_key:
        return _Skipped("name reduced to nothing")

    return [
        Listen(
            played_at=played_at,
            artist_key=artist_key,
            artist_name=artist_name,
            track_key=track_key,
            track_name=track_name,
            album_name=None,
            source=SOURCE_APPLE_EXPORT,
            player_id=None,
            duration_ms=None,
            played_ms=per_play_ms,
            fully_played=fully_played or None,
            # The day and hour are Apple's own; the minute is ours, invented to keep repeat
            # plays distinct. Say so, rather than claim the precision of a real timestamp.
            confidence=0.8,
        )
        for played_at in _play_times(day, hours, play_count)
    ]


def _play_times(day: datetime, hours: list[int], play_count: int) -> list[int]:
    """
    Place ``play_count`` plays into the hours Apple listed, at distinct minutes.

    Plays are dealt round-robin over the hours, so a count of 2 across hours 1 and 4 is one play
    in each rather than two in the first. Within an hour they are spread evenly, because the
    store dedupes to the minute and stacked plays would collapse into one.
    """
    per_hour: dict[int, int] = {}
    for i in range(play_count):
        hour = hours[i % len(hours)]
        per_hour[hour] = per_hour.get(hour, 0) + 1
    times: list[int] = []
    for hour, count in per_hour.items():
        step = 60 // count
        times.extend(
            int((day + timedelta(hours=hour, minutes=index * step)).timestamp())
            for index in range(count)
        )
    return sorted(times)


def _parse_day_number(value: str) -> datetime | None:
    """Parse this file's ``YYYYMMDD`` date into midnight UTC on that day."""
    if len(value) != 8 or not value.isdigit():
        return None
    try:
        return datetime(int(value[0:4]), int(value[4:6]), int(value[6:8]), tzinfo=UTC)
    except ValueError:
        return None


def _parse_hours(value: str) -> list[int]:
    """Parse this file's ``Hours`` column, which is a comma-separated list like ``1, 4``."""
    hours: list[int] = []
    for raw in value.split(","):
        part = raw.strip()
        if not part.isdigit():
            continue
        hour = int(part)
        if 0 <= hour <= 23 and hour not in hours:
            hours.append(hour)
    return hours


__all__ = ["DAILY_TRACKS_COLUMNS", "is_daily_tracks_header", "parse_daily_tracks"]
