"""
Last.fm ``user.getRecentTracks`` importer (§3.8).

Paginates a public Last.fm profile's recent-tracks history through the injected
:class:`~music_assistant.controllers.genome.http.HttpClient`, resuming from the last imported
timestamp so a scheduled poll only fetches what is new.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from music_assistant_models.helpers import create_safe_string

from music_assistant.controllers.genome.constants import (
    LASTFM_BASE_URL,
    LASTFM_INTER_PAGE_DELAY_SECONDS,
    LASTFM_PAGE_LIMIT,
    LOGGER,
    SOURCE_LASTFM,
)
from music_assistant.controllers.genome.models import GenomeImportResult, Listen
from music_assistant.helpers.util import parse_title_and_version

if TYPE_CHECKING:
    from music_assistant.controllers.genome.controller import _GenomeStoreProtocol
    from music_assistant.controllers.genome.http import HttpClient


@dataclass(slots=True, frozen=True)
class LastfmPage:
    """One decoded page of ``user.getRecentTracks``."""

    listens: tuple[Listen, ...]
    page: int
    total_pages: int
    # total `track` entries in the page, including the now-playing entry and any that failed to
    # parse; `len(listens)` only counts the ones that became a usable Listen
    raw_count: int


class LastfmImporter:
    """Fetches and normalizes a public Last.fm profile's recent-tracks history."""

    def __init__(self, client: HttpClient, username: str, api_key: str) -> None:
        """
        Initialize the importer.

        :param client: The :class:`HttpClient` to issue requests through.
        :param username: The public Last.fm username to import.
        :param api_key: A Last.fm API key. Never logged, never echoed back to a caller.
        """
        self._client = client
        self._username = username
        self._api_key = api_key

    async def fetch_recent(
        self, page: int, limit: int = LASTFM_PAGE_LIMIT, from_ts: int | None = None
    ) -> LastfmPage:
        """
        Fetch and normalize a single page of recent tracks.

        :param page: The 1-based page number to fetch.
        :param limit: Tracks per page (Last.fm's own maximum is 200).
        :param from_ts: Only return scrobbles after this unix timestamp, when given.
        """
        params = {
            "method": "user.getRecentTracks",
            "user": self._username,
            "api_key": self._api_key,
            "format": "json",
            "limit": str(limit),
            "page": str(page),
        }
        if from_ts is not None:
            params["from"] = str(from_ts)
        data = await self._client.get_json(LASTFM_BASE_URL, params=params)
        return self._parse_page(data)

    async def import_since(
        self, store: _GenomeStoreProtocol, *, listener: str, max_pages: int = 0
    ) -> GenomeImportResult:
        """
        Import every scrobble newer than the most recently stored Last.fm listen.

        Pages from the most recent scrobble backwards (Last.fm's default order) and stops once a
        page contains nothing newer than the resume point, or ``max_pages`` is reached.

        :param store: The ``GenomeStore``-shaped object to read the resume point from and write
            into.
        :param listener: The listener partition to attribute these listens to.
        :param max_pages: Stop after this many pages; ``0`` means no limit.
        """
        resume_from = await self._resume_timestamp(store, listener)
        LOGGER.info(
            "Last.fm import starting for %s (resuming from %s)",
            self._username,
            resume_from or "the beginning",
        )
        result: GenomeImportResult = {
            "source": SOURCE_LASTFM,
            "rows_read": 0,
            "rows_imported": 0,
            "rows_skipped": 0,
            "rows_duplicate": 0,
            "first_played_at": None,
            "last_played_at": None,
            "warnings": [],
        }
        page_number = 1
        total_pages = 1
        while page_number <= total_pages:
            try:
                page = await self.fetch_recent(page_number, from_ts=resume_from or None)
            except Exception as err:
                LOGGER.warning("Last.fm import: page %d failed: %s", page_number, err)
                result["warnings"].append(f"page {page_number}: {err}")
                break
            total_pages = page.total_pages or 1
            LOGGER.debug(
                "Last.fm import: fetched page %d/%d (%d tracks)",
                page_number,
                total_pages,
                page.raw_count,
            )
            result["rows_read"] += page.raw_count
            result["rows_skipped"] += page.raw_count - len(page.listens)
            if page.listens:
                batch_result = await store.add_listens(list(page.listens), listener=listener)
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
            page_number += 1
            if max_pages and page_number > max_pages:
                break
            if page_number <= total_pages:
                await asyncio.sleep(LASTFM_INTER_PAGE_DELAY_SECONDS)
        LOGGER.info(
            "Last.fm import finished for %s: %d pages fetched, %d rows added, %d skipped, "
            "%d duplicate",
            self._username,
            page_number - 1,
            result["rows_imported"],
            result["rows_skipped"],
            result["rows_duplicate"],
        )
        return result

    async def _resume_timestamp(self, store: _GenomeStoreProtocol, listener: str) -> int:
        """Return the newest stored Last.fm ``played_at`` for ``listener``, or ``0``."""
        latest = 0
        async for listen in store.iter_listens(listener):
            if listen.source == SOURCE_LASTFM and listen.played_at > latest:
                latest = listen.played_at
        return latest

    def _parse_page(self, data: Any) -> LastfmPage:
        """Convert a raw ``user.getRecentTracks`` JSON payload into a :class:`LastfmPage`."""
        payload = data.get("recenttracks", {}) if isinstance(data, dict) else {}
        attrs = payload.get("@attr", {})
        page = int(attrs.get("page", 1) or 1)
        total_pages = int(attrs.get("totalPages", 1) or 1)
        raw_tracks = payload.get("track", [])
        listens: list[Listen] = []
        for track in raw_tracks:
            if track.get("@attr", {}).get("nowplaying") == "true":
                # the currently-playing track has no `date` and is not a completed listen
                continue
            date = track.get("date")
            if not date or "uts" not in date:
                continue
            try:
                played_at = int(date["uts"])
            except TypeError, ValueError:
                continue
            artist_name = track.get("artist", {}).get("#text", "")
            track_name = track.get("name", "")
            title, _version = parse_title_and_version(track_name, strip_for_search=True)
            artist_key = create_safe_string(artist_name)
            track_key = create_safe_string(title)
            if not artist_key or not track_key:
                continue
            album_name = track.get("album", {}).get("#text") or None
            listens.append(
                Listen(
                    played_at=played_at,
                    artist_key=artist_key,
                    artist_name=artist_name,
                    track_key=track_key,
                    track_name=track_name,
                    album_name=album_name,
                    source=SOURCE_LASTFM,
                    player_id=None,
                    duration_ms=None,
                    played_ms=None,
                    fully_played=True,
                    confidence=1.0,
                )
            )
        return LastfmPage(
            listens=tuple(listens),
            page=page,
            total_pages=total_pages,
            raw_count=len(raw_tracks),
        )


__all__ = ["LastfmImporter", "LastfmPage"]
