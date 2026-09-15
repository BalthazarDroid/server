"""
Last.fm ``user.getRecentTracks`` importer (§3.8).

Paginates a public Last.fm profile's recent-tracks history through the injected
:class:`~music_assistant.controllers.genome.http.HttpClient`, resuming from the last imported
timestamp so a scheduled poll only fetches what is new.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from music_assistant_models.helpers import create_safe_string

from music_assistant.controllers.genome.constants import (
    LASTFM_BASE_URL,
    LASTFM_INTER_PAGE_DELAY_SECONDS,
    LASTFM_PAGE_LIMIT,
    LASTFM_RETRY_BASE_DELAY_SECONDS,
    LASTFM_RETRY_MAX_ATTEMPTS,
    LASTFM_RETRY_MAX_DELAY_SECONDS,
    LOGGER,
    SOURCE_LASTFM,
)
from music_assistant.controllers.genome.errors import LastfmApiError
from music_assistant.controllers.genome.models import GenomeImportResult, Listen
from music_assistant.helpers.util import parse_title_and_version

# A 429/500/502/503/504 is Last.fm (or the network between us and it) having a bad moment, not
# a reason to give up on the whole import - retried with backoff (P2). A network-level failure
# with no HTTP status at all (timeout, dropped connection) is treated the same way.
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# A bad API key or an unknown username can never succeed on retry - fail fast instead of
# burning through the retry budget for nothing.
_PERMANENT_STATUS_CODES = frozenset({401, 403, 404})

# Last.fm's own `format=json` error convention: a 200 response whose body is
# `{"error": <code>, "message": "..."}` rather than a `recenttracks` payload. Codes worth a
# specific hint are listed here; anything else falls back to Last.fm's own message text.
_LASTFM_ERROR_HINTS: dict[int, str] = {
    6: "Check the Last.fm username configured for the import.",
    10: "Check the Last.fm API key configured for the import.",
    26: "Check the Last.fm API key configured for the import.",
}


def _describe_api_error(data: dict[str, Any]) -> str:
    """Build a safe, actionable message from a Last.fm ``{"error": ..., "message": ...}`` body."""
    code = data.get("error")
    message = data.get("message") or "Last.fm rejected the request."
    hint = _LASTFM_ERROR_HINTS.get(code) if isinstance(code, int) else None
    return f"{message} {hint}" if hint else str(message)


def _describe_fetch_error(err: Exception) -> str:
    """
    Build a safe, actionable message from a failed Last.fm HTTP request.

    Deliberately never includes ``str(err)`` for a network/HTTP-layer exception: the request
    URL (which carries the API key as a query parameter) can end up inside an HTTP client
    exception's own string representation, and that must never reach a log line or a
    user-facing message. Only well-known, non-secret attributes are read off ``err``.
    """
    status = getattr(err, "status", None)
    if status == 403:
        return "Last.fm rejected the request (403 Forbidden) — check the configured API key."
    if status == 404:
        return "Last.fm could not find that user (404 Not Found) — check the username."
    if isinstance(err, TimeoutError):
        return "Last.fm did not respond in time — try the import again in a moment."
    if isinstance(status, int):
        return f"Last.fm request failed (HTTP {status})."
    return f"Could not reach Last.fm ({type(err).__name__})."


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
        data = await self._fetch_with_retry(page, params)
        if not isinstance(data, dict) or "recenttracks" not in data:
            if isinstance(data, dict) and "error" in data:
                raise LastfmApiError(_describe_api_error(data))
            raise LastfmApiError("Last.fm returned an unexpected response; try again in a moment.")
        return self._parse_page(data)

    async def import_since(
        self, store: _GenomeStoreProtocol, *, listener: str, max_pages: int = 0
    ) -> GenomeImportResult:
        """
        Import Last.fm history, in one of two explicit modes (§3.8, P1).

        **Backfill mode** - the one-time full-history sweep
        (:meth:`~music_assistant.controllers.genome.store.GenomeStore.lastfm_backfill_done`)
        has never completed: pages from page 1 through the last page with **no** ``from_ts`` at
        all, so an interrupted first sweep can always be resumed by simply running the import
        again - the store's existing ``dedupe_key`` uniqueness absorbs whatever a prior partial
        run already imported, and the sweep continues past however far it previously got.
        Backfill completion is recorded only when the sweep actually reaches the last page, in
        the ``settings`` table (no schema change, no migration - see ``store.py``).

        **Incremental mode** - backfill has completed at least once: resumes from the newest
        stored Last.fm timestamp, exactly as before, so a scheduled poll only fetches what is
        new.

        A page-1 failure (in either mode) still raises - nothing was imported yet, so a silent
        "0 rows" success would hide a bad API key or username. A later-page failure (after
        retries are exhausted - see :meth:`_fetch_with_retry`) returns a partial result instead:
        earlier pages already made it into the store, and - critically - backfill is *not*
        marked complete, so the next run resumes the sweep rather than switching to incremental
        mode and leaving the remaining history unreachable forever (the bug this fixes: a
        real-hardware import that hit a transient 500 on page 62/275 could never make progress
        past that point, because the old code always resumed from the newest stored timestamp).

        :param store: The ``GenomeStore``-shaped object to read the resume point/backfill state
            from and write into.
        :param listener: The listener partition to attribute these listens to.
        :param max_pages: Stop after this many pages; ``0`` means no limit. An explicit limit is
            always treated as an intentionally partial run - it never marks backfill complete.
        """
        backfilled = await store.lastfm_backfill_done()
        resume_from = await self._resume_timestamp(store, listener) if backfilled else None
        LOGGER.info(
            "Last.fm import starting for %s (mode=%s%s)",
            self._username,
            "incremental" if backfilled else "backfill",
            f", resuming from {resume_from}" if resume_from else "",
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
        swept_fully = False
        while page_number <= total_pages:
            try:
                page = await self.fetch_recent(page_number, from_ts=resume_from)
            except LastfmApiError as err:
                LOGGER.warning("Last.fm import: page %d failed: %s", page_number, err)
                if page_number == 1:
                    # nothing was imported at all - a silent "0 rows" success would hide a
                    # bad API key or username from the user, so surface it as a real failure
                    raise
                # later-page failure, retries already exhausted: earlier pages already made it
                # into the store, so treat this as a partial import rather than discarding what
                # already succeeded - and never mark backfill complete over an incomplete sweep
                result["warnings"].append(f"page {page_number}: {err}")
                if not backfilled:
                    result["warnings"].append(
                        f"Import stopped at page {page_number}/{total_pages}; run it again "
                        "later to continue the full-history sweep from where it left off."
                    )
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
        else:
            # the `while` condition went false on its own (no `break` above) - the sweep ran
            # all the way to the last page without an unresolved failure or an artificial cap
            swept_fully = True
        if not backfilled and swept_fully:
            await store.mark_lastfm_backfill_done()
            LOGGER.info("Last.fm full-history backfill complete for %s", self._username)
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

    async def _fetch_with_retry(self, page: int, params: dict[str, str]) -> Any:
        """
        Issue the request, retrying transient failures with capped exponential backoff (§3.8, P2).

        A permanent failure (401/403/404 - a bad key or an unknown user) is raised immediately,
        never retried. Anything else - a transient HTTP status (429/500/502/503/504) or a
        network-level error with no HTTP status at all (a timeout, a dropped connection) - is
        retried up to :data:`LASTFM_RETRY_MAX_ATTEMPTS` times before giving up.

        :param page: The page number being fetched, used only to label log lines.
        :param params: The request's query parameters (already includes the API key).
        """
        delay = LASTFM_RETRY_BASE_DELAY_SECONDS
        last_err: Exception | None = None
        for attempt in range(1, LASTFM_RETRY_MAX_ATTEMPTS + 1):
            try:
                return await self._client.get_json(LASTFM_BASE_URL, params=params)
            except Exception as err:
                status = getattr(err, "status", None)
                if status in _PERMANENT_STATUS_CODES:
                    raise LastfmApiError(_describe_fetch_error(err)) from err
                last_err = err
                transient = status is None or status in _TRANSIENT_STATUS_CODES
                if not transient or attempt == LASTFM_RETRY_MAX_ATTEMPTS:
                    break
                LOGGER.debug(
                    "Last.fm page %d: attempt %d/%d failed (status=%s), retrying in %.1fs",
                    page,
                    attempt,
                    LASTFM_RETRY_MAX_ATTEMPTS,
                    status,
                    delay,
                )
                await asyncio.sleep(delay * random.uniform(0.85, 1.15))
                delay = min(delay * 2, LASTFM_RETRY_MAX_DELAY_SECONDS)
        assert last_err is not None  # the loop only exits without returning via `break` above
        LOGGER.warning(
            "Last.fm page %d: giving up after %d attempt(s): %s",
            page,
            LASTFM_RETRY_MAX_ATTEMPTS,
            _describe_fetch_error(last_err),
        )
        raise LastfmApiError(_describe_fetch_error(last_err)) from last_err

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
            except (TypeError, ValueError):
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
