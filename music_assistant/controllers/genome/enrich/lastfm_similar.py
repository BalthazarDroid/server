"""
Last.fm ``artist.getSimilar`` enrichment for the discovery feature (§3.8, D-16).

Network-only module: it exists to be driven by the background discovery pass
(:meth:`GenomeController._background_discovery`), never by a read path. Every request goes
through the injected :class:`~music_assistant.controllers.genome.http.HttpClient`, so tests
answer it from fixtures and never touch the network.

The API key is sent as a query parameter, so no failure path here may stringify an exception
that could carry the request URL — every message is built by
:func:`~music_assistant.controllers.genome.importers.lastfm.describe_fetch_error`, which reads
only the HTTP status, and by Last.fm's own JSON error body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from music_assistant.controllers.genome.constants import (
    LASTFM_BASE_URL,
    LASTFM_SIMILAR_METHOD,
)
from music_assistant.controllers.genome.errors import LastfmApiError
from music_assistant.controllers.genome.importers.lastfm import describe_fetch_error

if TYPE_CHECKING:
    from music_assistant.controllers.genome.http import HttpClient


@dataclass(slots=True, frozen=True)
class SimilarArtist:
    """One entry of a Last.fm ``artist.getSimilar`` response."""

    name: str
    mbid: str | None
    match: float


async def fetch_similar_artists(
    artist_name: str,
    *,
    client: HttpClient,
    api_key: str,
    limit: int,
) -> tuple[SimilarArtist, ...]:
    """
    Fetch the artists Last.fm considers similar to ``artist_name``.

    Raises :class:`~music_assistant.controllers.genome.errors.LastfmApiError` when Last.fm
    rejects or fails the request, so the caller can record the seed as failed and apply its
    cooldown. A well-formed response that simply lists no similar artists is not an error and
    comes back as an empty tuple.

    :param artist_name: The seed artist name, as the household's own listens spell it.
    :param client: The :class:`HttpClient` to issue the request through.
    :param api_key: A Last.fm API key. Never logged and never included in any raised message.
    :param limit: The maximum number of similar artists to ask Last.fm for.
    """
    params = {
        "method": LASTFM_SIMILAR_METHOD,
        "artist": artist_name,
        "api_key": api_key,
        "format": "json",
        "limit": str(limit),
        "autocorrect": "1",
    }
    try:
        data = await client.get_json(LASTFM_BASE_URL, params=params)
    except Exception as err:
        raise LastfmApiError(describe_fetch_error(err)) from err
    if isinstance(data, dict) and data.get("error") is not None:
        # Last.fm's `format=json` convention: a 200 whose body is an error object. The message
        # is Last.fm's own text; the request (and therefore the key) is never part of it.
        raise LastfmApiError(str(data.get("message") or "Last.fm rejected the request."))
    return _parse_similar(data, limit=limit)


def _parse_similar(data: Any, *, limit: int) -> tuple[SimilarArtist, ...]:
    """Normalize a ``similarartists`` payload, skipping entries without a usable name."""
    container = data.get("similarartists") if isinstance(data, dict) else None
    entries = container.get("artist") if isinstance(container, dict) else None
    if isinstance(entries, dict):
        # Last.fm collapses a single-entry list into a bare object
        entries = [entries]
    if not isinstance(entries, list):
        return ()
    results: list[SimilarArtist] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        mbid = str(entry.get("mbid") or "").strip() or None
        results.append(SimilarArtist(name=name, mbid=mbid, match=_parse_match(entry.get("match"))))
        if len(results) >= limit:
            break
    return tuple(results)


def _parse_match(value: Any) -> float:
    """Parse Last.fm's ``match`` field (a stringified 0..1 float) into a clamped float."""
    try:
        match = float(value)
    except TypeError, ValueError:
        return 0.0
    return min(max(match, 0.0), 1.0)


__all__ = ["SimilarArtist", "fetch_similar_artists"]
