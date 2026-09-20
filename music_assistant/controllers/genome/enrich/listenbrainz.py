"""
ListenBrainz artist popularity enrichment (§1.9, §3.8).

No client for this exists in MA yet, so this module owns it outright (unlike the MusicBrainz
enrichment, which reuses MA's own client per D-08). No auth token is required for this endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from music_assistant.controllers.genome.constants import (
    LISTENBRAINZ_POPULARITY_BATCH_SIZE,
    LISTENBRAINZ_POPULARITY_URL,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from music_assistant.controllers.genome.http import HttpClient


@dataclass(slots=True, frozen=True)
class Popularity:
    """One artist's global ListenBrainz popularity."""

    listeners: int
    listen_count: int


async def artist_popularity(mbids: Sequence[str], *, client: HttpClient) -> dict[str, Popularity]:
    """
    Fetch global popularity for a batch of MusicBrainz artist IDs.

    Requests are batched at :data:`LISTENBRAINZ_POPULARITY_BATCH_SIZE` MBIDs per call. An MBID
    with no entry in the response (an artist ListenBrainz has no data for) is simply absent from
    the returned mapping rather than an error.

    :param mbids: The MusicBrainz artist IDs to look up. Duplicates are only requested once.
    :param client: The :class:`HttpClient` to issue requests through.
    """
    unique_mbids = list(dict.fromkeys(mbids))
    result: dict[str, Popularity] = {}
    for start in range(0, len(unique_mbids), LISTENBRAINZ_POPULARITY_BATCH_SIZE):
        batch = unique_mbids[start : start + LISTENBRAINZ_POPULARITY_BATCH_SIZE]
        if not batch:
            continue
        data = await client.post_json(LISTENBRAINZ_POPULARITY_URL, json={"artist_mbids": batch})
        for item in _rows(data):
            mbid = item.get("artist_mbid")
            if not mbid:
                continue
            result[mbid] = Popularity(
                listeners=int(item.get("total_user_count", 0) or 0),
                listen_count=int(item.get("total_listen_count", 0) or 0),
            )
    return result


def _rows(data: Any) -> list[dict[str, Any]]:
    """
    Pull the artist rows out of a popularity response, whatever envelope they arrive in.

    ``/1/popularity/artist`` answers a bare JSON array. This module assumed the
    ``{"payload": [...]}`` shape that ListenBrainz's STATISTICS endpoints use, so every backfill
    died on ``AttributeError: 'list' object has no attribute 'get'`` - which is why no artist
    ever got a listener count and the obscurity index sat at 0% with low confidence. The
    baseline builder had the identical assumption and was fixed without anyone checking here.

    Accepting every observed shape costs nothing and means a future envelope change degrades to
    an empty result rather than killing the pass.

    :param data: The decoded JSON body.
    """
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return []
    payload = data.get("payload", data)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        artists = payload.get("artists")
        if isinstance(artists, list):
            return [row for row in artists if isinstance(row, dict)]
    return []


__all__ = ["Popularity", "artist_popularity"]
