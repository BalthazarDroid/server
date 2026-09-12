"""
ListenBrainz artist popularity enrichment (§1.9, §3.8).

No client for this exists in MA yet, so this module owns it outright (unlike the MusicBrainz
enrichment, which reuses MA's own client per D-08). No auth token is required for this endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
        for item in (data or {}).get("payload", []):
            mbid = item.get("artist_mbid")
            if not mbid:
                continue
            result[mbid] = Popularity(
                listeners=int(item.get("total_user_count", 0) or 0),
                listen_count=int(item.get("total_listen_count", 0) or 0),
            )
    return result


__all__ = ["Popularity", "artist_popularity"]
