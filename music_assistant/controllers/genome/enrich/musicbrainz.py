"""
MusicBrainz artist enrichment (§1.9, §3.8, D-08).

Resolves an artist name to a MusicBrainz ID, tags, life-span begin year and country. Prefers the
already-loaded ``musicbrainz`` provider (throttled, MA's own mirror, 30-day HTTP cache) and falls
back to the injected :class:`~music_assistant.controllers.genome.http.HttpClient` — the seam tests
use, since MusicBrainz is unreachable from this workspace (BRIEF.md).

Contract gap (see ``docs/STATUS.md`` "Contract gaps"): ``providers/musicbrainz/provider.py``
exposes no plain "search by artist name" method — its ``search()`` needs a track/album context,
and ``get_artist_details()`` needs an MBID already in hand. D-08 says to reuse the provider rather
than write a second, unthrottled client, so this module reaches for the provider's own
``_api_client.get_data(...)`` (the same throttled, cached MusicBrainz HTTP client the provider
itself calls) when the provider is loaded, and the plain ``HttpClient`` otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from music_assistant_models.helpers import create_safe_string

from music_assistant.constants import DEFAULT_GENRE_MAPPING
from music_assistant.controllers.genome.constants import (
    LOGGER,
    RESOLVE_STATE_ERROR,
    RESOLVE_STATE_NOT_FOUND,
    RESOLVE_STATE_OK,
)

if TYPE_CHECKING:
    from music_assistant.controllers.genome.controller import _GenomeStoreProtocol
    from music_assistant.controllers.genome.http import HttpClient
    from music_assistant.mass import MusicAssistant

# MA's own MusicBrainz mirror (see providers/musicbrainz/api_client.py::MB_BASE_URL, which this
# duplicates rather than imports: importing that module pulls in the whole webserver stack for a
# single URL string, and it is not part of the frozen contract this package may edit).
_MB_BASE_URL = "https://musicbrainz-mirror.music-assistant.io/ws/2"

# from providers/musicbrainz/constants.py::LUCENE_SPECIAL, duplicated for the same reason.
_LUCENE_SPECIAL = r'([+\-&|!(){}\[\]\^"~*?:\\\/])'

_MIN_MATCH_SCORE = 85
_MAX_GENRES_PER_ARTIST = 3


@dataclass(slots=True, frozen=True)
class ArtistMetaUpdate:
    """The MusicBrainz-derived fields for one artist, ready to hand to ``GenomeStore``."""

    mbid: str | None
    mb_tags: tuple[tuple[str, int], ...]
    genres: tuple[str, ...]
    begin_year: int | None
    country: str | None


def _normalize_for_match(value: str) -> str:
    """Fold a genre/tag name for alias matching: lowercase, ``&``/``_``/``-`` normalized."""
    value = value.lower().strip().replace("&", "and")
    value = re.sub(r"[-_]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _build_alias_lookup() -> dict[str, str]:
    """Build the normalized-alias -> ``translation_key`` lookup from ``genre_mapping.json``."""
    lookup: dict[str, str] = {}
    for entry in DEFAULT_GENRE_MAPPING:
        translation_key = entry["translation_key"]
        for alias in (*entry.get("aliases", ()), entry["genre"]):
            lookup.setdefault(_normalize_for_match(alias), translation_key)
    return lookup


_ALIAS_LOOKUP = _build_alias_lookup()


async def resolve_artist(
    name: str,
    *,
    client: HttpClient,
    mass: MusicAssistant | None = None,
) -> ArtistMetaUpdate | None:
    """
    Resolve an artist name to MusicBrainz metadata, or ``None`` if no confident match exists.

    A transport failure or an unexpected response shape propagates as an exception rather than
    being swallowed here, so :func:`enrich_pending_artists` can tell "not found" (``None``, cheap
    30-day recheck) apart from "MusicBrainz did not answer" (exception, ``resolve_state="error"``,
    no cooldown — see §3.8). Callers that do not go through :func:`enrich_pending_artists` must
    apply the same distinction themselves; a rebuild must never fail outright because MusicBrainz
    is briefly unreachable.

    :param name: The artist name as reported by a listen.
    :param client: The :class:`HttpClient` fallback, used when no ``musicbrainz`` provider is
        loaded.
    :param mass: The running :class:`MusicAssistant` instance, used to prefer the loaded provider.
    """
    mbid = await _search_artist(name, client=client, mass=mass)
    if mbid is None:
        return None
    return await _lookup_artist(mbid, client=client, mass=mass)


async def enrich_pending_artists(
    store: _GenomeStoreProtocol,
    *,
    client: HttpClient,
    mass: MusicAssistant | None = None,
    limit: int = 200,
) -> int:
    """
    Resolve and store MusicBrainz metadata for pending artists, one at a time.

    Every artist :meth:`GenomeStore.pending_artist_keys` returns is resolved independently,
    never letting a single failure abort the batch.

    :param store: The ``GenomeStore``-shaped object to read pending artists from and write
        results into.
    :param client: The :class:`HttpClient` fallback for artists without a loaded provider.
    :param mass: The running :class:`MusicAssistant` instance, used to prefer the loaded provider.
    :param limit: The maximum number of artists to resolve in this pass.
    :return: The number of artists successfully resolved (``resolve_state="ok"``).
    """
    pending = await store.pending_artist_keys(limit=limit)
    if not pending:
        return 0
    LOGGER.info("MusicBrainz enrichment pass starting: %d pending artists", len(pending))
    resolved = 0
    for artist_key, artist_name in pending:
        try:
            update = await resolve_artist(artist_name, client=client, mass=mass)
        except Exception as err:
            LOGGER.debug("MusicBrainz lookup failed for %r: %s", artist_name, err)
            await store.upsert_artist_meta_full(
                [{"artist_key": artist_key, "artist_name": artist_name}], state=RESOLVE_STATE_ERROR
            )
            continue
        if update is None:
            await store.upsert_artist_meta_full(
                [{"artist_key": artist_key, "artist_name": artist_name}],
                state=RESOLVE_STATE_NOT_FOUND,
            )
            continue
        await store.upsert_artist_meta_full(
            [
                {
                    "artist_key": artist_key,
                    "artist_name": artist_name,
                    "mbid": update.mbid,
                    "mb_tags": [{"name": name, "count": count} for name, count in update.mb_tags],
                    "genres": list(update.genres),
                    "begin_year": update.begin_year,
                    "first_release_year": None,
                    "country": update.country,
                }
            ],
            state=RESOLVE_STATE_OK,
        )
        resolved += 1
    LOGGER.info(
        "MusicBrainz enrichment pass finished: %d/%d artists resolved", resolved, len(pending)
    )
    return resolved


async def _search_artist(
    name: str, *, client: HttpClient, mass: MusicAssistant | None
) -> str | None:
    """Search by artist name and return the best matching MBID, or ``None``."""
    escaped = re.sub(_LUCENE_SPECIAL, r"\\\1", name)
    query = f'artist:"{escaped}"'
    data = await _get("artist", {"query": query, "limit": "5"}, client=client, mass=mass)
    candidates = data.get("artists", []) if isinstance(data, dict) else []
    safe_name = create_safe_string(name)
    for candidate in candidates:
        score = candidate.get("score", 0) or 0
        if score >= _MIN_MATCH_SCORE and create_safe_string(candidate.get("name", "")) == safe_name:
            mbid: str = candidate["id"]
            return mbid
    return None


async def _lookup_artist(
    mbid: str, *, client: HttpClient, mass: MusicAssistant | None
) -> ArtistMetaUpdate:
    """Fetch full artist details for a known MBID and build an :class:`ArtistMetaUpdate`."""
    data = await _get(f"artist/{mbid}", {"inc": "tags+genres"}, client=client, mass=mass)
    tags = sorted((data.get("tags") or []), key=lambda tag: tag.get("count", 0), reverse=True)
    mb_tags = tuple((tag["name"], int(tag.get("count", 0))) for tag in tags if tag.get("name"))
    genres = _map_tags_to_genres(name for name, _count in mb_tags)
    life_span = data.get("life-span") or {}
    return ArtistMetaUpdate(
        mbid=mbid,
        mb_tags=mb_tags,
        genres=genres,
        begin_year=_parse_year(life_span.get("begin")),
        country=data.get("country"),
    )


async def _get(
    endpoint: str, params: dict[str, str], *, client: HttpClient, mass: MusicAssistant | None
) -> Any:
    """Issue one MusicBrainz GET, preferring the loaded provider's throttled client."""
    provider = mass.get_provider("musicbrainz") if mass is not None else None
    api_client = getattr(provider, "_api_client", None)
    if api_client is not None:
        return await api_client.get_data(endpoint, **params)
    url = f"{_MB_BASE_URL}/{endpoint}"
    return await client.get_json(url, params={**params, "fmt": "json"})


def _map_tags_to_genres(tag_names: Any) -> tuple[str, ...]:
    """Map tag names (most-confident first) to up to 3 distinct ``genre_mapping.json`` keys."""
    genres: list[str] = []
    for tag_name in tag_names:
        translation_key = _ALIAS_LOOKUP.get(_normalize_for_match(tag_name))
        if translation_key and translation_key not in genres:
            genres.append(translation_key)
        if len(genres) >= _MAX_GENRES_PER_ARTIST:
            break
    return tuple(genres)


def _parse_year(value: Any) -> int | None:
    """Parse a MusicBrainz ``life-span.begin`` value (``"1994"`` or ``"1994-03-01"``) to a year."""
    if not value or not isinstance(value, str):
        return None
    try:
        return int(value[:4])
    except ValueError:
        return None


__all__ = ["ArtistMetaUpdate", "enrich_pending_artists", "resolve_artist"]
