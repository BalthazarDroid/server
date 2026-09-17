"""
Discovery: cold library corners and Last.fm similar-artist suggestions (D-16).

Two sources, one result. The first is entirely local — artists already in the Music Assistant
library that the household has barely or never played, ranked by how well they fit the genres
where the household diverges most from the baseline. The second is a Last.fm
``artist.getSimilar`` walk seeded from the household's own top artists in those same genres;
it runs only in the background pass and is served from the store afterwards.

Everything in this module is either pure or reads the local MA library database. It issues no
outbound request, by construction: the network half lives in
:mod:`music_assistant.controllers.genome.enrich.lastfm_similar` and is reached only from the
controller's background pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from music_assistant_models.helpers import create_safe_string

from music_assistant.constants import DB_TABLE_ARTISTS
from music_assistant.controllers.genome.constants import LOGGER
from music_assistant.controllers.genome.enrich.musicbrainz import map_genre_names
from music_assistant.controllers.genome.models import ColdArtist
from music_assistant.helpers.json import json_loads

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from music_assistant.controllers.genome.models import GenomeResult
    from music_assistant.mass import MusicAssistant


@dataclass(slots=True, frozen=True)
class LibraryArtist:
    """One artist in the MA library, with the play count and genres the library itself holds."""

    artist_key: str
    artist_name: str
    plays: int
    genres: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class DivergentGenre:
    """One genre the household over-expresses relative to the baseline, and by how much."""

    key: str
    label: str
    contribution: float


@dataclass(slots=True, frozen=True)
class DiscoverySeed:
    """One of the household's own artists, chosen to seed a Last.fm similar-artist lookup."""

    artist_key: str
    artist_name: str
    genre: DivergentGenre


def divergent_genres(genome: GenomeResult | None, *, limit: int = 5) -> list[DivergentGenre]:
    """
    Return the genres the household over-expresses most, strongest first.

    Reuses the engine's own divergence decomposition rather than recomputing anything:
    ``divergence.top_over`` is already the per-genre Jensen-Shannon contribution, sorted, and
    D-06 is explicit that those terms are the mathematically real attribution of the headline
    score to individual genres.

    :param genome: A cached :class:`GenomeResult`, or ``None`` when none has been built yet.
    :param limit: The maximum number of genres to return.
    """
    if not genome:
        return []
    top_over = genome.get("divergence", {}).get("top_over") or []
    return [
        DivergentGenre(
            key=str(share["key"]),
            label=str(share.get("label") or share["key"]),
            contribution=float(share.get("contribution") or 0.0),
        )
        for share in top_over[:limit]
        if share.get("key")
    ]


def rank_cold_corners(
    library_artists: Sequence[LibraryArtist],
    genres: Sequence[DivergentGenre],
    *,
    max_plays: int,
    limit: int,
    extra_plays: Mapping[str, int] | None = None,
) -> list[ColdArtist]:
    """
    Rank barely-played library artists by how well they fit the household's divergent genres.

    An artist's play count is the highest evidence available for it: the library's own
    ``play_count`` and, when higher, the number of listens Genome has stored for the same
    artist key. Taking the maximum keeps an artist with a deep imported Apple/Last.fm history
    but a zeroed library counter from being presented as untouched.

    :param library_artists: Every artist in the MA library, as :func:`read_library_artists`
        returns them.
    :param genres: The divergent genres to match against, strongest first.
    :param max_plays: The highest play count that still counts as a cold corner.
    :param limit: The maximum number of rows to return.
    :param extra_plays: Additional per-``artist_key`` play counts to take the maximum with.
    """
    if not genres:
        return []
    by_key = {genre.key: genre for genre in genres}
    extra_plays = extra_plays or {}
    matched: list[tuple[float, int, str, ColdArtist]] = []
    for artist in library_artists:
        plays = max(artist.plays, extra_plays.get(artist.artist_key, 0))
        if plays > max_plays:
            continue
        best = max(
            (by_key[key] for key in artist.genres if key in by_key),
            key=lambda genre: genre.contribution,
            default=None,
        )
        if best is None:
            continue
        matched.append(
            (
                -best.contribution,
                plays,
                artist.artist_name.casefold(),
                ColdArtist(
                    artist_key=artist.artist_key,
                    artist_name=artist.artist_name,
                    plays=plays,
                    genre_key=best.key,
                    genre_label=best.label,
                ),
            )
        )
    matched.sort(key=lambda row: row[:3])
    return [row[3] for row in matched[:limit]]


def select_seeds(
    genome: GenomeResult | None,
    genres: Sequence[DivergentGenre],
    *,
    limit: int,
) -> list[DiscoverySeed]:
    """
    Pick the household's own top artists that sit in a divergent genre, to seed Last.fm with.

    Taken in ``top_artists`` order, which the engine already sorts by recency-weighted share,
    so the strongest evidence of taste seeds the walk first.

    :param genome: A cached :class:`GenomeResult`, or ``None`` when none has been built yet.
    :param genres: The divergent genres to match against.
    :param limit: The maximum number of seeds to return.
    """
    if not genome or not genres:
        return []
    by_key = {genre.key: genre for genre in genres}
    seeds: list[DiscoverySeed] = []
    for artist in genome.get("top_artists") or []:
        best = max(
            (by_key[key] for key in (artist.get("genres") or []) if key in by_key),
            key=lambda genre: genre.contribution,
            default=None,
        )
        if best is None:
            continue
        seeds.append(
            DiscoverySeed(
                artist_key=str(artist["artist_key"]),
                artist_name=str(artist["name"]),
                genre=best,
            )
        )
        if len(seeds) >= limit:
            break
    return seeds


async def read_library_artists(mass: MusicAssistant) -> list[LibraryArtist]:
    """
    Read every artist in the MA library, with its play count and mapped genres.

    A local database read: the MA library lives on disk and nothing here reaches the network.
    Genre strings come from the library item's own metadata and are folded into the same
    59-key vocabulary the divergence maths uses (D-07), so an artist can be matched against a
    genome genre key directly.

    :param mass: The running :class:`MusicAssistant` instance.
    """
    rows = await mass.music.database.get_rows_from_query(
        f"SELECT name, play_count, metadata FROM {DB_TABLE_ARTISTS}", {}, limit=0
    )
    artists: list[LibraryArtist] = []
    for row in rows:
        name = str(row["name"] or "").strip()
        artist_key = create_safe_string(name) if name else ""
        if not artist_key:
            continue
        artists.append(
            LibraryArtist(
                artist_key=artist_key,
                artist_name=name,
                plays=int(row["play_count"] or 0),
                genres=map_genre_names(_metadata_genres(row["metadata"])),
            )
        )
    return artists


def _metadata_genres(raw: Any) -> list[str]:
    """Extract the genre-name list from a library item's stored ``metadata`` JSON blob."""
    if not raw:
        return []
    try:
        metadata = json_loads(raw) if isinstance(raw, str | bytes) else raw
    except Exception:  # pragma: no cover - defensive, malformed library metadata
        LOGGER.debug("Skipping unreadable library artist metadata", exc_info=True)
        return []
    if not isinstance(metadata, dict):
        return []
    genres = metadata.get("genres") or []
    if isinstance(genres, str):
        return [genres]
    if not isinstance(genres, list | set | tuple):
        return []
    return [str(genre) for genre in genres if genre]


__all__ = [
    "DiscoverySeed",
    "DivergentGenre",
    "LibraryArtist",
    "divergent_genres",
    "rank_cold_corners",
    "read_library_artists",
    "select_seeds",
]
