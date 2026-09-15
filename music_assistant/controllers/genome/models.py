"""
Data model for the Listening Genome controller.

This module is the single frozen contract shared by every work package (see
``docs/ARCHITECTURE.md`` Part 4): the ``TypedDict``s mirror the ``genome/get`` JSON contract
(§3.4) 1:1 with ``frontend/src/composables/genome/useGenome.ts``, and the ``@dataclass``
types are the pure ``engine.py`` inputs (§3.6). Nothing in this file performs I/O.

Ambiguities resolved (simplest reading chosen where §3.4/§3.6 left a detail open):

- ``GenomeStats.coverage_by_source`` and ``DivergenceFacts``/``ObscurityFacts``/``EraFacts``/
  ``LoyaltyFacts`` are plain ``TypedDict``s (not frozen dataclasses) since §3.4 defines the whole
  JSON contract as ``TypedDict``s and these are only ever produced as JSON-serializable results.
- ``ArtistMeta.genres`` and ``Baseline.genre_shares``/``era_shares``/``listener_percentiles`` use
  ``tuple``/``Mapping`` exactly as written in §3.6 to keep the frozen dataclasses hashable-friendly
  and to signal read-only intent to callers in ``engine.py``.
- ``GenomeSettings.last_rebuild_at`` and every ``*_at``/``*_played`` timestamp are unix seconds
  (``int``), consistent with ``played_at`` elsewhere in the schema (§3.1, §3.4).
- ``ArtistFact.genres`` is ``list[str]`` of already-normalized ``translation_key`` strings (not
  ``GenreShare``), since §3.4 does not ask for a full share breakdown per artist row, only the
  genre tags for display.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

from mashumaro import DataClassDictMixin

# ============================================================================================
# §3.6 — engine inputs (pure dataclasses, no I/O)
# ============================================================================================


@dataclass(slots=True, frozen=True)
class Listen:
    """A single normalized listen event, as stored in ``genome_listens`` (§3.1)."""

    played_at: int
    artist_key: str
    artist_name: str
    track_key: str
    track_name: str
    album_name: str | None
    source: str
    player_id: str | None
    duration_ms: int | None
    played_ms: int | None
    fully_played: bool | None
    confidence: float


@dataclass(slots=True, frozen=True)
class ArtistMeta:
    """Enrichment metadata for one artist, keyed by ``artist_key`` (§3.1 ``genome_artist_meta``)."""

    artist_key: str
    artist_name: str
    mbid: str | None
    genres: tuple[str, ...]
    first_release_year: int | None
    lb_listeners: int | None
    lb_listen_count: int | None
    # the artist's life-span begin year (MusicBrainz), used by era_facts only as a fallback proxy
    # for first_release_year - see EraFacts.artist_year_share.
    begin_year: int | None = None


@dataclass(slots=True, frozen=True)
class Baseline:
    """The shipped average-listener baseline used for divergence and obscurity (§3.7)."""

    version: str
    genre_shares: Mapping[str, float]
    era_shares: Mapping[int, float]
    listener_percentiles: Mapping[int, int]
    concentration: float


@dataclass(slots=True, frozen=True)
class EngineParams:
    """Rebuild-time parameters that are not part of the stored data (§3.2, §3.5)."""

    now: int
    half_life_days: int
    obscurity_percentile: int
    min_seconds_played: int
    top_n: int = 20
    new_artist_window_days: int = 90


@dataclass(slots=True, frozen=True)
class GenomeInputs:
    """Everything ``engine.build_genome`` needs to compute a :class:`GenomeResult`."""

    listener: str
    listens: Sequence[Listen]
    artist_meta: Mapping[str, ArtistMeta]
    baseline: Baseline
    params: EngineParams
    player_names: Mapping[str, str]


# ============================================================================================
# §3.4 — the `genome/get` JSON contract (TypedDicts, total=True unless noted)
# ============================================================================================


class GenreShare(TypedDict):
    """
    One genre's share of household listening versus the baseline.

    ``ratio`` (``share / max(baseline_share, 1e-6)``, clamped to 99.0) is present on every
    row — the "Listening Genome" visual's per-genre Overexpressed/Stable/Underexpressed status
    is derived from it client-side; the server does not invent the thresholds.

    ``base_mix`` supports the "four bases" DNA visual (see ``GenomeResult.bases``): for a
    non-base genre it is that genre's affinity to each of the (up to four) base genres, as
    fractions of the same length and order as ``GenomeResult.bases`` that sum to ``1.0``.
    It is an empty list when the affinity cannot be honestly computed — see
    :func:`.engine.base_mix_for_genres` for exactly which cases those are — and always an
    empty list on a base genre's own row (a base's affinity to itself is not a meaningful
    figure the frontend needs).
    """

    key: str
    label: str
    share: float
    baseline_share: float
    ratio: float
    contribution: float
    base_mix: list[float]


class ArtistFact(TypedDict):
    """A single row in ``GenomeResult.top_artists``."""

    name: str
    artist_key: str
    mbid: str | None
    plays: int
    weight: float
    share: float
    lb_listeners: int | None
    obscurity: float | None
    ratio_vs_average: float | None
    genres: list[str]


class TrackFact(TypedDict):
    """A single row in ``GenomeResult.top_tracks``."""

    name: str
    artist: str
    track_key: str
    plays: int
    weight: float
    share: float
    year: int | None


class EraBucket(TypedDict):
    """A single decade's share of household listening versus the baseline."""

    decade: int
    share: float
    baseline_share: float


class RhythmCell(TypedDict):
    """A single weekday/hour cell of the 7x24 listening-rhythm heatmap."""

    weekday: int
    hour: int
    weight: float
    share: float


class PlayerSplit(TypedDict):
    """A single player/room's share of household listening."""

    player_id: str
    name: str
    share: float


class GenomeStats(TypedDict):
    """Volume and coverage summary for a :class:`GenomeResult`."""

    total_listens: int
    weighted_listens: float
    distinct_artists: int
    distinct_tracks: int
    first_listen: int | None
    last_listen: int | None
    coverage_by_source: dict[str, int]
    enrichment_coverage: float
    artists_pending: int  # known artists not yet resolved (pending/error resolve_state)
    artists_resolved: int  # known artists with a final resolution (ok/not_found resolve_state)


class DivergenceFacts(TypedDict):
    """The headline "off-mainstream" score and its top contributing genres."""

    score: float
    percent: int
    top_over: list[GenreShare]
    top_under: list[GenreShare]


class ObscurityFacts(TypedDict):
    """How much of household listening sits on below-percentile-popularity artists."""

    index: float
    percentile: int
    threshold_listeners: int
    known_share: float


class EraFacts(TypedDict):
    """The weighted distribution of listening across release decades."""

    center_of_mass: float
    spread: float
    buckets: list[EraBucket]
    known_share: float
    # share of `known_share`'s weight that used an artist's life-span begin year as a proxy for
    # its release year (no `first_release_year` on file) - see engine.py::era_facts.
    artist_year_share: float


class LoyaltyFacts(TypedDict):
    """
    Exploration versus repeat-listening behavior.

    ``exploration_ratio``/``new_artists_90d`` measure how much of the recency-weighted
    listening (or how many artists) are *new*; on a library imported in bulk from years of
    history, almost nothing is "new" and this reads as ~0 without being wrong - see
    ``effective_genres``/``effective_artists`` below for a figure that stays meaningful
    however the history was accumulated.

    ``effective_genres``/``effective_artists``/``baseline_effective_genres`` are each the
    effective number of categories (``exp(H)`` of the Shannon entropy of a share
    distribution - the Hill number of order 1; see ``engine.py::effective_count``), read as
    "this household listens to the equivalent of N genres/artists, evenly". Unlike
    ``exploration_ratio`` it does not care how long the history spans, so it stays meaningful
    on a deep, one-shot-imported library.
    """

    exploration_ratio: float
    concentration: float
    top_artist_share: float
    new_artists_90d: int
    repeat_rate: float
    effective_genres: float
    effective_artists: float
    baseline_effective_genres: float


class GenomeResult(TypedDict):
    """The full ``genome/get`` response payload."""

    schema_version: int
    engine_version: str
    baseline_version: str
    listener: str
    generated_at: int
    stale: bool
    half_life_days: int
    stats: GenomeStats
    genres: list[GenreShare]
    bases: list[GenreShare]  # top 4 genres by share; the DNA visual's four "bases" (0-4 entries)
    divergence: DivergenceFacts
    obscurity: ObscurityFacts
    era: EraFacts
    loyalty: LoyaltyFacts
    top_artists: list[ArtistFact]
    top_tracks: list[TrackFact]
    rhythm: list[RhythmCell]
    players: list[PlayerSplit]


class GenomeRebuildResult(TypedDict):
    """Return value of ``genome/rebuild``."""

    listener: str
    listens_scanned: int
    duration_ms: int
    genome: GenomeResult


class GenomeImportResult(TypedDict):
    """Return value of any import operation (Apple CSV, Last.fm, MA backfill)."""

    source: str
    rows_read: int
    rows_imported: int
    rows_skipped: int
    rows_duplicate: int
    first_played_at: int | None
    last_played_at: int | None
    warnings: list[str]


class GenomeSettings(TypedDict):
    """Return value of ``genome/settings`` — never includes the Last.fm API key itself."""

    half_life_days: int
    lastfm_username: str
    lastfm_configured: bool
    lastfm_poll_enabled: bool
    enrich_enabled: bool
    obscurity_percentile: int
    min_seconds_played: int
    apple_import_dir: str
    baseline_version: str
    last_rebuild_at: int | None


@dataclass
class GenomeSettingsPatch(DataClassDictMixin):
    """
    Partial update accepted by ``genome/settings/set``; every field is optional.

    Deliberately a mashumaro dataclass rather than a ``TypedDict``, unlike the rest of the
    §3.4 contract. Music Assistant parses incoming api_command arguments with
    ``helpers/api.py::parse_value``, which ends in ``isinstance(value, value_type)`` — and a
    ``TypedDict`` cannot be used with ``isinstance`` at all ("TypedDict does not support
    instance and class checks"), so every call raised before reaching the handler. MA does
    support any type exposing ``from_dict``, which ``DataClassDictMixin`` provides, and the
    wire format is unchanged: still a plain JSON object of the same keys.

    ``None`` means "leave this setting alone" — it is never written through as a value.
    """

    half_life_days: int | None = None
    lastfm_username: str | None = None
    lastfm_api_key: str | None = None
    lastfm_poll_enabled: bool | None = None
    enrich_enabled: bool | None = None
    obscurity_percentile: int | None = None
    min_seconds_played: int | None = None
    apple_import_dir: str | None = None
