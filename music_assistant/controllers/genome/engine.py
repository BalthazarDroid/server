"""
Pure computation engine for the Listening Genome (§3.6).

Every function here is a pure function of its arguments: no I/O, no ``mass``, no wall-clock
reads (``EngineParams.now`` is passed in). This keeps the whole module trivially unit-testable
and safe to run off the event loop. Only stdlib + the frozen dataclasses/TypedDicts from
``models.py`` are used — no ``numpy`` (see ``docs/ARCHITECTURE.md`` §3.6: pulling it into this
hot pure module would hurt testability and import time).

Ambiguities resolved (§3.6 leaves these open; simplest/most defensible reading chosen):

- ``build_genome`` takes an optional ``tz_offset_seconds`` keyword (default ``0``, i.e. UTC).
  §3.6 declares both ``rhythm_grid(inputs, tz_offset_seconds)`` and ``build_genome(inputs)``
  with no way for the latter to supply the former's second argument, and ``EngineParams`` has no
  timezone field. Rather than change the frozen ``EngineParams`` dataclass, ``build_genome``
  accepts the offset directly and defaults to UTC; ``controller.py`` passes the server's real
  local offset. See ``docs/STATUS.md`` "Contract gaps".
- Per-artist ``ArtistFact.obscurity`` (0..1, continuous) is derived by log-linearly interpolating
  the artist's ``lb_listeners`` against ``Baseline.listener_percentiles`` to an approximate
  percentile rank, then ``obscurity = 1 - rank / 100``. §3.6 only specifies the aggregate
  ``obscurity_index``; this reuses the same baseline table for a consistent per-artist figure.
- ``ArtistFact.ratio_vs_average`` reads "popularity ratio for the vs. average chip" as the
  artist's ``lb_listeners`` divided by the baseline's median (p50) listener count — the only
  "average listener" figure the baseline carries.
- ``GenomeStats.distinct_tracks`` and track aggregation key tracks by ``track_key`` alone (not
  ``(artist_key, track_key)``), matching the ``TrackFact``/``GenomeStats`` contract literally;
  a title shared by two different artists is a rare v1 undercount, noted here rather than
  silently diverging from the documented field name.
- ``player_split`` shares are normalized over the weight of listens with a known ``player_id``
  only (listens with no player are excluded, mirroring how the genre vector excludes listens
  with no resolvable genre), so the returned shares sum to 1.
- The "Listening Genome" DNA visual's four "bases" (``GenomeResult.bases``) are simply the top
  four entries of the already-share-sorted, already-nonzero ``genres`` list — no new selection
  rule beyond what ``build_genome`` already computes. ``GenreShare.base_mix`` (a non-base
  genre's affinity to each base) is computed by :func:`base_mix_for_genres` from the same
  ``listens``/``artist_meta`` the rest of the engine already has in hand; see that function's
  docstring for the degenerate cases (single-genre artists, zero overlap, unenriched artists).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from typing import TYPE_CHECKING

from music_assistant.constants import DEFAULT_GENRE_MAPPING

from .constants import ENGINE_VERSION, GENOME_RESULT_SCHEMA_VERSION
from .models import (
    ArtistFact,
    DivergenceFacts,
    EraBucket,
    EraFacts,
    GenomeResult,
    GenomeStats,
    GenreShare,
    LoyaltyFacts,
    ObscurityFacts,
    PlayerSplit,
    RhythmCell,
    TrackFact,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .models import EngineParams, GenomeInputs, Listen

__all__ = [
    "ENGINE_VERSION",
    "base_mix_for_genres",
    "build_genome",
    "build_genre_vector",
    "divergence_facts",
    "era_facts",
    "js_contributions",
    "js_divergence",
    "listen_weight",
    "loyalty_facts",
    "obscurity_index",
    "player_split",
    "rhythm_grid",
    "select_bases",
    "top_artists",
    "top_tracks",
]

# the DNA visual always has (at most) this many "bases" — the user's top genres by share.
_BASE_COUNT = 4

_ROUND_DP = 4
# genre translation_key -> English display name, built once from the shipped 59-key taxonomy
# (§1.6). Reused for every ``GenreShare.label`` so the household/baseline comparison always
# speaks the canonical genre vocabulary.
_GENRE_LABELS: Mapping[str, str] = {
    entry["translation_key"]: entry["genre"] for entry in DEFAULT_GENRE_MAPPING
}
_REQUIRED_PERCENTILES = (5, 10, 25, 50, 75, 90)


def listen_weight(listen: Listen, params: EngineParams) -> float:
    """
    Return the recency/confidence/completion-decayed weight of a single listen (§3.5).

    :param listen: The listen to weight.
    :param params: Rebuild-time parameters, including ``half_life_days`` (``0`` disables decay)
        and ``now`` (the rebuild's reference "current" unix timestamp).
    """
    if params.half_life_days <= 0:
        w_recency = 1.0
    else:
        age_days = max(0.0, (params.now - listen.played_at) / 86400)
        w_recency = 0.5 ** (age_days / params.half_life_days)
    if listen.fully_played:
        completion = 1.0
    elif listen.played_ms is not None and listen.duration_ms is not None and listen.duration_ms > 0:
        completion = _clamp(listen.played_ms / listen.duration_ms, 0.0, 1.0)
    else:
        completion = 1.0
    return w_recency * listen.confidence * completion


def build_genre_vector(inputs: GenomeInputs) -> dict[str, float]:
    """
    Return the household's weighted genre-share vector, normalized to sum to 1 (§3.6).

    Each listen's weight is split evenly across its resolved genres (a 3-genre listen gives
    ``w / 3`` to each). Listens whose artist is unknown or has no resolved genre are excluded
    entirely — that excluded weight is reported elsewhere as ``1 - enrichment_coverage``.

    :param inputs: The full genome computation input.
    """
    totals: dict[str, float] = {}
    for listen in inputs.listens:
        meta = inputs.artist_meta.get(listen.artist_key)
        if meta is None or not meta.genres:
            continue
        w = listen_weight(listen, inputs.params)
        share = w / len(meta.genres)
        for genre in meta.genres:
            totals[genre] = totals.get(genre, 0.0) + share
    return _normalize(totals)


def js_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """
    Return the Jensen-Shannon divergence (base 2, 0..1) between two share distributions.

    ``JSD(p‖q) = 1/2 * sum(p_i * log2(p_i / m_i)) + 1/2 * sum(q_i * log2(q_i / m_i))`` with
    ``m = (p + q) / 2``; a term with a zero numerator contributes 0.

    :param p: The first distribution, keyed by genre, summing to 1 (or 0 for an empty vector).
    :param q: The second distribution, same shape as ``p``.
    """
    return sum(js_contributions(p, q).values())


def js_contributions(p: Mapping[str, float], q: Mapping[str, float]) -> dict[str, float]:
    """
    Return each key's term of the Jensen-Shannon divergence; the values sum to ``js_divergence``.

    :param p: The first distribution, keyed by genre.
    :param q: The second distribution, same shape as ``p``.
    """
    keys = sorted(set(p) | set(q))
    if sum(p.values()) <= 0 or sum(q.values()) <= 0:
        # one side carries no data at all (e.g. an empty genome) — there is nothing to
        # compare yet, so report "no divergence" rather than treating the missing side as
        # the zero vector (which would read as maximally divergent from any real distribution).
        return dict.fromkeys(keys, 0.0)
    terms: dict[str, float] = {}
    for key in keys:
        pi = p.get(key, 0.0)
        qi = q.get(key, 0.0)
        mi = (pi + qi) / 2
        if mi <= 0:
            terms[key] = 0.0
            continue
        term = 0.0
        if pi > 0:
            term += 0.5 * pi * math.log2(pi / mi)
        if qi > 0:
            term += 0.5 * qi * math.log2(qi / mi)
        terms[key] = term
    return terms


def divergence_facts(
    p: Mapping[str, float], q: Mapping[str, float], labels: Mapping[str, str]
) -> DivergenceFacts:
    """
    Return the headline divergence score plus the genres that drive it most (§3.6).

    :param p: The household's genre-share vector.
    :param q: The baseline (average-listener) genre-share vector.
    :param labels: ``translation_key -> English display name`` for every key in ``p``/``q``.
    """
    terms = js_contributions(p, q)
    jsd = sum(terms.values())
    score = math.sqrt(max(jsd, 0.0))
    # one side has no data at all (e.g. an empty genome): there is nothing to call "over" or
    # "under" represented yet, even though _genre_shares would otherwise show every baseline
    # genre as "under" simply because the household side is all zeros.
    if sum(p.values()) <= 0 or sum(q.values()) <= 0:
        return DivergenceFacts(score=0.0, percent=0, top_over=[], top_under=[])
    shares = _genre_shares(p, q, terms, jsd, labels)
    top_over = sorted(
        (s for s in shares if s["share"] > s["baseline_share"]),
        key=lambda s: s["contribution"],
        reverse=True,
    )[:5]
    top_under = sorted(
        (s for s in shares if s["share"] < s["baseline_share"]),
        key=lambda s: s["contribution"],
        reverse=True,
    )[:5]
    return DivergenceFacts(
        score=_r4(score),
        percent=round(score * 100),
        top_over=top_over,
        top_under=top_under,
    )


def obscurity_index(inputs: GenomeInputs) -> ObscurityFacts:
    """
    Return the household's obscurity index against the baseline's listener-count percentiles.

    :param inputs: The full genome computation input.
    """
    percentile = inputs.params.obscurity_percentile
    threshold = inputs.baseline.listener_percentiles.get(percentile, 0)
    total_w = 0.0
    known_w = 0.0
    below_w = 0.0
    for listen in inputs.listens:
        w = listen_weight(listen, inputs.params)
        total_w += w
        meta = inputs.artist_meta.get(listen.artist_key)
        if meta is None or meta.lb_listeners is None:
            continue
        known_w += w
        if meta.lb_listeners < threshold:
            below_w += w
    index = below_w / known_w if known_w > 0 else 0.0
    known_share = known_w / total_w if total_w > 0 else 0.0
    return ObscurityFacts(
        index=_r4(index),
        percentile=percentile,
        threshold_listeners=threshold,
        known_share=_r4(known_share),
    )


def era_facts(inputs: GenomeInputs) -> EraFacts:
    """
    Return the weighted release-decade distribution versus the baseline (§3.6).

    :param inputs: The full genome computation input.
    """
    total_w = 0.0
    known: list[tuple[float, int]] = []
    for listen in inputs.listens:
        w = listen_weight(listen, inputs.params)
        total_w += w
        meta = inputs.artist_meta.get(listen.artist_key)
        if meta is None or meta.first_release_year is None:
            continue
        known.append((w, meta.first_release_year))
    known_w = sum(w for w, _ in known)
    if known_w > 0:
        com = sum(w * year for w, year in known) / known_w
        variance = sum(w * (year - com) ** 2 for w, year in known) / known_w
        spread = math.sqrt(variance)
    else:
        com = 0.0
        spread = 0.0
    bucket_w: dict[int, float] = {}
    for w, year in known:
        decade = (year // 10) * 10
        bucket_w[decade] = bucket_w.get(decade, 0.0) + w
    buckets = [
        EraBucket(
            decade=decade,
            share=_r4(w / known_w) if known_w > 0 else 0.0,
            baseline_share=_r4(inputs.baseline.era_shares.get(decade, 0.0)),
        )
        for decade, w in sorted(bucket_w.items())
    ]
    known_share = known_w / total_w if total_w > 0 else 0.0
    return EraFacts(
        center_of_mass=round(com, 1),
        spread=_r4(spread),
        buckets=buckets,
        known_share=_r4(known_share),
    )


def rhythm_grid(inputs: GenomeInputs, tz_offset_seconds: int) -> list[RhythmCell]:
    """
    Return the full 168-cell (7 weekday x 24 hour) listening-rhythm grid, zero-filled (§3.6).

    :param inputs: The full genome computation input.
    :param tz_offset_seconds: Offset added to each ``played_at`` (unix seconds, UTC) before
        deriving the local weekday/hour.
    """
    cells: dict[tuple[int, int], float] = {
        (weekday, hour): 0.0 for weekday in range(7) for hour in range(24)
    }
    for listen in inputs.listens:
        w = listen_weight(listen, inputs.params)
        local_ts = listen.played_at + tz_offset_seconds
        days, seconds_of_day = divmod(local_ts, 86400)
        # 1970-01-01 (unix epoch day 0) was a Thursday == weekday index 3 (Mon=0).
        weekday = (days + 3) % 7
        hour = seconds_of_day // 3600
        cells[(weekday, hour)] += w
    total = sum(cells.values())
    return [
        RhythmCell(
            weekday=weekday,
            hour=hour,
            weight=_r4(w),
            share=_r4(w / total) if total > 0 else 0.0,
        )
        for (weekday, hour), w in sorted(cells.items())
    ]


def loyalty_facts(inputs: GenomeInputs) -> LoyaltyFacts:
    """
    Return exploration-vs-repeat listening behavior (§3.6).

    :param inputs: The full genome computation input.
    """
    params = inputs.params
    first_heard: dict[str, int] = {}
    for listen in inputs.listens:
        prev = first_heard.get(listen.artist_key)
        if prev is None or listen.played_at < prev:
            first_heard[listen.artist_key] = listen.played_at

    artist_w: dict[str, float] = {}
    total_w = 0.0
    new_w = 0.0
    window_seconds = params.new_artist_window_days * 86400
    for listen in inputs.listens:
        w = listen_weight(listen, params)
        total_w += w
        artist_w[listen.artist_key] = artist_w.get(listen.artist_key, 0.0) + w
        if params.now - first_heard[listen.artist_key] < window_seconds:
            new_w += w

    shares = [w / total_w for w in artist_w.values()] if total_w > 0 else []
    n = len(shares)
    if n > 1:
        herfindahl = sum(s * s for s in shares)
        concentration = (herfindahl - 1 / n) / (1 - 1 / n)
    elif n == 1:
        concentration = 1.0
    else:
        concentration = 0.0

    new_artists_90d = sum(
        1 for first in first_heard.values() if params.now - first < window_seconds
    )
    total_listens = len(inputs.listens)
    distinct_tracks = len({listen.track_key for listen in inputs.listens})
    repeat_rate = 1 - distinct_tracks / total_listens if total_listens > 0 else 0.0

    return LoyaltyFacts(
        exploration_ratio=_r4(new_w / total_w) if total_w > 0 else 0.0,
        concentration=_r4(concentration),
        top_artist_share=_r4(max(shares)) if shares else 0.0,
        new_artists_90d=new_artists_90d,
        repeat_rate=_r4(repeat_rate),
    )


def top_artists(inputs: GenomeInputs) -> list[ArtistFact]:
    """
    Return the top ``params.top_n`` artists by decayed weight, richest first (§3.6).

    :param inputs: The full genome computation input.
    """
    params = inputs.params
    plays: dict[str, int] = {}
    weights: dict[str, float] = {}
    names: dict[str, str] = {}
    total_w = 0.0
    for listen in inputs.listens:
        w = listen_weight(listen, params)
        total_w += w
        key = listen.artist_key
        plays[key] = plays.get(key, 0) + 1
        weights[key] = weights.get(key, 0.0) + w
        names[key] = listen.artist_name
    percentiles = inputs.baseline.listener_percentiles
    median_listeners = percentiles.get(50, 0)
    facts: list[ArtistFact] = []
    for key, weight in weights.items():
        meta = inputs.artist_meta.get(key)
        lb_listeners = meta.lb_listeners if meta is not None else None
        obscurity = (
            _r4(1 - _percentile_rank(lb_listeners, percentiles) / 100)
            if lb_listeners is not None
            else None
        )
        ratio_vs_average = (
            _r4(lb_listeners / median_listeners)
            if lb_listeners is not None and median_listeners > 0
            else None
        )
        facts.append(
            ArtistFact(
                name=names[key],
                artist_key=key,
                mbid=meta.mbid if meta is not None else None,
                plays=plays[key],
                weight=_r4(weight),
                share=_r4(weight / total_w) if total_w > 0 else 0.0,
                lb_listeners=lb_listeners,
                obscurity=obscurity,
                ratio_vs_average=ratio_vs_average,
                genres=list(meta.genres[:3]) if meta is not None else [],
            )
        )
    facts.sort(key=lambda fact: fact["weight"], reverse=True)
    return facts[: params.top_n]


def top_tracks(inputs: GenomeInputs) -> list[TrackFact]:
    """
    Return the top ``params.top_n`` tracks by decayed weight, richest first (§3.6).

    :param inputs: The full genome computation input.
    """
    params = inputs.params
    plays: dict[str, int] = {}
    weights: dict[str, float] = {}
    names: dict[str, tuple[str, str, str]] = {}
    total_w = 0.0
    for listen in inputs.listens:
        w = listen_weight(listen, params)
        total_w += w
        key = listen.track_key
        plays[key] = plays.get(key, 0) + 1
        weights[key] = weights.get(key, 0.0) + w
        names[key] = (listen.track_name, listen.artist_name, listen.artist_key)
    facts: list[TrackFact] = []
    for key, weight in weights.items():
        track_name, artist_name, artist_key = names[key]
        meta = inputs.artist_meta.get(artist_key)
        facts.append(
            TrackFact(
                name=track_name,
                artist=artist_name,
                track_key=key,
                plays=plays[key],
                weight=_r4(weight),
                share=_r4(weight / total_w) if total_w > 0 else 0.0,
                year=meta.first_release_year if meta is not None else None,
            )
        )
    facts.sort(key=lambda fact: fact["weight"], reverse=True)
    return facts[: params.top_n]


def player_split(inputs: GenomeInputs) -> list[PlayerSplit]:
    """
    Return each player/room's share of household listening, summing to 1 (§3.6).

    Listens with no known ``player_id`` are excluded from both the numerator and denominator.

    :param inputs: The full genome computation input.
    """
    weights: dict[str, float] = {}
    for listen in inputs.listens:
        if not listen.player_id:
            continue
        weights[listen.player_id] = weights.get(listen.player_id, 0.0) + listen_weight(
            listen, inputs.params
        )
    total = sum(weights.values())
    splits = [
        PlayerSplit(
            player_id=player_id,
            name=inputs.player_names.get(player_id, player_id),
            share=_r4(weight / total) if total > 0 else 0.0,
        )
        for player_id, weight in weights.items()
    ]
    splits.sort(key=lambda split: split["share"], reverse=True)
    return splits


def select_bases(genres: list[GenreShare]) -> list[GenreShare]:
    """
    Return the "Listening Genome" DNA visual's four bases: the top genres by share, in order.

    ``genres`` is expected already sorted by ``share`` descending with zero-share entries
    dropped (exactly what ``build_genome`` passes in). Exactly four are returned when at least
    four such genres exist; fewer when the user has fewer non-zero genres. Never padded with
    fabricated entries — the frontend must handle 0-4 bases.

    :param genres: The household's non-zero genre shares, sorted by ``share`` descending.
    """
    return genres[:_BASE_COUNT]


def base_mix_for_genres(inputs: GenomeInputs, base_keys: Sequence[str]) -> dict[str, list[float]]:
    """
    Return each genre's affinity to the base genres, as fractions summing to 1.0.

    For a genre ``G``, take the recency-weighted listens whose artist carries ``G``; for each
    base ``B`` in ``base_keys``, sum the weight of those same listens whose artist *also*
    carries ``B``; normalize the per-base sums across ``base_keys`` so they sum to 1.0.

    Degenerate cases, resolved deliberately rather than silently:

    - An artist resolved to only one genre contributes to no overlap for that genre (it cannot
      simultaneously carry any base genre), so its listens count toward ``G``'s pool but never
      toward any base's overlap sum.
    - A genre with zero measured overlap with every base returns an empty list, never a uniform
      split — a uniform 25/25/25/25 (or 1/n) would be a fabricated claim of "we don't know", not
      a real affinity.
    - Unenriched artists (no resolved ``ArtistMeta``, or a resolved row with no genres at all)
      are excluded from the computation entirely, for every genre — never treated as "no
      overlap" for a genre they were never counted toward in the first place.

    :param inputs: The full genome computation input (already ``_filter_eligible``-d).
    :param base_keys: The base genres' ``translation_key`` values, in display order. The
        returned lists follow this same order. Empty input yields an empty result.
    """
    if not base_keys:
        return {}
    overlap: dict[str, dict[str, float]] = {}
    for listen in inputs.listens:
        meta = inputs.artist_meta.get(listen.artist_key)
        if meta is None or not meta.genres:
            continue
        genre_set = set(meta.genres)
        w = listen_weight(listen, inputs.params)
        for genre in genre_set:
            bucket = overlap.setdefault(genre, {})
            for base in base_keys:
                if base in genre_set:
                    bucket[base] = bucket.get(base, 0.0) + w
    result: dict[str, list[float]] = {}
    for genre, bucket in overlap.items():
        total = sum(bucket.get(base, 0.0) for base in base_keys)
        if total <= 0:
            result[genre] = []
            continue
        result[genre] = _round_fractions([bucket.get(base, 0.0) / total for base in base_keys])
    return result


def build_genome(inputs: GenomeInputs, *, tz_offset_seconds: int = 0) -> GenomeResult:
    """
    Compose the full ``genome/get`` payload from raw inputs (§3.6).

    :param inputs: The full genome computation input.
    :param tz_offset_seconds: Offset used to localize :func:`rhythm_grid`; see the module
        docstring's "Ambiguities resolved" note. Defaults to UTC.
    """
    filtered = _filter_eligible(inputs)
    listens = filtered.listens
    total_w = sum(listen_weight(listen, filtered.params) for listen in listens)

    genre_vector = build_genre_vector(filtered)
    baseline_vector = dict(filtered.baseline.genre_shares)
    terms = js_contributions(genre_vector, baseline_vector)
    jsd = sum(terms.values())
    genres = sorted(
        (
            share
            for share in _genre_shares(genre_vector, baseline_vector, terms, jsd, _GENRE_LABELS)
            if share["share"] > 0
        ),
        key=lambda share: share["share"],
        reverse=True,
    )
    bases = select_bases(genres)
    base_keys = [base["key"] for base in bases]
    mix_by_genre = base_mix_for_genres(filtered, base_keys)
    genres = [
        share
        if share["key"] in base_keys
        else _with_base_mix(share, mix_by_genre.get(share["key"], []))
        for share in genres
    ]
    bases = genres[: len(bases)]

    return GenomeResult(
        schema_version=GENOME_RESULT_SCHEMA_VERSION,
        engine_version=ENGINE_VERSION,
        baseline_version=filtered.baseline.version,
        listener=filtered.listener,
        generated_at=filtered.params.now,
        stale=False,
        half_life_days=filtered.params.half_life_days,
        stats=_build_stats(filtered, total_w),
        genres=genres,
        bases=bases,
        divergence=divergence_facts(genre_vector, baseline_vector, _GENRE_LABELS),
        obscurity=obscurity_index(filtered),
        era=era_facts(filtered),
        loyalty=loyalty_facts(filtered),
        top_artists=top_artists(filtered),
        top_tracks=top_tracks(filtered),
        rhythm=rhythm_grid(filtered, tz_offset_seconds),
        players=player_split(filtered),
    )


def _filter_eligible(inputs: GenomeInputs) -> GenomeInputs:
    """Drop listens whose known play duration does not clear ``min_seconds_played`` (§3.2)."""
    floor_ms = inputs.params.min_seconds_played * 1000
    eligible = [
        listen
        for listen in inputs.listens
        if listen.played_ms is None or listen.played_ms >= floor_ms
    ]
    return replace(inputs, listens=eligible)


def _build_stats(inputs: GenomeInputs, total_w: float) -> GenomeStats:
    """Build the volume/coverage summary block of a :class:`GenomeResult`."""
    listens = inputs.listens
    coverage_by_source: dict[str, int] = {}
    known_genre_w = 0.0
    for listen in listens:
        coverage_by_source[listen.source] = coverage_by_source.get(listen.source, 0) + 1
        meta = inputs.artist_meta.get(listen.artist_key)
        if meta is not None and meta.genres:
            known_genre_w += listen_weight(listen, inputs.params)
    return GenomeStats(
        total_listens=len(listens),
        weighted_listens=_r4(total_w),
        distinct_artists=len({listen.artist_key for listen in listens}),
        distinct_tracks=len({listen.track_key for listen in listens}),
        first_listen=min((listen.played_at for listen in listens), default=None),
        last_listen=max((listen.played_at for listen in listens), default=None),
        coverage_by_source=coverage_by_source,
        enrichment_coverage=_r4(known_genre_w / total_w) if total_w > 0 else 0.0,
        # the pure engine has no visibility into `genome_artist_meta.resolve_state` across the
        # *whole* store (only per-listen artist_meta lookups); the controller overwrites these
        # with real counts from GenomeStore.artist_resolution_counts() after a rebuild (§3.8, P3)
        artists_pending=0,
        artists_resolved=0,
    )


def _genre_shares(
    p: Mapping[str, float],
    q: Mapping[str, float],
    terms: Mapping[str, float],
    jsd: float,
    labels: Mapping[str, str],
) -> list[GenreShare]:
    """Build one :class:`GenreShare` row per key in the union of ``p`` and ``q``."""
    shares: list[GenreShare] = []
    for key in sorted(set(p) | set(q)):
        share = p.get(key, 0.0)
        baseline_share = q.get(key, 0.0)
        ratio = min(share / max(baseline_share, 1e-6), 99.0)
        contribution = terms.get(key, 0.0) / jsd if jsd > 1e-9 else 0.0
        shares.append(
            GenreShare(
                key=key,
                label=labels.get(key, key),
                share=_r4(share),
                baseline_share=_r4(baseline_share),
                ratio=_r4(ratio),
                contribution=_r4(contribution),
                # filled in by ``build_genome`` for the main ``genres`` list only (§ DNA visual);
                # left empty here since this helper also builds ``divergence_facts``'s
                # top_over/top_under rows, which have no artist-level data to compute it from.
                base_mix=[],
            )
        )
    return shares


def _percentile_rank(listener_count: int, percentiles: Mapping[int, int]) -> float:
    """
    Approximate ``listener_count``'s percentile rank (0..100) against a percentile table.

    Log-linear interpolation between the table's known points; clamped to the table's ends.
    """
    points = sorted(percentiles.items())
    if not points:
        return 50.0
    if listener_count <= points[0][1]:
        return float(points[0][0])
    if listener_count >= points[-1][1]:
        return float(points[-1][0])
    for (pct_lo, count_lo), (pct_hi, count_hi) in itertools.pairwise(points):
        if count_lo <= listener_count <= count_hi:
            if count_hi == count_lo:
                return float(pct_lo)
            log_lo = math.log1p(count_lo)
            log_hi = math.log1p(count_hi)
            log_val = math.log1p(listener_count)
            fraction = (log_val - log_lo) / (log_hi - log_lo) if log_hi > log_lo else 0.0
            return pct_lo + fraction * (pct_hi - pct_lo)
    return 50.0


def _with_base_mix(share: GenreShare, base_mix: list[float]) -> GenreShare:
    """Return a copy of ``share`` with ``base_mix`` set, leaving every other field unchanged."""
    return GenreShare(
        key=share["key"],
        label=share["label"],
        share=share["share"],
        baseline_share=share["baseline_share"],
        ratio=share["ratio"],
        contribution=share["contribution"],
        base_mix=base_mix,
    )


def _round_fractions(values: list[float]) -> list[float]:
    """
    Round fractions that sum to ~1.0 to 4dp while keeping their sum exactly 1.0.

    Rounding each entry independently (as ``_r4`` does elsewhere) can drift the sum by more
    than a rounding error once several entries are involved; the last entry instead absorbs
    whatever the first ``n - 1`` rounded entries leave, so ``base_mix`` always sums to exactly
    1.0 rather than "close to" it.

    :param values: Fractions that sum to 1.0 (within floating-point error). Must be non-empty.
    """
    rounded = [_r4(value) for value in values[:-1]]
    rounded.append(_r4(1.0 - sum(rounded)))
    return rounded


def _normalize(values: dict[str, float]) -> dict[str, float]:
    """Return ``values`` scaled so its entries sum to 1, or unchanged if the total is 0."""
    total = sum(values.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in values.items()}


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the inclusive ``[low, high]`` range."""
    return max(low, min(high, value))


def _r4(value: float) -> float:
    """Round ``value`` to 4 decimal places (§3.4: every emitted float is rounded to 4dp)."""
    return round(value, _ROUND_DP)
