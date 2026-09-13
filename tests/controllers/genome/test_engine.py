"""
Unit tests for the pure genome computation engine (``music_assistant/controllers/genome/engine.py``).

Covers every assertion listed in ``docs/ARCHITECTURE.md`` Part 4 "Engine tests must include"
plus targeted coverage of the remaining public functions. Pure-engine tests need no ``mass``
fixture (§1.11).
"""

from __future__ import annotations

from music_assistant.controllers.genome import engine
from music_assistant.controllers.genome.models import (
    ArtistMeta,
    Baseline,
    EngineParams,
    GenomeInputs,
    Listen,
)

NOW = 1_757_000_000
DAY = 86400


def _listen(**overrides: object) -> Listen:
    base: dict[str, object] = {
        "played_at": NOW,
        "artist_key": "artist",
        "artist_name": "Artist",
        "track_key": "track",
        "track_name": "Track",
        "album_name": None,
        "source": "ma_playlog",
        "player_id": None,
        "duration_ms": None,
        "played_ms": None,
        "fully_played": True,
        "confidence": 1.0,
    }
    base.update(overrides)
    return Listen(**base)  # type: ignore[arg-type]


def _baseline(**overrides: object) -> Baseline:
    base: dict[str, object] = {
        "version": "test",
        "genre_shares": {"rock": 0.5, "pop": 0.5},
        "era_shares": {1990: 1.0},
        "listener_percentiles": {5: 10, 10: 20, 25: 50, 50: 200, 75: 1000, 90: 5000},
        "concentration": 0.1,
    }
    base.update(overrides)
    return Baseline(**base)  # type: ignore[arg-type]


def _params(**overrides: object) -> EngineParams:
    base: dict[str, object] = {
        "now": NOW,
        "half_life_days": 548,
        "obscurity_percentile": 25,
        "min_seconds_played": 30,
    }
    base.update(overrides)
    return EngineParams(**base)  # type: ignore[arg-type]


def _inputs(
    listens: list[Listen],
    *,
    artist_meta: dict[str, ArtistMeta] | None = None,
    baseline: Baseline | None = None,
    params: EngineParams | None = None,
    player_names: dict[str, str] | None = None,
) -> GenomeInputs:
    return GenomeInputs(
        listener="household",
        listens=listens,
        artist_meta=artist_meta or {},
        baseline=baseline or _baseline(),
        params=params or _params(),
        player_names=player_names or {},
    )


# ---------------------------------------------------------------------------------------
# Required assertions (Part 4 "Engine tests must include")
# ---------------------------------------------------------------------------------------


def test_jsd_identical_distributions_is_zero() -> None:
    """Test jsd identical distributions is zero."""
    p = {"rock": 0.5, "pop": 0.5}
    assert engine.js_divergence(p, dict(p)) == 0.0


def test_jsd_disjoint_distributions_is_one() -> None:
    """Test jsd disjoint distributions is one."""
    p = {"rock": 1.0}
    q = {"pop": 1.0}
    assert abs(engine.js_divergence(p, q) - 1.0) < 1e-9


def test_js_contributions_sum_to_js_divergence() -> None:
    """Test js contributions sum to js divergence."""
    p = {"rock": 0.7, "pop": 0.3}
    q = {"rock": 0.2, "jazz": 0.8}
    jsd = engine.js_divergence(p, q)
    contributions = engine.js_contributions(p, q)
    assert abs(sum(contributions.values()) - jsd) < 1e-9


def test_half_life_548_halves_weight_at_548_days() -> None:
    """Test half life 548 halves weight at 548 days."""
    params = _params(half_life_days=548)
    listen = _listen(played_at=NOW - 548 * DAY)
    assert abs(engine.listen_weight(listen, params) - 0.5) < 1e-6


def test_half_life_zero_disables_decay() -> None:
    """Test half life zero disables decay."""
    params = _params(half_life_days=0)
    listen = _listen(played_at=NOW - 10_000 * DAY)
    assert engine.listen_weight(listen, params) == 1.0


def test_three_genre_listen_splits_weight_evenly() -> None:
    """Test three genre listen splits weight evenly."""
    meta = ArtistMeta(
        artist_key="artist",
        artist_name="Artist",
        mbid=None,
        genres=("rock", "pop", "jazz"),
        first_release_year=2000,
        lb_listeners=100,
        lb_listen_count=1000,
    )
    inputs = _inputs([_listen()], artist_meta={"artist": meta})
    vector = engine.build_genre_vector(inputs)
    assert abs(vector["rock"] - 1 / 3) < 1e-9
    assert abs(vector["pop"] - 1 / 3) < 1e-9
    assert abs(vector["jazz"] - 1 / 3) < 1e-9


def test_obscurity_known_share_zero_does_not_divide_by_zero() -> None:
    """Test obscurity known share zero does not divide by zero."""
    inputs = _inputs([_listen(artist_key="unknown_artist")], artist_meta={})
    facts = engine.obscurity_index(inputs)
    assert facts["index"] == 0.0
    assert facts["known_share"] == 0.0


def test_rhythm_grid_always_returns_168_cells() -> None:
    """Test rhythm grid always returns 168 cells."""
    inputs = _inputs([])
    assert len(engine.rhythm_grid(inputs, 0)) == 168
    inputs_with_data = _inputs([_listen(), _listen(played_at=NOW - DAY)])
    assert len(engine.rhythm_grid(inputs_with_data, 3600)) == 168


def test_concentration_is_one_for_single_artist() -> None:
    """Test concentration is one for single artist."""
    inputs = _inputs([_listen(), _listen(played_at=NOW - DAY)])
    facts = engine.loyalty_facts(inputs)
    assert facts["concentration"] == 1.0


def test_empty_input_returns_well_formed_zeroed_genome_result() -> None:
    """Test empty input returns well formed zeroed genome result."""
    inputs = _inputs([])
    result = engine.build_genome(inputs)
    assert result["stats"]["total_listens"] == 0
    assert result["stats"]["weighted_listens"] == 0.0
    assert result["genres"] == []
    assert result["bases"] == []
    assert result["divergence"]["score"] == 0.0
    assert result["divergence"]["percent"] == 0
    assert result["divergence"]["top_over"] == []
    assert result["divergence"]["top_under"] == []
    assert result["obscurity"]["index"] == 0.0
    assert result["era"]["buckets"] == []
    assert result["loyalty"]["concentration"] == 0.0
    assert result["top_artists"] == []
    assert result["top_tracks"] == []
    assert len(result["rhythm"]) == 168
    assert result["players"] == []
    assert result["stats"]["first_listen"] is None
    assert result["stats"]["last_listen"] is None


# ---------------------------------------------------------------------------------------
# Additional coverage
# ---------------------------------------------------------------------------------------


def test_listen_weight_uses_completion_when_not_fully_played() -> None:
    """Test listen weight uses completion when not fully played."""
    params = _params(half_life_days=0)
    listen = _listen(fully_played=False, duration_ms=200_000, played_ms=100_000)
    assert engine.listen_weight(listen, params) == 0.5


def test_listen_weight_completion_defaults_to_one_when_unknown() -> None:
    """Test listen weight completion defaults to one when unknown."""
    params = _params(half_life_days=0)
    listen = _listen(fully_played=False, duration_ms=None, played_ms=None)
    assert engine.listen_weight(listen, params) == 1.0


def test_listen_weight_applies_confidence() -> None:
    """Test listen weight applies confidence."""
    params = _params(half_life_days=0)
    listen = _listen(confidence=0.5)
    assert engine.listen_weight(listen, params) == 0.5


def test_genre_vector_excludes_listens_with_no_genre_meta() -> None:
    """Test genre vector excludes listens with no genre meta."""
    inputs = _inputs([_listen(artist_key="unknown")], artist_meta={})
    assert engine.build_genre_vector(inputs) == {}


def test_genre_vector_normalizes_across_multiple_listens() -> None:
    """Test genre vector normalizes across multiple listens."""
    meta_a = ArtistMeta("a", "A", None, ("rock",), 2000, 100, 1000)
    meta_b = ArtistMeta("b", "B", None, ("pop",), 2000, 100, 1000)
    inputs = _inputs(
        [
            _listen(artist_key="a", track_key="t1"),
            _listen(artist_key="a", track_key="t2"),
            _listen(artist_key="b", track_key="t3"),
        ],
        artist_meta={"a": meta_a, "b": meta_b},
    )
    vector = engine.build_genre_vector(inputs)
    assert abs(vector["rock"] - 2 / 3) < 1e-9
    assert abs(vector["pop"] - 1 / 3) < 1e-9


def test_divergence_facts_top_over_and_under_and_labels() -> None:
    """Test divergence facts top over and under and labels."""
    p = {"rock": 0.8, "pop": 0.2}
    q = {"rock": 0.2, "pop": 0.8}
    labels = {"rock": "Rock", "pop": "Pop"}
    facts = engine.divergence_facts(p, q, labels)
    assert facts["score"] > 0
    assert facts["percent"] == round(facts["score"] * 100)
    assert facts["top_over"][0]["key"] == "rock"
    assert facts["top_over"][0]["label"] == "Rock"
    assert facts["top_under"][0]["key"] == "pop"
    total_contribution = sum(s["contribution"] for s in facts["top_over"] + facts["top_under"])
    assert 0 < total_contribution <= 1.0001


def test_obscurity_index_counts_below_threshold_weight() -> None:
    """Test obscurity index counts below threshold weight."""
    meta_obscure = ArtistMeta("obscure", "Obscure", None, (), None, 5, 50)
    meta_popular = ArtistMeta("popular", "Popular", None, (), None, 100_000, 1_000_000)
    inputs = _inputs(
        [_listen(artist_key="obscure"), _listen(artist_key="popular", track_key="t2")],
        artist_meta={"obscure": meta_obscure, "popular": meta_popular},
    )
    facts = engine.obscurity_index(inputs)
    assert facts["known_share"] == 1.0
    assert facts["index"] == 0.5
    assert facts["threshold_listeners"] == 50


def test_era_facts_center_of_mass_and_buckets() -> None:
    """Test era facts center of mass and buckets."""
    meta_a = ArtistMeta("a", "A", None, (), 1990, 100, 1000)
    meta_b = ArtistMeta("b", "B", None, (), 2010, 100, 1000)
    inputs = _inputs(
        [
            _listen(artist_key="a", track_key="t1"),
            _listen(artist_key="b", track_key="t2"),
        ],
        artist_meta={"a": meta_a, "b": meta_b},
        params=_params(half_life_days=0),
    )
    facts = engine.era_facts(inputs)
    assert facts["center_of_mass"] == 2000.0
    assert facts["known_share"] == 1.0
    decades = {bucket["decade"]: bucket["share"] for bucket in facts["buckets"]}
    assert decades[1990] == 0.5
    assert decades[2010] == 0.5


def test_era_facts_handles_no_known_years() -> None:
    """Test era facts handles no known years."""
    inputs = _inputs([_listen(artist_key="unknown")], artist_meta={})
    facts = engine.era_facts(inputs)
    assert facts["center_of_mass"] == 0.0
    assert facts["spread"] == 0.0
    assert facts["known_share"] == 0.0
    assert facts["buckets"] == []


def test_loyalty_exploration_ratio_for_new_artist() -> None:
    """Test loyalty exploration ratio for new artist."""
    inputs = _inputs(
        [_listen(artist_key="new_artist", played_at=NOW - DAY)],
        params=_params(half_life_days=0, now=NOW),
    )
    facts = engine.loyalty_facts(inputs)
    assert facts["exploration_ratio"] == 1.0
    assert facts["new_artists_90d"] == 1


def test_loyalty_repeat_rate() -> None:
    """Test loyalty repeat rate."""
    inputs = _inputs(
        [
            _listen(track_key="t1"),
            _listen(track_key="t1"),
            _listen(track_key="t2"),
        ],
        params=_params(half_life_days=0),
    )
    facts = engine.loyalty_facts(inputs)
    assert facts["repeat_rate"] == round(1 - 2 / 3, 4)


def test_top_artists_sorted_and_limited() -> None:
    """Test top artists sorted and limited."""
    params = _params(half_life_days=0, top_n=1)
    inputs = _inputs(
        [
            _listen(artist_key="a", artist_name="A", track_key="t1"),
            _listen(artist_key="a", artist_name="A", track_key="t2"),
            _listen(artist_key="b", artist_name="B", track_key="t3"),
        ],
        params=params,
    )
    top = engine.top_artists(inputs)
    assert len(top) == 1
    assert top[0]["artist_key"] == "a"
    assert top[0]["plays"] == 2


def test_top_artists_obscurity_and_ratio_vs_average() -> None:
    """Test top artists obscurity and ratio vs average."""
    meta = ArtistMeta("a", "A", None, (), None, 5, 50)
    inputs = _inputs(
        [_listen(artist_key="a")],
        artist_meta={"a": meta},
        params=_params(half_life_days=0),
    )
    top = engine.top_artists(inputs)
    assert top[0]["obscurity"] is not None
    assert 0 <= top[0]["obscurity"] <= 1
    assert top[0]["ratio_vs_average"] is not None


def test_top_tracks_sorted_and_limited() -> None:
    """Test top tracks sorted and limited."""
    params = _params(half_life_days=0, top_n=1)
    inputs = _inputs(
        [
            _listen(track_key="t1", track_name="T1"),
            _listen(track_key="t1", track_name="T1"),
            _listen(track_key="t2", track_name="T2"),
        ],
        params=params,
    )
    top = engine.top_tracks(inputs)
    assert len(top) == 1
    assert top[0]["track_key"] == "t1"
    assert top[0]["plays"] == 2


def test_player_split_excludes_unknown_player_and_normalizes() -> None:
    """Test player split excludes unknown player and normalizes."""
    inputs = _inputs(
        [
            _listen(player_id="kitchen", track_key="t1"),
            _listen(player_id="kitchen", track_key="t2"),
            _listen(player_id=None, track_key="t3"),
            _listen(player_id="office", track_key="t4"),
        ],
        params=_params(half_life_days=0),
        player_names={"kitchen": "Kitchen", "office": "Office"},
    )
    splits = engine.player_split(inputs)
    total_share = sum(split["share"] for split in splits)
    assert abs(total_share - 1.0) < 1e-6
    by_id = {split["player_id"]: split for split in splits}
    assert by_id["kitchen"]["share"] == round(2 / 3, 4)
    assert by_id["kitchen"]["name"] == "Kitchen"


def test_build_genome_min_seconds_played_filters_short_listens() -> None:
    """Test build genome min seconds played filters short listens."""
    params = _params(half_life_days=0, min_seconds_played=30)
    inputs = _inputs(
        [
            _listen(track_key="short", played_ms=5_000, duration_ms=200_000),
            _listen(track_key="long", played_ms=60_000, duration_ms=200_000),
        ],
        params=params,
    )
    result = engine.build_genome(inputs)
    assert result["stats"]["total_listens"] == 1


def test_build_genome_respects_tz_offset_for_rhythm() -> None:
    # 1970-01-01T00:00:00Z is a Thursday (weekday index 3); a -1h offset should land on
    # 1969-12-31 23:00 UTC-equivalent local time, i.e. Wednesday (index 2) hour 23.
    """Test build genome respects tz offset for rhythm."""
    listen = _listen(played_at=0, fully_played=True)
    inputs = _inputs([listen], params=_params(now=0, half_life_days=0))
    cells_utc = engine.rhythm_grid(inputs, 0)
    cells_offset = engine.rhythm_grid(inputs, -3600)
    utc_hit = next(c for c in cells_utc if c["weight"] > 0)
    offset_hit = next(c for c in cells_offset if c["weight"] > 0)
    assert utc_hit["weekday"] == 3
    assert utc_hit["hour"] == 0
    assert offset_hit["weekday"] == 2
    assert offset_hit["hour"] == 23


def test_build_genome_full_result_shape_is_stable() -> None:
    """Test build genome full result shape is stable."""
    meta = ArtistMeta("a", "A", None, ("rock",), 1999, 200, 2000)
    inputs = _inputs(
        [_listen(artist_key="a", artist_name="A", player_id="kitchen")],
        artist_meta={"a": meta},
        player_names={"kitchen": "Kitchen"},
    )
    result = engine.build_genome(inputs)
    assert result["schema_version"] == 1
    assert result["engine_version"] == engine.ENGINE_VERSION
    assert result["listener"] == "household"
    assert result["stats"]["total_listens"] == 1
    assert result["genres"][0]["key"] == "rock"
    assert result["top_artists"][0]["artist_key"] == "a"
    assert result["players"][0]["player_id"] == "kitchen"


# ---------------------------------------------------------------------------------------
# "Four bases" DNA visual: select_bases / base_mix_for_genres / ratio exposure
# ---------------------------------------------------------------------------------------


def test_select_bases_returns_top_four_in_order() -> None:
    """Test select bases returns top four in order when five or more genres exist."""
    genres = [
        {"key": key, "share": share}  # type: ignore[typeddict-item]
        for key, share in (
            ("rock", 0.4),
            ("pop", 0.3),
            ("jazz", 0.15),
            ("folk", 0.1),
            ("soul", 0.05),
        )
    ]
    bases = engine.select_bases(genres)  # type: ignore[arg-type]
    assert [b["key"] for b in bases] == ["rock", "pop", "jazz", "folk"]


def test_select_bases_returns_exactly_four_when_exactly_four_exist() -> None:
    """Test select bases returns exactly four when exactly four genres exist."""
    genres = [
        {"key": key, "share": share}  # type: ignore[typeddict-item]
        for key, share in (("rock", 0.4), ("pop", 0.3), ("jazz", 0.2), ("folk", 0.1))
    ]
    bases = engine.select_bases(genres)  # type: ignore[arg-type]
    assert len(bases) == 4
    assert [b["key"] for b in bases] == ["rock", "pop", "jazz", "folk"]


def test_select_bases_returns_fewer_than_four_without_padding() -> None:
    """Test select bases never pads with fabricated entries when fewer than four exist."""
    genres = [
        {"key": key, "share": share}  # type: ignore[typeddict-item]
        for key, share in (("rock", 0.6), ("pop", 0.4))
    ]
    bases = engine.select_bases(genres)  # type: ignore[arg-type]
    assert len(bases) == 2
    assert [b["key"] for b in bases] == ["rock", "pop"]

    assert engine.select_bases([]) == []  # type: ignore[arg-type]


def test_base_mix_sums_to_one() -> None:
    """Test a genre with real overlap against multiple bases gets a base_mix summing to 1.0."""
    # "indie" artist also tagged "rock"; another "indie" artist also tagged "pop" — indie's
    # affinity should split across both bases proportional to listen weight.
    meta = {
        "a1": ArtistMeta("a1", "A1", None, ("indie", "rock"), None, None, None),
        "a2": ArtistMeta("a2", "A2", None, ("indie", "pop"), None, None, None),
        "a3": ArtistMeta("a3", "A3", None, ("rock",), None, None, None),
        "a4": ArtistMeta("a4", "A4", None, ("pop",), None, None, None),
        # give the engine two more, higher-share genres so "indie" itself isn't a base — it
        # should be the one *measured* genre whose base_mix we assert on.
        "a5": ArtistMeta("a5", "A5", None, ("jazz",), None, None, None),
        "a6": ArtistMeta("a6", "A6", None, ("folk",), None, None, None),
    }
    inputs = _inputs(
        [
            _listen(artist_key="a1", track_key="t1"),
            _listen(artist_key="a2", track_key="t2"),
            _listen(artist_key="a3", track_key="t3"),
            _listen(artist_key="a4", track_key="t4"),
            _listen(artist_key="a5", track_key="t5a"),
            _listen(artist_key="a5", track_key="t5b"),
            _listen(artist_key="a6", track_key="t6a"),
            _listen(artist_key="a6", track_key="t6b"),
        ],
        artist_meta=meta,
        params=_params(half_life_days=0),
    )
    result = engine.build_genome(inputs)
    indie = next(g for g in result["genres"] if g["key"] == "indie")
    assert "indie" not in {b["key"] for b in result["bases"]}
    assert indie["base_mix"]
    assert abs(sum(indie["base_mix"]) - 1.0) < 1e-9
    assert len(indie["base_mix"]) == len(result["bases"])
    # indie has one listen overlapping "rock" and one overlapping "pop", none overlapping the
    # other two bases => an even split across exactly those two positions.
    base_order = [b["key"] for b in result["bases"]]
    by_base = dict(zip(base_order, indie["base_mix"], strict=True))
    assert by_base["rock"] == 0.5
    assert by_base["pop"] == 0.5
    assert by_base["jazz"] == 0.0
    assert by_base["folk"] == 0.0


def test_base_mix_zero_overlap_is_empty_not_uniform() -> None:
    """Test a genre with zero overlap with any base gets an empty base_mix, not a 25pct split."""
    meta = {
        "rock": ArtistMeta("rock", "Rock", None, ("rock",), None, None, None),
        "pop": ArtistMeta("pop", "Pop", None, ("pop",), None, None, None),
        "jazz": ArtistMeta("jazz", "Jazz", None, ("jazz",), None, None, None),
        "folk": ArtistMeta("folk", "Folk", None, ("folk",), None, None, None),
        # "ambient" only ever co-occurs with "drone", neither of which is a base below.
        "ambient1": ArtistMeta(
            "ambient1", "Ambient1", None, ("ambient", "drone"), None, None, None
        ),
        "drone": ArtistMeta("drone", "Drone", None, ("drone",), None, None, None),
    }
    listens = (
        [_listen(artist_key="rock", track_key=f"r{i}") for i in range(4)]
        + [_listen(artist_key="pop", track_key=f"p{i}") for i in range(3)]
        + [_listen(artist_key="jazz", track_key=f"j{i}") for i in range(2)]
        + [_listen(artist_key="folk", track_key=f"f{i}") for i in range(2)]
        + [_listen(artist_key="drone", track_key="d0")]
        + [_listen(artist_key="ambient1", track_key="a0")]
    )
    inputs = _inputs(listens, artist_meta=meta, params=_params(half_life_days=0))
    result = engine.build_genome(inputs)
    base_keys = {b["key"] for b in result["bases"]}
    assert base_keys == {"rock", "pop", "jazz", "folk"}
    assert "ambient" not in base_keys
    assert "drone" not in base_keys
    ambient = next(g for g in result["genres"] if g["key"] == "ambient")
    assert ambient["base_mix"] == []


def test_base_mix_single_genre_artist_contributes_no_overlap() -> None:
    """Test an artist resolved to only one genre contributes to no overlap for that genre."""
    meta = {
        # every "shoegaze" listen comes from single-genre artists, so shoegaze can never
        # overlap with any base no matter how many listens it accumulates.
        "a1": ArtistMeta("a1", "A1", None, ("shoegaze",), None, None, None),
        "a2": ArtistMeta("a2", "A2", None, ("rock",), None, None, None),
        "a3": ArtistMeta("a3", "A3", None, ("pop",), None, None, None),
    }
    inputs = _inputs(
        [
            _listen(artist_key="a1", track_key="t1"),
            _listen(artist_key="a2", track_key="t2"),
            _listen(artist_key="a3", track_key="t3"),
        ],
        artist_meta=meta,
        params=_params(half_life_days=0),
    )
    mix = engine.base_mix_for_genres(inputs, ["rock", "pop"])
    assert mix["shoegaze"] == []


def test_base_mix_excludes_unenriched_artists() -> None:
    """Test listens from an unresolved (no genre) artist never enter the overlap calculation."""
    meta = {
        "a1": ArtistMeta("a1", "A1", None, ("indie", "rock"), None, None, None),
        "a2": ArtistMeta("a2", "A2", None, ("rock",), None, None, None),
        # "a3" is unenriched: no meta row at all.
    }
    inputs = _inputs(
        [
            _listen(artist_key="a1", track_key="t1"),
            _listen(artist_key="a2", track_key="t2"),
            _listen(artist_key="a3", track_key="t3"),
        ],
        artist_meta=meta,
        params=_params(half_life_days=0),
    )
    mix = engine.base_mix_for_genres(inputs, ["rock"])
    # a3 has no meta at all, so it must not silently register as "no overlap" weight for
    # any genre — "indie" only ever came from a1, which fully overlaps "rock".
    assert mix["indie"] == [1.0]


def test_base_mix_for_genres_empty_base_keys_returns_empty_dict() -> None:
    """Test base_mix_for_genres with no bases (e.g. a brand-new listener) returns {}."""
    inputs = _inputs([_listen()], params=_params(half_life_days=0))
    assert engine.base_mix_for_genres(inputs, []) == {}


def test_genre_share_ratio_exposed_on_every_genre_not_only_top_over_under() -> None:
    """Test every entry in the genres list carries its own ratio, not only the top over/under."""
    meta = {
        "a1": ArtistMeta("a1", "A1", None, ("rock", "jazz"), None, None, None),
    }
    inputs = _inputs(
        [_listen(artist_key="a1")],
        artist_meta=meta,
        baseline=_baseline(genre_shares={"rock": 0.9, "jazz": 0.1}),
        params=_params(half_life_days=0),
    )
    result = engine.build_genome(inputs)
    assert len(result["genres"]) == 2
    for genre in result["genres"]:
        assert "ratio" in genre
        expected = min(genre["share"] / max(genre["baseline_share"], 1e-6), 99.0)
        assert abs(genre["ratio"] - round(expected, 4)) < 1e-6


def test_base_genre_rows_carry_empty_base_mix() -> None:
    """Test a genre that is itself one of the bases never gets a self-affinity base_mix."""
    meta = {
        "a1": ArtistMeta("a1", "A1", None, ("rock",), None, None, None),
    }
    inputs = _inputs(
        [_listen(artist_key="a1")],
        artist_meta=meta,
        params=_params(half_life_days=0),
    )
    result = engine.build_genome(inputs)
    assert result["bases"][0]["key"] == "rock"
    assert result["bases"][0]["base_mix"] == []
