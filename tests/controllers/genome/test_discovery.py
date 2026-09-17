"""
Tests for the discovery feature: cold library corners and Last.fm suggestions (D-16).

The load-bearing test here is :func:`test_discovery_read_path_never_touches_the_network`.
D-16 exists because a websocket read fell through to an inline MusicBrainz pass and hung the
page for an hour, and every other test in this file would still pass if that happened again.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import pytest

from music_assistant.controllers.genome.constants import (
    DISCOVERY_STATE_PENDING,
    DISCOVERY_STATE_READY,
    DISCOVERY_STATE_UNAVAILABLE,
    GENOME_DISCOVERY_SEED_ERROR_COOLDOWN_HOURS,
)
from music_assistant.controllers.genome.discovery import (
    DivergentGenre,
    LibraryArtist,
    divergent_genres,
    rank_cold_corners,
    select_seeds,
)
from music_assistant.controllers.genome.enrich.lastfm_similar import fetch_similar_artists
from music_assistant.controllers.genome.errors import LastfmApiError

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from music_assistant.controllers.genome.controller import GenomeController
    from music_assistant.controllers.genome.models import GenomeResult
    from tests.controllers.genome.conftest import FixtureHttpClient, StubGenomeStore

_API_KEY = "a" * 32


# ---------------------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------------------


def _genome(**overrides: Any) -> GenomeResult:
    """Build a genome result carrying just the fields discovery reads."""
    result: dict[str, Any] = {
        "divergence": {
            "score": 0.42,
            "percent": 42,
            "top_over": [
                {"key": "ambient", "label": "Ambient", "contribution": 0.31},
                {"key": "rock", "label": "Rock", "contribution": 0.12},
            ],
            "top_under": [],
        },
        "top_artists": [
            {"name": "Sigur Rós", "artist_key": "sigurros", "genres": ["rock", "ambient"]},
            {"name": "AC/DC", "artist_key": "acdc", "genres": ["metal"]},
            {"name": "Portishead", "artist_key": "portishead", "genres": ["rock"]},
        ],
    }
    result.update(overrides)
    return cast("GenomeResult", result)


_LIBRARY = [
    # never played, in the strongest divergent genre — the headline cold corner
    LibraryArtist("brianeno", "Brian Eno", 0, ("ambient",)),
    # played twice, still cold, but in the weaker divergent genre
    LibraryArtist("thecure", "The Cure", 2, ("rock", "pop")),
    # never played, but in no divergent genre at all
    LibraryArtist("dollyparton", "Dolly Parton", 0, ("country",)),
    # a real habit, not a corner
    LibraryArtist("sigurros", "Sigur Rós", 41, ("ambient", "rock")),
    # zero library plays, but a deep imported history — not cold
    LibraryArtist("portishead", "Portishead", 0, ("rock",)),
    # never played, ambient again: ranks with Brian Eno, tie-broken on name
    LibraryArtist("aphextwin", "Aphex Twin", 1, ("ambient",)),
]


class _ExplodingHttpClient:
    """An ``HttpClient`` that fails the test if anything asks it for a request."""

    async def get_json(self, url: str, **_kwargs: Any) -> Any:
        """Fail: a read path must never issue a GET."""
        raise AssertionError(f"discovery read path issued a network GET to {url}")

    async def post_json(self, url: str, **_kwargs: Any) -> Any:
        """Fail: a read path must never issue a POST."""
        raise AssertionError(f"discovery read path issued a network POST to {url}")


class _ExplodingSession:
    """An ``aiohttp`` session double that fails the test if anything opens a request on it."""

    def get(self, url: str, **_kwargs: Any) -> Any:
        """Fail: a read path must never open a GET."""
        raise AssertionError(f"discovery read path opened a session GET to {url}")

    def post(self, url: str, **_kwargs: Any) -> Any:
        """Fail: a read path must never open a POST."""
        raise AssertionError(f"discovery read path opened a session POST to {url}")


def _wire(
    controller: GenomeController,
    *,
    library: list[LibraryArtist] | None = None,
    api_key: str = "",
    client: Any = None,
) -> None:
    """Point a controller at an in-memory library, a fixed API key and an injected HTTP client."""
    artists = list(library if library is not None else _LIBRARY)

    async def _reader(_mass: Any) -> list[LibraryArtist]:
        return artists

    controller._library_artists_reader = _reader  # type: ignore[assignment]
    if client is not None:
        controller._lastfm_similar_client_factory = lambda _mass: client  # type: ignore[assignment]
    original = controller.get_config_value

    def _get_config_value(key: str, default: Any = None, *, return_type: Any = None) -> Any:
        if key == "lastfm_api_key":
            return api_key
        return original(key, default, return_type=return_type)

    controller.get_config_value = _get_config_value  # type: ignore[method-assign]


# ---------------------------------------------------------------------------------------
# divergent genres + cold-corner ranking (pure)
# ---------------------------------------------------------------------------------------


def test_divergent_genres_reuses_the_engines_own_decomposition() -> None:
    """The divergent genres are the engine's `divergence.top_over` terms, in its own order."""
    genres = divergent_genres(_genome())
    assert [(g.key, g.label) for g in genres] == [("ambient", "Ambient"), ("rock", "Rock")]
    assert genres[0].contribution == pytest.approx(0.31)


def test_divergent_genres_handles_no_genome() -> None:
    """With no cached genome at all there is nothing to diverge from."""
    assert divergent_genres(None) == []


def test_rank_cold_corners_ranks_by_divergent_genre_then_plays() -> None:
    """Cold artists sort by their genre's divergence contribution, then by how little played."""
    ranked = rank_cold_corners(
        _LIBRARY,
        divergent_genres(_genome()),
        max_plays=2,
        limit=25,
        extra_plays={"portishead": 300},
    )
    assert [(row.artist_name, row.plays, row.genre_key) for row in ranked] == [
        ("Brian Eno", 0, "ambient"),
        ("Aphex Twin", 1, "ambient"),
        ("The Cure", 2, "rock"),
    ]
    assert ranked[0].genre_label == "Ambient"


def test_rank_cold_corners_uses_the_highest_known_play_count() -> None:
    """A zeroed library counter must not present an artist with a deep import as untouched."""
    ranked = rank_cold_corners(
        [LibraryArtist("portishead", "Portishead", 0, ("rock",))],
        divergent_genres(_genome()),
        max_plays=2,
        limit=25,
        extra_plays={"portishead": 300},
    )
    assert ranked == []


def test_rank_cold_corners_without_divergent_genres_is_empty() -> None:
    """With no divergent genres there is nothing to rank a cold artist against."""
    assert rank_cold_corners(_LIBRARY, [], max_plays=2, limit=25) == []


def test_rank_cold_corners_respects_the_limit() -> None:
    """The result is capped at `limit`, keeping the best-fitting rows."""
    ranked = rank_cold_corners(_LIBRARY, divergent_genres(_genome()), max_plays=2, limit=1)
    assert [row.artist_name for row in ranked] == ["Brian Eno"]


def test_select_seeds_picks_top_artists_in_divergent_genres() -> None:
    """Seeds come from the household's own top artists, best-fitting divergent genre first."""
    seeds = select_seeds(_genome(), divergent_genres(_genome()), limit=8)
    assert [(seed.artist_name, seed.genre.key) for seed in seeds] == [
        ("Sigur Rós", "ambient"),
        ("Portishead", "rock"),
    ]


def test_select_seeds_respects_the_limit() -> None:
    """A large genome must not seed an unbounded number of Last.fm requests."""
    assert len(select_seeds(_genome(), divergent_genres(_genome()), limit=1)) == 1


# ---------------------------------------------------------------------------------------
# the read path — `genome/discovery`
# ---------------------------------------------------------------------------------------


async def test_discovery_read_path_never_touches_the_network(
    genome_controller: GenomeController, genome_store: StubGenomeStore, mass_stub: MagicMock
) -> None:
    """
    The single most important guarantee in this feature (D-16).

    Both seams a network call could come through — the injected HTTP client and the shared
    aiohttp session — raise on use. A read that reaches either one fails here rather than
    hanging a user's page.
    """
    genome_store.cache["household"] = _genome()
    genome_store.artist_plays = {"portishead": 300}
    genome_store.discovery["household"] = {
        "generated_at": 1_700_000_000.0,
        "suggested": [
            {
                "artist_name": "Jónsi",
                "mbid": None,
                "seed_artist": "Sigur Rós",
                "genre_key": "ambient",
                "genre_label": "Ambient",
                "match": 1.0,
            }
        ],
        "seed_failures": {},
    }
    mass_stub.http_session = _ExplodingSession()
    _wire(genome_controller, api_key=_API_KEY, client=_ExplodingHttpClient())

    result = await genome_controller.discovery()

    assert result.suggested_state == DISCOVERY_STATE_READY
    assert [row.artist_name for row in result.in_library] == [
        "Brian Eno",
        "Aphex Twin",
        "The Cure",
    ]
    assert [row.artist_name for row in result.suggested] == ["Jónsi"]
    assert result.generated_at == 1_700_000_000.0


async def test_discovery_never_rebuilds_the_genome(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """A read must serve the cached genome, never fall through to a recompute."""
    genome_store.cache["household"] = _genome()
    _wire(genome_controller, api_key=_API_KEY)
    calls: list[str] = []
    original_rebuild = genome_controller._rebuild

    async def _tracking_rebuild(listener: str) -> Any:
        calls.append(listener)
        return await original_rebuild(listener)

    genome_controller._rebuild = _tracking_rebuild  # type: ignore[method-assign]

    await genome_controller.discovery()

    assert calls == []


async def test_discovery_without_lastfm_key_is_unavailable_not_an_error(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """No API key is a normal state: the cold-corner half of the card still renders."""
    genome_store.cache["household"] = _genome()
    genome_store.artist_plays = {"portishead": 300}
    _wire(genome_controller, api_key="")

    result = await genome_controller.discovery()

    assert result.suggested_state == DISCOVERY_STATE_UNAVAILABLE
    assert result.suggested == []
    assert result.generated_at is None
    assert [row.artist_name for row in result.in_library] == [
        "Brian Eno",
        "Aphex Twin",
        "The Cure",
    ]


async def test_discovery_before_the_first_pass_is_pending(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """A configured key with nothing stored yet reads as pending, not as an empty result."""
    genome_store.cache["household"] = _genome()
    _wire(genome_controller, api_key=_API_KEY)

    result = await genome_controller.discovery()

    assert result.suggested_state == DISCOVERY_STATE_PENDING
    assert result.generated_at is None


async def test_discovery_with_no_cached_genome_is_empty(
    genome_controller: GenomeController,
) -> None:
    """No genome means no divergent genres, which is an honest empty state, not a rebuild."""
    _wire(genome_controller, api_key=_API_KEY)

    result = await genome_controller.discovery()

    assert result.in_library == []
    assert result.suggested == []
    assert result.suggested_state == DISCOVERY_STATE_PENDING


async def test_discovery_with_an_empty_library_is_empty(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """A genome with divergent genres but nothing in the library yields no cold corners."""
    genome_store.cache["household"] = _genome()
    _wire(genome_controller, library=[], api_key=_API_KEY)

    result = await genome_controller.discovery()

    assert result.in_library == []


async def test_discovery_with_no_divergent_genres_is_empty(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """A household indistinguishable from the baseline has no corners to point at."""
    genome_store.cache["household"] = _genome(
        divergence={"score": 0.0, "percent": 0, "top_over": [], "top_under": []}
    )
    _wire(genome_controller, api_key=_API_KEY)

    result = await genome_controller.discovery()

    assert result.in_library == []


async def test_discovery_survives_an_unreadable_library(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """A library read failure yields an empty card, never an error popup at the user."""
    genome_store.cache["household"] = _genome()
    _wire(genome_controller, api_key=_API_KEY)

    async def _broken(_mass: Any) -> list[LibraryArtist]:
        raise RuntimeError("library unavailable")

    genome_controller._library_artists_reader = _broken  # type: ignore[assignment]

    result = await genome_controller.discovery()

    assert result.in_library == []


# ---------------------------------------------------------------------------------------
# `genome/discovery_refresh` — dispatch only, never awaited
# ---------------------------------------------------------------------------------------


async def test_discovery_refresh_dispatches_without_blocking(
    genome_controller: GenomeController, genome_store: StubGenomeStore, mass_stub: MagicMock
) -> None:
    """The refresh command hands the pass to the background and returns straight away."""
    genome_store.cache["household"] = _genome()
    dispatched: list[Any] = []

    def _capture(coro: Any, *_a: Any, **_kw: Any) -> None:
        dispatched.append(coro)
        coro.close()

    mass_stub.create_task = _capture
    _wire(genome_controller, api_key=_API_KEY)

    assert await genome_controller.discovery_refresh() is True
    assert len(dispatched) == 1


async def test_discovery_refresh_without_a_key_does_nothing(
    genome_controller: GenomeController, mass_stub: MagicMock
) -> None:
    """With no Last.fm key there is nothing to dispatch, and that is not an error."""
    dispatched: list[Any] = []

    def _capture(coro: Any, *_a: Any, **_kw: Any) -> None:
        dispatched.append(coro)
        coro.close()

    mass_stub.create_task = _capture
    _wire(genome_controller, api_key="")

    assert await genome_controller.discovery_refresh() is False
    assert dispatched == []


# ---------------------------------------------------------------------------------------
# the Last.fm similar-artist fetch
# ---------------------------------------------------------------------------------------


async def test_fetch_similar_artists_normalizes_the_payload(
    fixture_http_client: FixtureHttpClient,
) -> None:
    """Entries without a name are dropped, an unparseable match becomes 0.0, mbids stay optional."""
    similar = await fetch_similar_artists(
        "Sigur Rós", client=fixture_http_client, api_key=_API_KEY, limit=20
    )
    assert [(row.name, row.match) for row in similar] == [
        ("Jónsi", 1.0),
        ("Ólafur Arnalds", pytest.approx(0.812345)),
        ("Radiohead", 0.5),
        ("Amiina", 0.0),
    ]
    assert similar[3].mbid is None


async def test_fetch_similar_artists_accepts_a_single_entry_object(
    fixture_http_client: FixtureHttpClient,
) -> None:
    """Last.fm collapses a one-entry list into a bare object; that is not an empty result."""
    similar = await fetch_similar_artists(
        "Portishead", client=fixture_http_client, api_key=_API_KEY, limit=20
    )
    assert [row.name for row in similar] == ["Ólafur Arnalds"]


async def test_fetch_similar_artists_never_leaks_the_api_key_on_an_http_error() -> None:
    """A transport failure is described from its status alone — never from the request URL."""

    class _Forbidden(Exception):
        status = 403

    class _FailingClient:
        async def get_json(self, url: str, **_kwargs: Any) -> Any:
            raise _Forbidden(f"403, message='Forbidden', url='{url}?api_key={_API_KEY}'")

        async def post_json(self, url: str, **_kwargs: Any) -> Any:
            raise AssertionError("unexpected POST")

    with pytest.raises(LastfmApiError) as excinfo:
        await fetch_similar_artists(
            "Sigur Rós", client=_FailingClient(), api_key=_API_KEY, limit=20
        )
    assert _API_KEY not in str(excinfo.value)
    assert "403" in str(excinfo.value)


async def test_fetch_similar_artists_raises_on_a_json_error_body() -> None:
    """Last.fm's 200-with-an-error-object convention is a failure, not an empty result."""

    class _ErrorBodyClient:
        async def get_json(self, url: str, **_kwargs: Any) -> Any:
            return {"error": 10, "message": "Invalid API key"}

        async def post_json(self, url: str, **_kwargs: Any) -> Any:
            raise AssertionError("unexpected POST")

    with pytest.raises(LastfmApiError, match="Invalid API key"):
        await fetch_similar_artists(
            "Sigur Rós", client=_ErrorBodyClient(), api_key=_API_KEY, limit=20
        )


# ---------------------------------------------------------------------------------------
# the background pass
# ---------------------------------------------------------------------------------------


async def test_background_pass_stores_suggestions_and_filters_the_library(
    genome_controller: GenomeController,
    genome_store: StubGenomeStore,
    fixture_http_client: FixtureHttpClient,
) -> None:
    """A successful pass persists suggestions, excluding anything the household already has."""
    genome_store.cache["household"] = _genome()
    genome_store.artist_plays = {"radiohead": 12}
    _wire(genome_controller, api_key=_API_KEY, client=fixture_http_client)

    await genome_controller._run_discovery_pass("household")

    blob = genome_store.discovery["household"]
    names = [row["artist_name"] for row in blob["suggested"]]
    # Radiohead is already in the household's history and Ólafur Arnalds is deduped across
    # the two seeds, keeping the higher Portishead match
    assert names == ["Jónsi", "Ólafur Arnalds", "Amiina"]
    assert blob["suggested"][1]["seed_artist"] == "Portishead"
    assert blob["suggested"][1]["match"] == pytest.approx(0.93)
    assert blob["suggested"][0]["genre_key"] == "ambient"
    assert blob["seed_failures"] == {}
    assert blob["generated_at"] > 0


async def test_background_pass_result_is_served_by_the_read_path(
    genome_controller: GenomeController,
    genome_store: StubGenomeStore,
    fixture_http_client: FixtureHttpClient,
) -> None:
    """What the pass stores is exactly what `genome/discovery` hands back afterwards."""
    genome_store.cache["household"] = _genome()
    _wire(genome_controller, api_key=_API_KEY, client=fixture_http_client)

    await genome_controller._run_discovery_pass("household")
    result = await genome_controller.discovery()

    assert result.suggested_state == DISCOVERY_STATE_READY
    assert result.generated_at is not None
    assert result.suggested[0].artist_name == "Jónsi"
    assert result.suggested[0].seed_artist == "Sigur Rós"


async def test_background_pass_records_a_failed_seed_and_keeps_going(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """One seed's HTTP error must not lose the other seed's results."""
    genome_store.cache["household"] = _genome()

    class _PartlyFailingClient:
        async def get_json(self, url: str, *, params: Any = None, **_kwargs: Any) -> Any:
            if (params or {}).get("artist") == "Sigur Rós":
                err = Exception("boom")
                err.status = 500  # type: ignore[attr-defined]
                raise err
            from tests.controllers.genome.conftest import load_fixture  # noqa: PLC0415

            return load_fixture("lastfm_similar_portishead")

        async def post_json(self, url: str, **_kwargs: Any) -> Any:
            raise AssertionError("unexpected POST")

    _wire(genome_controller, api_key=_API_KEY, client=_PartlyFailingClient())

    await genome_controller._run_discovery_pass("household")

    blob = genome_store.discovery["household"]
    assert list(blob["seed_failures"]) == ["sigurros"]
    assert [row["artist_name"] for row in blob["suggested"]] == ["Ólafur Arnalds"]
    # the pass completed, so the card reads "ready with what we have" rather than pending forever
    assert blob["generated_at"] > 0


async def test_background_pass_respects_the_seed_error_cooldown(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """A seed that failed moments ago is skipped, not retried on the very next pass."""
    genome_store.cache["household"] = _genome()
    genome_store.discovery["household"] = {
        "generated_at": time.time() - 60,
        "suggested": [],
        "seed_failures": {"sigurros": time.time() - 60},
    }
    queried: list[str] = []

    class _RecordingClient:
        async def get_json(self, url: str, *, params: Any = None, **_kwargs: Any) -> Any:
            queried.append((params or {})["artist"])
            from tests.controllers.genome.conftest import load_fixture  # noqa: PLC0415

            return load_fixture("lastfm_similar_portishead")

        async def post_json(self, url: str, **_kwargs: Any) -> Any:
            raise AssertionError("unexpected POST")

    _wire(genome_controller, api_key=_API_KEY, client=_RecordingClient())

    await genome_controller._run_discovery_pass("household")

    assert queried == ["Portishead"]
    assert list(genome_store.discovery["household"]["seed_failures"]) == ["sigurros"]


async def test_background_pass_retries_a_seed_once_the_cooldown_expires(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """The cooldown is a delay, not a permanent ban — an expired failure is retried and cleared."""
    genome_store.cache["household"] = _genome()
    expired = time.time() - (GENOME_DISCOVERY_SEED_ERROR_COOLDOWN_HOURS * 3600) - 60
    genome_store.discovery["household"] = {
        "generated_at": expired,
        "suggested": [],
        "seed_failures": {"sigurros": expired},
    }
    queried: list[str] = []

    class _RecordingClient:
        async def get_json(self, url: str, *, params: Any = None, **_kwargs: Any) -> Any:
            queried.append((params or {})["artist"])
            from tests.controllers.genome.conftest import load_fixture  # noqa: PLC0415

            return load_fixture("lastfm_similar_portishead")

        async def post_json(self, url: str, **_kwargs: Any) -> Any:
            raise AssertionError("unexpected POST")

    _wire(genome_controller, api_key=_API_KEY, client=_RecordingClient())

    await genome_controller._run_discovery_pass("household")

    assert queried == ["Sigur Rós", "Portishead"]
    assert genome_store.discovery["household"]["seed_failures"] == {}


async def test_background_pass_without_a_key_stores_nothing(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """An unconfigured Last.fm must leave the stored result untouched rather than blanking it."""
    genome_store.cache["household"] = _genome()
    _wire(genome_controller, api_key="", client=_ExplodingHttpClient())

    await genome_controller._run_discovery_pass("household")

    assert "household" not in genome_store.discovery


async def test_background_pass_with_no_seeds_still_finishes(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """A household with no divergent genres gets a completed, empty pass — never a stuck one."""
    genome_store.cache["household"] = _genome(
        divergence={"score": 0.0, "percent": 0, "top_over": [], "top_under": []}
    )
    _wire(genome_controller, api_key=_API_KEY, client=_ExplodingHttpClient())

    await genome_controller._run_discovery_pass("household")

    blob = genome_store.discovery["household"]
    assert blob["suggested"] == []
    assert blob["generated_at"] > 0


async def test_background_discovery_never_raises(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """A background pass must not crash the task loop, whatever Last.fm does."""
    genome_store.cache["household"] = _genome()

    class _AlwaysFailingClient:
        async def get_json(self, url: str, **_kwargs: Any) -> Any:
            raise RuntimeError("network on fire")

        async def post_json(self, url: str, **_kwargs: Any) -> Any:
            raise RuntimeError("network on fire")

    _wire(genome_controller, api_key=_API_KEY, client=_AlwaysFailingClient())

    await genome_controller._background_discovery("household")

    assert list(genome_store.discovery["household"]["seed_failures"]) == [
        "sigurros",
        "portishead",
    ]


def test_divergent_genre_is_hashable_and_frozen() -> None:
    """The ranking helpers rely on these being immutable value objects."""
    genre = DivergentGenre(key="ambient", label="Ambient", contribution=0.31)
    with pytest.raises(AttributeError):
        genre.key = "rock"  # type: ignore[misc]
