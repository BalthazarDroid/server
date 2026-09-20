"""
Tests for the Listening Genome baseline builder's response handling.

These cover the LIVE-fetch path's response parsing, which is the part of this script that no
test previously touched: the fixture path was the only one exercised in development, because
ListenBrainz is unreachable from that workspace. The live path shipped with an envelope
assumption that was wrong for one of the two endpoints, and it failed on the user's machine
partway through a real run. That is the project's recurring failure mode - a correct function
reached through a path no test walks - so the parsing is now separated from the I/O and tested
against the real shapes both endpoints return.
"""

from __future__ import annotations

from typing import Any, Self

import aiohttp
import pytest

from scripts import build_genome_baseline
from scripts.build_genome_baseline import _request_json, _unwrap


def test_unwrap_accepts_a_bare_list() -> None:
    """``/1/popularity/artist`` answers a bare JSON array with no envelope at all."""
    assert _unwrap([{"artist_mbid": "a"}, {"artist_mbid": "b"}]) == [
        {"artist_mbid": "a"},
        {"artist_mbid": "b"},
    ]


def test_unwrap_accepts_a_payload_holding_a_list() -> None:
    """The same endpoint has also been observed wrapping its array in ``payload``."""
    assert _unwrap({"payload": [{"artist_mbid": "c"}]}) == [{"artist_mbid": "c"}]


def test_unwrap_accepts_a_payload_holding_a_keyed_list() -> None:
    """``/1/stats/sitewide/artists`` answers ``{"payload": {"artists": [...]}}``."""
    assert _unwrap({"payload": {"artists": [{"artist_mbid": "d"}]}}, "artists") == [
        {"artist_mbid": "d"}
    ]


def test_unwrap_returns_empty_for_an_unexpected_shape() -> None:
    """
    An envelope change upstream must degrade to an empty result, not a traceback.

    A run that has already spent time on the network should not be lost to a shape this
    script did not anticipate.
    """
    assert _unwrap(None) == []
    assert _unwrap("nonsense") == []
    assert _unwrap({"payload": {}}, "artists") == []
    assert _unwrap({"payload": {"artists": "not a list"}}, "artists") == []
    assert _unwrap({"unexpected": 1}, "artists") == []


def test_unwrap_discards_non_object_entries() -> None:
    """Callers index into each record, so anything that is not an object has to go."""
    assert _unwrap(["junk", 7, None, {"artist_mbid": "e"}]) == [{"artist_mbid": "e"}]


def test_unwrap_needs_a_key_to_reach_into_a_keyed_payload() -> None:
    """Without a key there is no way to guess which of several lists was meant."""
    assert _unwrap({"payload": {"artists": [{"artist_mbid": "f"}]}}) == []


async def _no_pace() -> None:
    """Skip the courtesy delay; these tests exercise paging, not politeness."""


class _FakeResponse:
    """Minimal stand-in for an aiohttp response."""

    def __init__(self, status: int, body: Any, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self._body = body
        self.headers = headers or {}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(None, (), status=self.status)

    async def json(self) -> Any:
        return self._body


class _FakeSession:
    """Replays a scripted sequence of responses and records how many requests were made."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def request(self, _method: str, _url: str, **_kwargs: Any) -> Any:
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


async def test_request_json_returns_the_body_on_first_success() -> None:
    """The ordinary case must not pay for the retry machinery."""
    session = _FakeSession([_FakeResponse(200, {"ok": True})])
    assert await _request_json(session, "GET", "https://example.invalid/x") == {"ok": True}
    assert session.calls == 1


async def test_request_json_retries_a_transient_status_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A 503 partway through must cost one retry, not the whole run.

    This is the behaviour LIVE_MAX_ATTEMPTS always claimed in a comment while the code around
    it did exactly one attempt.
    """
    monkeypatch.setattr(build_genome_baseline, "LIVE_RETRY_BACKOFF_SECONDS", 0)
    session = _FakeSession([_FakeResponse(503, None), _FakeResponse(200, [{"a": 1}])])
    assert await _request_json(session, "GET", "https://example.invalid/x") == [{"a": 1}]
    assert session.calls == 2


async def test_request_json_gives_up_after_the_attempt_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host that is genuinely down must not be hammered indefinitely."""
    monkeypatch.setattr(build_genome_baseline, "LIVE_RETRY_BACKOFF_SECONDS", 0)
    session = _FakeSession([_FakeResponse(503, None)] * build_genome_baseline.LIVE_MAX_ATTEMPTS)
    with pytest.raises(OSError, match="after 4 attempts"):
        await _request_json(session, "GET", "https://example.invalid/x")
    assert session.calls == build_genome_baseline.LIVE_MAX_ATTEMPTS


async def test_request_json_does_not_retry_a_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 is the request being wrong; repeating it spends someone else's rate limit."""
    monkeypatch.setattr(build_genome_baseline, "LIVE_RETRY_BACKOFF_SECONDS", 0)
    session = _FakeSession([_FakeResponse(404, None)])
    with pytest.raises(OSError, match="HTTP 404"):
        await _request_json(session, "GET", "https://example.invalid/x")
    assert session.calls == 1


async def test_request_json_never_puts_the_url_in_the_retry_error() -> None:
    """
    The retry exception carries a status and nothing else.

    URLs in this script carry no credentials today, but the Last.fm work in this feature has
    already had one near-miss with a key reaching a log line, so the rule is kept uniformly.
    """
    err = build_genome_baseline._TransientFetchError(503)
    assert str(err) == "HTTP 503"


def test_user_agent_identifies_the_client() -> None:
    """MusicBrainz refuses anonymous clients, and aiohttp's default is anonymous."""
    assert "MusicAssistant" in build_genome_baseline.USER_AGENT
    assert "http" in build_genome_baseline.USER_AGENT


def test_default_musicbrainz_host_is_not_the_music_assistant_mirror() -> None:
    """
    The mirror is the MA project's own, for running MA instances to resolve metadata.

    It answers 403 to anything not identifying as Music Assistant, and spoofing that
    User-Agent to push ~1500 lookups of one-off bulk traffic through infrastructure someone
    else pays for is not a thing to do quietly in a script. The canonical API is where this
    belongs; --mb-base-url exists for anyone running their own.
    """
    assert "music-assistant.io" not in build_genome_baseline.MUSICBRAINZ_BASE_URL
    assert build_genome_baseline.MUSICBRAINZ_BASE_URL.startswith("https://musicbrainz.org/")


def test_pacing_respects_the_documented_courtesy_limit() -> None:
    """MusicBrainz documents ~1 req/sec; the delay is a ceiling, not a target."""
    assert build_genome_baseline.LIVE_REQUEST_DELAY_SECONDS >= 1.0


def test_parse_retry_after_reads_seconds() -> None:
    """MusicBrainz sends a plain seconds value when it is rate-limiting."""
    assert build_genome_baseline._parse_retry_after("5") == 5.0
    assert build_genome_baseline._parse_retry_after(" 2.5 ") == 2.5


def test_parse_retry_after_ignores_what_it_cannot_use() -> None:
    """An HTTP-date or junk falls back to the script's own backoff rather than failing."""
    assert build_genome_baseline._parse_retry_after(None) is None
    assert build_genome_baseline._parse_retry_after("") is None
    assert build_genome_baseline._parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
    assert build_genome_baseline._parse_retry_after("0") is None
    assert build_genome_baseline._parse_retry_after("-3") is None


def test_parse_retry_after_caps_an_implausible_wait() -> None:
    """A server asking for an hour is likelier misconfigured than serious."""
    assert build_genome_baseline._parse_retry_after("99999") == 60.0


async def test_request_json_waits_as_long_as_the_server_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that states its own interval knows better than our backoff curve."""
    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(build_genome_baseline.asyncio, "sleep", _fake_sleep)
    session = _FakeSession(
        [_FakeResponse(503, None, {"Retry-After": "7"}), _FakeResponse(200, {"ok": True})]
    )
    assert await _request_json(session, "GET", "https://example.invalid/x") == {"ok": True}
    assert slept == [7.0]


async def test_request_json_backs_off_harder_each_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The backoff grows by the square, not linearly.

    A linear curve topped out at six seconds across four attempts, which a rate limiter does
    not notice; every attempt then failed the same way and the run was lost.
    """
    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(build_genome_baseline.asyncio, "sleep", _fake_sleep)
    session = _FakeSession([_FakeResponse(503, None)] * build_genome_baseline.LIVE_MAX_ATTEMPTS)
    with pytest.raises(OSError, match="after 4 attempts"):
        await _request_json(session, "GET", "https://example.invalid/x")
    assert slept == sorted(slept)
    assert slept[-1] > slept[0]


class _PagingSession:
    """
    Serves ListenBrainz's real shape: a separate, capped chart per time range.

    Each range holds ``per_range`` artists, and the ranges overlap by ``shared`` artists at the
    head - which is what makes widening by range worth anything and also what stops it scaling
    linearly.
    """

    def __init__(self, per_range: int, shared: int = 0, page_limit: int | None = None) -> None:
        self.per_range = per_range
        self.shared = shared
        self.page_limit = page_limit or build_genome_baseline.LISTENBRAINZ_PAGE_LIMIT
        self.requests: list[tuple[int, int, str]] = []

    def request(self, _method: str, _url: str, **kwargs: Any) -> Any:
        params = kwargs.get("params") or {}
        count = int(params.get("count", 25))
        offset = int(params.get("offset", 0))
        time_range = params.get("range", "all_time")
        self.requests.append((count, offset, time_range))
        served = min(count, self.page_limit, max(0, self.per_range - offset))
        artists = []
        for i in range(served):
            index = offset + i
            # The first `shared` of every range are the same artists everywhere.
            key = f"shared-{index}" if index < self.shared else f"{time_range}-{index}"
            artists.append({"artist_mbid": key, "artist_name": key})
        return _FakeResponse(200, {"payload": {"artists": artists}})


async def test_top_artists_widens_across_ranges_when_one_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The bug this exists for.

    ListenBrainz computes only the top 1000 artists PER TIME RANGE, so offset paging within
    one range stops at 1000 no matter what is asked for - a --sample of 5000 returned 993 and
    looked like a finished run. Reaching further means asking other ranges.
    """
    monkeypatch.setattr(build_genome_baseline, "_pace", _no_pace)
    session = _PagingSession(per_range=1000)
    got = await build_genome_baseline._fetch_top_artists(session, 2500)
    assert len(got) == 2500
    assert len({a["artist_mbid"] for a in got}) == 2500
    assert len({r[2] for r in session.requests}) >= 3


async def test_top_artists_stops_when_every_range_is_used_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for more than the whole source holds must end, not loop."""
    monkeypatch.setattr(build_genome_baseline, "_pace", _no_pace)
    session = _PagingSession(per_range=50)
    got = await build_genome_baseline._fetch_top_artists(session, 100_000)
    assert len(got) == 50 * len(build_genome_baseline.LISTENBRAINZ_RANGES)


async def test_top_artists_counts_an_artist_once_across_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The ranges share their head, which is the whole reason this does not scale linearly.

    A sample inflated by the same popular artists appearing in nine charts would be worse than
    a smaller honest one: those artists' genres would be counted nine times over.
    """
    monkeypatch.setattr(build_genome_baseline, "_pace", _no_pace)
    session = _PagingSession(per_range=100, shared=80)
    got = await build_genome_baseline._fetch_top_artists(session, 5000)
    keys = [a["artist_mbid"] for a in got]
    assert len(keys) == len(set(keys))
    # 80 shared + 20 unique from each of the nine ranges.
    assert len(keys) == 80 + 20 * len(build_genome_baseline.LISTENBRAINZ_RANGES)


async def test_top_artists_starts_with_the_all_time_chart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All-time is the most durable reference; the transient ranges only fill the tail."""
    monkeypatch.setattr(build_genome_baseline, "_pace", _no_pace)
    session = _PagingSession(per_range=1000)
    await build_genome_baseline._fetch_top_artists(session, 1500)
    assert session.requests[0][2] == "all_time"


async def test_top_artists_needs_only_one_request_for_a_small_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sample inside one range's chart must not pay for a second round trip."""
    monkeypatch.setattr(build_genome_baseline, "_pace", _no_pace)
    session = _PagingSession(per_range=1000)
    got = await build_genome_baseline._fetch_top_artists(session, 400)
    assert len(got) == 400
    assert len(session.requests) == 1
    assert session.requests[0][:2] == (400, 0)


async def test_top_artists_tags_each_artist_with_the_range_it_came_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The tag is what keeps a range-scoped listen count out of an all-time average.

    _fetch_live falls back to the chart's own listen_count when the popularity endpoint has
    nothing, and that figure covers only the chart's window.
    """
    monkeypatch.setattr(build_genome_baseline, "_pace", _no_pace)
    session = _PagingSession(per_range=10)
    got = await build_genome_baseline._fetch_top_artists(session, 30)
    assert all(a["_range"] in build_genome_baseline.LISTENBRAINZ_RANGES for a in got)
    assert got[0]["_range"] == "all_time"


async def test_top_artists_discards_duplicates_when_offset_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An endpoint that ignored offset would otherwise inflate the sample with repeats."""
    monkeypatch.setattr(build_genome_baseline, "_pace", _no_pace)

    class _IgnoresOffset(_PagingSession):
        def request(self, method: str, url: str, **kwargs: Any) -> Any:
            kwargs["params"] = {**(kwargs.get("params") or {}), "offset": "0"}
            return super().request(method, url, **kwargs)

    session = _IgnoresOffset(per_range=500)
    got = await build_genome_baseline._fetch_top_artists(session, 2500)
    keys = [a["artist_mbid"] for a in got]
    assert len(keys) == len(set(keys))
