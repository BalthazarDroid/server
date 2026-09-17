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


class _FakeResponse:
    """Minimal stand-in for an aiohttp response."""

    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self._body = body

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
