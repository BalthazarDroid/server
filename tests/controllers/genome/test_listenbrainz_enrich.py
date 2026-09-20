"""
Tests for the ListenBrainz popularity enrichment.

This module had no tests at all, and shipped assuming the wrong response envelope: every
backfill died on ``AttributeError: 'list' object has no attribute 'get'``. Nothing surfaced it,
because the failure was caught and logged as a warning while the page went on rendering an
obscurity index of 0% with "low confidence" - which reads as a finding about the household
rather than as a component that has never once succeeded.

The identical assumption existed in the baseline builder and was fixed there first, without
anyone checking whether the server made the same mistake. It did.
"""

from __future__ import annotations

from typing import Any

import pytest

from music_assistant.controllers.genome.enrich.listenbrainz import (
    Popularity,
    _rows,
    artist_popularity,
)


class _StubClient:
    """Records the bodies posted and replays a scripted response per call."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.posted: list[list[str]] = []

    async def post_json(self, _url: str, *, json: dict[str, Any]) -> Any:
        self.posted.append(list(json["artist_mbids"]))
        return self._responses.pop(0)


def test_rows_accepts_the_bare_list_the_endpoint_actually_returns() -> None:
    """The regression. ``/1/popularity/artist`` answers an array, not an envelope."""
    assert _rows([{"artist_mbid": "a"}]) == [{"artist_mbid": "a"}]


def test_rows_accepts_a_payload_envelope() -> None:
    """The shape this module always expected; it must keep working."""
    assert _rows({"payload": [{"artist_mbid": "b"}]}) == [{"artist_mbid": "b"}]


def test_rows_accepts_a_keyed_payload() -> None:
    """The statistics endpoints nest their rows one level deeper again."""
    assert _rows({"payload": {"artists": [{"artist_mbid": "c"}]}}) == [{"artist_mbid": "c"}]


@pytest.mark.parametrize(
    "body",
    [None, "nonsense", 7, {}, {"payload": {}}, {"payload": {"artists": "not a list"}}],
)
def test_rows_returns_empty_for_anything_it_cannot_read(body: Any) -> None:
    """An envelope change must cost the pass its results, never raise and kill it."""
    assert _rows(body) == []


def test_rows_discards_non_object_entries() -> None:
    """Callers index into each row, so anything that is not an object has to go."""
    assert _rows(["junk", None, {"artist_mbid": "d"}]) == [{"artist_mbid": "d"}]


async def test_artist_popularity_reads_a_bare_list_response() -> None:
    """
    End to end over the real response shape.

    This is the test that would have caught it: the parsing helper alone can be right while the
    caller still unpacks the body its own way.
    """
    client = _StubClient(
        [
            [
                {"artist_mbid": "mb-1", "total_user_count": 40_000, "total_listen_count": 900_000},
                {"artist_mbid": "mb-2", "total_user_count": 120, "total_listen_count": 3_000},
            ]
        ]
    )
    got = await artist_popularity(["mb-1", "mb-2"], client=client)
    assert got == {
        "mb-1": Popularity(listeners=40_000, listen_count=900_000),
        "mb-2": Popularity(listeners=120, listen_count=3_000),
    }


async def test_artist_popularity_survives_an_unreadable_response() -> None:
    """A shape we cannot read yields nothing for that batch, not an exception."""
    client = _StubClient([{"unexpected": True}])
    assert await artist_popularity(["mb-1"], client=client) == {}


async def test_artist_popularity_omits_an_artist_listenbrainz_has_no_data_for() -> None:
    """An absent artist is a normal answer, not a zeroed reading."""
    client = _StubClient([[{"artist_mbid": "mb-1", "total_user_count": 5}]])
    got = await artist_popularity(["mb-1", "mb-missing"], client=client)
    assert set(got) == {"mb-1"}


async def test_artist_popularity_requests_each_mbid_once() -> None:
    """Duplicates in the backlog must not be paid for twice against a public API."""
    client = _StubClient([[]])
    await artist_popularity(["mb-1", "mb-1", "mb-1"], client=client)
    assert client.posted == [["mb-1"]]


async def test_artist_popularity_tolerates_missing_counts() -> None:
    """A row without counts is zero, not a KeyError mid-batch."""
    client = _StubClient([[{"artist_mbid": "mb-1"}]])
    got = await artist_popularity(["mb-1"], client=client)
    assert got["mb-1"] == Popularity(listeners=0, listen_count=0)


async def test_artist_popularity_makes_no_request_for_an_empty_backlog() -> None:
    """Nothing to look up means nothing asked of ListenBrainz."""
    client = _StubClient([])
    assert await artist_popularity([], client=client) == {}
    assert client.posted == []
