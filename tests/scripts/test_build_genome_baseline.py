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

from scripts.build_genome_baseline import _unwrap


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
