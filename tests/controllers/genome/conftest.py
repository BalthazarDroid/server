"""Shared fixtures for Listening Genome (WP-A) tests: a fixture-backed :class:`HttpClient`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from music_assistant.helpers.json import json_loads

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "genome"

# maps an artist name (as it appears in a Lucene `artist:"<name>"` search query) to the fixture
# file slug used for that artist's search/lookup pair
_ARTIST_SLUGS: dict[str, str] = {
    "Sigur Rós": "sigur_ros",
    "AC/DC": "ac_dc",
    "Radiohead": "radiohead",
    "Boards of Canada": "boards_of_canada",
    "Portishead": "portishead",
    "Nick Drake": "nick_drake",
    "Kasabian": "kasabian",
}

_QUERY_NAME_PATTERN = re.compile(r'artist:"(.+)"$')


def load_fixture(name: str) -> Any:
    """Load and JSON-decode a fixture file by its filename (with or without ``.json``)."""
    filename = name if name.endswith(".json") else f"{name}.json"
    return json_loads((FIXTURES_DIR / filename).read_bytes())


class FixtureHttpClient:
    """
    An :class:`~music_assistant.controllers.genome.http.HttpClient` backed entirely by fixtures.

    No test using this ever makes a network call — every request is answered from
    ``tests/fixtures/genome/``, matched by URL shape rather than a literal mapping, so the
    importer/enrichment code under test does not need to know it is not talking to the internet.
    """

    def __init__(self) -> None:
        """Initialize with an empty call log."""
        self.calls: list[tuple[str, str, Any]] = []

    async def get_json(
        self, url: str, *, params: dict[str, str] | None = None, headers: Any = None
    ) -> Any:
        """Answer a ``GET`` from fixtures, matched by URL shape."""
        params = params or {}
        self.calls.append(("GET", url, params))
        if url.endswith("/artist"):
            return self._search_artist(params.get("query", ""))
        if "/artist/" in url:
            mbid = url.rsplit("/artist/", maxsplit=1)[1]
            return load_fixture(f"musicbrainz_artist_lookup_{mbid}")
        if "audioscrobbler.com" in url:
            page = params.get("page", "1")
            return load_fixture(f"lastfm_recent_page{page}")
        raise AssertionError(f"FixtureHttpClient: no fixture mapped for GET {url} {params}")

    async def post_json(self, url: str, *, json: Any, headers: Any = None) -> Any:
        """Answer a ``POST`` from fixtures, matched by URL shape."""
        self.calls.append(("POST", url, json))
        if "popularity/artist" in url:
            return load_fixture("listenbrainz_popularity")
        raise AssertionError(f"FixtureHttpClient: no fixture mapped for POST {url} {json}")

    def _search_artist(self, query: str) -> Any:
        """Resolve a Lucene ``artist:"<name>"`` search query to its fixture search result."""
        match = _QUERY_NAME_PATTERN.search(query)
        name = match.group(1).replace("\\", "") if match else ""
        slug = _ARTIST_SLUGS.get(name)
        if slug is None:
            return load_fixture("musicbrainz_artist_search_notfound")
        return load_fixture(f"musicbrainz_artist_search_{slug}")


@pytest.fixture
def fixture_http_client() -> FixtureHttpClient:
    """A fresh :class:`FixtureHttpClient` for a single test."""
    return FixtureHttpClient()


__all__ = ["FIXTURES_DIR", "FixtureHttpClient", "fixture_http_client", "load_fixture"]
