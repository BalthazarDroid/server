"""
Shared fixtures for Listening Genome tests.

Combines WP-A's fixture-backed :class:`HttpClient` with WP-B's in-memory stub of the
frozen ``GenomeStore`` interface.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from music_assistant.controllers.genome.controller import GenomeController
from music_assistant.controllers.genome.models import GenomeImportResult
from music_assistant.helpers.json import json_loads

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from music_assistant.controllers.genome.models import ArtistMeta, GenomeResult, Listen


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
    """Build a fresh :class:`FixtureHttpClient` for a single test."""
    return FixtureHttpClient()


__all__ = ["FIXTURES_DIR", "FixtureHttpClient", "fixture_http_client", "load_fixture"]


class StubGenomeStore:
    """
    Minimal in-memory implementation of the frozen ``GenomeStore`` surface (Part 4).

    Used so ``GenomeController`` tests never depend on ``store.py``, which is owned by a
    different work package and may not exist in this checkout.
    """

    def __init__(self) -> None:
        """Start with no listens, no cache and no artist metadata."""
        self.listens: list[Listen] = []
        self.artist_meta: dict[str, ArtistMeta] = {}
        self.cache: dict[str, GenomeResult] = {}

    async def setup(self) -> None:
        """No-op: nothing to open."""

    async def close(self) -> None:
        """No-op: nothing to close."""

    async def add_listens(
        self, listens: Sequence[Listen], *, listener: str, ma_userid: str | None = None
    ) -> GenomeImportResult:
        """Append every listen unconditionally (no dedup — tests control their own input)."""
        self.listens.extend(listens)
        played_ats = [listen.played_at for listen in listens]
        return GenomeImportResult(
            source=listens[0].source if listens else "test",
            rows_read=len(listens),
            rows_imported=len(listens),
            rows_skipped=0,
            rows_duplicate=0,
            first_played_at=min(played_ats) if played_ats else None,
            last_played_at=max(played_ats) if played_ats else None,
            warnings=[],
        )

    async def iter_listens(self, listener: str, *, since: int = 0) -> AsyncIterator[Listen]:
        """Yield every stored listen with ``played_at >= since``."""
        for listen in self.listens:
            if listen.played_at >= since:
                yield listen

    async def count_listens(self, listener: str) -> int:
        """Return the number of stored listens."""
        return len(self.listens)

    async def get_artist_meta(self, artist_keys: Sequence[str]) -> dict[str, ArtistMeta]:
        """Return known metadata for the requested artist keys."""
        return {key: self.artist_meta[key] for key in artist_keys if key in self.artist_meta}

    async def upsert_artist_meta(self, rows: Sequence[ArtistMeta], *, state: str) -> None:
        """Store artist metadata rows, keyed by ``artist_key``."""
        for row in rows:
            self.artist_meta[row.artist_key] = row

    async def pending_artist_keys(self, limit: int = 200) -> list[tuple[str, str]]:
        """No pending enrichment in the stub by default."""
        return []

    async def get_cached_genome(self, listener: str) -> GenomeResult | None:
        """Return the cached genome for ``listener``, if any."""
        return self.cache.get(listener)

    async def set_cached_genome(self, listener: str, genome: GenomeResult) -> None:
        """Cache ``genome`` for ``listener``."""
        self.cache[listener] = genome

    async def clear(self, listener: str | None = None) -> None:
        """Drop all stored listens, metadata and cache."""
        self.listens.clear()
        self.artist_meta.clear()
        self.cache.clear()

    async def source_counts(self, listener: str) -> dict[str, int]:
        """Return listen counts grouped by source."""
        counts: dict[str, int] = {}
        for listen in self.listens:
            counts[listen.source] = counts.get(listen.source, 0) + 1
        return counts

    async def player_names(self) -> dict[str, str]:
        """No known player names in the stub by default."""
        return {}

    async def update_lb_popularity(self, rows: dict[str, tuple[int, int]]) -> None:
        """No-op: the stub does not model ListenBrainz enrichment."""

    async def backfill_done(self) -> bool:
        """Report the one-time MA backfill as already done, so unit tests never trigger it."""
        return True

    async def mark_backfill_done(self) -> None:
        """No-op: nothing to persist in the stub."""

    async def lastfm_backfill_done(self) -> bool:
        """Report the one-time Last.fm sweep as already done by default (incremental mode)."""
        return True

    async def mark_lastfm_backfill_done(self) -> None:
        """No-op: nothing to persist in the stub."""


@pytest.fixture
def genome_store() -> StubGenomeStore:
    """Build a fresh in-memory ``GenomeStore`` stub."""
    return StubGenomeStore()


@pytest.fixture
def mass_stub() -> MagicMock:
    """Build a minimal ``MusicAssistant`` double with just what ``GenomeController`` touches."""
    mass = MagicMock()
    mass.storage_path = tempfile.mkdtemp()
    mass.tasks.register_scheduled_task = MagicMock()
    mass.config.get_raw_core_config_value = MagicMock(return_value="GLOBAL")
    return mass


@pytest.fixture
def genome_controller(mass_stub: MagicMock, genome_store: StubGenomeStore) -> GenomeController:
    """Build a ``GenomeController`` wired to the in-memory store stub, with defaulted config values."""
    controller = GenomeController(mass_stub, store=genome_store)
    # signature must match GenomeController.get_config_value exactly - callers pass return_type
    # by keyword, so its unused params here cannot be underscore-prefixed
    controller.get_config_value = (  # type: ignore[method-assign]
        lambda key, default=None, *, return_type=None: default  # noqa: ARG005
    )
    return controller
