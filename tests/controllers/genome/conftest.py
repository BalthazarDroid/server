"""Shared fixtures for genome controller tests: an in-memory stub of the frozen GenomeStore."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from music_assistant.controllers.genome.controller import GenomeController
from music_assistant.controllers.genome.models import GenomeImportResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from music_assistant.controllers.genome.models import ArtistMeta, GenomeResult, Listen


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


@pytest.fixture
def genome_store() -> StubGenomeStore:
    """A fresh in-memory ``GenomeStore`` stub."""
    return StubGenomeStore()


@pytest.fixture
def mass_stub() -> MagicMock:
    """A minimal ``MusicAssistant`` double with just what ``GenomeController`` touches."""
    mass = MagicMock()
    mass.storage_path = tempfile.mkdtemp()
    mass.tasks.register_scheduled_task = MagicMock()
    mass.config.get_raw_core_config_value = MagicMock(return_value="GLOBAL")
    return mass


@pytest.fixture
def genome_controller(mass_stub: MagicMock, genome_store: StubGenomeStore) -> GenomeController:
    """A ``GenomeController`` wired to the in-memory store stub, with defaulted config values."""
    controller = GenomeController(mass_stub, store=genome_store)
    controller.get_config_value = lambda key, default=None, *, return_type=None: default  # type: ignore[method-assign]
    return controller
