"""Tests for the Last.fm ``user.getRecentTracks`` importer (§3.8, §3.9)."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

from music_assistant.controllers.genome.importers.lastfm import LastfmImporter
from music_assistant.controllers.genome.store import GenomeStore

if TYPE_CHECKING:
    from pathlib import Path

    from tests.controllers.genome.conftest import FixtureHttpClient


async def _new_store(tmp_path: Path) -> GenomeStore:
    mass = types.SimpleNamespace(storage_path=str(tmp_path), players=None)
    store = GenomeStore(mass)
    await store.setup()
    return store


async def test_fetch_recent_skips_nowplaying_entry(fixture_http_client: FixtureHttpClient) -> None:
    """The now-playing entry (no `date`) on page 1 must never become a Listen."""
    importer = LastfmImporter(fixture_http_client, "testuser", "fake-key")
    page = await importer.fetch_recent(1)
    assert page.page == 1
    assert page.total_pages == 2
    assert page.raw_count == 20
    assert len(page.listens) == 19


async def test_fetch_recent_page_two(fixture_http_client: FixtureHttpClient) -> None:
    """Page 2 has exactly the 3 tracks the fixture defines."""
    importer = LastfmImporter(fixture_http_client, "testuser", "fake-key")
    page = await importer.fetch_recent(2)
    assert page.page == 2
    assert len(page.listens) == 3


async def test_import_since_paginates_and_stores_everything(
    tmp_path: Path, fixture_http_client: FixtureHttpClient
) -> None:
    """A first import walks every page and stores all non-nowplaying scrobbles."""
    store = await _new_store(tmp_path)
    try:
        importer = LastfmImporter(fixture_http_client, "testuser", "fake-key")
        result = await importer.import_since(store, listener="household")
        assert result["source"] == "lastfm"
        assert result["rows_read"] == 23
        assert result["rows_skipped"] == 1
        assert result["rows_imported"] == 22
        assert await store.count_listens("household") == 22
    finally:
        await store.close()


async def test_import_since_resumes_without_reimporting(
    tmp_path: Path, fixture_http_client: FixtureHttpClient
) -> None:
    """A second import against the same (unchanged) fixture pages dedupes to zero new rows."""
    store = await _new_store(tmp_path)
    try:
        importer = LastfmImporter(fixture_http_client, "testuser", "fake-key")
        await importer.import_since(store, listener="household")
        second = await importer.import_since(store, listener="household")
        assert second["rows_imported"] == 0
        assert second["rows_duplicate"] == 22
    finally:
        await store.close()


async def test_import_since_respects_max_pages(
    tmp_path: Path, fixture_http_client: FixtureHttpClient
) -> None:
    """max_pages=1 must stop after the first page."""
    store = await _new_store(tmp_path)
    try:
        importer = LastfmImporter(fixture_http_client, "testuser", "fake-key")
        result = await importer.import_since(store, listener="household", max_pages=1)
        assert result["rows_read"] == 20
        assert result["rows_imported"] == 19
    finally:
        await store.close()


async def test_api_key_never_appears_in_result(
    tmp_path: Path, fixture_http_client: FixtureHttpClient
) -> None:
    """A GenomeImportResult must never leak the Last.fm API key (§3.8)."""
    store = await _new_store(tmp_path)
    try:
        importer = LastfmImporter(fixture_http_client, "testuser", "super-secret-key")
        result = await importer.import_since(store, listener="household")
        assert "super-secret-key" not in repr(result)
    finally:
        await store.close()
