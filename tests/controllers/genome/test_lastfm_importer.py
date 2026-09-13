"""Tests for the Last.fm ``user.getRecentTracks`` importer (§3.8, §3.9)."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import pytest

from music_assistant.controllers.genome.errors import LastfmApiError
from music_assistant.controllers.genome.importers.lastfm import LastfmImporter
from music_assistant.controllers.genome.store import GenomeStore

if TYPE_CHECKING:
    from pathlib import Path

    from tests.controllers.genome.conftest import FixtureHttpClient


class _RaisingHttpClient:
    """An ``HttpClient`` whose ``get_json`` always raises ``to_raise``."""

    def __init__(self, to_raise: Exception) -> None:
        self._to_raise = to_raise

    async def get_json(self, url: str, *, params: Any = None, headers: Any = None) -> Any:
        raise self._to_raise

    async def post_json(self, url: str, *, json: Any, headers: Any = None) -> Any:
        raise self._to_raise


class _StatusError(Exception):
    """
    Mimics an aiohttp ``ClientResponseError`` closely enough for these tests.

    Carries a bare status code and nothing else, so a test can prove the importer never
    needs (or leaks) more.
    """

    def __init__(self, status: int) -> None:
        super().__init__(f"<url containing the api_key query param>, status={status}")
        self.status = status


class _JsonHttpClient:
    """An ``HttpClient`` whose ``get_json`` always returns a fixed payload."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    async def get_json(self, url: str, *, params: Any = None, headers: Any = None) -> Any:
        return self._payload

    async def post_json(self, url: str, *, json: Any, headers: Any = None) -> Any:
        return self._payload


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


# ---------------------------------------------------------------------------------------
# Network/API failures (§3.8) — a bad key, an unknown user, a timeout or a malformed
# response must be logged with a readable reason and re-raised, never left to look like a
# silent "0 rows imported" success.
# ---------------------------------------------------------------------------------------


async def test_bad_api_key_raises_readable_error_without_leaking_the_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A 403 (bad API key) must raise a helpful message and never leak the key or the URL."""
    store = await _new_store(tmp_path)
    try:
        client = _RaisingHttpClient(_StatusError(403))
        importer = LastfmImporter(client, "testuser", "super-secret-key")
        with caplog.at_level("WARNING"), pytest.raises(LastfmApiError) as excinfo:
            await importer.import_since(store, listener="household")
        assert "api key" in str(excinfo.value).lower()
        assert "super-secret-key" not in str(excinfo.value)
        assert "super-secret-key" not in caplog.text
    finally:
        await store.close()


async def test_unknown_user_raises_readable_error(tmp_path: Path) -> None:
    """A 404 (unknown user) must say to check the username, not leak a raw traceback."""
    store = await _new_store(tmp_path)
    try:
        client = _RaisingHttpClient(_StatusError(404))
        importer = LastfmImporter(client, "no-such-user", "fake-key")
        with pytest.raises(LastfmApiError) as excinfo:
            await importer.import_since(store, listener="household")
        assert "username" in str(excinfo.value).lower()
    finally:
        await store.close()


async def test_timeout_raises_readable_error(tmp_path: Path) -> None:
    """A timeout must raise a readable, retry-suggesting message."""
    store = await _new_store(tmp_path)
    try:
        client = _RaisingHttpClient(TimeoutError())
        importer = LastfmImporter(client, "testuser", "fake-key")
        with pytest.raises(LastfmApiError) as excinfo:
            await importer.import_since(store, listener="household")
        assert "did not respond" in str(excinfo.value).lower()
    finally:
        await store.close()


async def test_malformed_json_raises_readable_error(tmp_path: Path) -> None:
    """A response that is not the expected shape must raise, not silently look like 0 listens."""
    store = await _new_store(tmp_path)
    try:
        client = _JsonHttpClient("<html>not json-shaped</html>")
        importer = LastfmImporter(client, "testuser", "fake-key")
        with pytest.raises(LastfmApiError) as excinfo:
            await importer.import_since(store, listener="household")
        assert "unexpected response" in str(excinfo.value).lower()
    finally:
        await store.close()


async def test_lastfm_error_body_raises_readable_error(tmp_path: Path) -> None:
    """Last.fm's own `{"error": ..., "message": ...}` convention must also be surfaced."""
    store = await _new_store(tmp_path)
    try:
        client = _JsonHttpClient({"error": 10, "message": "Invalid API key"})
        importer = LastfmImporter(client, "testuser", "fake-key")
        with pytest.raises(LastfmApiError) as excinfo:
            await importer.import_since(store, listener="household")
        assert "invalid api key" in str(excinfo.value).lower()
        assert "api key" in str(excinfo.value).lower()
    finally:
        await store.close()


async def test_failure_logged_before_it_propagates(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The failure must actually reach the log, not just the raised exception (the reported bug)."""
    store = await _new_store(tmp_path)
    try:
        client = _RaisingHttpClient(_StatusError(403))
        importer = LastfmImporter(client, "testuser", "fake-key")
        with caplog.at_level("WARNING"), pytest.raises(LastfmApiError):
            await importer.import_since(store, listener="household")
        assert any("page 1 failed" in record.message for record in caplog.records)
    finally:
        await store.close()
