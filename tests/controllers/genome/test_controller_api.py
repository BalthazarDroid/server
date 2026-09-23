"""
Tests for ``GenomeController``'s API commands, config entries and Apple upload protocol (§3.2/§3.3).

Runs entirely against the in-memory :class:`StubGenomeStore` from ``conftest.py`` — never
against the real ``GenomeStore`` (owned by a separate work package).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import sqlite3
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import ConfigEntryType
from music_assistant_models.errors import InvalidDataError

from music_assistant.controllers.genome import controller as controller_module
from music_assistant.controllers.genome.constants import (
    CONF_ACTION_CLEAR_GENOME_DATA,
    CONF_ACTION_EXPORT_DB,
    CONF_ACTION_REBUILD_NOW,
    CONF_APPLE_IMPORT_DIR,
    CONF_LASTFM_API_KEY,
    CONF_LASTFM_USERNAME,
    CONF_RECENCY_HALF_LIFE_DAYS,
    LISTENER_HOUSEHOLD,
)
from music_assistant.controllers.genome.controller import GenomeController
from music_assistant.controllers.genome.errors import LastfmNotConfiguredError
from music_assistant.controllers.genome.jobs import (
    JOB_EXPORT_DB,
    JOB_LASTFM_IMPORT,
)
from music_assistant.controllers.genome.models import (
    GenomeImportResult,
    GenomeSettingsPatch,
    Listen,
)
from tests.controllers.genome.conftest import FixtureHttpClient, StubGenomeStore

CSV_CHUNK_1 = base64.b64encode(b"Song Name,Artist Name,Event Start Timestamp\n").decode()
CSV_CHUNK_2 = base64.b64encode(b"Test Song,Test Artist,2024-01-01T00:00:00Z\n").decode()


def _listen(**overrides: object) -> Listen:
    base: dict[str, object] = {
        "played_at": 1_700_000_000,
        "artist_key": "artist",
        "artist_name": "Artist",
        "track_key": "track",
        "track_name": "Track",
        "album_name": None,
        "source": "ma_playlog",
        "player_id": None,
        "duration_ms": None,
        "played_ms": None,
        "fully_played": True,
        "confidence": 1.0,
    }
    base.update(overrides)
    return Listen(**base)  # type: ignore[arg-type]


async def _fake_apple_parser(path: str, *, min_seconds: int, stats=None):  # noqa: ARG001
    # signature must match GenomeController._apple_parser exactly (positional `path`,
    # keyword-only `min_seconds`/`stats`) - the fake never needs path/min_seconds itself
    if stats is not None:
        stats.rows_read = 1
    yield _listen(source="apple_export")


async def _fake_apple_parser_with_skips(path: str, *, min_seconds: int, stats=None):  # noqa: ARG001
    # simulates a CSV with 3 raw rows where 2 were filtered out by the parser itself (bad
    # data, or below min_seconds) - only the store-level dedupe accounting is visible unless
    # this stats sidecar is threaded through and merged into the returned GenomeImportResult
    if stats is not None:
        stats.rows_read = 3
        stats.rows_skipped = 2
        stats.warnings.append("row 2: missing artist name")
    yield _listen(source="apple_export")


# ---------------------------------------------------------------------------------------
# Config entries (§3.2)
# ---------------------------------------------------------------------------------------


async def test_config_entries_cover_every_documented_key(
    genome_controller: GenomeController,
) -> None:
    """Test config entries cover every documented key."""
    entries = await genome_controller.get_config_entries()
    keys = {entry.key for entry in entries}
    assert keys == {
        "recency_half_life_days",
        "lastfm_username",
        "lastfm_api_key",
        "lastfm_poll_enabled",
        "lastfm_poll_interval_hours",
        "apple_import_dir",
        "enrich_enabled",
        "obscurity_percentile",
        "min_seconds_played",
        "rebuild_schedule_hour",
        CONF_ACTION_REBUILD_NOW,
        CONF_ACTION_EXPORT_DB,
        CONF_ACTION_CLEAR_GENOME_DATA,
    }


async def test_config_entries_never_hardcode_label_or_description(
    genome_controller: GenomeController,
) -> None:
    """Test config entries never hardcode label or description."""
    entries = await genome_controller.get_config_entries()
    for entry in entries:
        assert entry.label is None
        assert entry.description is None


async def test_lastfm_api_key_is_secure_string(genome_controller: GenomeController) -> None:
    """Test lastfm api key is secure string."""
    entries = await genome_controller.get_config_entries()
    entry = next(e for e in entries if e.key == CONF_LASTFM_API_KEY)
    assert entry.type == ConfigEntryType.SECURE_STRING


async def test_rebuild_now_action_rebuilds(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """Test rebuild now action rebuilds."""
    genome_store.listens.append(_listen())
    result = await genome_controller.handle_config_action(CONF_ACTION_REBUILD_NOW)
    assert result is not None
    assert genome_store.cache[LISTENER_HOUSEHOLD]["stats"]["total_listens"] == 1


async def test_rebuild_now_action_reports_listens_used(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """A populated store's rebuild result must say how many listens it actually used."""
    genome_store.listens.append(_listen())
    result = await genome_controller.handle_config_action(CONF_ACTION_REBUILD_NOW)
    assert result is not None
    assert result.translation_key == f"{CONF_ACTION_REBUILD_NOW}.result"
    assert result.translation_args == ["1"]


async def test_rebuild_now_action_on_empty_store_says_so(
    genome_controller: GenomeController,
) -> None:
    """
    An empty-store rebuild must be distinguishable from a real failure.

    Before this fix, `rebuild_now` over an empty store returned the exact same generic
    "rebuilt" result as a real rebuild, with zero indication that nothing was actually there
    to rebuild from.
    """
    result = await genome_controller.handle_config_action(CONF_ACTION_REBUILD_NOW)
    assert result is not None
    assert result.translation_key == f"{CONF_ACTION_REBUILD_NOW}.result_empty"
    assert result.translation_key != f"{CONF_ACTION_REBUILD_NOW}.result"


async def test_clear_genome_data_action_clears_store(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """Test clear genome data action clears store."""
    genome_store.listens.append(_listen())
    await genome_controller.handle_config_action(CONF_ACTION_CLEAR_GENOME_DATA)
    assert genome_store.listens == []


# ---------------------------------------------------------------------------------------
# genome/get, genome/rebuild (§3.3, §3.4)
# ---------------------------------------------------------------------------------------


async def test_get_genome_empty_state(genome_controller: GenomeController) -> None:
    """Test get genome empty state."""
    result = await genome_controller.get_genome()
    assert result["stats"]["total_listens"] == 0
    assert result["listener"] == LISTENER_HOUSEHOLD


async def test_unresolved_artists_returns_store_rows(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """`genome/unresolved_artists` passes through whatever the store reports as failed."""
    genome_store.failed_artists = [
        {"artist_key": "a", "artist_name": "Artist A", "resolved_at": 200},
        {"artist_key": "b", "artist_name": "Artist B", "resolved_at": 100},
    ]
    result = await genome_controller.unresolved_artists()
    assert result == genome_store.failed_artists


async def test_unresolved_artists_passes_limit_through(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """The `limit` argument reaches the store unchanged."""
    genome_store.failed_artists = [
        {"artist_key": "a", "artist_name": "Artist A", "resolved_at": 200},
        {"artist_key": "b", "artist_name": "Artist B", "resolved_at": 100},
    ]
    result = await genome_controller.unresolved_artists(limit=1)
    assert result == [genome_store.failed_artists[0]]


async def test_unresolved_artists_empty_by_default(genome_controller: GenomeController) -> None:
    """No artists have failed in the stub by default."""
    assert await genome_controller.unresolved_artists() == []


# ---------------------------------------------------------------------------------------
# genome/retry_artists, genome/dismiss_unresolved
# ---------------------------------------------------------------------------------------


async def test_retry_artists_none_retries_every_failed_artist(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """Passing no keys retries every currently-failed artist."""
    genome_store.failed_keys = {"a", "b"}
    result = await genome_controller.retry_artists()
    assert result == 2
    assert genome_store.failed_keys == set()
    assert sorted(genome_store.retried_keys) == ["a", "b"]


async def test_retry_artists_with_keys_retries_only_those(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """Passing explicit keys only retries the intersection with the failed set."""
    genome_store.failed_keys = {"a", "b", "c"}
    result = await genome_controller.retry_artists(artist_keys=["a", "z"])
    assert result == 1
    assert genome_store.failed_keys == {"b", "c"}
    assert genome_store.retried_keys == ["a"]


async def test_retry_artists_returns_zero_when_nothing_failed(
    genome_controller: GenomeController,
) -> None:
    """No failed artists in the stub by default means nothing to retry."""
    assert await genome_controller.retry_artists() == 0


async def test_retry_artists_does_no_network_work(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """`genome/retry_artists` only touches store state - never anything network-shaped."""
    genome_store.failed_keys = {"a"}
    with mock.patch("music_assistant.controllers.genome.http.AiohttpClient") as client:
        await genome_controller.retry_artists()
    client.assert_not_called()


async def test_dismiss_unresolved_records_current_failed_set(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """Dismissing records exactly the artists currently failing, and reports itself dismissed."""
    genome_store.failed_keys = {"a", "b"}
    result = await genome_controller.dismiss_unresolved()
    assert result is True
    assert genome_store.dismissed_keys == frozenset({"a", "b"})


async def test_dismiss_unresolved_returns_to_false_when_a_new_artist_fails(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """A newly-failing artist changes the current set, so the dismissal no longer matches."""
    genome_store.failed_keys = {"a"}
    await genome_controller.dismiss_unresolved()
    genome_store.failed_keys.add("b")
    assert await genome_controller._unresolved_dismissed() is False


async def test_dismiss_unresolved_with_no_failures_dismisses_the_empty_set(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """Dismissing with nothing currently failed still records (and matches) the empty set."""
    result = await genome_controller.dismiss_unresolved()
    assert result is True
    assert genome_store.dismissed_keys == frozenset()


async def test_rebuild_stats_reflect_dismissal_state(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """`stats.unresolved_dismissed` is computed fresh on every rebuild."""
    genome_store.failed_keys = {"a"}
    result = await genome_controller.get_genome(refresh=True)
    assert result["stats"]["unresolved_dismissed"] is False
    await genome_controller.dismiss_unresolved()
    result = await genome_controller.get_genome(refresh=True)
    assert result["stats"]["unresolved_dismissed"] is True


async def test_get_genome_serves_fresh_cache(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """Test get genome serves fresh cache."""
    genome_store.listens.append(_listen())
    first = await genome_controller.get_genome(refresh=True)
    second = await genome_controller.get_genome()
    assert second["stale"] is False
    assert second["generated_at"] == first["generated_at"]


async def test_get_genome_flags_stale_cache(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """Test get genome flags stale cache."""
    await genome_controller.get_genome(refresh=True)
    genome_store.listens.append(_listen())
    result = await genome_controller.get_genome()
    assert result["stale"] is True
    assert result["stats"]["total_listens"] == 0  # still serving the old cached payload


async def test_get_genome_refresh_forces_rebuild(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """Test get genome refresh forces rebuild."""
    await genome_controller.get_genome(refresh=True)
    genome_store.listens.append(_listen())
    result = await genome_controller.get_genome(refresh=True)
    assert result["stale"] is False
    assert result["stats"]["total_listens"] == 1


async def test_rebuild_returns_counts_and_caches(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """Test rebuild returns counts and caches."""
    genome_store.listens.extend([_listen(), _listen(track_key="t2")])
    result = await genome_controller.rebuild()
    assert result["listens_scanned"] == 2
    assert result["genome"]["stats"]["total_listens"] == 2
    assert genome_store.cache[LISTENER_HOUSEHOLD] == result["genome"]


# ---------------------------------------------------------------------------------------
# Apple upload protocol (§3.3)
# ---------------------------------------------------------------------------------------


async def test_apple_upload_non_final_chunk_returns_empty_result(
    genome_controller: GenomeController,
) -> None:
    """Test apple upload non final chunk returns empty result."""
    genome_controller._apple_parser = _fake_apple_parser
    result = await genome_controller.import_apple("upload-1", 0, CSV_CHUNK_1, final=False)
    assert result["rows_imported"] == 0


async def test_apple_upload_final_chunk_ingests(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """Test apple upload final chunk ingests."""
    genome_controller._apple_parser = _fake_apple_parser
    await genome_controller.import_apple("upload-1", 0, CSV_CHUNK_1, final=False)
    result = await genome_controller.import_apple(
        "upload-1", 1, CSV_CHUNK_2, final=True, filename="Apple Music Play Activity.csv"
    )
    assert result["rows_imported"] == 1
    assert len(genome_store.listens) == 1
    # the partial upload state is cleared after a final chunk
    assert "upload-1" not in genome_controller._uploads


async def test_apple_upload_out_of_order_chunk_raises(genome_controller: GenomeController) -> None:
    """Test apple upload out of order chunk raises."""
    with pytest.raises(InvalidDataError):
        await genome_controller.import_apple("upload-2", 5, CSV_CHUNK_1, final=False)


async def test_apple_upload_wrong_first_seq_raises(genome_controller: GenomeController) -> None:
    """Test apple upload wrong first seq raises."""
    with pytest.raises(InvalidDataError):
        await genome_controller.import_apple("upload-3", 1, CSV_CHUNK_1, final=False)


async def test_apple_upload_out_of_order_after_start_raises(
    genome_controller: GenomeController,
) -> None:
    """Test apple upload out of order after start raises."""
    await genome_controller.import_apple("upload-4", 0, CSV_CHUNK_1, final=False)
    with pytest.raises(InvalidDataError):
        await genome_controller.import_apple("upload-4", 3, CSV_CHUNK_2, final=False)


async def test_apple_upload_rejects_non_csv_filename(genome_controller: GenomeController) -> None:
    """Test apple upload rejects non csv filename."""
    genome_controller._apple_parser = _fake_apple_parser
    with pytest.raises(InvalidDataError):
        await genome_controller.import_apple(
            "upload-5", 0, CSV_CHUNK_1, final=True, filename="not-a-csv.txt"
        )


async def test_apple_import_via_server_side_dir(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """Test apple import via server side dir."""
    genome_controller._apple_parser = _fake_apple_parser
    genome_controller.get_config_value = lambda key, default=None, *, return_type=None: (  # noqa: ARG005
        "/some/dir" if key == CONF_APPLE_IMPORT_DIR else default
    )
    result = await genome_controller.import_apple(
        "", 0, "", final=True, filename="Apple Music Play Activity.csv"
    )
    assert result["rows_imported"] == 1
    assert len(genome_store.listens) == 1


async def test_apple_import_without_upload_id_or_dir_raises(
    genome_controller: GenomeController,
) -> None:
    """Test apple import without upload id or dir raises."""
    with pytest.raises(InvalidDataError):
        await genome_controller.import_apple("", 0, "", final=True, filename="export.csv")


async def test_apple_import_propagates_parser_warnings_and_rows_skipped(
    genome_controller: GenomeController, genome_store: StubGenomeStore
) -> None:
    """The CSV parser's own rows_skipped/warnings must reach the returned GenomeImportResult."""
    genome_controller._apple_parser = _fake_apple_parser_with_skips
    genome_controller.get_config_value = lambda key, default=None, *, return_type=None: (  # noqa: ARG005
        "/some/dir" if key == CONF_APPLE_IMPORT_DIR else default
    )
    result = await genome_controller.import_apple(
        "", 0, "", final=True, filename="Apple Music Play Activity.csv"
    )
    assert result["rows_read"] == 3
    assert result["rows_skipped"] == 2
    assert result["rows_imported"] == 1
    assert result["warnings"] == ["row 2: missing artist name"]
    assert len(genome_store.listens) == 1


async def test_apple_import_writes_its_warnings_to_the_log(
    genome_controller: GenomeController,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    An import that explains itself only to the page explains itself to nobody.

    A real 306,140-row export imported nothing twice over. The parser had been taught to say
    why - an unrecognised header, a ranked tally of skip reasons - and every word of it went
    into the result handed back to the import panel, which showed three counts and no text.
    The add-on log, the one place anyone looks when an import fails, said only "0 imported".
    """
    genome_controller._apple_parser = _fake_apple_parser_with_skips
    genome_controller.get_config_value = lambda key, default=None, *, return_type=None: (  # noqa: ARG005
        "/some/dir" if key == CONF_APPLE_IMPORT_DIR else default
    )
    with caplog.at_level(logging.WARNING):
        await genome_controller.import_apple(
            "", 0, "", final=True, filename="Apple Music Play Activity.csv"
        )
    assert any("missing artist name" in record.getMessage() for record in caplog.records)


async def test_apple_import_logs_the_skip_tally_the_controller_now_builds(
    genome_controller: GenomeController,
    genome_store: StubGenomeStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    The ranked tally lived in ``import_play_activity``, which this controller never calls.

    So the counts were gathered on every real import and thrown away. The controller has to
    close the tally itself, or the diagnostic exists only on a code path nothing walks.
    """

    async def parser(path: str, *, min_seconds: int, stats=None):  # noqa: ARG001
        if stats is not None:
            stats.rows_read = 4
            stats.note_skip("no artist name in the export")
            stats.note_skip("no artist name in the export")
            stats.note_skip("not audio")
        yield _listen(source="apple_export")

    genome_controller._apple_parser = parser
    genome_controller.get_config_value = lambda key, default=None, *, return_type=None: (  # noqa: ARG005
        "/some/dir" if key == CONF_APPLE_IMPORT_DIR else default
    )
    with caplog.at_level(logging.WARNING):
        result = await genome_controller.import_apple(
            "", 0, "", final=True, filename="Apple Music Play Activity.csv"
        )
    tally = next(w for w in result["warnings"] if w.startswith("Skipped rows by reason:"))
    assert "no artist name in the export (2)" in tally
    assert any("Skipped rows by reason" in record.getMessage() for record in caplog.records)
    assert len(genome_store.listens) == 1


# ---------------------------------------------------------------------------------------
# genome/settings get/set (§3.3)
# ---------------------------------------------------------------------------------------


async def test_get_settings_never_exposes_api_key(genome_controller: GenomeController) -> None:
    """Test get settings never exposes api key."""
    settings = await genome_controller.get_settings()
    assert "lastfm_api_key" not in settings
    assert settings["lastfm_configured"] is False


async def test_set_settings_round_trips_through_config(genome_controller: GenomeController) -> None:
    """Test set settings round trips through config."""
    saved: dict[str, object] = {}

    async def fake_save_core_config(_domain: str, values: dict[str, object]) -> None:
        saved.update(values)

    genome_controller.mass.config.save_core_config = fake_save_core_config
    await genome_controller.set_settings(GenomeSettingsPatch(half_life_days=365))
    assert saved[CONF_RECENCY_HALF_LIFE_DAYS] == 365


async def test_set_settings_empty_patch_does_not_touch_config(
    genome_controller: GenomeController,
) -> None:
    """Test set settings empty patch does not touch config."""
    called = False

    async def fake_save_core_config(_domain: str, _values: dict[str, object]) -> None:
        nonlocal called
        called = True

    genome_controller.mass.config.save_core_config = fake_save_core_config
    await genome_controller.set_settings(GenomeSettingsPatch())
    assert called is False


# ---------------------------------------------------------------------------------------
# genome/import_lastfm — missing credentials must be logged, not just raised (the reported bug:
# GenomeController.import_lastfm raised InvalidDataError before any logging happened, so a
# user-facing failure left nothing in the server log)
# ---------------------------------------------------------------------------------------


async def test_import_lastfm_without_config_raises_and_logs(
    genome_controller: GenomeController, caplog: pytest.LogCaptureFixture
) -> None:
    """Test import lastfm without config raises and logs."""
    with caplog.at_level("WARNING"), pytest.raises(LastfmNotConfiguredError):
        await genome_controller.import_lastfm()
    assert any("genome/import_lastfm" in record.message for record in caplog.records)


async def test_import_lastfm_without_config_message_says_what_is_missing(
    genome_controller: GenomeController,
) -> None:
    """The error must tell the user what to fix, not just that something is wrong."""
    with pytest.raises(LastfmNotConfiguredError) as excinfo:
        await genome_controller.import_lastfm()
    message = str(excinfo.value).lower()
    assert "username" in message
    assert "api key" in message


async def test_import_lastfm_with_only_username_reports_missing_api_key(
    genome_controller: GenomeController,
) -> None:
    """Test import lastfm with only username reports missing api key."""
    genome_controller.get_config_value = (  # type: ignore[method-assign]
        lambda key, default=None, *, return_type=None: (  # noqa: ARG005
            "Bob_Baird" if key == CONF_LASTFM_USERNAME else default
        )
    )
    with pytest.raises(LastfmNotConfiguredError) as excinfo:
        await genome_controller.import_lastfm()
    message = str(excinfo.value).lower()
    assert "api key" in message
    assert "username is missing" not in message


async def test_import_lastfm_unexpected_failure_logged_as_error(
    genome_controller: GenomeController, mass_stub: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """An unanticipated failure (a bug, not a user-correctable problem) gets ERROR + traceback."""
    genome_controller.get_config_value = (  # type: ignore[method-assign]
        lambda key, default=None, *, return_type=None: (  # noqa: ARG005
            # a well-formed (if fake) 32-hex key, so this test exercises the failure path it
            # is actually about rather than tripping the API-key format guard first
            "someuser" if key == CONF_LASTFM_USERNAME else "0" * 32
        )
    )

    def _boom(*_args: object, **_kwargs: object) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    genome_controller._lastfm_importer_factory = _boom  # type: ignore[method-assign]
    with caplog.at_level("DEBUG"):
        await genome_controller.import_lastfm()
        await mass_stub.created_tasks[-1]
    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any("genome/import_lastfm" in r.message for r in error_records)
    assert any(r.exc_info is not None for r in error_records)


async def test_failed_background_import_records_the_reason_not_just_a_failure(
    genome_controller: GenomeController, mass_stub: MagicMock
) -> None:
    """
    A background job that dies has to leave the reason behind, or nobody ever learns it.

    The whole point of moving the import off the websocket request is that the user can navigate
    away. If the job records only "error", the one thing they came back for - why it failed - is
    exactly what is missing, and the old bug (a progress bar and no explanation) is back.
    """
    genome_controller.get_config_value = (  # type: ignore[method-assign]
        lambda key, default=None, *, return_type=None: (  # noqa: ARG005
            "someuser" if key == CONF_LASTFM_USERNAME else "0" * 32
        )
    )

    def _boom(*_args: object, **_kwargs: object) -> None:
        msg = "last.fm said 403 Forbidden"
        raise RuntimeError(msg)

    genome_controller._lastfm_importer_factory = _boom  # type: ignore[method-assign]
    await genome_controller.import_lastfm()
    await mass_stub.created_tasks[-1]

    job = (await genome_controller.get_jobs())[JOB_LASTFM_IMPORT]
    assert job["state"] == "error"
    assert "403 Forbidden" in job["message"]
    assert job["finished_at"] is not None


async def test_import_lastfm_returns_immediately_with_a_running_job(
    genome_controller: GenomeController, mass_stub: MagicMock
) -> None:
    """
    A paginated history sweep must not be what the websocket request is waiting on.

    It can run for minutes; the request has to hand back the job handle and let the page move on.
    """
    genome_controller.get_config_value = (  # type: ignore[method-assign]
        lambda key, default=None, *, return_type=None: (  # noqa: ARG005
            "someuser" if key == CONF_LASTFM_USERNAME else "0" * 32
        )
    )
    started = asyncio.Event()
    release = asyncio.Event()

    def _factory(_mass: object, **_kw: object) -> object:
        class _Importer:
            async def import_since(self, _store: object, **_kw: object) -> GenomeImportResult:
                started.set()
                await release.wait()
                return GenomeImportResult(
                    source="lastfm",
                    rows_read=0,
                    rows_imported=0,
                    rows_skipped=0,
                    rows_duplicate=0,
                    first_played_at=None,
                    last_played_at=None,
                    warnings=[],
                )

        return _Importer()

    genome_controller._lastfm_importer_factory = _factory  # type: ignore[assignment]

    state = await genome_controller.import_lastfm()

    assert state["state"] == "running"
    # the import is genuinely still in flight - the request did not secretly await it
    await started.wait()
    assert not mass_stub.created_tasks[-1].done()
    release.set()
    await mass_stub.created_tasks[-1]


async def test_import_lastfm_records_the_row_counts_a_person_needs(
    genome_controller: GenomeController, mass_stub: MagicMock
) -> None:
    """
    The recorded message is the whole result once the page that asked for it is gone.

    "Import finished" would be useless: what the user came back for is how many rows landed.
    """
    genome_controller.get_config_value = (  # type: ignore[method-assign]
        lambda key, default=None, *, return_type=None: (  # noqa: ARG005
            "someuser" if key == CONF_LASTFM_USERNAME else "0" * 32
        )
    )

    def _factory(_mass: object, **_kw: object) -> object:
        class _Importer:
            async def import_since(self, _store: object, **_kw: object) -> GenomeImportResult:
                return GenomeImportResult(
                    source="lastfm",
                    rows_read=120,
                    rows_imported=97,
                    rows_skipped=20,
                    rows_duplicate=3,
                    first_played_at=None,
                    last_played_at=None,
                    warnings=["3 scrobbles had no timestamp"],
                )

        return _Importer()

    genome_controller._lastfm_importer_factory = _factory  # type: ignore[assignment]
    await genome_controller.import_lastfm()
    await mass_stub.created_tasks[-1]

    job = (await genome_controller.get_jobs())[JOB_LASTFM_IMPORT]
    assert job["state"] == "ok"
    assert "97 imported" in job["message"]
    assert "20 skipped" in job["message"]
    assert "3 duplicate" in job["message"]
    assert "3 scrobbles had no timestamp" in job["message"]


# ---------------------------------------------------------------------------------------
# genome/settings/set — the Last.fm API key must be settable the same way every other
# setting is, and genome/settings must still never echo it back (§3.3)
# ---------------------------------------------------------------------------------------


async def test_set_settings_can_set_lastfm_api_key(genome_controller: GenomeController) -> None:
    """Test set settings can set lastfm api key."""
    saved: dict[str, object] = {}

    async def fake_save_core_config(_domain: str, values: dict[str, object]) -> None:
        saved.update(values)

    genome_controller.mass.config.save_core_config = fake_save_core_config
    await genome_controller.set_settings(GenomeSettingsPatch(lastfm_api_key="super-secret-key"))
    assert saved[CONF_LASTFM_API_KEY] == "super-secret-key"


async def test_get_settings_still_redacts_a_freshly_set_api_key(
    genome_controller: GenomeController,
) -> None:
    """A key just written through set_settings must still never come back from get_settings."""
    saved: dict[str, object] = {}

    async def fake_save_core_config(_domain: str, values: dict[str, object]) -> None:
        saved.update(values)

    genome_controller.mass.config.save_core_config = fake_save_core_config
    genome_controller.get_config_value = (  # type: ignore[method-assign]
        lambda key, default=None, *, return_type=None: saved.get(key, default)  # noqa: ARG005
    )
    settings = await genome_controller.set_settings(
        GenomeSettingsPatch(lastfm_api_key="super-secret-key")
    )
    assert "lastfm_api_key" not in settings
    assert "super-secret-key" not in repr(settings)
    assert settings["lastfm_configured"] is True


@pytest.mark.asyncio
async def test_import_lastfm_rejects_malformed_api_key(
    genome_controller: GenomeController,
) -> None:
    """
    A mis-pasted API key fails locally, with a message that names the real problem.

    Regression test for a real incident: a URL was pasted into the API-key field, went out as a
    query parameter, and came back as an opaque `403 Forbidden` whose error text embedded the
    whole request URL. Validating the shape here means no network call and no secret in flight.
    """
    values = {
        CONF_LASTFM_USERNAME: "Bob_Baird",
        CONF_LASTFM_API_KEY: " http://192.168.4.11:8095/setup",
    }
    genome_controller.get_config_value = (  # type: ignore[method-assign]
        lambda key, default=None, **_kw: values.get(key, default)
    )
    with pytest.raises(LastfmNotConfiguredError) as excinfo:
        await genome_controller.import_lastfm()
    message = str(excinfo.value)
    assert "32 hex characters" in message
    # the key must never be echoed back, only its length
    assert "192.168.4.11" not in message
    assert "setup" not in message


@pytest.mark.asyncio
async def test_import_lastfm_accepts_a_well_formed_key_with_whitespace(
    genome_controller: GenomeController, mass_stub: MagicMock
) -> None:
    """A key that is valid apart from stray copy-paste whitespace is accepted and trimmed."""
    key = "a" * 32
    values = {CONF_LASTFM_USERNAME: "Bob_Baird", CONF_LASTFM_API_KEY: f"  {key}\n"}
    genome_controller.get_config_value = (  # type: ignore[method-assign]
        lambda k, default=None, **_kw: values.get(k, default)
    )
    captured: dict[str, str] = {}

    def _factory(_mass: object, *, username: str, api_key: str) -> object:
        captured["api_key"] = api_key
        captured["username"] = username

        class _Importer:
            async def import_since(self, _store: object, **_kw: object) -> GenomeImportResult:
                return GenomeImportResult(
                    source="lastfm",
                    rows_read=0,
                    rows_imported=0,
                    rows_duplicate=0,
                    rows_skipped=0,
                    first_played_at=None,
                    last_played_at=None,
                    warnings=[],
                )

        return _Importer()

    genome_controller._lastfm_importer_factory = _factory  # type: ignore[assignment]
    await genome_controller.import_lastfm()
    await mass_stub.created_tasks[-1]
    assert captured["api_key"] == key


async def test_get_genome_never_enriches(
    genome_controller: GenomeController,
    genome_store: StubGenomeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A read must never wait on MusicBrainz.

    Regression: `genome/get` fell through to a rebuild that resolved pending artists inline.
    On a real backlog that meant a MusicBrainz pass - paced at roughly one artist per second,
    and answered with minute-long penalties when it slips - inside the websocket call the page
    was waiting on. The page simply never finished loading.
    """
    calls = 0

    async def _tripwire(*_args: object, **_kwargs: object) -> int:
        nonlocal calls
        calls += 1
        return 0

    monkeypatch.setattr(genome_controller, "_enrich_pending", _tripwire)
    genome_store.listens.append(_listen())
    await genome_controller.get_genome(refresh=True)
    await genome_controller.get_genome()
    assert calls == 0


async def test_rebuild_dispatches_enrichment_without_awaiting_it(
    genome_controller: GenomeController,
    genome_store: StubGenomeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Rebuild returns the recomputed genome now and leaves enrichment running behind it.

    The command must not block on the pass: `stats.artists_pending` is how the caller learns
    that more is still resolving.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_pass(*_args: object, **_kwargs: object) -> int:
        started.set()
        await release.wait()
        return 7

    monkeypatch.setattr(genome_controller, "_enrich_pending", _slow_pass)
    genome_store.listens.append(_listen())

    result = await asyncio.wait_for(genome_controller.rebuild(), timeout=5)
    assert result["listens_scanned"] == 1
    await asyncio.wait_for(started.wait(), timeout=5)  # dispatched, still running
    release.set()


def _patch_listenbrainz_client(
    monkeypatch: pytest.MonkeyPatch, fixture_http_client: FixtureHttpClient
) -> None:
    """Make every ``AiohttpClient(...)`` construction in genome/* return ``fixture_http_client``."""
    monkeypatch.setattr(
        "music_assistant.controllers.genome.http.AiohttpClient",
        lambda *_args, **_kwargs: fixture_http_client,
    )


async def test_enrich_pending_drains_popularity_backlog_independent_of_resolution(
    genome_controller: GenomeController,
    genome_store: StubGenomeStore,
    fixture_http_client: FixtureHttpClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Regression: lb_listeners was only ever looked up for artists the current pass resolved.

    An artist that already has an mbid from an earlier pass (or a pass whose ListenBrainz lookup
    failed) but never got popularity must still be picked up here, even when nothing is pending
    MusicBrainz resolution at all.
    """
    _patch_listenbrainz_client(monkeypatch, fixture_http_client)
    genome_store.popularity_backlog = [("sigurros", "f6f2326f-6b25-4170-b89d-e235b25508e8")]

    resolved = await genome_controller._enrich_pending()

    assert resolved == 0  # nothing was pending MusicBrainz resolution
    assert genome_store.lb_popularity_updates["sigurros"] == (118422, 4821334)
    assert genome_store.popularity_attempted == []


async def test_enrich_pending_marks_unresolved_popularity_as_attempted(
    genome_controller: GenomeController,
    genome_store: StubGenomeStore,
    fixture_http_client: FixtureHttpClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An artist ListenBrainz has no data for is marked attempted, not retried every pass."""
    _patch_listenbrainz_client(monkeypatch, fixture_http_client)
    genome_store.popularity_backlog = [("ghost", "00000000-missing-mbid")]

    await genome_controller._enrich_pending()

    assert genome_store.lb_popularity_updates == {}
    assert genome_store.popularity_attempted == ["ghost"]


async def test_enrich_pending_caps_popularity_backlog_at_limit(
    genome_controller: GenomeController,
    genome_store: StubGenomeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The popularity backfill pass is capped by the same ``limit`` as the MusicBrainz pass."""
    captured: dict[str, int] = {}

    async def _fake_pending_popularity_keys(limit: int = 200) -> list[tuple[str, str]]:
        captured["limit"] = limit
        return []

    monkeypatch.setattr(genome_store, "pending_popularity_keys", _fake_pending_popularity_keys)

    await genome_controller._enrich_pending(limit=5)

    assert captured["limit"] == 5


async def test_enrich_pending_is_noop_with_no_backlog(
    genome_controller: GenomeController,
) -> None:
    """An empty popularity backlog (the default stub state) must not raise."""
    resolved = await genome_controller._enrich_pending()
    assert resolved == 0


async def test_export_db_writes_a_readable_snapshot(
    genome_controller: GenomeController, tmp_path: Path
) -> None:
    """
    The one step of the HACS migration that cannot be undone if it is skipped.

    Most of genome.db is rebuildable from its sources - the Last.fm and Apple imports are
    repeatable. The live-captured plays are not: they were recorded from the playlog as they
    happened and exist nowhere else.
    """
    source = tmp_path / "genome.db"
    connection = sqlite3.connect(source)
    connection.execute("CREATE TABLE listens (id INTEGER PRIMARY KEY, name TEXT)")
    connection.execute("INSERT INTO listens (name) VALUES ('kept')")
    connection.commit()
    connection.close()

    destination = tmp_path / "share"
    destination.mkdir()
    genome_controller.store.db_path = str(source)  # type: ignore[misc]

    result = await genome_controller._run_export_db_job(directory=str(destination))

    assert result is not None
    written = Path(result["path"])
    assert written.parent == destination
    assert result["bytes"] > 0
    # A snapshot that cannot be opened is worse than no snapshot, because it looks like one.
    copy = sqlite3.connect(written)
    try:
        assert copy.execute("SELECT name FROM listens").fetchall() == [("kept",)]
    finally:
        copy.close()


async def test_export_db_refuses_a_directory_that_is_not_there(
    genome_controller: GenomeController, tmp_path: Path
) -> None:
    """A typo in the destination must fail loudly, not silently write nowhere."""
    source = tmp_path / "genome.db"
    sqlite3.connect(source).close()
    genome_controller.store.db_path = str(source)  # type: ignore[misc]

    with pytest.raises(InvalidDataError, match="not a directory this app can write to"):
        await genome_controller._do_export_db(directory=str(tmp_path / "nope"))


async def test_export_db_refuses_when_there_is_no_database_yet(
    genome_controller: GenomeController, tmp_path: Path
) -> None:
    """Exporting nothing would hand back an empty file that reads as a successful backup."""
    genome_controller.store.db_path = str(tmp_path / "missing.db")  # type: ignore[misc]

    with pytest.raises(InvalidDataError, match="No genome database"):
        await genome_controller._do_export_db(directory=str(tmp_path))


async def test_export_db_action_button_starts_the_job_without_waiting(
    genome_controller: GenomeController, tmp_path: Path
) -> None:
    """
    The command needs a button, or it is not reachable from anywhere but a terminal.

    The button must dispatch rather than await: copying a 200,000-row database inside the
    request is the same stall that made the first version of this look hung forever. The path
    lands on the job, whose message outlives both the dialog and the page.
    """
    source = tmp_path / "genome.db"
    sqlite3.connect(source).close()
    genome_controller.store.db_path = str(source)  # type: ignore[misc]

    result = await genome_controller.handle_config_action(CONF_ACTION_EXPORT_DB)

    assert result is not None
    assert result.translation_key == f"{CONF_ACTION_EXPORT_DB}.started"
    assert genome_controller.jobs.get_all()[JOB_EXPORT_DB]["state"] == "running"


async def test_export_db_finds_a_mount_when_none_is_named(
    genome_controller: GenomeController, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The default must be discovered, not assumed.

    Hardcoding "/share" cost a round trip: the MA DEV app does not map it, so the export failed
    with "the data provided is invalid" and the reason only existed in the app log. Which folders
    an app maps is not knowable from inside it, so the command looks.
    """
    source = tmp_path / "genome.db"
    sqlite3.connect(source).close()
    genome_controller.store.db_path = str(source)  # type: ignore[misc]
    mount = tmp_path / "media"
    mount.mkdir()
    monkeypatch.setattr(controller_module, "GENOME_EXPORT_DIRS", ("/nope", str(mount)))

    result = await genome_controller._do_export_db()

    assert Path(result["path"]).parent == mount


async def test_export_db_says_what_is_missing_when_nothing_is_mounted(
    genome_controller: GenomeController, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An app with no shared folder has to say so, and name what it looked for."""
    source = tmp_path / "genome.db"
    sqlite3.connect(source).close()
    genome_controller.store.db_path = str(source)  # type: ignore[misc]
    monkeypatch.setattr(controller_module, "GENOME_EXPORT_DIRS", ("/nope", "/also-nope"))

    with pytest.raises(InvalidDataError, match="no shared folder"):
        await genome_controller._do_export_db()
