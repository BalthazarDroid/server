"""
Tests for ``GenomeController``'s API commands, config entries and Apple upload protocol (§3.2/§3.3).

Runs entirely against the in-memory :class:`StubGenomeStore` from ``conftest.py`` — never
against the real ``GenomeStore`` (owned by a separate work package).
"""

from __future__ import annotations

import base64

import pytest
from music_assistant_models.enums import ConfigEntryType
from music_assistant_models.errors import InvalidDataError

from music_assistant.controllers.genome.constants import (
    CONF_ACTION_CLEAR_GENOME_DATA,
    CONF_ACTION_REBUILD_NOW,
    CONF_APPLE_IMPORT_DIR,
    CONF_LASTFM_API_KEY,
    CONF_LASTFM_USERNAME,
    CONF_RECENCY_HALF_LIFE_DAYS,
    LISTENER_HOUSEHOLD,
)
from music_assistant.controllers.genome.controller import GenomeController
from music_assistant.controllers.genome.errors import LastfmNotConfiguredError
from music_assistant.controllers.genome.models import (
    GenomeImportResult,
    GenomeSettingsPatch,
    Listen,
)
from tests.controllers.genome.conftest import StubGenomeStore

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
    genome_controller: GenomeController, caplog: pytest.LogCaptureFixture
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
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError):
        await genome_controller.import_lastfm()
    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any("genome/import_lastfm" in r.message for r in error_records)
    assert any(r.exc_info is not None for r in error_records)


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
    genome_controller: GenomeController,
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
                    rows_added=0,
                    rows_duplicate=0,
                    rows_skipped=0,
                    warnings=[],
                )

        return _Importer()

    genome_controller._lastfm_importer_factory = _factory  # type: ignore[assignment]
    await genome_controller.import_lastfm()
    assert captured["api_key"] == key
