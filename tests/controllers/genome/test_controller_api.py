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
    CONF_RECENCY_HALF_LIFE_DAYS,
    LISTENER_HOUSEHOLD,
)
from music_assistant.controllers.genome.controller import GenomeController
from music_assistant.controllers.genome.models import Listen
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


async def _fake_apple_parser(path: str, *, min_seconds: int):  # noqa: ARG001
    # signature must match GenomeController._apple_parser exactly (positional `path`,
    # keyword-only `min_seconds`) - the fake never needs either value itself
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
    await genome_controller.set_settings({"half_life_days": 365})
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
    await genome_controller.set_settings({})
    assert called is False
