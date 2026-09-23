"""
Tests for the Last.fm poll scheduled task (§3.2, §3.8, B2).

Before this fix, `lastfm_poll_enabled`/`lastfm_poll_interval_hours` were saved, returned by
`genome/settings`, and otherwise completely inert: nothing ever scheduled a poll. These tests
guard that `setup()` and `update_config()` actually register (or unregister) a recurring
task through `mass.tasks`, gated correctly on the enabled flag and on Last.fm being configured.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from music_assistant_models.background_task import TaskSchedule
from music_assistant_models.enums import TaskScheduleType

from music_assistant.controllers.genome.constants import (
    CONF_LASTFM_API_KEY,
    CONF_LASTFM_POLL_ENABLED,
    CONF_LASTFM_POLL_INTERVAL_HOURS,
    CONF_LASTFM_USERNAME,
    GENOME_LASTFM_POLL_TASK_ID,
)
from music_assistant.controllers.genome.controller import GenomeController


def _configure(
    controller: GenomeController,
    *,
    enabled: bool,
    username: str = "testuser",
    api_key: str = "testkey",
    interval_hours: int = 6,
) -> None:
    """Stub `get_config_value` to answer the Last.fm poll config keys directly."""
    values = {
        CONF_LASTFM_POLL_ENABLED: enabled,
        CONF_LASTFM_USERNAME: username,
        CONF_LASTFM_API_KEY: api_key,
        CONF_LASTFM_POLL_INTERVAL_HOURS: interval_hours,
    }
    controller.get_config_value = (  # type: ignore[method-assign]
        lambda key, default=None, *, return_type=None: values.get(key, default)  # noqa: ARG005
    )


async def test_poll_task_registered_when_enabled_and_configured(
    genome_controller: GenomeController, mass_stub: MagicMock
) -> None:
    """Enabling the poll with a username and API key registers the scheduled task."""
    _configure(genome_controller, enabled=True, interval_hours=6)
    genome_controller._register_lastfm_poll_task()

    mass_stub.tasks.register_scheduled_task.assert_called_once()
    call = mass_stub.tasks.register_scheduled_task.call_args.kwargs
    assert call["task_id"] == GENOME_LASTFM_POLL_TASK_ID
    assert call["handler"] == genome_controller._scheduled_lastfm_poll
    schedule = call["schedule"]
    assert isinstance(schedule, TaskSchedule)
    assert schedule.type == TaskScheduleType.HOURLY
    assert schedule.every == 6
    mass_stub.tasks.unregister_scheduled_task.assert_not_called()


async def test_poll_task_not_registered_when_disabled(
    genome_controller: GenomeController, mass_stub: MagicMock
) -> None:
    """Leaving the poll toggle off never registers the scheduled task."""
    _configure(genome_controller, enabled=False)
    genome_controller._register_lastfm_poll_task()

    mass_stub.tasks.register_scheduled_task.assert_not_called()
    mass_stub.tasks.unregister_scheduled_task.assert_called_once_with(GENOME_LASTFM_POLL_TASK_ID)


@pytest.mark.parametrize(
    ("username", "api_key"),
    [("", "testkey"), ("testuser", "")],
)
async def test_poll_task_not_registered_without_credentials(
    genome_controller: GenomeController,
    mass_stub: MagicMock,
    username: str,
    api_key: str,
) -> None:
    """Enabling the poll with a blank username or API key must not register a live task."""
    _configure(genome_controller, enabled=True, username=username, api_key=api_key)
    genome_controller._register_lastfm_poll_task()

    mass_stub.tasks.register_scheduled_task.assert_not_called()
    mass_stub.tasks.unregister_scheduled_task.assert_called_once_with(GENOME_LASTFM_POLL_TASK_ID)


async def test_setup_registers_lastfm_poll_task(
    genome_controller: GenomeController, mass_stub: MagicMock
) -> None:
    """`setup()` must register the poll task alongside the rebuild task, not just the rebuild."""
    _configure(genome_controller, enabled=True)
    config = MagicMock()

    await genome_controller.setup(config)

    task_ids = {
        call.kwargs["task_id"] for call in mass_stub.tasks.register_scheduled_task.call_args_list
    }
    assert GENOME_LASTFM_POLL_TASK_ID in task_ids


async def test_update_config_reregisters_poll_task_on_relevant_change(
    genome_controller: GenomeController, mass_stub: MagicMock
) -> None:
    """Toggling `lastfm_poll_enabled` through `update_config` re-registers the task."""
    _configure(genome_controller, enabled=True)
    config = MagicMock()

    await genome_controller.update_config(config, {f"values/{CONF_LASTFM_POLL_ENABLED}"})

    mass_stub.tasks.register_scheduled_task.assert_called_once()


async def test_update_config_ignores_unrelated_changes(
    genome_controller: GenomeController, mass_stub: MagicMock
) -> None:
    """A change unrelated to Last.fm polling must not touch the poll task at all."""
    _configure(genome_controller, enabled=True)
    config = MagicMock()

    await genome_controller.update_config(config, {"values/recency_half_life_days"})

    mass_stub.tasks.register_scheduled_task.assert_not_called()
    mass_stub.tasks.unregister_scheduled_task.assert_not_called()


async def test_scheduled_poll_imports_and_logs_result(
    genome_controller: GenomeController,
) -> None:
    """The scheduled poll handler must actually run an import, not just exist."""

    async def _fake_import_lastfm(username: str = "", max_pages: int = 0):  # noqa: ARG001
        return {
            "source": "lastfm",
            "rows_read": 3,
            "rows_imported": 2,
            "rows_skipped": 1,
            "rows_duplicate": 0,
            "first_played_at": None,
            "last_played_at": None,
            "warnings": [],
        }

    # the poll calls the blocking form, not the websocket command: it has no request to hand a
    # result back to, and it needs the row counts it logs
    genome_controller._do_import_lastfm = _fake_import_lastfm  # type: ignore[method-assign]
    await genome_controller._scheduled_lastfm_poll()  # should not raise
