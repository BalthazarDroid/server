"""
Tests for the persisted job tracker (``controllers/genome/jobs.py``) and its controller wiring.

Every test here exists because of the same pair of failures: a long import reported its result
only as the return value of the websocket request that started it, so navigating away lost the
result forever, and a stall left a progress bar that never finished with nothing logged.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
import types
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from music_assistant.controllers.genome.jobs import (
    ALL_JOBS,
    JOB_APPLE_IMPORT,
    JOB_EXPORT_DB,
    JOB_LASTFM_IMPORT,
    JobState,
    JobTracker,
)
from music_assistant.controllers.genome.store import GenomeStore
from tests.controllers.genome.conftest import StubGenomeStore

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from music_assistant.controllers.genome.controller import GenomeController


@pytest.fixture
def tracker(genome_store: StubGenomeStore) -> JobTracker:
    """Build a tracker over the in-memory store stub."""
    return JobTracker(genome_store)


# ---------------------------------------------------------------------------------------
# JobState — the wire form is a frontend contract, so its shape is asserted directly
# ---------------------------------------------------------------------------------------


def test_job_state_round_trips_through_its_dict_form() -> None:
    """The dict form is both the wire format and the stored format; it must lose nothing."""
    state = JobState(
        job=JOB_LASTFM_IMPORT,
        state="ok",
        message="Last.fm import finished: 97 imported",
        started_at=1_700_000_000,
        finished_at=1_700_000_060,
        progress=100,
    )
    assert JobState.from_dict(state.to_dict()) == state


def test_job_state_survives_a_malformed_stored_row() -> None:
    """
    A corrupt status row must never be able to stop the controller from starting.

    It is status, not data: degrading to "idle" costs the user one forgotten result, while
    raising here would cost them the whole Genome controller.
    """
    state = JobState.from_dict(
        {"job": JOB_EXPORT_DB, "state": "who knows", "started_at": "not a number"}
    )
    assert state.state == "idle"
    assert state.started_at is None


# ---------------------------------------------------------------------------------------
# Persistence — the reason this module exists at all
# ---------------------------------------------------------------------------------------


async def test_every_known_job_has_a_state_before_anything_has_run(tracker: JobTracker) -> None:
    """The frontend renders all three jobs unconditionally, so all three must always be there."""
    assert set(tracker.get_all()) == set(ALL_JOBS)
    assert all(state["state"] == "idle" for state in tracker.get_all().values())


async def test_finished_job_state_survives_a_tracker_reload(
    genome_store: StubGenomeStore,
) -> None:
    """
    The result of a four-minute import has to outlive the page that asked for it.

    This is the navigation bug in miniature: a fresh tracker over the same store stands in for
    the user coming back to a newly-mounted component, or to a restarted server.
    """
    first = JobTracker(genome_store)
    await first.start(JOB_LASTFM_IMPORT, "Importing…")
    await first.finish(JOB_LASTFM_IMPORT, "Last.fm import finished: 97 imported, 20 skipped")

    second = JobTracker(genome_store)
    await second.load()

    job = second.get_all()[JOB_LASTFM_IMPORT]
    assert job["state"] == "ok"
    assert job["message"] == "Last.fm import finished: 97 imported, 20 skipped"
    assert job["finished_at"] is not None


async def test_job_left_running_by_a_dead_process_becomes_an_error_on_load(
    genome_store: StubGenomeStore,
) -> None:
    """
    This is the exact bug the tracker was built to fix.

    A job still marked "running" at startup was owned by a process that no longer exists;
    nothing will ever finish it. Left as-is it renders as a progress bar that never completes,
    forever. It has to become an honest error that says the run was interrupted.
    """
    dying = JobTracker(genome_store)
    await dying.start(JOB_APPLE_IMPORT, "Importing Apple Music history…")
    assert genome_store.jobs[JOB_APPLE_IMPORT]["state"] == "running"

    restarted = JobTracker(genome_store)
    await restarted.load()

    job = restarted.get_all()[JOB_APPLE_IMPORT]
    assert job["state"] == "error"
    assert "interrupted" in job["message"].lower()
    assert job["finished_at"] is not None
    # and the downgrade is itself persisted, so a second restart does not have to redo it
    assert genome_store.jobs[JOB_APPLE_IMPORT]["state"] == "error"


async def test_load_leaves_a_finished_job_alone(genome_store: StubGenomeStore) -> None:
    """Only "running" is stale. Downgrading a completed result would destroy the thing kept."""
    first = JobTracker(genome_store)
    await first.finish(JOB_EXPORT_DB, "Exported 1200 listens to /share/genome-export.db")

    second = JobTracker(genome_store)
    await second.load()

    assert second.get_all()[JOB_EXPORT_DB]["state"] == "ok"


async def test_update_amends_a_running_job_without_restarting_it(tracker: JobTracker) -> None:
    """Progress reporting must not move ``started_at``, or elapsed time reads as zero forever."""
    await tracker.start(JOB_LASTFM_IMPORT, "Importing…")
    started_at = tracker.get_all()[JOB_LASTFM_IMPORT]["started_at"]

    await tracker.update(JOB_LASTFM_IMPORT, message="Page 4 of 12", progress=33)

    job = tracker.get_all()[JOB_LASTFM_IMPORT]
    assert job["started_at"] == started_at
    assert job["state"] == "running"
    assert job["message"] == "Page 4 of 12"
    assert job["progress"] == 33


async def test_starting_a_job_clears_the_previous_runs_outcome(tracker: JobTracker) -> None:
    """A stale "ok" shown next to a run in flight is worse than no message at all."""
    await tracker.finish(JOB_EXPORT_DB, "Exported 1200 listens")
    await tracker.start(JOB_EXPORT_DB, "Exporting the Genome database…")

    job = tracker.get_all()[JOB_EXPORT_DB]
    assert job["state"] == "running"
    assert job["finished_at"] is None
    assert "1200" not in job["message"]


async def test_reads_never_touch_the_store(genome_store: StubGenomeStore) -> None:
    """
    ``genome/jobs`` is polled by an open page, so it must be a pure in-memory read (D-16).

    A read that fell through to the database would put a query on the path of every poll tick.
    """
    tracker = JobTracker(genome_store)
    await tracker.start(JOB_LASTFM_IMPORT, "Importing…")

    async def _explode() -> dict:
        raise AssertionError("genome/jobs must not read the database")

    genome_store.get_jobs = _explode  # type: ignore[method-assign]
    assert tracker.get_all()[JOB_LASTFM_IMPORT]["state"] == "running"


async def test_a_store_that_cannot_persist_does_not_fail_the_job(
    genome_store: StubGenomeStore,
) -> None:
    """Losing the status of an import is bad; failing the import over it is worse."""
    tracker = JobTracker(genome_store)

    async def _explode(_data: dict) -> None:
        msg = "disk full"
        raise RuntimeError(msg)

    genome_store.set_jobs = _explode  # type: ignore[method-assign]
    await tracker.finish(JOB_EXPORT_DB, "Exported 1200 listens")
    assert tracker.get_all()[JOB_EXPORT_DB]["state"] == "ok"


# ---------------------------------------------------------------------------------------
# GenomeStore.set_jobs / get_jobs — the real json settings row, not the stub
# ---------------------------------------------------------------------------------------


async def test_store_round_trips_the_job_map(tmp_path: Path) -> None:
    """
    The persistence has to work against the real settings table, not just the test stub.

    A tracker that only survives a reload in-process would still lose everything on the server
    restart that half of this feature is about.
    """
    mass = types.SimpleNamespace(storage_path=str(tmp_path), players=None)
    store = GenomeStore(mass)
    await store.setup()
    try:
        assert await store.get_jobs() == {}
        payload = {JOB_EXPORT_DB: JobState(job=JOB_EXPORT_DB, state="ok", message="x").to_dict()}
        await store.set_jobs(payload)
        assert await store.get_jobs() == payload
    finally:
        await store.close()


async def test_store_job_map_survives_reopening_the_database(tmp_path: Path) -> None:
    """The server restart case, end to end: a closed and reopened genome.db still knows."""
    mass = types.SimpleNamespace(storage_path=str(tmp_path), players=None)
    store = GenomeStore(mass)
    await store.setup()
    tracker = JobTracker(store)
    await tracker.start(JOB_LASTFM_IMPORT, "Importing…")
    await store.close()

    reopened = GenomeStore(mass)
    await reopened.setup()
    try:
        restarted = JobTracker(reopened)
        await restarted.load()
        assert restarted.get_all()[JOB_LASTFM_IMPORT]["state"] == "error"
    finally:
        await reopened.close()


# ---------------------------------------------------------------------------------------
# Controller wiring — genome/jobs, and the two commands that became background jobs
# ---------------------------------------------------------------------------------------


async def test_genome_jobs_returns_every_job(genome_controller: GenomeController) -> None:
    """The frontend asks once and renders three rows; a missing key is a crash in the page."""
    jobs = await genome_controller.get_jobs()
    assert set(jobs) == {JOB_LASTFM_IMPORT, JOB_APPLE_IMPORT, JOB_EXPORT_DB}
    for job_id, job in jobs.items():
        assert job["job"] == job_id
        assert set(job) == {
            "job",
            "state",
            "message",
            "started_at",
            "finished_at",
            "progress",
        }


async def test_export_db_returns_immediately_instead_of_awaiting_the_copy(
    genome_controller: GenomeController, mass_stub: MagicMock, tmp_path: Path
) -> None:
    """
    A snapshot of a real database takes time; the websocket request must not be what waits.

    The copy is deliberately made to block here, so a request that secretly awaited it could
    not pass this test.
    """
    source = tmp_path / "genome.db"
    sqlite3.connect(source).close()
    genome_controller.store.db_path = str(source)  # type: ignore[misc]
    release = asyncio.Event()
    started = asyncio.Event()

    async def _slow_export(_directory: str = "") -> dict:
        started.set()
        await release.wait()
        return {"path": str(tmp_path / "out.db"), "bytes": 10, "listens": 7}

    genome_controller._do_export_db = _slow_export  # type: ignore[method-assign]

    state = await genome_controller.export_db(directory=str(tmp_path))

    assert state["job"] == JOB_EXPORT_DB
    assert state["state"] == "running"
    await started.wait()
    assert not mass_stub.created_tasks[-1].done()

    release.set()
    await mass_stub.created_tasks[-1]
    job = (await genome_controller.get_jobs())[JOB_EXPORT_DB]
    assert job["state"] == "ok"
    # the path and the listen count are the whole point of the message: a backup nobody can
    # find is not a backup
    assert str(tmp_path / "out.db") in job["message"]
    assert "7 listens" in job["message"]


async def test_export_db_records_why_it_failed(
    genome_controller: GenomeController, mass_stub: MagicMock, tmp_path: Path
) -> None:
    """A background failure the user never sees raised must at least leave the reason behind."""
    genome_controller.store.db_path = str(tmp_path / "missing.db")  # type: ignore[misc]

    await genome_controller.export_db(directory=str(tmp_path))
    await mass_stub.created_tasks[-1]

    job = (await genome_controller.get_jobs())[JOB_EXPORT_DB]
    assert job["state"] == "error"
    assert "No genome database" in job["message"]


async def test_import_lastfm_records_a_missing_credential_before_raising(
    genome_controller: GenomeController,
) -> None:
    """
    The caller sees the error now; a page opened later still needs to learn why nothing ran.

    Without this the job would sit at whatever it last said, which is indistinguishable from
    the import having simply not been started.
    """
    with pytest.raises(Exception, match="not configured"):
        await genome_controller.import_lastfm()

    job = (await genome_controller.get_jobs())[JOB_LASTFM_IMPORT]
    assert job["state"] == "error"
    assert "not configured" in job["message"]


async def test_apple_import_result_survives_navigation(
    genome_controller: GenomeController, tmp_path: Path
) -> None:
    """
    The Apple ingest of a ~300k-row export is the longest of the three and the easiest to lose.

    Its chunk upload stays in the request (its progress is client-side), but the ingest result
    has to be written down or a user who switched tabs never learns whether it worked.
    """
    csv_path = tmp_path / "history.csv"
    csv_path.write_text("Song Name,Artist Name\n")
    genome_controller.get_config_value = (  # type: ignore[method-assign]
        lambda key, default=None, *, return_type=None: (  # noqa: ARG005
            str(tmp_path) if key == "apple_import_dir" else default
        )
    )

    async def _parser(path: str, *, min_seconds: int, stats=None):  # noqa: ARG001
        if stats is not None:
            stats.rows_read = 2
            stats.rows_skipped = 1
            stats.warnings.append("row 2: missing artist name")
        yield _apple_listen()

    genome_controller._apple_parser = _parser  # type: ignore[assignment]

    result = await genome_controller.import_apple("", 0, "", final=True, filename="history.csv")

    # the command's own return value is unchanged - the current frontend still reads it
    assert result["rows_read"] == 2
    job = (await genome_controller.get_jobs())[JOB_APPLE_IMPORT]
    assert job["state"] == "ok"
    assert "1 imported" in job["message"]
    assert "missing artist name" in job["message"]


async def test_apple_import_failure_is_recorded_too(
    genome_controller: GenomeController, tmp_path: Path
) -> None:
    """A failed ingest leaves the reason behind for the same reason a successful one does."""
    csv_path = tmp_path / "history.csv"
    csv_path.write_text("Song Name,Artist Name\n")
    genome_controller.get_config_value = (  # type: ignore[method-assign]
        lambda key, default=None, *, return_type=None: (  # noqa: ARG005
            str(tmp_path) if key == "apple_import_dir" else default
        )
    )

    async def _parser(path: str, *, min_seconds: int, stats=None):  # noqa: ARG001
        msg = "line 40120: unterminated quoted field"
        raise ValueError(msg)
        yield  # pragma: no cover - unreachable, keeps this an async generator

    genome_controller._apple_parser = _parser  # type: ignore[assignment]

    with pytest.raises(ValueError, match="unterminated"):
        await genome_controller.import_apple("", 0, "", final=True, filename="history.csv")

    job = (await genome_controller.get_jobs())[JOB_APPLE_IMPORT]
    assert job["state"] == "error"
    assert "unterminated quoted field" in job["message"]


def _apple_listen():
    """Build one minimal Apple-sourced listen for the ingest tests."""
    from music_assistant.controllers.genome.models import Listen  # noqa: PLC0415

    return Listen(
        played_at=int(time.time()),
        artist_key="artist",
        artist_name="Artist",
        track_key="track",
        track_name="Track",
        album_name=None,
        source="apple_export",
        player_id=None,
        duration_ms=None,
        played_ms=None,
        fully_played=True,
        confidence=1.0,
    )
