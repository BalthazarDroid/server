"""
Persisted job tracker for the Listening Genome's long-running operations.

Three operations — the Last.fm import, the Apple Music export ingest and the ``genome.db``
snapshot — can run for minutes. Before this module they reported their outcome only as the
return value of the websocket request that started them, which produced two failures a user
actually hit:

* Navigating away lost the result forever. The Vue component that issued the request unmounts,
  the reply lands nowhere, and the import that just ran for four minutes is indistinguishable
  from one that never happened.
* A stalled request left a progress bar that never finished and nothing in the server log,
  because the only record of the operation was the pending future.

So the outcome is written down instead of returned: :class:`JobTracker` keeps the last known
state of every job, persists it through the store (so it survives both navigation and a server
restart), and serves it from memory so a read never touches the database.

The stale-job rule in :meth:`JobTracker.load` is the other half of that fix. A job still marked
``running`` when the process starts belongs to a process that no longer exists — nothing will
ever finish it — so it is converted to ``error`` at load time rather than left running forever.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from music_assistant.controllers.genome.constants import LOGGER

if TYPE_CHECKING:
    from collections.abc import Mapping

# job ids. These are the keys of the `genome/jobs` payload and are part of the frontend
# contract, so they are spelled out here rather than derived from anything.
JOB_LASTFM_IMPORT = "lastfm_import"
JOB_APPLE_IMPORT = "apple_import"
JOB_EXPORT_DB = "export_db"

#: every job the tracker knows about, in the order the UI shows them
ALL_JOBS: tuple[str, ...] = (JOB_LASTFM_IMPORT, JOB_APPLE_IMPORT, JOB_EXPORT_DB)

JOB_STATE_IDLE = "idle"
JOB_STATE_RUNNING = "running"
JOB_STATE_OK = "ok"
JOB_STATE_ERROR = "error"

_VALID_STATES = frozenset({JOB_STATE_IDLE, JOB_STATE_RUNNING, JOB_STATE_OK, JOB_STATE_ERROR})

# what a job left `running` by a dead process is told when this process loads it. Phrased for a
# user reading it in the UI: it says what happened, not what the code noticed.
INTERRUPTED_MESSAGE = (
    "Interrupted: the server restarted while this was running, so it never finished. "
    "Start it again."
)


@dataclass(frozen=True)
class JobState:
    """
    The last known state of one long-running Genome operation.

    Frozen because every transition goes through :class:`JobTracker`, which persists on each
    change: an in-place mutation somewhere else would be a state the database never sees.

    :param job: The job id (one of :data:`ALL_JOBS`).
    :param state: ``"idle"``, ``"running"``, ``"ok"`` or ``"error"``.
    :param message: The human-readable outcome — the same text a person needs, not a code.
    :param started_at: Unix timestamp of the last start, or ``None`` if never started.
    :param finished_at: Unix timestamp of the last finish, or ``None`` while running/never run.
    :param progress: Percentage 0-100, or ``None`` meaning "indeterminate" (most of these jobs
        cannot know their own total until they are done).
    """

    job: str
    state: str = JOB_STATE_IDLE
    message: str = ""
    started_at: int | None = None
    finished_at: int | None = None
    progress: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe wire form served by ``genome/jobs`` and written to the store."""
        return {
            "job": self.job,
            "state": self.state,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobState:
        """
        Rebuild a :class:`JobState` from its stored form, defensively.

        Anything unreadable (a truncated write, a shape from an older version) degrades to a
        plain idle job rather than raising: a corrupt status row must not stop the controller
        from starting up.

        :param data: A mapping as produced by :meth:`to_dict`.
        """
        job = str(data.get("job") or "")
        state = str(data.get("state") or JOB_STATE_IDLE)
        if state not in _VALID_STATES:
            state = JOB_STATE_IDLE
        return cls(
            job=job,
            state=state,
            message=str(data.get("message") or ""),
            started_at=_int_or_none(data.get("started_at")),
            finished_at=_int_or_none(data.get("finished_at")),
            progress=_int_or_none(data.get("progress")),
        )


def _int_or_none(value: Any) -> int | None:
    """Coerce a stored value to ``int``, or ``None`` when it is absent or not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except TypeError, ValueError:  # pragma: no cover - defensive, malformed settings row
        return None


class JobTracker:
    """
    Remembers what the long-running Genome operations did, across navigation and restarts.

    Every job in :data:`ALL_JOBS` always has a state, so the frontend can render all three
    without special-casing "never run". Reads come from the in-memory copy and never touch the
    database; writes update memory first and then persist through the store.
    """

    def __init__(self, store: Any) -> None:
        """
        Initialize the tracker with every known job idle.

        :param store: A ``GenomeStore``-shaped object exposing ``get_jobs``/``set_jobs``.
        """
        self._store = store
        self._jobs: dict[str, JobState] = {job: JobState(job=job) for job in ALL_JOBS}

    def get_all(self) -> dict[str, dict[str, Any]]:
        """Return every job's state in wire form. Pure and instant — never touches the DB (D-16)."""
        return {job: state.to_dict() for job, state in self._jobs.items()}

    def get(self, job: str) -> dict[str, Any]:
        """Return one job's state in wire form, creating an idle entry for an unknown id."""
        return self._jobs.setdefault(job, JobState(job=job)).to_dict()

    async def load(self) -> None:
        """
        Restore persisted job state, downgrading anything a dead process left ``running``.

        A ``running`` job read at startup was owned by the process that just went away: nothing
        is going to finish it, and leaving it running is the exact bug this tracker exists to
        fix (a progress bar that never completes). It becomes an ``error`` saying so.
        """
        try:
            stored = await self._store.get_jobs()
        except Exception:  # pragma: no cover - defensive, a broken store must not block setup
            LOGGER.warning("Could not load Genome job state; starting from idle", exc_info=True)
            return
        if not isinstance(stored, dict):  # pragma: no cover - defensive, malformed settings row
            return
        interrupted: list[str] = []
        for job, raw in stored.items():
            if not isinstance(raw, dict):  # pragma: no cover - defensive
                continue
            state = JobState.from_dict({**raw, "job": job})
            if state.state == JOB_STATE_RUNNING:
                state = replace(
                    state,
                    state=JOB_STATE_ERROR,
                    message=INTERRUPTED_MESSAGE,
                    finished_at=int(time.time()),
                    progress=None,
                )
                interrupted.append(job)
            self._jobs[job] = state
        for job in interrupted:
            LOGGER.warning(
                "Genome job %r was still running at shutdown: %s", job, INTERRUPTED_MESSAGE
            )
        if interrupted:
            await self._persist()

    async def start(self, job: str, message: str = "") -> dict[str, Any]:
        """
        Mark ``job`` as running now, clearing the previous run's outcome.

        :param job: The job id.
        :param message: What is happening, in the words a user would read.
        """
        self._jobs[job] = JobState(
            job=job,
            state=JOB_STATE_RUNNING,
            message=message,
            started_at=int(time.time()),
            finished_at=None,
            progress=None,
        )
        await self._persist()
        return self._jobs[job].to_dict()

    async def update(
        self, job: str, *, message: str | None = None, progress: int | None = None
    ) -> dict[str, Any]:
        """
        Amend a running job's message and/or progress, leaving its timestamps alone.

        :param job: The job id.
        :param message: Replacement message; ``None`` keeps the current one.
        :param progress: Replacement percentage (clamped to 0-100); ``None`` keeps the current
            value, which for most of these jobs is "indeterminate".
        """
        current = self._jobs.setdefault(job, JobState(job=job))
        self._jobs[job] = replace(
            current,
            message=current.message if message is None else message,
            progress=current.progress if progress is None else max(0, min(100, int(progress))),
        )
        await self._persist()
        return self._jobs[job].to_dict()

    async def finish(self, job: str, message: str) -> dict[str, Any]:
        """
        Record ``job`` as having succeeded, with the outcome a person needs to read.

        :param job: The job id.
        :param message: The result — rows imported, the export path, the listen count. This is
            what survives navigation, so it has to be the whole answer, not "done".
        """
        return await self._settle(job, JOB_STATE_OK, message)

    async def fail(self, job: str, message: str) -> dict[str, Any]:
        """
        Record ``job`` as having failed, with the actual reason.

        :param job: The job id.
        :param message: Why it failed, in terms the user can act on.
        """
        return await self._settle(job, JOB_STATE_ERROR, message)

    async def _settle(self, job: str, state: str, message: str) -> dict[str, Any]:
        """Apply a terminal state to ``job`` and persist it."""
        current = self._jobs.setdefault(job, JobState(job=job))
        self._jobs[job] = replace(
            current,
            state=state,
            message=message,
            finished_at=int(time.time()),
            progress=100 if state == JOB_STATE_OK else None,
        )
        await self._persist()
        return self._jobs[job].to_dict()

    async def _persist(self) -> None:
        """
        Write the whole job map through the store.

        Failures here are logged and swallowed on purpose: losing the status of an import is
        bad, but failing the import itself because its status could not be written is worse.
        """
        try:
            await self._store.set_jobs(self.get_all())
        except Exception:  # pragma: no cover - defensive, a broken store must not fail the job
            LOGGER.warning("Could not persist Genome job state", exc_info=True)
