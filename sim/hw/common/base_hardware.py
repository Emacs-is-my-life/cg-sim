from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from sim.core.sim_object import SimObject
from sim.core.log import Log, TrackID

if TYPE_CHECKING:
    from sim.core.job.job import BaseJob


"""
class WorkRate:
    def __init__(self):
        self.compute: float = 0    # AU / microsecond
        self.read_from: float = 0  # KB / microsecond
        self.write_to: float = 0   # KB / microsecond
        self.rw_total: float = 0   # KB / microsecond
"""


class BaseHardware(SimObject):
    """Abstract base class for all hardwares in simulator"""
    def __init__(self, obj_id: int, name: str, log: Log):
        super().__init__(obj_id, name, log)

        self.job_running: list["BaseJob"] = []

        # Create logging tracks
        log.record(Log.subtrack(TrackID.Event, self.id, self.name))
        log.record(Log.subtrack(TrackID.Counter, self.id, self.name))
        log.record(Log.subtrack(TrackID.State, self.id, self.name))
        return

    def run(self, job: "BaseJob"):
        """Put a job in this hardware's running slot."""
        self.job_running.append(job)
        return

    def retire(self, retired_job: "BaseJob"):
        """Retire a job from this hardware."""
        self.job_running = [job for job in self.job_running if job.id != retired_job.id]
        return

    @abstractmethod
    def can_run(self, job: "BaseJob") -> bool:
        """Return if this hardware can accept this new job or not"""
        pass

    # TODO: Should we move onto constraints based optimization?
    @abstractmethod
    def max_work_rate(self) -> float:
        """
        Based on current running jobs in self.job_running,
        return the max work_rate this hardware can provide to them in total.
        """
        pass
