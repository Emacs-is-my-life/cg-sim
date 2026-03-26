from abc import abstractmethod

from sim.core import SimObject
from sim.core.log import Log, TrackID
from sim.core.engine import Engine
from sim.core.job import BaseJob


class WorkRate:
    def __init__(self):
        self.compute: float = 0    # AU / microsecond
        self.read_from: float = 0  # KB / microsecond
        self.write_to: float = 0   # KB / microsecond
        self.rw_total: float = 0   # KB / microsecond


class BaseHardware(SimObject):
    """Abstract base class for all hardwares in simulator"""
    def __init__(self, obj_id: int, name: str, log: Log):
        super().__init__(obj_id, name, log)
        self._engine: Engine | None = None   # Will be injected later, by engine

        self.job_running: list[BaseJob] = []
        self.max_rate: WorkRate = WorkRate()

        # Create logging tracks
        log.record(Log.subtrack(TrackID.Event, self.id, self.name))
        log.record(Log.subtrack(TrackID.Counter, self.id, self.name))
        log.record(Log.subtrack(TrackID.State, self.id, self.name))
        return

    def schedule(self, job: BaseJob):
        """Schedule a job in this hardware."""
        self.job_running.append(job)
        return

    def retire(self, retired_job: BaseJob):
        """Retire a job from this hardware."""
        self.job_running = [job for job in self.job_running if job.id != retired_job.id]
        return

    @abstractmethod
    def is_avail(self) -> bool:
        """Return if this hardware can accept a new job"""
        pass

    @abstractmethod
    def update_work_rate(self) -> None:
        """
        Based on current running jobs in self.job_running,
        update self.max_rate based on hardware characteristics
        """
        pass
