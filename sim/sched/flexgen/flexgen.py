from typing import Any

from sim.sched.common import BaseScheduler
from sim.core.log import Log
from sim.core import System
from sim.core.job import BaseJob, ComputeJob, ClaimJob, ReleaseJob, TransferJob
from sim.hw.compute.common import BaseCompute
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage


class FlexGen(BaseScheduler):
    """
    FlexGen scheduler, implementing:
    https://dl.acm.org/doi/abs/10.5555/3618408.3619696
    """

    def __init__(self, obj_id: int, name: str, log: Log, sys: System, args: dict[str, Any] | None = None):
        super().__init__(obj_id, name, log, sys)
        return

    def compile(self) -> None:
        """No compilation"""
        return

    def layout(self) -> None:
        pass

    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        pass
