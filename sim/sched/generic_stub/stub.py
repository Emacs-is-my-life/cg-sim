from __future__ import annotations

from typing import Any, TYPE_CHECKING

from sim.sched.common import BaseScheduler
from sim.core.log import Log
from sim.core.trace import Trace
from sim.core.job import BaseJob
from sim.hw.storage.common import BaseStorage

if TYPE_CHECKING:
    from sim.core.system import System


class Stub(BaseScheduler):
    """
    No-op scheduler.

    Lets a simulation be spun up for configs whose intended scheduler is
    not available on this branch (e.g. DeviceAwareVanillaAsync), so the
    config can be loaded and the compile / layout stages can be reached
    for inspection. Runtime aborts — the Stub is not meant to drive an
    actual run.
    """

    def __init__(self, obj_id: int, name: str, log: Log, sys: System, args: dict[str, Any] | None = None):
        super().__init__(obj_id, name, log, sys, args)
        return

    def compile(self, trace: Trace) -> None:
        return

    def layout(self, init_storage: BaseStorage) -> bool:
        return True

    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        self.sys.abort({
            "from": self.name,
            "error": "STUB_NOT_RUNNABLE",
            "msg": "Stub scheduler is not intended for runtime execution; "
                   "use it only to spin up a simulation for inspection.",
        })
        return
