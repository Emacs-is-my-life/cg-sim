import uuid
from typing import Any

from sim.core.log import Log
from sim.core.trace import Trace, Node, Tensor
from sim.core.engine import EngineSignal, Engine
from sim.hw.common import BaseHardware, DataRegion
from sim.hw.compute.common import BaseCompute
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage

from sim.core.job import ComputeJob, ClaimJob, ReleaseJob, TransferJob


class System:
    """
    System represents REAL stuffs in the simulation.

    - Trace
    - Hardwares

    and API for scheduler

    - compute()
    - claim()
    - find()     # This is not a job dispatch. Just a helper method.
    - release()
    - transfer()
    """

    def __init__(self, trace: Trace, hw: dict[str, BaseHardware]):
        self.engine: Engine | None = None
        self.trace: Trace = trace
        self.hw: dict[str, BaseHardware] = hw
        return

    def compute(self, hw: BaseCompute, node: Node) -> uuid.UUID:
        job = ComputeJob(hw, node)
        self.engine.submit(job)
        return job.id

    def claim(self, hw: BaseMemory | BaseStorage, tensor: Tensor, page_idx_start: int = -1) -> uuid.UUID:
        job = ClaimJob(hw, tensor, page_idx_start)
        self.engine.submit(job)
        return job.id

    def find(self, hw: BaseMemory | BaseStorage, tensor: Tensor | int) -> list[DataRegion]:
        tensor_id = tensor
        if isinstance(tensor, Tensor):
            tensor_id = tensor.id

        regions = hw.space.get_by_tensor_id(tensor_id)
        return regions

    def release(self, region: DataRegion) -> uuid.UUID:
        job = ReleaseJob(region)
        self.engine.submit(job)
        return job.id

    def transfer(self, batch: list[tuple[DataRegion, DataRegion]]) -> uuid.UUID:
        job = TransferJob(batch)
        self.engine.submit(job)
        return job.id

    # These are engine signal to control simulation run
    def end_stage(self) -> None:
        self.engine.signal(EngineSignal.END_STAGE)
        return

    def abort(self, args: dict[str, Any] | None = None) -> None:
        args = args if args is not None else {}

        self.engine._log_abort(args)
        self.engine.signal(EngineSignal.ABORT)
        return
