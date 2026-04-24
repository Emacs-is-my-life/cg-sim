import uuid
from typing import Any

from sim.core.log import Log
from sim.core.trace import Trace, Node, NodeStatus, Tensor
from sim.core.engine import EngineSignal, Engine
from sim.hw.common.base_hardware import BaseHardware
from sim.hw.common.data_region import DataRegion
from sim.hw.compute.common import BaseCompute
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage

from sim.core.job import ComputeJob, ClaimJob, ReleaseJob, TransferJob

from sim.core.job.assertion import claim_assertion, release_assertion
from sim.core.job.mutation import claim_mutation, release_mutation
from sim.core.job.logging import claim_log, release_log


class System:
    """
    System represents REAL stuffs in the simulation.

    - Trace
    - Hardwares

    and API for scheduler

    # Job: takes time to execute
    - compute()
    - transfer()

    # Immediate Actions: returns to Scheduler immediately
    - claim()
    - release()
    - find()
    """

    def __init__(self, trace: Trace, hw: dict[str, BaseHardware]):
        self.engine: Engine | None = None
        self.trace: Trace = trace
        self.hw: dict[str, BaseHardware] = hw
        return

    def compute(self, hw: BaseCompute, node: Node) -> uuid.UUID:
        node.status = NodeStatus.WAITING
        job = ComputeJob(hw, node)
        self.engine.submit(job)
        return job.id

    def claim(self, hw: BaseMemory | BaseStorage, tensor: Tensor, page_idx_start: int = -1) -> DataRegion | None:
        job = ClaimJob(hw, tensor, page_idx_start)
        if not claim_assertion(job, self):
            args = {
                "from": self.engine.name,
                "error": "Job Pre-Execution Assertion Failure",
                "job_type": "ClaimJob",
                "msg": f"Cannot claim a data region from {hw.name}."
            }
            self.abort(args)
            return None

        job.timestamp_begin = self.engine.timestamp_now
        claim_mutation(job, self)
        claim_log(job, self.engine.log)
        return job.region

    def find(self, hw: BaseMemory | BaseStorage, tensor: Tensor | int) -> list[DataRegion]:
        tensor_id = tensor
        if isinstance(tensor, Tensor):
            tensor_id = tensor.id

        regions = hw.space.get_by_tensor_id(tensor_id)
        return regions

    def release(self, region: DataRegion) -> None:
        job = ReleaseJob(region)
        if not release_assertion(job, self):
            args = {
                "from": self.engine.name,
                "error": "Job Pre-Execution Assertion Failure",
                "job_type": "ReleaseJob",
                "msg": f"Cannot release a data region from {region.hw.name}, access_status: {region.access_status.name}"
            }
            self.abort(args)
            return

        job.timestamp_begin = self.engine.timestamp_now
        release_mutation(job, self)
        release_log(job, self.engine.log)
        return

    def transfer(self, batch: list[tuple[DataRegion, DataRegion]]) -> uuid.UUID:
        job = TransferJob(batch)
        self.engine.submit(job)
        return job.id

    # These are engine signal to control simulation run
    def end_stage(self) -> None:
        self.engine.signal(EngineSignal.END_STAGE)
        return

    def abort(self, args: dict[str, Any] | None = None) -> None:
        self.engine.log.on = True
        print("[System] Abort called!")
        args = args if args is not None else {}

        if self.engine is not None:
            self.engine._log_abort(args)
            self.engine.signal(EngineSignal.ABORT)
        return
