from sim.core.trace import Trace, Node, Tensor
from sim.core.engine import Engine
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
    - release()
    - transfer()
    """

    def __init__(self, trace: Trace, hw: dict[str, BaseHardware]):
        self.engine: Engine = None
        self.trace: Trace = trace
        self.hw: dict[str, BaseHardware] = hw
        return

    def compute(self, hw: BaseCompute, node: Node) -> None:
        job = ComputeJob(hw, node)
        self.engine.submit(job)
        return

    def claim(self, hw: BaseMemory | BaseStorage, tensor: Tensor, page_idx_start: int = -1) -> None:
        job = ClaimJob(hw, tensor, page_idx_start)
        self.engine.submit(job)
        return

    def find(self, hw: BaseMemory | BaseStorage, tensor: Tensor | int) -> list[DataRegion]:
        tensor_id = tensor
        if isinstance(tensor, Tensor):
            tensor_id = tensor.id

        regions = hw.space.get_by_tensor_id(tensor_id)
        return regions

    def release(self, region: DataRegion) -> None:
        job = ReleaseJob(region)
        self.engine.submit(job)
        return

    def transfer(self, batch: list[tuple[DataRegion, DataRegion]]) -> None:
        job = TransferJob(batch)
        self.engine.submit(job)
        return
