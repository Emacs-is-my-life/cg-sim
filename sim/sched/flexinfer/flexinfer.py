from __future__ import annotations

from enum import Enum, auto
from typing import Any, TYPE_CHECKING

from sim.sched.common import BaseScheduler
from sim.core.log import Log
from sim.core.trace import Trace
from sim.core.job import BaseJob, ComputeJob, ClaimJob, ReleaseJob, TransferJob
from sim.hw.compute.common import BaseCompute
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage

if TYPE_CHECKING:
    from sim.core.system import System


class FlexInferMode(Enum):
    MEMORY_SUFFICIENT = auto()
    MEMORY_INTERMEDIATE = auto()
    MEMROY_LIMITED = auto()


class FlexInfer(BaseScheduler):
    """
    Implementation of, FlexInfer: Breaking Memory Constraint via Flexible and Efficient Offloading for On-Device LLM Inference
    Published in EuroMLSys '25
    """

    def __init__(self, obj_id: int, name: str, log: Log, sys: System, args: dict[str, Any] | None = None):
        super().__init__(obj_id, name, log, sys, args)

        """
        Assumption: Expect one of each hardware:

        - One compute unit
        - One memory unit
        - One storage unit
        """
        for hw in sys.hw.values():
            if isinstance(hw, BaseCompute):
                self.compute = hw
            elif isinstance(hw, BaseMemory):
                self.memory = hw
            elif isinstance(hw, BaseStorage):
                self.storage = hw

        # Check tensors in sys.trace.tensor_map,
        # see if memory.space size is sufficient,
        # set self.mode: FlexInferOpMode = <MODE> accordingly.
        return

    def compile(self, trace: Trace) -> None:
        """No compilation"""
        return

    def layout(self, init_storage: BaseStorage) -> None:
        """
        Input: Attention tensor size 𝑠𝑖𝑧𝑒𝑎𝑡𝑡𝑒 , FFN tensor size
        𝑠𝑖𝑧𝑒𝐹 𝐹 𝑁 , Layer number 𝑁 , Memory budget 𝑠𝑖𝑧𝑒𝑚𝑒𝑚 ,
        Output: Tensor preservation plan 𝑃
        1: if 𝑠𝑖𝑧𝑒𝑚𝑒𝑚 >= 𝑠𝑖𝑧𝑒𝐹 𝐹 𝑁 ∗ 𝑁 ∗ 3 + 𝑠𝑖𝑧𝑒𝑎𝑡𝑡𝑛 ∗ 𝑁 ∗ 2 then
        2:     Set all FFN tensors for all layers
        3: else
        4:     if 𝑠𝑖𝑧𝑒𝑚𝑒𝑚 >= 𝑠𝑖𝑧𝑒𝐹 𝐹 𝑁 ∗ 𝑁 ∗ 2 then
        5:         Set two FFN tensor for all layers
        6:     else
        7:         if 𝑠𝑖𝑧𝑒𝑚𝑒𝑚 >= 𝑠𝑖𝑧𝑒𝐹 𝐹 𝑁 ∗ 𝑁 then
        8:             Set one FFN tensor for all layers
        9:         end if
        10:    end if
        11: end if
        12: Set as much as possible attention tensors one by one
        13: return 𝑃
        """

        """
        - Considering self.mode,
        - see tensor name and type,
        - load tensors from init_storage to memory
        - according to FlexInfer layout algorithm above
        """

        return

    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        """
        - Don't touch MemoryRegions for pinned tensors
        - Release MemoryRegions holding spent tensors (and will not be used soon)
        - Claim Memory Regions which will be used in future, as much as memory permits
        - Submit compute jobs, which can be run
        """

        return
