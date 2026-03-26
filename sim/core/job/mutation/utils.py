from sim.core import System
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage


def invalidate(sys: System, tensor_id: int) -> None:
    for hw in sys.hw.values():
        if isinstance(hw, BaseMemory) or isinstance(BaseStorage):
            # Invalidate
            candidates = hw.space.get_by_tensor_id(tensor_id)

            for data_region in candidates:
                data_region.is_latest = False
    return
