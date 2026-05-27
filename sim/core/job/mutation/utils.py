from __future__ import annotations

from typing import TYPE_CHECKING

from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage

if TYPE_CHECKING:
    from sim.core.system import System


def invalidate(sys: System, tensor_id: int) -> None:
    for hw in sys.hw.values():
        if isinstance(hw, (BaseMemory, BaseStorage)):
            candidates = hw.space.get_by_tensor_id(tensor_id)

            for data_region in candidates:
                # Mark these regions as stale copy of data
                data_region.is_latest = False
    return
