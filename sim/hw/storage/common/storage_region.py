from __future__ import annotations

from typing import TYPE_CHECKING

from sim.hw.common.data_region import DataRegion

if TYPE_CHECKING:
    from .base_storage import BaseStorage


class StorageRegion(DataRegion):
    """
    StorageRegion is a continuous space in Storage, holding a Tensor

    Let's say storage has infinite size
    """
    def __init__(self, hw: BaseStorage, num_pages: int, tensor_id: int):
        super().__init__(hw, num_pages, tensor_id)
        return


class StorageSpace:
    """
    Has many StorageRegions
    """

    def __init__(self, hw: BaseStorage):
        self.hw: BaseStorage = hw
        self._regions: list[StorageRegion] = []
        return

    def get_by_tensor_id(self, tensor_id: int) -> list[StorageRegion]:
        regions = []
        for stor_region in self._regions:
            if stor_region.tensor_id == tensor_id:
                regions.append(stor_region)

        return regions

    def check_avail(self) -> bool:
        """
        Check if StorageRegion is claim-able
        """
        return True

    def claim(self, tensor_id: int, page_idx_start: int, num_pages: int) -> StorageRegion | None:
        # Ignore page_idx_start

        new_region = StorageRegion(self.hw, num_pages, tensor_id)
        self._regions.append(new_region)
        return new_region

    def release(self, free_region: StorageRegion) -> None:
        self._regions = [stor_region for stor_region in self._regions if stor_region.id != free_region.id]
        return
