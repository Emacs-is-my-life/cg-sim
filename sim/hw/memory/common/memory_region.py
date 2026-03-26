from sortedcontainers import SortedDict

from sim.hw.common import DataRegion


class MemoryRegion(DataRegion):
    """
    MemoryRegion represents a continuous space in linear memory,
    composed of one or more pages.

    A memory region is: [page_idx_start, page_idx_end)
    So, region [0, 10) and [10, 20) are non-overlapping.
    """

    def __init__(self, page_idx_start: int, num_pages: int, tensor_id: int):
        super().__init__(tensor_id)

        self.page_idx_start = page_idx_start
        self.page_idx_end = page_idx_start + num_pages
        self.num_pages = num_pages
        return


class MemorySpace:
    """
    Tracks allocated memory regions, in a memory hardware.

    Invariants:
    - One MemoryRegion is owned by a Tensor
    - MemoryRegions are stored in this map, sorted by their page_idx_start
    - No two MemoryRegions can overlap
    """

    def __init__(self, num_total_pages: int):
        self.num_total_pages: int = num_total_pages
        self.num_used_pages: int = 0
        self.peak_num_used_pages: int = 0

        self._regions_by_page_idx_start: SortedDict[int, MemoryRegion] = SortedDict()
        return

    def _find_neighbors(self, page_idx_start: int) -> (MemoryRegion | None, MemoryRegion | None):
        """
        Returns: (prev_region, next_region) when,

        - [prev_region)
        -       <-------- page_idx_start here
        - [next_region)
        """

        idx = self._regions_by_page_idx_start.bisect_right(page_idx_start)
        prev_region = None
        next_region = None

        if idx - 1 >= 0 and len(self._regions_by_page_idx_start) > 0:
            prev_key = self._regions_by_page_idx_start.keys()[idx - 1]
            prev_region = self._regions_by_page_idx_start[prev_key]

        if idx < len(self._regions_by_page_idx_start):
            next_key = self._regions_by_page_idx_start.keys()[idx]
            next_region = self._regions_by_page_idx_start[next_key]

        return (prev_region, next_region)

    def check_avail(self, page_idx_start: int, num_pages: int) -> bool:
        """
        Check whether [page_idx_start, page_idx_start + num_pages) is claim-able
        """
        page_idx_end = page_idx_start + num_pages

        if page_idx_start < 0 or page_idx_start >= self.num_total_pages:
            return False
        if num_pages <= 0 or num_pages > self.num_total_pages:
            return False
        if page_idx_end > self.num_total_pages:
            return False

        prev_region, next_region = self._find_neighbors(page_idx_start)

        # Check overlap with the previous region
        if prev_region is not None and prev_region.end_page_idx > page_idx_start:
            return False

        # Check overlap with the next region
        if next_region is not None and next_region.start_page_idx < page_idx_end:
            return False

        return True

    def get_by_tensor_id(self, tensor_id: int) -> list[MemoryRegion]:
        """
        Find all MemoryRegion who holds tensor with tensor_id
        """

        regions: list[MemoryRegion] = []
        for mem_region in self._regions_by_page_idx_start.values():
            if mem_region.tensor_id == tensor_id:
                regions.append(mem_region)

        return regions

    def claim(self, tensor_id: int, page_idx_start: int, num_pages: int) -> MemoryRegion | None:
        """
        Try to allocate a new MemoryRegion and assign it to a Tensor
        - Success: returns MemoryRegion
        - Failure: returns None
        """

        if not self.check_avail(page_idx_start, num_pages):
            return None

        new_region = MemoryRegion(page_idx_start, num_pages, tensor_id)
        self._regions_by_start_page_idx[page_idx_start] = new_region
        self.num_used_pages += num_pages

        if self.num_used_pages > self.peak_num_used_pages:
            self.peak_num_used_pages = self.num_used_pages

        return new_region

    def release(self, free_region: MemoryRegion) -> None:
        """
        Release a MemoryRegion reserved for certain tensor,
        freeing space for other tensors
        """

        for mem_region in self._regions_by_page_idx_start.values():
            if mem_region.id == free_region.id:
                del mem_region
                self.num_used_pages -= mem_region.num_pages
                break

        return
