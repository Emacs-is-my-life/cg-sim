from __future__ import annotations

from typing import TYPE_CHECKING

from sortedcontainers import SortedDict

from sim.hw.common.data_region import DataRegion

if TYPE_CHECKING:
    from .base_memory import BaseMemory


class MemoryRegion(DataRegion):
    """
    MemoryRegion represents a continuous space in linear memory,
    composed of one or more pages.

    A memory region is: [page_idx_start, page_idx_end)
    So, region [0, 10) and [10, 20) are non-overlapping.
    """

    def __init__(self, hw: BaseMemory, page_idx_start: int, num_pages: int, tensor_id: int):
        super().__init__(hw, num_pages, tensor_id)

        self.page_idx_start = page_idx_start
        self.page_idx_end = page_idx_start + num_pages
        return


class MemorySpace:
    """
    Tracks allocated memory regions, in a memory hardware.

    Invariants:
    - One MemoryRegion is owned by a Tensor
    - MemoryRegions are stored in this map, sorted by their page_idx_start
    - No two MemoryRegions can overlap
    """

    def __init__(self, hw: BaseMemory, num_total_pages: int):
        self.hw: BaseMemory = hw
        self.num_total_pages: int = num_total_pages
        self.num_used_pages: int = 0
        self.peak_num_used_pages: int = 0

        self._regions_by_page_idx_start: SortedDict[int, MemoryRegion] = SortedDict()
        # Secondary index: tensor_id -> regions holding it. A region's
        # tensor_id is immutable, so membership only changes in claim()/
        # release() (the only mutators of the primary map). This turns
        # get_by_tensor_id from an O(all regions) linear scan into an
        # O(regions-for-this-tid) lookup — the dominant cost in offload
        # schedulers, which call it ~10x per node (input residency, output
        # claim, eviction, refcount release) across traces with thousands
        # of live regions (e.g. SD3: ~6k VRAM regions).
        self._regions_by_tid: dict[int, list[MemoryRegion]] = {}
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
        if prev_region is not None and prev_region.page_idx_end > page_idx_start:
            return False

        # Check overlap with the next region
        if next_region is not None and next_region.page_idx_start < page_idx_end:
            return False

        return True

    def get_by_tensor_id(self, tensor_id: int) -> list[MemoryRegion]:
        """
        Find all MemoryRegion who holds tensor with tensor_id.

        Returns them in page_idx_start order — identical to the previous
        linear scan over the page-sorted primary map — so callers that pick
        the first matching region (begin_mutation, _dispatch_transfer, ...)
        choose the same region as before; placement/peak stay deterministic.
        Per-tid region count is tiny (1-2), so the sort is negligible.
        """
        regions = self._regions_by_tid.get(tensor_id)
        if not regions:
            return []
        if len(regions) == 1:
            return list(regions)
        return sorted(regions, key=lambda r: r.page_idx_start)

    def claim(self, tensor_id: int, page_idx_start: int, num_pages: int) -> MemoryRegion | None:
        """
        Try to allocate a new MemoryRegion and assign it to a Tensor
        - Success: returns MemoryRegion
        - Failure: returns None
        """

        if not self.check_avail(page_idx_start, num_pages):
            return None

        new_region = MemoryRegion(self.hw, page_idx_start, num_pages, tensor_id)
        self._regions_by_page_idx_start[page_idx_start] = new_region
        self._regions_by_tid.setdefault(tensor_id, []).append(new_region)
        self.num_used_pages += num_pages

        if self.num_used_pages > self.peak_num_used_pages:
            self.peak_num_used_pages = self.num_used_pages

        return new_region

    def release(self, free_region: MemoryRegion) -> None:
        """
        Release a MemoryRegion reserved for certain tensor,
        freeing space for other tensors.

        Look the region up by its own (immutable) page_idx_start key
        rather than scanning the whole map for a matching id — equivalent,
        since keys are unique and non-overlapping, but O(log N) not O(N).
        The id guard keeps the original "no-op if not present" semantics.
        """
        key = free_region.page_idx_start
        existing = self._regions_by_page_idx_start.get(key)
        if existing is None or existing.id != free_region.id:
            return

        self.num_used_pages -= existing.num_pages
        del self._regions_by_page_idx_start[key]

        # Maintain the tensor_id index.
        lst = self._regions_by_tid.get(existing.tensor_id)
        if lst is not None:
            for i, r in enumerate(lst):
                if r.id == existing.id:
                    lst.pop(i)
                    break
            if not lst:
                del self._regions_by_tid[existing.tensor_id]

        return
