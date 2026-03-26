from sim.core import System
from sim.core.job import TransferJob

from sim.hw.common import DataRegionAccess, DataRegion


def begin_mutation(job: TransferJob, sys: System) -> None:
    batch: list[(DataRegion, DataRegion)] = job.batch
    for src_region, dest_region in batch:
        # 0. Update Src Region
        src_region.access_status = DataRegionAccess.BEING_READ
        src_region.access_count += 1

        # 1. Update Dest Region
        dest_region.access_status = DataRegionAccess.BEING_WRITTEN
        dest_region.is_ready = False
        dest_region.is_latest = src_region.is_latest

    return


def end_mutation(job: TransferJob, sys: System) -> None:
    batch: list[(DataRegion, DataRegion)] = job.batch
    for src_region, dest_region in batch:
        # 0. Update Src Region
        src_region.access_count -= 1
        if src_region.access_count == 0:
            src_region.access_status = DataRegionAccess.IDLE

        # 1. Update Dest Region
        dest_region.access_status = DataRegionAccess.IDLE
        dest_region.is_ready = True

    return
