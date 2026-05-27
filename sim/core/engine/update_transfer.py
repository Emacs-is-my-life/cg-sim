from collections import defaultdict

from sim.core.job import TransferJob

from sim.hw.common.data_region import DataRegion
from sim.hw.common.base_hardware import BaseHardware

from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage


def update_transfer_jobs(transfer_jobs: list[TransferJob], timestamp_now: float) -> None:
    # Initial bandwidth allocations per job
    bandwidth = {job: 0.0 for job in transfer_jobs}
    active_transfers = set(transfer_jobs)
    hardwares = {hw for job in transfer_jobs for hw in job.running_on}
    hw_capacity = {hw: hw.max_work_rate() for hw in hardwares}

    def weight(job: TransferJob, hardware: BaseMemory | BaseStorage) -> int:
        return sum(1 for hw in job.running_on if hw is hardware)

    while active_transfers:
        # Current load per Hardware
        load = defaultdict(float)
        for job in transfer_jobs:
            for hw in set(job.running_on):
                load[hw] += weight(job, hw) * bandwidth[job]

        # Growth rate per hardware (sum of weights of all active jobs)
        growth = defaultdict(float)
        for job in active_transfers:
            for hw in set(job.running_on):
                growth[hw] += weight(job, hw)

        # Find the smallest possible increment
        delta = float("inf")
        bottlenecked = []

        for hw in hardwares:
            if growth[hw] == 0:
                continue

            remaining = hw_capacity[hw] - load[hw]
            if remaining < 1e-12:
                continue
            t = remaining / growth[hw]
            if t < delta - 1e-12:
                delta = t
                bottlenecked = [hw]
            elif abs(t - delta) <= 1e-12:
                bottlenecked.append(hw)

        if delta == float("inf"):
            break  # No transfer stream can grow more

        # Increase all active jobs
        for job in active_transfers:
            bandwidth[job] += delta

        # Freeze jobs that touch any saturated hardwares
        to_freeze = set()
        for hw in bottlenecked:
            for job in active_transfers:
                if hw in job.running_on:
                    to_freeze.add(job)

        active_transfers -= to_freeze

    # Update all transfer jobs
    for job in transfer_jobs:
        job.update_ETA(timestamp_now, bandwidth[job])

    return
