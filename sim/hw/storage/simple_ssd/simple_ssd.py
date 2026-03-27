from typing import Any

from sim.core.log import Log
from sim.core.job import TransferJob
from sim.hw.common import DataRegion
from sim.hw.storage.common import BaseStorage


class SimpleSSD(BaseStorage):
    """
    Simple model of SSD.
    Has following tunable parameters:

    - fixed_latency_micros: latency overhead for accessing SSD no matter what, in microseconds.
    - read_io_curve_KBps: a list of (io_size_KB, bandwidth_KBps) tuples
    - write_io_curve_KBps: a list of (io_size_KB, bandwidth_KBps) tuples

    io_latency = fixed_latency + (io_size / io_bandwidth)
    io_bandwidth is dependent on io_size. Bigger io_size results in  higher io_bandwidth
    """

    def __init__(self, obj_id: int, name: str, log: Log, args: dict[str, Any]):
        super().__init__(obj_id, name, log)

        # Load fixed_latency_micros
        fixed_latency_micros = float(args["fixed_latency_micros"])
        if fixed_latency_micros < 0:
            raise ValueError(f"[Storage] Fixed latency cannot be: {fixed_latency_micros} micro-seconds")

        # Load io performance curve
        read_io_curve_KBps: list[tuple[float, float]] = [(float(io_size), float(bandwidth)) for io_size, bandwidth in args["read_io_curve_KBps"]]
        write_io_curve_KBps: list[tuple[float, float]] = [(float(io_size), float(bandwidth)) for io_size, bandwidth in args["write_io_curve_KBps"]]
        SimpleSSD._check_list_of_tuples(read_io_curve_KBps)
        SimpleSSD._check_list_of_tuples(write_io_curve_KBps)

        self.fixed_latency_micros = fixed_latency_micros
        self.read_io_curve_KBps = read_io_curve_KBps
        self.write_io_curve_KBps = write_io_curve_KBps
        return

    @staticmethod
    def _check_list_of_tuples(lst_tuple):
        for tup in lst_tuple:
            io_size, bandwidth = tup
            if io_size <= 0:
                raise ValueError(f"[Storage] IO size cannot be: {io_size} KB")
            if bandwidth <= 0:
                raise ValueError(f"[Storage] Bandwidth cannot be: {bandwidth} KB")

        return

    @staticmethod
    def _get_total_io_size_KB(job: TransferJob) -> float:
        """When storage job is given, get a sum of IO size from it's batch"""
        batch: list[(DataRegion, DataRegion)] = job.batch

        total_io_size_KB = float(0.0)
        for src_region, dest_region in batch:
            total_io_size_KB += 4 * src_region.num_pages

        return total_io_size_KB

    @staticmethod
    def _get_bandwidth_KBps(total_io_size_KB: float, io_curve_KBps: list[tuple[float, float]]) -> float:
        """When total io size, and io performance curve of SSD is given, returns the io bandwidth"""
        if not io_curve_KBps:
            raise ValueError("[Storage] IO characteristic curve is empty")

        # Clamp to boundaries
        if total_io_size_KB <= io_curve_KBps[0][0]:
            return io_curve_KBps[0][1]
        if total_io_size_KB >= io_curve_KBps[-1][0]:
            return io_curve_KBps[-1][1]

        # Find "bounding" points
        for (x0, y0), (x1, y1) in zip(io_curve_KBps, io_curve_KBps[1:]):
            if x0 <= total_io_size_KB <= x1:
                if x0 == x1:
                    return y0

                # Linear interpolation
                t = (total_io_size_KB - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)

        raise ValueError("[Storage] IO characteristic curve is not properly sorted!")

    def is_avail(self) -> bool:
        return len(self.job_running) == 0

    def update_work_rate(self) -> None:
        if self.job_running:
            job = self.job_running[0]
            total_io_size_KB = SimpleSSD._get_total_io_size_KB(job)
            batch: list[(DataRegion, DataRegion)] = job.batch
            src0, dest0 = batch[0]

            if src0.hw.id == self.id:     # Use Read IO curve
                io_bandwidth_KBps = SimpleSSD._get_bandwidth_KBps(total_io_size_KB, self.read_io_curve_KBps)
                self.max_rate.read_from = (io_bandwidth_KBps / 1_000_000)  # KB per microsecond
                self.max_rate.write_to = 0
            elif dest0.hw.id == self.id:  # Use Write IO curve
                io_bandwidth_KBps = SimpleSSD._get_bandwidth_KBps(total_io_size_KB, self.write_io_curve_KBps)
                self.max_rate.read_from = 0
                self.max_rate.write_to = (io_bandwidth_KBps / 1_000_000)   # KB per microsecond

        return
