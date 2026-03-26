from sim.core.trace import Trace
from sim.hw.common import BaseHardware


class System:
    """
    System represents REAL stuffs in the simulation.

    - Trace
    - Hardwares

    and API for scheduler

    - compute()
    - claim()
    - find()
    - release()
    - transfer()

    """

    def __init__(self, trace: Trace, hw: dict[str, BaseHardware]):
        self.trace: Trace = trace
        self.hw: dict[str, BaseHardware] = hw
        return

    # TODO
    def compute() -> None:
        return

    def claim() -> None:
        return

    def find() -> None:
        return

    def release() -> None:
        return

    def transfer() -> None:
        return
