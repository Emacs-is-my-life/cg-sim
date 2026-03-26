from sim.core.trace import Trace
from sim.hw.common import BaseHardware


class System:
    """
    System represents REAL stuffs in the simulation.
    - Trace
    - Hardwares
    """

    def __init__(self, trace: Trace, hw: dict[str, BaseHardware]):
        self.trace = trace
        self.hw = hw
        return
