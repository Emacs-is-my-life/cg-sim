import sys
from hydra import initialize, compose
from omegaconf import OmegaConf
from pathlib import Path

from sim.core import System
from sim.core.log import Log
from sim.core.engine import Engine

from .init.trace import LOAD_TRACE_CLASS
from .init.compute import LOAD_COMPUTE_CLASS
from .init.memory import LOAD_MEMORY_CLASS
from .init.storage import LOAD_STORAGE_CLASS
from .init.scheduler import LOAD_SCHEDULER_CLASS


def parse_config(config_file_path: str):
    config_file_path = Path(config_file_path)
    config_file_dir = str(config_file_path.parent)
    config_file_name = config_file_path.stem

    # CLI options will override config file
    overrides = sys.argv[1:]

    # Initialize Hydra
    cfg = None
    with initialize(version_base=None, config_path=config_file_dir):
        cfg = compose(config_name=config_file_name, overrides=overrides)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return cfg_dict


class SimIdentityMgr:
    def __init__(self):
        self.counter: int = 10
        self.names: list[str] = []
        return

    def get_id(self) -> int:
        num = self.counter
        self.counter += 10
        return num

    def check_name(self, name: str) -> None:
        if name in self.names:
            raise Exception(f"[Simulator] SimObject with name: {name} exists!")

        self.names.append(name)
        return


class Simulator:
    """
    Compute Graph Simulator
    """

    def __init__(self, config_file_path: str):
        # Read input file(input.yaml), and parse fields
        cfg = parse_config(config_file_path)

        sim_id = SimIdentityMgr()

        # Logger
        l_cfg = cfg["logger"]
        l_cfg["args"]["input_path"] = config_file_path  # Supply input_path
        log = Log(l_cfg["args"])
        log.start()

        # Trace
        t_cfg = cfg["trace"]
        t_cfg["args"]["input_path"] = config_file_path  # Supply input_path
        TraceLoaderClass = LOAD_TRACE_CLASS(t_cfg["type"])
        name = "Trace"
        sim_id.check_name(name)
        trace_loader = TraceLoaderClass(sim_id.get_id(), name, log, t_cfg["args"])
        trace = trace_loader.load()

        # Hardware Dictionary
        hw = {}

        # Storage
        for s_cfg in cfg["hardware"]["storage"]:
            StorageClass = LOAD_STORAGE_CLASS(s_cfg["type"])
            name = s_cfg["name"]
            sim_id.check_name(name)
            storage_hw = StorageClass(sim_id.get_id(), name, log, s_cfg["args"])
            hw[storage_hw.name] = storage_hw

        # Memory
        for m_cfg in cfg["hardware"]["memory"]:
            MemoryClass = LOAD_MEMORY_CLASS(m_cfg["type"])
            name = m_cfg["name"]
            sim_id.check_name(name)
            memory_hw = MemoryClass(sim_id.get_id(), name, log, m_cfg["args"])
            hw[memory_hw.name] = memory_hw

        # Compute - Compute units must be initialized after memory units
        for c_cfg in cfg["hardware"]["compute"]:
            ComputeClass = LOAD_COMPUTE_CLASS(c_cfg["type"])
            local_memory = hw[c_cfg["args"]["memory"]]
            name = c_cfg["name"]
            sim_id.check_name(name)
            compute_hw = ComputeClass(sim_id.get_id(), name, log, local_memory, c_cfg["args"])
            hw[compute_hw.name] = compute_hw

        # System
        sys = System(trace, hw)

        # Scheduler
        sched_cfg = cfg["scheduler"]
        SchedulerClass = LOAD_SCHEDULER_CLASS(sched_cfg["type"])
        name = "Scheduler"
        sim_id.check_name(name)
        sched = SchedulerClass(sim_id.get_id(), name, log, sys, sched_cfg["args"])

        # Engine
        name = "Engine"
        sim_id.check_name(name)
        self.engine = Engine(sim_id.get(), name, log, sys, sched)
        return

    def run(self):
        print("Simulation is starting ...")
        self.engine.run()
        print("Simulation is finished.")
        return
