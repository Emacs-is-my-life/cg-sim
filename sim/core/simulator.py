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


class SimID:
    def __init__(self):
        self.counter = 10
        return

    def get(self):
        num = self.counter
        self.counter += 10
        return num


class Simulator:
    """
    Compute Graph Simulator
    """

    def __init__(self, config_file_path: str):
        # Read input file(input.yaml), and parse fields
        cfg = parse_config(config_file_path)

        sim_id = SimID()

        # Logger
        cfg["logger"]["args"]["input_path"] = config_file_path  # Supply input_path
        log = Log(cfg["logger"]["args"])
        log.start()

        # Trace
        cfg["trace"]["args"]["input_path"] = config_file_path  # Supply input_path
        TraceLoaderClass = LOAD_TRACE_CLASS(cfg["trace"]["type"])
        trace_loader = TraceLoaderClass(sim_id.get(), "Trace", log, cfg["trace"]["args"])
        trace = trace_loader.load()

        hw = {}
        # Storage
        StorageClass = LOAD_STORAGE_CLASS(cfg["storage"]["type"])
        storage = StorageClass(sim_id.get(), cfg["storage"]["name"], log, cfg["storage"]["args"])
        hw[storage.name] = storage

        # Memory
        MemoryClass = LOAD_MEMORY_CLASS(cfg["memory"]["type"])
        memory = MemoryClass(sim_id.get(), cfg["memory"]["name"], log, cfg["memory"]["args"])
        hw[memory.name] = memory

        # Compute
        ComputeClass = LOAD_COMPUTE_CLASS(cfg["compute"]["type"])
        compute = ComputeClass(sim_id.get(), cfg["compute"]["name"], log, cfg["compute"]["args"])
        compute.memory = memory
        hw[compute.name] = compute

        # System
        sys = System(trace, hw)

        # Scheduler
        SchedulerClass = LOAD_SCHEDULER_CLASS(cfg["scheduler"]["type"])
        sched = SchedulerClass(sim_id.get(), "Scheduler", log, sys, cfg["scheduler"]["args"])

        # Engine
        self.engine = Engine(sim_id.get(), "Engine", log, sys, sched)

        return

    def run(self):
        print("Simulation is starting ...")
        self.engine.run()
        print("Simulation is finished.")
        return
