import importlib


def LOAD_MEMORY_CLASS(memory_type: str):
    # Live lookup so `restart_simulation(reload=True)` can hot-reload
    # user-edited memory hardware code: after sys.modules invalidation,
    # the next import_module re-executes `sim/hw/memory/__init__.py`
    # (which walks subpackages via pkgutil) and yields fresh class objects.
    pkg = importlib.import_module("sim.hw.memory")
    memory = getattr(pkg, memory_type, None)
    if memory is None:
        raise Exception(f"[init] Memory Hardware of type: {memory_type} does not exist.")
    return memory
