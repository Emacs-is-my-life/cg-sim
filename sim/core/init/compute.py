import importlib


def LOAD_COMPUTE_CLASS(compute_type: str):
    # Live lookup so `restart_simulation(reload=True)` can hot-reload
    # user-edited compute hardware code: after sys.modules invalidation,
    # the next import_module re-executes `sim/hw/compute/__init__.py`
    # (which walks subpackages via pkgutil) and yields fresh class objects.
    pkg = importlib.import_module("sim.hw.compute")
    compute = getattr(pkg, compute_type, None)
    if compute is None:
        raise Exception(f"[init] Compute Hardware of type: {compute_type} does not exist.")
    return compute
