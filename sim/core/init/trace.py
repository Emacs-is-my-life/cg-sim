import importlib


def LOAD_TRACE_CLASS(loader_type: str):
    # Live lookup so `restart_simulation(reload=True)` can hot-reload
    # user-edited trace loader code: after sys.modules invalidation, the
    # next import_module re-executes `sim/load/__init__.py` (which walks
    # subpackages via pkgutil) and yields fresh class objects.
    pkg = importlib.import_module("sim.load")
    trace_loader_class = getattr(pkg, loader_type, None)
    if trace_loader_class is None:
        raise Exception(f"[init] Trace loader of type: {loader_type} does not exist.")
    return trace_loader_class
