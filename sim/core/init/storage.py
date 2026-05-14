import importlib


def LOAD_STORAGE_CLASS(storage_type: str):
    # Live lookup so `restart_simulation(reload=True)` can hot-reload
    # user-edited storage hardware code: after sys.modules invalidation,
    # the next import_module re-executes `sim/hw/storage/__init__.py`
    # (which walks subpackages via pkgutil) and yields fresh class objects.
    pkg = importlib.import_module("sim.hw.storage")
    storage = getattr(pkg, storage_type, None)
    if storage is None:
        raise Exception(f"[init] Storage Hardware of type: {storage_type} does not exist.")
    return storage
