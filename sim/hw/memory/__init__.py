import pkgutil
import importlib

__all__ = []

for _, name, _ in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{__name__}.{name}")

    if hasattr(module, "__all__"):
        for attr in module.__all__:
            globals()[attr] = getattr(module, attr)
        __all__ += module.__all__
