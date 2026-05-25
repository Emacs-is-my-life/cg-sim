"""HF accelerate / diffusers CPU-offload schedulers.

Each HF offload API maps to its own scheduler entry point. The CLI
modules below hardcode the semantics of the corresponding HF function
— no ``--mode`` switch, no ``--d2h-evict`` / ``--sync-calls`` /
``--lookahead`` knobs that change *what* HF does. Run a specific mode
via e.g. ``python -m graph_modifiers.schedulers.hf_accelerate.sequential``.

| Module | HF API |
|---|---|
| ``sequential`` | ``pipe.enable_sequential_cpu_offload()`` |
| ``module`` | ``accelerate.cpu_offload(model)`` |
| ``model`` | ``pipe.enable_model_cpu_offload()`` |
| ``module_hook`` | chained ``accelerate.cpu_offload_with_hook(...)`` |
| ``group`` | ``diffusers.apply_group_offloading(block_level, ...)`` |
"""

from .scheduler import HFAccelerateKnobs, solve, write_eager_schedule

__all__ = ["HFAccelerateKnobs", "solve", "write_eager_schedule"]
