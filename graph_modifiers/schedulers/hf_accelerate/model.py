"""``enable_model_cpu_offload`` scheduler.

Models ``pipe.enable_model_cpu_offload()`` from diffusers, which
chains ``accelerate.cpu_offload_with_hook`` per top-level component
(text_encoder / unet / vae / ...) with ``prev_module_hook=`` pointing
at the previous component. Two consequences:

- **Per-component granularity** (depth:1 in module-path terms): when
  any leaf of component N runs, the whole component lives on GPU.
- **Default-stream serialization**: at component N's pre_forward,
  ``prev_module_hook.offload()`` queues the D2H of N-1 first, then
  ``module.to(execution_device)`` queues the H2D of N — both on the
  default CUDA stream, so D2H runs to completion before H2D starts.
  Peak is therefore ``max(N-1, N)``, NOT ``N-1 + N``. Modelled with
  ``lookahead=0``: the D2H fires at N-1's last GPU node retire and
  N's prefetch fires at N's cpu_dispatcher (after N-1 is done), so
  the sim's transfer subsystem serializes them naturally on PCIe.
- **D2H on offload**: ``CpuOffload`` does ``module.to("cpu")`` which is
  a real per-parameter ``cudaMemcpyAsync`` D2H.
- **Last component also offloaded**: ``pipe.__call__`` invokes
  ``maybe_free_model_hooks`` at the end, which calls
  ``self.to("cpu")`` on every offloaded component, including the last
  one (no ``omit_last_unit_d2h``).
"""

from __future__ import annotations

from graph_modifiers.schedulers.hf_accelerate._cli import (
    base_parser, keep_substrings, run,
)
from graph_modifiers.schedulers.hf_accelerate.scheduler import (
    HFAccelerateKnobs,
)


def main() -> None:
    p = base_parser(
        "HF diffusers `enable_model_cpu_offload` "
        "(chained `cpu_offload_with_hook` per top-level component + "
        "`maybe_free_model_hooks`). Depth:1 granularity, lookahead=0 "
        "(default-stream serialization of D2H-prev + H2D-next), real "
        "per-param D2H on offload, last component is also D2H'd via "
        "maybe_free_model_hooks."
    )
    args = p.parse_args()
    knobs = HFAccelerateKnobs(
        lookahead=0,
        granularity="depth:1",
        keep_substrings=keep_substrings(args),
        include_buffers=True,
        sync_calls=True,
        emit_d2h_evict=True,
        omit_last_unit_d2h=False,
    )
    run(args, knobs)


if __name__ == "__main__":
    main()
