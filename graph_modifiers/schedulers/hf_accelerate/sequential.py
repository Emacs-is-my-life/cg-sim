"""``enable_sequential_cpu_offload`` scheduler.

Models ``pipe.enable_sequential_cpu_offload()`` from diffusers, which
calls ``accelerate.cpu_offload(component, offload_buffers=True)`` on
every top-level component. Underneath, ``cpu_offload`` attaches
``AlignDevicesHook`` to every leaf ``nn.Module``:

- ``pre_forward`` synchronously H2Ds the leaf's params (+ buffers) to
  GPU before the leaf's first kernel launches.
- ``post_forward`` releases the GPU storage; the original CPU copy
  was preserved in the hook's ``weights_map`` so no D2H is needed.

Because the hook fires per-leaf and the H2D is on the default stream,
sequential offload is strictly synchronous from the host's
perspective — no lookahead, no concurrent residency across leaves.
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
        "HF accelerate sequential CPU offload "
        "(`enable_sequential_cpu_offload` / `cpu_offload` with "
        "`offload_buffers=True`). Per-leaf AlignDevicesHook, synchronous "
        "H2D, no D2H emission (cpu copy preserved by the hook)."
    )
    args = p.parse_args()
    knobs = HFAccelerateKnobs(
        lookahead=0,
        granularity="leaf",
        keep_substrings=keep_substrings(args),
        include_buffers=True,
        sync_calls=True,
        emit_d2h_evict=False,
        omit_last_unit_d2h=False,
    )
    run(args, knobs)


if __name__ == "__main__":
    main()
