"""``accelerate.cpu_offload`` scheduler.

Models a direct call to ``accelerate.cpu_offload(model)`` (without the
``enable_sequential_cpu_offload`` wrapper). Same underlying mechanism
as sequential — ``AlignDevicesHook`` per leaf, synchronous H2D, no
D2H — but with the accelerate default ``offload_buffers=False``: only
parameters are streamed, buffers stay where they are.
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
        "HF accelerate per-module CPU offload "
        "(`accelerate.cpu_offload(model)`). Per-leaf AlignDevicesHook, "
        "synchronous H2D, no D2H, buffers NOT streamed "
        "(offload_buffers=False default)."
    )
    args = p.parse_args()
    knobs = HFAccelerateKnobs(
        lookahead=0,
        granularity="leaf",
        keep_substrings=keep_substrings(args),
        include_buffers=False,
        sync_calls=True,
        emit_d2h_evict=False,
        omit_last_unit_d2h=False,
    )
    run(args, knobs)


if __name__ == "__main__":
    main()
