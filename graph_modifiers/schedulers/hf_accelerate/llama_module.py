"""Llama harness ``--offload-mode module`` scheduler.

Models ``run_llama_accelerate_cpu_offload.py --offload-mode module``:
the harness calls ``accelerate.cpu_offload`` on the top-level children
of ``LlamaForCausalLM`` (``model`` and ``lm_head``), with
``offload_buffers=True``.

The resulting Accelerate hooks are still per direct tensor-bearing
submodule (Linear / Embedding / RMSNorm / buffer-only helpers), but the
H2D copies are ordinary default-stream ``set_module_tensor_to_device``
copies. In cg-sim we issue them after the previous GPU node retires so
future decoder-layer weights do not pile up while older kernels are
still running.
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
        "Llama direct HF Accelerate module CPU offload "
        "(`cpu_offload` on top-level LlamaForCausalLM children, "
        "`offload_buffers=True`). Per-leaf AlignDevicesHook, "
        "default-stream serialized synchronous H2D, no D2H."
    )
    args = p.parse_args()
    knobs = HFAccelerateKnobs(
        lookahead=0,
        granularity="leaf",
        keep_substrings=keep_substrings(args),
        include_buffers=True,
        sync_calls=True,
        h2d_after_prev_gpu=True,
        emit_d2h_evict=False,
        omit_last_unit_d2h=False,
    )
    run(args, knobs)


if __name__ == "__main__":
    main()
