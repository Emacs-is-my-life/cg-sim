"""``cpu_offload_with_hook`` chained scheduler.

Models the lower-level call pattern that ``enable_model_cpu_offload``
uses internally, but applied manually by the user::

    _, h1 = cpu_offload_with_hook(text_encoder, device)
    _, h2 = cpu_offload_with_hook(unet, device, prev_module_hook=h1)
    _, h3 = cpu_offload_with_hook(vae,  device, prev_module_hook=h2)

Behavioural differences from ``model.py``:

- Same default-stream serialization of D2H-prev + H2D-next at each
  ``pre_forward`` hook, modelled as ``lookahead=0`` (peak = max of
  any two adjacent components).
- There is **no** outer ``maybe_free_model_hooks`` wrapper, so the
  *last* component in the chain has no ``prev_module_hook`` successor
  and its offload hook is never fired — its weights stay resident on
  GPU after the pipeline finishes (``omit_last_unit_d2h=True``).
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
        "HF accelerate `cpu_offload_with_hook` chained per top-level "
        "component with `prev_module_hook`. Same prefetch/D2H semantics "
        "as `enable_model_cpu_offload` except the LAST component is not "
        "offloaded (no successor hook to fire its offload)."
    )
    args = p.parse_args()
    knobs = HFAccelerateKnobs(
        lookahead=0,
        granularity="depth:1",
        keep_substrings=keep_substrings(args),
        include_buffers=True,
        sync_calls=True,
        emit_d2h_evict=True,
        omit_last_unit_d2h=True,
    )
    run(args, knobs)


if __name__ == "__main__":
    main()
