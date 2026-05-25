"""``apply_group_offloading`` block-level scheduler.

Models diffusers' ``apply_group_offloading(offload_type="block_level",
num_blocks_per_group=N, use_stream=True, non_blocking=True)``:

- **Block granularity** — each immediate child of a ``ModuleList`` /
  ``Sequential`` / ``ModuleDict`` is its own offload group (e.g.
  every ``LlamaDecoderLayer`` under ``model.layers``, every UNet
  transformer block). Recovered from a ``module_hierarchy.json``
  built alongside the bundle.
- **Async via separate stream** — the H2D fires on a dedicated CUDA
  stream that overlaps with the prior block's compute on the default
  stream. Modelled as ``sync_calls=False`` + ``lookahead=1``.
- **No D2H bytes** — with ``use_stream=True`` the post-forward
  offload is a *pointer swap* to a cached cpu tensor preserved at
  hook setup. No ``cudaMemcpyAsync`` D2H bytes move, so
  ``emit_d2h_evict=False``.
- **Buffers stream too** — the whole offload group moves together.

The only user-side knob exposed is ``--num-blocks-per-group``
(coalesce N consecutive container children into one offload group),
which is a real diffusers argument with semantic effect on VRAM peak.
"""

from __future__ import annotations

import sys
from pathlib import Path

from graph_modifiers.schedulers.hf_accelerate._cli import (
    base_parser, keep_substrings, run,
)
from graph_modifiers.schedulers.hf_accelerate.scheduler import (
    HFAccelerateKnobs,
)


def _resolve_hierarchy(bundle: str, override: str | None) -> str | None:
    """Locate a ``module_hierarchy.json`` for the bundle.

    Order:
      1. Explicit ``--module-hierarchy`` flag.
      2. ``<bundle>/module_hierarchy.json``.
      3. ``<bundle>/../<bundle_basename>_module_hierarchy/module_hierarchy.json``
         (the layout the eager profile pipeline produces).
    """
    bundle_path = Path(bundle)
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    candidates.append(bundle_path / "module_hierarchy.json")
    sibling = bundle_path.parent / f"{bundle_path.name}_module_hierarchy"
    candidates.append(sibling / "module_hierarchy.json")
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def main() -> None:
    p = base_parser(
        "diffusers `apply_group_offloading(offload_type=\"block_level\", "
        "use_stream=True, non_blocking=True)`. Block-granular offload "
        "with prefetch on a separate cuda stream — H2D overlaps with "
        "the previous block's compute and offload is a pointer swap "
        "(no D2H bytes move)."
    )
    p.add_argument(
        "--num-blocks-per-group", type=int, default=1,
        help="Coalesce N consecutive ModuleList/Sequential entries into "
             "one offload group (`apply_group_offloading "
             "num_blocks_per_group`). Default 1 (one block per group).",
    )
    p.add_argument(
        "--module-hierarchy", dest="module_hierarchy", default=None,
        help="Path to module_hierarchy.json. Auto-discovered from "
             "<bundle>/module_hierarchy.json or "
             "<bundle>/../<bundle>_module_hierarchy/module_hierarchy.json "
             "if omitted. Required to locate real ModuleList/Sequential "
             "block boundaries.",
    )
    args = p.parse_args()

    hierarchy_path = _resolve_hierarchy(args.bundle, args.module_hierarchy)
    if hierarchy_path is None:
        print(
            "[hf_accelerate.group] ERROR: no module_hierarchy.json found "
            f"near bundle ({args.bundle}). apply_group_offloading needs "
            "the real nn.Module tree to identify ModuleList/Sequential "
            "block boundaries. Pass --module-hierarchy explicitly, or "
            "re-emit the bundle with a sibling _module_hierarchy "
            "directory.",
            file=sys.stderr, flush=True,
        )
        sys.exit(2)
    print(f"[hf_accelerate.group] using module hierarchy: {hierarchy_path}",
          flush=True)

    knobs = HFAccelerateKnobs(
        lookahead=1,
        granularity="block",
        keep_substrings=keep_substrings(args),
        include_buffers=True,
        sync_calls=False,
        emit_d2h_evict=False,
        omit_last_unit_d2h=False,
        module_hierarchy_path=hierarchy_path,
        num_blocks_per_group=int(args.num_blocks_per_group),
    )
    run(args, knobs)


if __name__ == "__main__":
    main()
