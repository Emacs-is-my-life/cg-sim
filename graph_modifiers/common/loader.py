"""Standalone wrapper around cg-sim's ``PytorchProfile`` TraceLoader.

``PytorchProfile`` is normally instantiated inside the full cg-sim ``Simulator``
boot sequence. Weight-streaming scheduling only needs the parsed ``Trace``
(nodes / tensors / edges) and the two PyTorch compile sidecars
(``compiled_launch_map_graph*.json`` and ``compiled_tensor_map_graph*.json``).

This module invokes ``PytorchProfile.load()`` with a minimal stub ``Log``
object, so callers outside the simulator (CLI scripts in this package)
can read a bundle without spinning up the event engine.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sim.core.log import Log
from sim.core.trace import Trace
from sim.load.pytorch_profile.pytorch_profile import PytorchProfile


class _NullLog(Log):
    """Log implementation that discards records.

    ``Log`` in cg-sim needs a concrete ``output_path`` etc. to write its
    chrome-trace, which we don't need for schedule generation. Subclassing
    and no-op'ing the recorder keeps ``SimObject`` construction happy.
    """

    def __init__(self) -> None:
        # Bypass the superclass __init__ that requires file handles. Keep
        # the attributes Engine / runtime callers touch.
        from sim.core.log.log import Level
        self._records: list = []
        self.on = False
        self.level = Level.ENGINE

    def record(self, *_args, **_kwargs) -> None:  # type: ignore[override]
        return

    def start(self) -> None:  # type: ignore[override]
        return

    def stop(self) -> None:  # type: ignore[override]
        return

    def get_trace_log(self, trace):  # type: ignore[override]
        return ({}, {})


def _build_loader(bundle_dir: Path, extra_args: dict[str, Any] | None = None) -> PytorchProfile:
    """Construct a ``PytorchProfile`` rooted at ``bundle_dir``.

    The loader resolves the manifest relative to ``args['input_path']``'s
    *parent*, so we set ``input_path`` to a path inside the bundle so
    ``input_dir = bundle_dir``. Then ``profile_dir = bundle_dir`` and
    ``bundle_manifest = 'manifest.json'``.
    """
    args = {
        "input_path": str(bundle_dir / "manifest.json"),
        "profile_dir": ".",
        "bundle_manifest": "manifest.json",
        "skip_zero_byte_tensors": True,
        "zero_wait_nodes": True,
        "strict_dot_validation": False,
    }
    if extra_args:
        args.update(extra_args)
    return PytorchProfile(obj_id=0, name="pytorch_profile", log=_NullLog(), args=args)


def load_trace_from_bundle(
    bundle_dir: str | Path, *, extra_args: dict[str, Any] | None = None
) -> Trace:
    """Parse a PyTorch profile bundle directory into a cg-sim ``Trace``."""
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"bundle dir not found: {bundle_dir}")
    loader = _build_loader(bundle_dir, extra_args=extra_args)
    return loader.load()


@dataclass
class BundleSidecars:
    launch_map: dict[str, Any] | None
    tensor_map: dict[str, Any] | None
    launch_map_path: Path | None
    tensor_map_path: Path | None


def load_sidecars(bundle_dir: str | Path) -> BundleSidecars:
    """Load the PyTorch compile sidecars (first graph only, for legacy)."""
    bundle_dir = Path(bundle_dir)
    launch_candidates = sorted(
        Path(p) for p in glob.glob(str(bundle_dir / "compiled_launch_map_graph*.json"))
    )
    tensor_candidates = sorted(
        Path(p) for p in glob.glob(str(bundle_dir / "compiled_tensor_map_graph*.json"))
    )
    launch_path = launch_candidates[0] if launch_candidates else None
    tensor_path = tensor_candidates[0] if tensor_candidates else None

    launch_map: dict[str, Any] | None = None
    tensor_map: dict[str, Any] | None = None
    if launch_path is not None:
        with open(launch_path) as f:
            launch_map = json.load(f)
    if tensor_path is not None:
        with open(tensor_path) as f:
            tensor_map = json.load(f)

    return BundleSidecars(
        launch_map=launch_map,
        tensor_map=tensor_map,
        launch_map_path=launch_path,
        tensor_map_path=tensor_path,
    )


@dataclass
class MultiGraphSidecars:
    """All compile sidecars in a bundle, keyed by graph_id.

    Use this when a pipeline has multiple compiled graphs (e.g. SDXL has
    text_encoder, text_encoder_2, unet, vae).
    """
    launch_maps: dict[int, dict[str, Any]]
    tensor_maps: dict[int, dict[str, Any]]


def load_multi_graph_sidecars(bundle_dir: str | Path) -> MultiGraphSidecars:
    """Load ALL compiled_{launch,tensor}_map_graphN.json sidecars in the bundle."""
    bundle_dir = Path(bundle_dir)
    launch_maps: dict[int, dict[str, Any]] = {}
    tensor_maps: dict[int, dict[str, Any]] = {}
    for p in sorted(glob.glob(str(bundle_dir / "compiled_launch_map_graph*.json"))):
        with open(p) as f:
            d = json.load(f)
        gid = int(d.get("graph_id", 0))
        launch_maps[gid] = d
    for p in sorted(glob.glob(str(bundle_dir / "compiled_tensor_map_graph*.json"))):
        with open(p) as f:
            d = json.load(f)
        gid = int(d.get("graph_id", 0))
        tensor_maps[gid] = d
    return MultiGraphSidecars(
        launch_maps=launch_maps, tensor_maps=tensor_maps,
    )
