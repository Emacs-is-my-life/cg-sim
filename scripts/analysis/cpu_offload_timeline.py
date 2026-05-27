#!/usr/bin/env python3
"""Per-tensor TRANSFER/RELEASE swimlane for cpu_offload-style schedulers.

Reads a single chrome-trace result.json and produces:
  1. ``cpu_offload_timeline.csv`` — one row per event (TRANSFER or RELEASE)
     with tensor_id, tensor_name, module_path, size_KB, src/dest hw, t_start,
     t_end (== t_start for RELEASE).
  2. ``cpu_offload_timeline.html`` — interactive plotly figure. y-axis is
     tensor (ordered by first-touch); TRANSFER drawn as a `[begin, end]`
     line segment, RELEASE as an instant marker; color by event type;
     hover shows module_path + tensor metadata.

Designed to validate a new accelerate-style cpu_offload scheduler: the
expected pattern is a clean per-module load → use → release cycle, which
should appear as contiguous module-coloured swimlanes when tensors are
ordered by first-touch.

Module attribution:
  Each WEIGHT tensor is attributed to a module by BFS up the consumer's
  parent chain until a node with ``args["module"]["module_path"]`` is
  found. Tensors whose chain has no module info (e.g., llamacpp traces)
  are labelled ``<no-module>`` and still plotted.

The script auto-discovers the input.yaml path from the SIM_CONFIG event
and reloads the Trace via the standard simulator init (skipping ``run``)
so node/tensor metadata comes from the live trace object, not from the
NODES/TENSORS event dumps. The reload is sent to a tmp logfile to avoid
clobbering the existing result.json.

Filtering signal:
  * Default ``--filter weight``: only ``tensor.args["tensor_type"] ==
    "WEIGHT"`` tensors. Canonical across pytorch_profile + llamacpp loaders.
  * ``--filter any``: every tensor that appears in a TRANSFER or RELEASE
    event (useful when validating against traces that don't classify
    tensor kinds — every llama-3-8B example does, though).

Usage:
    python3 scripts/analysis/cpu_offload_timeline.py PATH/TO/result.json
        [--out OUT_DIR]
        [--filter weight|any]

Default ``--out`` lands the CSV and HTML in the result.json's setup dir
(``output/<setup>/sim_results/result.json`` →
``output/<setup>/{analysis,plots}/``).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    find_runtime_start,
    load_events,
    parse_sim_config,
    parse_transfer_jobs,
    write_meta,
    write_table,
)

TRACK_EVENT = 1
NO_MODULE = "<no-module>"
# Initial x-axis zoom fraction: the figure opens showing the first
# INITIAL_X_FRACTION of the full runtime span — short TRANSFER segments
# are then visually distinct (not crushed into dots). User double-clicks
# the axis (or hits the modebar autoscale) to see the full timeline.
INITIAL_X_FRACTION = 0.05


def _parse_release_events(
    events: list[dict], t_start: float
) -> list[tuple[float, int, int]]:
    """List of (ts_us, tensor_id, hw_id) for RELEASE_JOB instant events."""
    out: list[tuple[float, int, int]] = []
    for ev in events:
        if ev.get("pid") != TRACK_EVENT or ev.get("ph") != "i":
            continue
        if ev.get("name") != "RELEASE_JOB":
            continue
        ts = float(ev.get("ts", 0.0))
        if ts < t_start:
            continue
        args = ev.get("args") or {}
        tid = args.get("tensor_id")
        if tid is None:
            continue
        out.append((ts, int(tid), int(ev.get("tid", -1))))
    out.sort()
    return out


def _load_trace_from_sim_config(sim_cfg: dict) -> Any:
    """Reinstantiate the Trace via Simulator() using the input.yaml path
    embedded in SIM_CONFIG. Logger is redirected to a tmp file so the
    existing result.json is not overwritten."""
    from sim.core import Simulator
    cfg = sim_cfg.get("config") or {}
    input_path = (
        (cfg.get("logger", {}).get("args", {}) or {}).get("input_path")
        or (cfg.get("trace", {}).get("args", {}) or {}).get("input_path")
    )
    if not input_path:
        raise SystemExit(
            "SIM_CONFIG has no input_path — cannot locate the source YAML "
            "for trace reload. Use --input-yaml to override."
        )
    # Resolve relative to repo root (input_path is what was passed to main.py).
    yaml_path = (REPO_ROOT / input_path) if not Path(input_path).is_absolute() else Path(input_path)
    if not yaml_path.exists():
        raise SystemExit(f"input.yaml from SIM_CONFIG not found: {yaml_path}")
    tmp_log = Path(tempfile.gettempdir()) / "cgsim_cpu_offload_timeline_probe.json"
    overrides = [f"logger.args.result_path={tmp_log}"]
    print(f"  reloading trace from {yaml_path} (probe log -> {tmp_log})")
    sim = Simulator(str(yaml_path), overrides=overrides)
    assert sim.engine is not None and sim.log is not None
    trace = sim.engine.sys.trace
    # Stop the log so the tmp file isn't held open.
    try:
        sim.log.stop()
    except Exception:
        pass
    return trace


def _build_consumer_map(trace: Any) -> dict[int, list[int]]:
    """tensor_id -> list of node_ids that consume it (as input)."""
    consumers: dict[int, list[int]] = defaultdict(list)
    for nid, n in trace.node_map.items():
        for tid in (n.input_tensors or []):
            tid = tid if isinstance(tid, int) else tid.id
            consumers[tid].append(nid)
    return consumers


def _walk_for_module(
    start_nid: int, node_map: dict, max_depth: int = 8
) -> str | None:
    """BFS up parent chain to find the first ancestor whose
    ``args["module"]["module_path"]`` is set."""
    seen: set[int] = set()
    frontier: list[tuple[int, int]] = [(start_nid, 0)]
    while frontier:
        nxt: list[tuple[int, int]] = []
        for nid, d in frontier:
            if nid in seen or d > max_depth:
                continue
            seen.add(nid)
            n = node_map.get(nid)
            if n is None:
                continue
            mod = (n.args or {}).get("module") if isinstance(n.args, dict) else None
            mp = mod.get("module_path") if isinstance(mod, dict) else None
            if mp:
                return mp
            for pid in (n.parent_nodes or []):
                pid_int = pid if isinstance(pid, int) else pid.id
                nxt.append((pid_int, d + 1))
        frontier = nxt
    return None


def _attribute_tensor(
    tid: int,
    consumers: dict[int, list[int]],
    node_map: dict,
) -> str:
    """Module path for ``tid`` via first-consumer-then-parent walk."""
    for cnid in consumers.get(tid, []):
        n = node_map.get(cnid)
        if n is None:
            continue
        mod = (n.args or {}).get("module") if isinstance(n.args, dict) else None
        mp = mod.get("module_path") if isinstance(mod, dict) else None
        if mp:
            return mp
        mp = _walk_for_module(cnid, node_map)
        if mp:
            return mp
    return NO_MODULE


def _resolve_out_dirs(
    result_path: Path, override: Path | None
) -> tuple[Path, Path]:
    if override is not None:
        return override / "analysis", override / "plots"
    setup_dir = result_path.resolve().parent.parent
    return setup_dir / "analysis", setup_dir / "plots"


def _figure(
    rows: list[dict],
    tensor_order: dict[int, int],
    tensor_info: dict[int, tuple[str, str, float]],  # name, kind, size_KB
    tensor_module: dict[int, str],
    max_time_us: float,
):
    import plotly.graph_objects as go

    fig = go.Figure()

    # TRANSFER events: line segments.
    xs_t, ys_t, hovers_t = [], [], []
    for r in rows:
        if r["event_type"] != "TRANSFER":
            continue
        y = tensor_order[r["tensor_id"]]
        name, kind, size = tensor_info.get(r["tensor_id"], ("", "", 0.0))
        mod = tensor_module.get(r["tensor_id"], NO_MODULE)
        hover = (
            f"<b>TRANSFER</b><br>"
            f"<b>name=</b>{name or '&lt;unnamed&gt;'}<br>"
            f"tensor_id={r['tensor_id']}<br>"
            f"kind={kind}  size={size:.1f} KB<br>"
            f"module={mod}<br>"
            f"{r['src_name']} → {r['dest_name']}<br>"
            f"t=[{r['t_start_us']:.0f}, {r['t_end_us']:.0f}] us "
            f"(dur={r['t_end_us'] - r['t_start_us']:.0f})"
        )
        xs_t += [r["t_start_us"], r["t_end_us"], None]
        ys_t += [y, y, None]
        hovers_t += [hover, hover, None]
    if xs_t:
        fig.add_trace(go.Scattergl(
            x=xs_t, y=ys_t,
            mode="lines",
            line=dict(color="rgba(31,119,180,0.85)", width=3),
            name="TRANSFER",
            hovertext=hovers_t,
            hoverinfo="text",
            connectgaps=False,
        ))

    # RELEASE events: instant markers.
    xs_r, ys_r, hovers_r = [], [], []
    for r in rows:
        if r["event_type"] != "RELEASE":
            continue
        y = tensor_order[r["tensor_id"]]
        name, kind, size = tensor_info.get(r["tensor_id"], ("", "", 0.0))
        mod = tensor_module.get(r["tensor_id"], NO_MODULE)
        hover = (
            f"<b>RELEASE</b><br>"
            f"<b>name=</b>{name or '&lt;unnamed&gt;'}<br>"
            f"tensor_id={r['tensor_id']}<br>"
            f"kind={kind}  size={size:.1f} KB<br>"
            f"module={mod}<br>"
            f"t={r['t_start_us']:.0f} us"
        )
        xs_r.append(r["t_start_us"])
        ys_r.append(y)
        hovers_r.append(hover)
    if xs_r:
        fig.add_trace(go.Scattergl(
            x=xs_r, y=ys_r,
            mode="markers",
            marker=dict(color="rgba(214,39,40,0.9)", size=3, symbol="circle"),
            name="RELEASE",
            hovertext=hovers_r,
            hoverinfo="text",
        ))

    initial_x_max = max_time_us * INITIAL_X_FRACTION if max_time_us > 0 else 1.0
    fig.update_layout(
        title="CPU-offload timeline — TRANSFER (line) & RELEASE (•) per tensor",
        xaxis=dict(title="time (us)", range=[0, initial_x_max]),
        yaxis=dict(
            title=f"tensor (ordered by first touch, n={len(tensor_order)})",
        ),
        hovermode="closest",
        template="plotly_white",
        autosize=True,
        margin=dict(l=80, r=20, t=60, b=50),
    )
    return fig


def main(
    result_path: Path,
    out_root: Path | None,
    *,
    filter_mode: str = "weight",
) -> None:
    if filter_mode not in {"weight", "any"}:
        raise SystemExit(f"--filter must be 'weight' or 'any' (got {filter_mode!r})")

    print(f"loading {result_path} ...")
    events = load_events(result_path)
    t_start = find_runtime_start(events)
    print(f"  runtime_start_us = {t_start:.0f}")

    sim_cfg = parse_sim_config(events)
    if not sim_cfg:
        raise SystemExit("SIM_CONFIG event not found — log may predate this contract")

    transfers = parse_transfer_jobs(events, dest_name=None, t_start=t_start)
    releases = _parse_release_events(events, t_start)
    print(f"  TRANSFER_JOB={len(transfers)}  RELEASE_JOB={len(releases)}")
    if not transfers and not releases:
        print("nothing to plot — scheduler emitted no TRANSFER/RELEASE events.")
        return

    # Universe of tensors that appear in events.
    in_events: set[int] = set()
    for tj in transfers:
        in_events.update(int(x) for x in tj.tensor_ids if x is not None)
    for _, tid, _ in releases:
        in_events.add(tid)

    # Reload trace for module info + tensor kind.
    trace = _load_trace_from_sim_config(sim_cfg)
    node_map = trace.node_map
    tensor_map = trace.tensor_map
    print(f"  trace reloaded: {len(node_map)} nodes, {len(tensor_map)} tensors")

    # Apply filter.
    def kind_of(tid: int) -> str:
        t = tensor_map.get(tid)
        if t is None:
            return ""
        return (t.args or {}).get("tensor_type", "") if isinstance(t.args, dict) else ""

    if filter_mode == "weight":
        universe = {tid for tid in in_events if kind_of(tid) == "WEIGHT"}
        excluded = in_events - universe
        print(f"  filter=weight: {len(universe)} tensors kept "
              f"({len(excluded)} non-WEIGHT excluded)")
    else:
        universe = set(in_events)
        print(f"  filter=any: {len(universe)} tensors kept")

    if not universe:
        print("nothing to plot — no qualifying tensors after filter.")
        return

    # Module attribution.
    consumers = _build_consumer_map(trace)
    tensor_module: dict[int, str] = {
        tid: _attribute_tensor(tid, consumers, node_map) for tid in universe
    }
    with_mod = sum(1 for v in tensor_module.values() if v != NO_MODULE)
    print(f"  module attribution: {with_mod}/{len(universe)} "
          f"({100*with_mod/len(universe):.1f}%)")

    # Tensor info table for hover (name, kind, size_KB).
    tensor_info: dict[int, tuple[str, str, float]] = {}
    for tid in universe:
        t = tensor_map.get(tid)
        if t is None:
            tensor_info[tid] = ("", "", 0.0)
            continue
        size_kb = (4 * (t.num_pages or 0))  # page size = 4KB
        kind = (t.args or {}).get("tensor_type", "") if isinstance(t.args, dict) else ""
        tensor_info[tid] = (str(t.name), str(kind), float(size_kb))

    # Build rows + first-touch order.
    rows: list[dict] = []
    first_touch: dict[int, float] = {}
    for tj in transfers:
        for tid_raw in tj.tensor_ids:
            if tid_raw is None:
                continue
            tid = int(tid_raw)
            if tid not in universe:
                continue
            rows.append({
                "event_type": "TRANSFER",
                "tensor_id": tid,
                "t_start_us": tj.begin_us,
                "t_end_us": tj.end_us,
                "src_name": tj.src_name,
                "dest_name": tj.dest_name,
            })
            if tj.begin_us < first_touch.get(tid, float("inf")):
                first_touch[tid] = tj.begin_us
    for ts, tid, _ in releases:
        if tid not in universe:
            continue
        rows.append({
            "event_type": "RELEASE",
            "tensor_id": tid,
            "t_start_us": ts,
            "t_end_us": ts,
            "src_name": "",
            "dest_name": "",
        })
        if ts < first_touch.get(tid, float("inf")):
            first_touch[tid] = ts

    ordered_tids = sorted(universe, key=lambda t: first_touch.get(t, float("inf")))
    tensor_order = {tid: i for i, tid in enumerate(ordered_tids)}

    # Write outputs.
    analysis_dir, plots_dir = _resolve_out_dirs(result_path, out_root)
    csv_rows = []
    for r in rows:
        name, kind, size = tensor_info.get(r["tensor_id"], ("", "", 0.0))
        csv_rows.append([
            r["tensor_id"],
            name,
            kind,
            f"{size:.3f}",
            tensor_module.get(r["tensor_id"], NO_MODULE),
            r["event_type"],
            f"{r['t_start_us']:.3f}",
            f"{r['t_end_us']:.3f}",
            r["src_name"],
            r["dest_name"],
        ])
    csv_path = write_table(
        analysis_dir,
        "cpu_offload_timeline",
        ["tensor_id", "tensor_name", "tensor_type", "size_KB",
         "module_path", "event_type", "t_start_us", "t_end_us",
         "src_name", "dest_name"],
        csv_rows,
    )
    meta_path = write_meta(
        analysis_dir,
        log_path=str(result_path),
        runtime_start_us=t_start,
        filter_mode=filter_mode,
        n_tensors=len(universe),
        n_transfer_events=sum(1 for r in rows if r["event_type"] == "TRANSFER"),
        n_release_events=sum(1 for r in rows if r["event_type"] == "RELEASE"),
        n_tensors_with_module=with_mod,
    )
    print(f"  csv:  {csv_path}")
    print(f"  meta: {meta_path}")

    max_time_us = 0.0
    for r in rows:
        if r["t_end_us"] > max_time_us:
            max_time_us = r["t_end_us"]
    fig = _figure(rows, tensor_order, tensor_info, tensor_module, max_time_us)
    plots_dir.mkdir(parents=True, exist_ok=True)
    html_path = plots_dir / "cpu_offload_timeline.html"
    fig_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        default_height="100vh",
        default_width="100%",
        config={"responsive": True},
    )
    html_path.write_text(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>cpu_offload_timeline</title>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; overflow: hidden; }}
  .plot-wrap {{ width: 100vw; height: 100vh; }}
</style>
</head><body>
<div class="plot-wrap">{fig_html}</div>
</body></html>
""")
    print(f"  html: {html_path}  (initial x-zoom: 0 .. {max_time_us * INITIAL_X_FRACTION:.0f} us "
          f"= {INITIAL_X_FRACTION*100:.1f}% of {max_time_us:.0f} us total)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_path", type=Path,
                    help="path to sim result.json (chrome-trace)")
    ap.add_argument("--out", type=Path, default=None,
                    help="override output root (defaults to result.json's "
                         "<setup>/ — i.e. one up from sim_results/)")
    ap.add_argument("--filter", choices=("weight", "any"), default="weight",
                    help="tensor filter signal (default: weight)")
    args = ap.parse_args()
    main(args.result_path, args.out, filter_mode=args.filter)
