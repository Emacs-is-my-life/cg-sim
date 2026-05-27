#!/usr/bin/env python3
"""Interactive Gantt of a single run: compute + transfers over time.

Reads a single analysis run dir produced with `--out`:
  * `<in_dir>/stalls.csv`           (gap intervals on the compute hw)
  * `<in_dir>/transfer_phases.csv`  (per-transfer phases on the link)
  * `<in_dir>/module_rollup.csv`    (for coloring blocked computes by module)

A standalone HTML file is written; open in browser to hover/zoom. Three
rows are drawn:
  - **compute:gaps**          stalls coloured by blocked module
  - **link:transfer**         per-transfer bars coloured by src→dest leg
  - **link:queue_wait**       queue-side waiting in front of each transfer

Best paired with `scripts/analysis/prefetch_quality.py --out`.

Usage:
    python plot_timeline.py <in_dir> [--out PATH.html]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import parse_out_path_flag, read_table  # noqa: E402


def _module_key(name: str, depth: int = 3) -> str:
    if not isinstance(name, str) or not name:
        return "<unknown>"
    parts = name.split(".")
    return ".".join(parts[:depth]) if parts else "<unknown>"


def main(
    in_dir: Path,
    *,
    out_path: Path | None = None,
    module_depth: int = 3,
) -> None:
    import plotly.graph_objects as go

    in_dir = Path(in_dir)
    stalls = read_table(in_dir, "stalls")
    transfers = read_table(in_dir, "transfer_phases")

    fig = go.Figure()

    # Row 1: stall gaps, coloured by blocked-node module.
    if not stalls.empty:
        modules = stalls["blocked_node_name"].fillna("").map(
            lambda n: _module_key(n, module_depth)
        )
        for mod, sub in stalls.assign(_mod=modules).groupby("_mod"):
            xs, ys, hovers = [], [], []
            for _, r in sub.iterrows():
                xs += [r["gap_start_us"], r["gap_end_us"], None]
                ys += ["compute:gaps", "compute:gaps", None]
                hovers += [
                    f"node={r['blocked_node_name']}<br>"
                    f"dur={r['dur_us']:.1f}us<br>"
                    f"blamed_size_KB={r.get('blamed_size_KB', '')}",
                    None,
                    None,
                ]
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(width=12),
                    name=f"stall {mod[:40]}",
                    hovertext=hovers, hoverinfo="text",
                )
            )

    # Row 2: per-transfer bars (xfer phase only, the actual byte movement).
    # Row 3: queue_wait phase (how long the transfer queued before its head).
    if not transfers.empty:
        # Bars per (src, dest) leg.
        for (src, dest), sub in transfers.groupby(["src_name", "dest_name"]):
            xs, ys, hovers = [], [], []
            for _, r in sub.iterrows():
                xs += [r["end_us"] - r["xfer_us"], r["end_us"], None]
                ys += ["link:transfer", "link:transfer", None]
                hovers += [
                    f"{src} → {dest}<br>"
                    f"size={r['size_KB']:.1f}KB<br>"
                    f"xfer={r['xfer_us']:.1f}us<br>"
                    f"rate={r['rate_KBps']:.0f}KB/s<br>"
                    f"tids={r['tensor_ids']}",
                    None,
                    None,
                ]
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(width=12),
                    name=f"{src}→{dest}",
                    hovertext=hovers, hoverinfo="text",
                )
            )

        xs_q, ys_q, hovers_q = [], [], []
        for _, r in transfers.iterrows():
            qw = float(r["queue_wait_us"])
            if qw <= 0:
                continue
            t_end = float(r["end_us"]) - float(r["xfer_us"]) - float(r["head_wait_us"])
            xs_q += [t_end - qw, t_end, None]
            ys_q += ["link:queue_wait", "link:queue_wait", None]
            hovers_q += [
                f"queue_wait={qw:.1f}us<br>size={r['size_KB']:.1f}KB",
                None,
                None,
            ]
        if xs_q:
            fig.add_trace(
                go.Scatter(
                    x=xs_q, y=ys_q, mode="lines",
                    line=dict(width=12, color="rgba(128,128,128,0.5)"),
                    name="queue_wait",
                    hovertext=hovers_q, hoverinfo="text",
                )
            )

    fig.update_layout(
        title=f"{in_dir.name}: timeline",
        xaxis_title="time (us)",
        yaxis=dict(categoryorder="array",
                   categoryarray=["link:queue_wait", "link:transfer", "compute:gaps"]),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )

    if out_path is None:
        out_path = in_dir / "plot_timeline.html"
    if not str(out_path).endswith(".html"):
        out_path = out_path.with_suffix(".html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        print("Usage: python plot_timeline.py <in_dir> [--out PATH.html]")
        sys.exit(2)
    in_dir = Path(argv[0])
    rest, out = parse_out_path_flag(argv[1:], __file__, in_dir)
    main(in_dir, out_path=out)
