"""Consistent visual style across plots.

Applied once at the top of each matplotlib script via
`apply_matplotlib_defaults()`. Plotly scripts use the same PALETTE.
"""
from __future__ import annotations


# Color-blind-safe Okabe-Ito palette. Stable across plots so the same
# series gets the same color in different figures of a paper.
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
]


def color_for(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def apply_matplotlib_defaults() -> None:
    """Paper-friendly defaults: medium fonts, tight layout, vector-friendly."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.figsize": (5.5, 3.8),
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.7,
            "lines.markersize": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
