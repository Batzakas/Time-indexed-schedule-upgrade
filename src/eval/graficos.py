"""

Two chart families, rendered from any summary CSV that has the right
columns (results/aggregated/congestion_<topology>.csv from
congestion_sweep.py, or results/aggregated/summary.csv from
aggregate_partials.py -- both now carry u_fraction/n_commodities/model-size
columns):

  - network_load_<label>.png   bottleneck/average link utilization
                                (_network_load in milp_makespan.py) and MILP
                                feasibility rate, vs --x-col (default:
                                n_commodities), one color per --series-col
                                value (default: u_fraction).
  - solver_<label>.png         solve runtime (log scale), MIP optimality
                                gap, and model size (constraint count),
                                against the same --x-col/--series-col axes.

Each chart is only rendered if its required columns are present and carry
at least one non-NaN value, so pointing this at an older summary.csv (before
u_fraction/model-size columns existed) just skips gracefully.

Usage:
    # congestion sweep (results/aggregated/congestion_<topology>.csv):
    # x=n_commodities, one series per u_fraction (both charts apply)
    python -m src.eval.graficos --summary-csv results/aggregated/congestion_dt12.csv \
        --topology-label dt12 --out-dir results/figures

    # run_pipeline.sh's M-sweep summary (results/aggregated/summary.csv):
    # x=M, one series per topology (graph_type)
    python -m src.eval.graficos --summary-csv results/aggregated/summary.csv \
        --topology-label all --out-dir results/figures \
        --x-col M --x-label "M (reroute penalty weight)" --series-col graph_type
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical palette (fixed order, validated for CVD-safety) and chart
# chrome, taken verbatim from the design-system reference used across this
# project's charts.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# Columns that identify an instance / a scenario rather than holding a
# numeric measurement -- left as strings by load_summary().
_NON_NUMERIC_COLS = {"graph_type", "instance_names"}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRIDLINE,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK_PRIMARY,
    "legend.frameon": False,
})


def load_summary(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k in _NON_NUMERIC_COLS:
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def _seed_label(instance_names: str) -> str:
    """'dt12_real_50_1' -> '1' (the trailing seed in the name every
    generator uses: <graph_type>_..._<seed>). Multiple pipe-joined names
    (a group that really did merge >1 seed) become e.g. '1,3'."""
    if not instance_names:
        return "?"
    seeds = []
    for name in instance_names.split("|"):
        parts = name.rsplit("_", 1)
        seeds.append(parts[1] if len(parts) == 2 else "?")
    return ",".join(seeds)


def _style_axis(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
        ax.spines[spine].set_linewidth(1.0)


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _series_colors(rows: List[dict], series_col: str) -> Dict[object, str]:
    values = sorted({r[series_col] for r in rows if series_col in r})
    if len(values) > len(CATEGORICAL):
        print(f"[warn] {len(values)} distinct {series_col!r} values but only {len(CATEGORICAL)} "
              f"categorical slots defined; extra ones will repeat colors.")
    return {v: CATEGORICAL[i % len(CATEGORICAL)] for i, v in enumerate(values)}


def _series_label(series_col: str, value) -> str:
    if isinstance(value, float):
        return f"{series_col}={value:g}"
    return str(value)


def _has_data(rows: List[dict], col: str) -> bool:
    return any(not _is_nan(r.get(col)) and r.get(col) is not None for r in rows)


def plot_network_load(
    rows: List[dict], out_dir: Path, topology_label: str,
    x_col: str = "n_commodities", x_label: Optional[str] = None, series_col: str = "u_fraction",
):
    x_label = x_label or x_col
    rows = [r for r in rows if x_col in r and series_col in r]
    rows = sorted(rows, key=lambda r: (r[series_col], r[x_col]))
    if not rows or not _has_data(rows, "bottleneck_util_mean"):
        print(f"[skip] network_load_{topology_label}.png: no bottleneck_util_mean data in CSV")
        return

    series_vals = sorted({r[series_col] for r in rows})
    colors = _series_colors(rows, series_col)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    for sv in series_vals:
        pts = [r for r in rows if r[series_col] == sv and r["feasible_rate"] > 0
               and not _is_nan(r.get("bottleneck_util_mean"))]
        xs = [r[x_col] for r in pts]
        ys = [r["bottleneck_util_mean"] for r in pts]
        yerr = [r["bottleneck_util_ci95"] for r in pts]
        ax.errorbar(xs, ys, yerr=yerr, fmt="o-", markersize=6, capsize=3, linewidth=1.6,
                    color=colors[sv], ecolor=colors[sv], alpha=0.9, label=_series_label(series_col, sv))
    ax.set_ylim(0, 1.05)
    _style_axis(ax, "Bottleneck utilization", x_label, "Worst (edge, h) utilization")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    for sv in series_vals:
        pts = [r for r in rows if r[series_col] == sv and r["feasible_rate"] > 0
               and not _is_nan(r.get("avg_util_mean"))]
        xs = [r[x_col] for r in pts]
        ys = [r["avg_util_mean"] for r in pts]
        yerr = [r["avg_util_ci95"] for r in pts]
        ax.errorbar(xs, ys, yerr=yerr, fmt="o-", markersize=6, capsize=3, linewidth=1.6,
                    color=colors[sv], ecolor=colors[sv], alpha=0.9, label=_series_label(series_col, sv))
    ax.set_ylim(bottom=0)
    _style_axis(ax, "Average utilization", x_label, "Mean over live (edge, h) slots")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    for sv in series_vals:
        pts = [r for r in rows if r[series_col] == sv]
        xs = [r[x_col] for r in pts]
        ys = [r["feasible_rate"] for r in pts]
        ax.plot(xs, ys, "o-", markersize=6, linewidth=1.6, color=colors[sv], alpha=0.9,
                label=_series_label(series_col, sv))
    ax.set_ylim(-0.05, 1.1)
    _style_axis(ax, "MILP feasibility rate", x_label, "Fraction solved feasibly within time limit")
    ax.legend(loc="lower left", fontsize=8)

    fig.suptitle(f"Network load: {topology_label} ({x_col} x {series_col})",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.text(0.5, -0.04,
              "A missing point = every seed at that cell failed to even generate (no feasible "
              "initial routing at h=0) -- the network can't carry that much traffic at all.",
              ha="center", fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    out_path = out_dir / f"network_load_{topology_label}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_solver_metrics(
    rows: List[dict], out_dir: Path, topology_label: str,
    x_col: str = "n_commodities", x_label: Optional[str] = None, series_col: str = "u_fraction",
    time_limit_line: Optional[float] = 3600.0,
):
    x_label = x_label or x_col
    rows = [r for r in rows if x_col in r and series_col in r]
    rows = sorted(rows, key=lambda r: (r[series_col], r[x_col]))
    if not rows:
        print(f"[skip] solver_{topology_label}.png: no rows in CSV")
        return

    series_vals = sorted({r[series_col] for r in rows})
    colors = _series_colors(rows, series_col)
    has_model_size = _has_data(rows, "num_constrs_mean")

    fig, axes = plt.subplots(1, 3 if has_model_size else 2, figsize=(16, 4.5) if has_model_size else (11, 4.5))

    ax = axes[0]
    for sv in series_vals:
        pts = [r for r in rows if r[series_col] == sv and not _is_nan(r.get("runtime_mean"))]
        xs = [r[x_col] for r in pts]
        ys = [max(r["runtime_mean"], 1e-2) for r in pts]
        yerr = [r["runtime_ci95"] for r in pts]
        ax.errorbar(xs, ys, yerr=yerr, fmt="o-", markersize=6, capsize=3, linewidth=1.6,
                    color=colors[sv], ecolor=colors[sv], alpha=0.9, label=_series_label(series_col, sv))
    ax.set_yscale("log")
    if time_limit_line:
        ax.axhline(time_limit_line, color=INK_MUTED, linestyle="--", linewidth=1, alpha=0.6)
        ax.text(rows[0][x_col], time_limit_line, " time limit", fontsize=7, color=INK_MUTED, va="bottom")
    _style_axis(ax, "Solve time", x_label, "Runtime (s, log)")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    for sv in series_vals:
        pts = [r for r in rows if r[series_col] == sv and r["feasible_rate"] > 0
               and not _is_nan(r.get("mip_gap_mean"))]
        xs = [r[x_col] for r in pts]
        ys = [r["mip_gap_mean"] for r in pts]
        yerr = [r["mip_gap_ci95"] for r in pts]
        ax.errorbar(xs, ys, yerr=yerr, fmt="o-", markersize=6, capsize=3, linewidth=1.6,
                    color=colors[sv], ecolor=colors[sv], alpha=0.9, label=_series_label(series_col, sv))
    ax.set_ylim(bottom=-0.02)
    _style_axis(ax, "MIP gap (feasible only)", x_label, "Optimality gap")
    ax.legend(loc="upper left", fontsize=8)

    if has_model_size:
        ax = axes[2]
        for sv in series_vals:
            pts = [r for r in rows if r[series_col] == sv and not _is_nan(r.get("num_constrs_mean"))]
            xs = [r[x_col] for r in pts]
            ys = [r["num_constrs_mean"] for r in pts]
            yerr = [r["num_constrs_ci95"] for r in pts]
            ax.errorbar(xs, ys, yerr=yerr, fmt="o-", markersize=6, capsize=3, linewidth=1.6,
                        color=colors[sv], ecolor=colors[sv], alpha=0.9, label=_series_label(series_col, sv))
        _style_axis(ax, "Model size", x_label, "Number of constraints")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"Solver metrics: {topology_label} ({x_col} x {series_col})",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = out_dir / f"solver_{topology_label}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render network-load and solver-diagnostic charts from a summary CSV")
    parser.add_argument("--summary-csv", type=str, required=True)
    parser.add_argument("--topology-label", type=str, required=True,
                         help="used in the title and output filenames, e.g. 'dt12' or 'all'")
    parser.add_argument("--out-dir", type=str, default="results/figures")
    parser.add_argument("--x-col", type=str, default="n_commodities",
                         help="numeric column for the x-axis (default: n_commodities)")
    parser.add_argument("--x-label", type=str, default=None,
                         help="axis label for --x-col (default: the column name)")
    parser.add_argument("--series-col", type=str, default="u_fraction",
                         help="column used to color/split series (default: u_fraction)")
    parser.add_argument("--time-limit-line", type=float, default=3600.0,
                         help="draw a dashed reference line at this runtime (s) on the solver-time "
                              "panel; pass 0 to disable")
    args = parser.parse_args()

    rows = load_summary(args.summary_csv)
    if not rows:
        print(f"No rows in {args.summary_csv}; nothing to plot.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_network_load(rows, out_dir, args.topology_label,
                       x_col=args.x_col, x_label=args.x_label, series_col=args.series_col)
    plot_solver_metrics(rows, out_dir, args.topology_label,
                         x_col=args.x_col, x_label=args.x_label, series_col=args.series_col,
                         time_limit_line=args.time_limit_line or None)


if __name__ == "__main__":
    main()
