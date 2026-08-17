"""
Turns results/aggregated/summary.csv (see aggregate_partials.py) into a
fixed set of PNG charts under results/figures/:

  - tradeoff.png       makespan and reroute count vs the reroute-penalty
                        weight M, one color per graph_type/topology.
                        (Two scenarios can share graph_type+M but differ in
                        Hmax -- different upgrade-type draws per seed give
                        a different worst-case Hmax -- so points are
                        plotted individually rather than connected as a
                        line.)
  - scalability.png    solve runtime vs Hmax (log-scale y), one color per
                        graph_type/topology -- how the model's difficulty
                        grows with the size of the time horizon.
  - diagnostics.png    feasible_rate and mean MIP gap per topology, as a
                        sanity check that the sweep actually solved cleanly.

Usage:
    python -m src.eval.graficos --summary-csv results/aggregated/summary.csv \
        --out-dir results/figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

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
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"

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
            if k == "graph_type":
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


def _topology_colors(rows: List[dict]) -> Dict[str, str]:
    topologies = sorted({r["graph_type"] for r in rows})
    if len(topologies) > len(CATEGORICAL):
        print(f"[warn] {len(topologies)} topologies but only {len(CATEGORICAL)} "
              f"categorical slots defined; extra ones will repeat colors.")
    return {t: CATEGORICAL[i % len(CATEGORICAL)] for i, t in enumerate(topologies)}


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


def plot_tradeoff(rows: List[dict], colors: Dict[str, str], out_dir: Path):
    feasible = [r for r in rows if r["feasible_rate"] > 0]
    if not feasible:
        print("[skip] tradeoff.png: no feasible scenarios")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    seen_labels = set()

    for ax, ycol, ycicol, title, ylabel in (
        (axes[0], "makespan_mean", "makespan_ci95", "Makespan vs. reroute penalty", "Makespan (weeks)"),
        (axes[1], "reroutes_mean", "reroutes_ci95", "Reroutes vs. reroute penalty", "Total reroutes"),
    ):
        for topo in sorted(colors):
            pts = [r for r in feasible if r["graph_type"] == topo]
            if not pts:
                continue
            # small horizontal jitter so scenarios that share (topology, M)
            # but differ in Hmax don't fully overlap
            n = len(pts)
            for i, r in enumerate(sorted(pts, key=lambda r: r["M"])):
                jitter = (i % 3 - 1) * 0.02 * max(r["M"], 1.0)
                label = topo if topo not in seen_labels else None
                ax.errorbar(
                    r["M"] + jitter, r[ycol], yerr=r[ycicol],
                    fmt="o", markersize=6, capsize=3, linewidth=1.4,
                    color=colors[topo], ecolor=colors[topo], alpha=0.9,
                    label=label,
                )
                ax.annotate(
                    _seed_label(r.get("instance_names", "")),
                    (r["M"] + jitter, r[ycol]), textcoords="offset points",
                    xytext=(5, 4), fontsize=7, color=INK_MUTED,
                )
                if label:
                    seen_labels.add(topo)
        _style_axis(ax, title, "M (reroute penalty weight)", ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               bbox_to_anchor=(0.5, -0.04))
    fig.text(0.5, -0.09, "labels = seed (data/instances/<topology>_..._<seed>.json)",
              ha="center", fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = out_dir / "tradeoff.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_scalability(rows: List[dict], colors: Dict[str, str], out_dir: Path):
    feasible = [r for r in rows if r["feasible_rate"] > 0]
    if not feasible:
        print("[skip] scalability.png: no feasible scenarios")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for topo in sorted(colors):
        pts = [r for r in feasible if r["graph_type"] == topo]
        if not pts:
            continue
        xs = [r["Hmax"] for r in pts]
        ys = [max(r["runtime_mean"], 1e-4) for r in pts]
        ax.scatter(xs, ys, s=48, color=colors[topo], alpha=0.9, label=topo,
                   edgecolors=SURFACE, linewidths=0.6)
        for r, x, y in zip(pts, xs, ys):
            ax.annotate(
                _seed_label(r.get("instance_names", "")), (x, y),
                textcoords="offset points", xytext=(5, 4), fontsize=7, color=INK_MUTED,
            )

    ax.set_yscale("log")
    _style_axis(ax, "Solve time vs. time-horizon size", "Hmax (weeks)", "Runtime (s, log scale)")
    ax.legend(loc="upper left", ncol=1)
    fig.text(0.5, -0.02, "labels = seed (data/instances/<topology>_..._<seed>.json)",
              ha="center", fontsize=8, color=INK_MUTED)
    fig.tight_layout()
    out_path = out_dir / "scalability.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_diagnostics(rows: List[dict], colors: Dict[str, str], out_dir: Path):
    topologies = sorted(colors)
    feas_by_topo = {t: [r["feasible_rate"] for r in rows if r["graph_type"] == t] for t in topologies}
    gap_by_topo = {
        t: [r["mip_gap_mean"] for r in rows if r["graph_type"] == t and r["feasible_rate"] > 0]
        for t in topologies
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    ax = axes[0]
    worst_feas = [min(feas_by_topo[t]) if feas_by_topo[t] else 0.0 for t in topologies]
    bar_colors = [STATUS_GOOD if v >= 0.999 else (STATUS_WARNING if v >= 0.5 else STATUS_CRITICAL) for v in worst_feas]
    bars = ax.bar(topologies, worst_feas, color=bar_colors, width=0.55, zorder=3)
    for b, v in zip(bars, worst_feas):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center",
                fontsize=9, color=INK_SECONDARY)
    ax.set_ylim(0, 1.15)
    _style_axis(ax, "Worst-case feasibility rate", "", "Feasible / solved")

    ax = axes[1]
    mean_gap = [sum(gap_by_topo[t]) / len(gap_by_topo[t]) if gap_by_topo[t] else 0.0 for t in topologies]
    bar_colors = [STATUS_GOOD if v <= 1e-6 else (STATUS_WARNING if v <= 0.05 else STATUS_CRITICAL) for v in mean_gap]
    bars = ax.bar(topologies, mean_gap, color=bar_colors, width=0.55, zorder=3)
    for b, v in zip(bars, mean_gap):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1%}", ha="center", va="bottom",
                fontsize=9, color=INK_SECONDARY)
    _style_axis(ax, "Mean MIP optimality gap", "", "Gap")
    ax.set_ylim(bottom=0)  # gap is structurally >= 0; don't let autoscale imply otherwise

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    out_path = out_dir / "diagnostics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Render charts from a summary CSV")
    parser.add_argument("--summary-csv", type=str, default="results/aggregated/summary.csv")
    parser.add_argument("--out-dir", type=str, default="results/figures")
    args = parser.parse_args()

    rows = load_summary(args.summary_csv)
    if not rows:
        print(f"No rows in {args.summary_csv}; nothing to plot.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = _topology_colors(rows)

    plot_tradeoff(rows, colors, out_dir)
    plot_scalability(rows, colors, out_dir)
    plot_diagnostics(rows, colors, out_dir)


if __name__ == "__main__":
    main()
