"""
Single-figure comparison of the two *real* topologies (test5 vs dt12) from
results/aggregated/summary.csv, ignoring the synthetic graph_types
(erdos_renyi, grid). Four panels:

  - makespan vs M
  - reroutes vs M
  - runtime vs Hmax (log scale)
  - feasibility rate / mean MIP gap, side by side per topology

Reuses the palette, chart chrome and helpers from graficos.py so the visual
language matches the other charts.

Usage:
    python -m src.eval.graficos_real --summary-csv results/aggregated/summary.csv \
        --out-dir results/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.graficos import (
    CATEGORICAL, INK_MUTED, INK_SECONDARY, SURFACE,
    STATUS_GOOD, STATUS_WARNING, STATUS_CRITICAL,
    _seed_label, _style_axis, load_summary,
)


def _real_rows(rows):
    return [r for r in rows if str(r["graph_type"]).startswith("real:")]


def plot_real_comparison(rows, out_dir: Path):
    real_rows = _real_rows(rows)
    if not real_rows:
        print("[skip] real_topologies.png: no real:* rows in summary")
        return

    topologies = sorted({r["graph_type"] for r in real_rows})
    colors = {t: CATEGORICAL[i % len(CATEGORICAL)] for i, t in enumerate(topologies)}
    feasible = [r for r in real_rows if r["feasible_rate"] > 0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # -- panel 1: makespan vs M --
    ax = axes[0][0]
    seen = set()
    for topo in topologies:
        pts = [r for r in feasible if r["graph_type"] == topo]
        for i, r in enumerate(sorted(pts, key=lambda r: r["M"])):
            jitter = 1.0 + (i % 3 - 1) * 0.03
            label = topo if topo not in seen else None
            ax.errorbar(
                max(r["M"], 0.1) * jitter, r["makespan_mean"], yerr=r["makespan_ci95"],
                fmt="o", markersize=6, capsize=3, linewidth=1.4,
                color=colors[topo], ecolor=colors[topo], alpha=0.9, label=label,
            )
            ax.annotate(_seed_label(r.get("instance_names", "")),
                        (max(r["M"], 0.1) * jitter, r["makespan_mean"]), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color=INK_MUTED)
            if label:
                seen.add(topo)
    ax.set_xscale("log")
    _style_axis(ax, "Makespan vs. reroute penalty", "M (log scale)", "Makespan (weeks)")

    # -- panel 2: reroutes vs M --
    ax = axes[0][1]
    seen = set()
    for topo in topologies:
        pts = [r for r in feasible if r["graph_type"] == topo]
        for i, r in enumerate(sorted(pts, key=lambda r: r["M"])):
            jitter = 1.0 + (i % 3 - 1) * 0.03
            label = topo if topo not in seen else None
            ax.errorbar(
                max(r["M"], 0.1) * jitter, r["reroutes_mean"], yerr=r["reroutes_ci95"],
                fmt="o", markersize=6, capsize=3, linewidth=1.4,
                color=colors[topo], ecolor=colors[topo], alpha=0.9, label=label,
            )
            ax.annotate(_seed_label(r.get("instance_names", "")),
                        (max(r["M"], 0.1) * jitter, r["reroutes_mean"]), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color=INK_MUTED)
            if label:
                seen.add(topo)
    ax.set_xscale("log")
    _style_axis(ax, "Reroutes vs. reroute penalty", "M (log scale)", "Total reroutes")

    # -- panel 3: runtime vs Hmax --
    ax = axes[1][0]
    for topo in topologies:
        pts = [r for r in feasible if r["graph_type"] == topo]
        xs = [r["Hmax"] for r in pts]
        ys = [max(r["runtime_mean"], 1e-4) for r in pts]
        ax.scatter(xs, ys, s=48, color=colors[topo], alpha=0.9, label=topo,
                   edgecolors=SURFACE, linewidths=0.6)
        for r, x, y in zip(pts, xs, ys):
            ax.annotate(_seed_label(r.get("instance_names", "")), (x, y),
                        textcoords="offset points", xytext=(5, 4), fontsize=7, color=INK_MUTED)
    ax.set_yscale("log")
    _style_axis(ax, "Solve time vs. time-horizon size", "Hmax (weeks)", "Runtime (s, log scale)")

    # -- panel 4: feasibility rate / mean gap, grouped bars per topology --
    ax = axes[1][1]
    x = range(len(topologies))
    width = 0.35
    feas_vals = [min([r["feasible_rate"] for r in real_rows if r["graph_type"] == t], default=0.0)
                 for t in topologies]
    gap_vals = [
        (lambda g: sum(g) / len(g) if g else 0.0)(
            [r["mip_gap_mean"] for r in real_rows if r["graph_type"] == t and r["feasible_rate"] > 0]
        )
        for t in topologies
    ]
    bars1 = ax.bar([i - width / 2 for i in x], feas_vals, width=width,
                    color=STATUS_GOOD, label="worst-case feasibility rate", zorder=3)
    bars2 = ax.bar([i + width / 2 for i in x], gap_vals, width=width,
                    color=STATUS_WARNING, label="mean MIP gap", zorder=3)
    for b, v in zip(bars1, feas_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center",
                fontsize=8, color=INK_SECONDARY)
    for b, v in zip(bars2, gap_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}", ha="center",
                fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(list(x))
    ax.set_xticklabels(topologies)
    ax.set_ylim(0, 1.3)
    _style_axis(ax, "Feasibility & optimality gap", "", "Value")
    ax.legend(loc="upper center", fontsize=8, ncol=2, bbox_to_anchor=(0.5, 1.14))

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Real topologies: test5 vs. dt12", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.text(0.5, -0.045, "labels = seed (data/instances/<topology>_..._<seed>.json)",
              ha="center", fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out_path = out_dir / "real_topologies.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare only the real topologies (test5 vs dt12)")
    parser.add_argument("--summary-csv", type=str, default="results/aggregated/summary.csv")
    parser.add_argument("--out-dir", type=str, default="results/figures")
    args = parser.parse_args()

    rows = load_summary(args.summary_csv)
    if not rows:
        print(f"No rows in {args.summary_csv}; nothing to plot.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_real_comparison(rows, out_dir)


if __name__ == "__main__":
    main()
