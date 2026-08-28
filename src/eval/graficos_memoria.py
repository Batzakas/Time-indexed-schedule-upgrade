"""
Renders an out-of-memory rate chart for a congestion sweep: what fraction of
seeds, at each (u_fraction, n_commodities) cell, failed specifically because
Gurobi exceeded MemLimit

Usage:
    python -m src.eval.graficos_memoria --partial-dir results/partial_results/congestion_dt12 \
        --topology-label dt12 --out-dir results/figures
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.graficos import CATEGORICAL, _style_axis
from src.eval.scalability_sweep import _u_fraction_bucket

_KNOWN_U_FRACTIONS = (0.3, 0.5, 0.7, 0.9, 1.0)


def load_oom_rates(partial_dir: str) -> List[dict]:
    records = []
    for path in glob.glob(str(Path(partial_dir) / "**" / "partial_*.json"), recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            try:
                records.append(json.load(f))
            except json.JSONDecodeError:
                print(f"[warn] skipping malformed JSON: {path}")

    groups: Dict[Tuple[float, int], List[dict]] = {}
    for r in records:
        key = (_u_fraction_bucket(r, list(_KNOWN_U_FRACTIONS)), int(r["n_commodities"]))
        groups.setdefault(key, []).append(r)

    rows = []
    for (u_fraction, n_commodities), recs in sorted(groups.items()):
        n = len(recs)
        n_oom = sum(1 for r in recs if r.get("error") == "Out of memory")
        # mean peak memory, only present on records solved after max_mem_used
        # was added to the pipeline -- absent (None) for older records
        mem_vals = [r["max_mem_used"] for r in recs if r.get("max_mem_used") is not None]
        rows.append({
            "u_fraction": u_fraction,
            "n_commodities": n_commodities,
            "n_samples": n,
            "oom_rate": n_oom / n if n else float("nan"),
            "max_mem_used_mean": (sum(mem_vals) / len(mem_vals)) if mem_vals else None,
            "max_mem_used_n": len(mem_vals),
        })
    return rows


def plot_oom_rate(rows: List[dict], out_dir: Path, topology_label: str):
    rows = sorted(rows, key=lambda r: (r["u_fraction"], r["n_commodities"]))
    if not rows:
        print(f"[skip] memory_{topology_label}.png: no rows")
        return

    u_fractions = sorted({r["u_fraction"] for r in rows})
    colors = {u: CATEGORICAL[i % len(CATEGORICAL)] for i, u in enumerate(u_fractions)}
    has_mem_data = any(r["max_mem_used_n"] > 0 for r in rows)

    fig, axes = plt.subplots(1, 2 if has_mem_data else 1, figsize=(11, 4.5) if has_mem_data else (6, 4.5))
    if not has_mem_data:
        axes = [axes]

    ax = axes[0]
    for u in u_fractions:
        pts = [r for r in rows if r["u_fraction"] == u]
        xs = [r["n_commodities"] for r in pts]
        ys = [r["oom_rate"] for r in pts]
        ax.plot(xs, ys, "o-", markersize=6, linewidth=1.6, color=colors[u], alpha=0.9,
                label=f"u_fraction={u:g}")
    ax.set_ylim(-0.05, 1.05)
    _style_axis(ax, "Out-of-memory rate", "n_commodities", "Fraction of seeds that hit MemLimit")
    ax.legend(loc="upper left", fontsize=8)

    if has_mem_data:
        ax = axes[1]
        for u in u_fractions:
            pts = [r for r in rows if r["u_fraction"] == u and r["max_mem_used_n"] > 0]
            xs = [r["n_commodities"] for r in pts]
            ys = [r["max_mem_used_mean"] for r in pts]
            ax.plot(xs, ys, "o-", markersize=6, linewidth=1.6, color=colors[u], alpha=0.9,
                    label=f"u_fraction={u:g}")
        _style_axis(ax, "Peak memory (partial coverage)", "n_commodities", "max_mem_used (GB)")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"Memory pressure: {topology_label} (fixed topology, u_fraction x n_commodities)",
                  fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.text(0.5, -0.04,
              "OOM rate = fraction of seeds where Gurobi's MemLimit was exceeded (proxy for memory "
              "pressure -- direct peak-memory readings were only added to the pipeline partway "
              "through this sweep, so they cover a subset of records where shown).",
              ha="center", fontsize=8, color="#898781")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    out_path = out_dir / f"memory_{topology_label}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Render an out-of-memory rate / peak-memory chart")
    parser.add_argument("--partial-dir", type=str, required=True)
    parser.add_argument("--topology-label", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="results/figures")
    args = parser.parse_args()

    rows = load_oom_rates(args.partial_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_oom_rate(rows, out_dir, args.topology_label)


if __name__ == "__main__":
    main()
