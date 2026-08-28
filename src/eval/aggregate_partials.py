"""
Aggregate partial per-instance JSON results into a summary CSV. For partial_*.json, group replicate seeds of the same (graph_type, n_nodes,
n_upgradeable, M, Hmax) scenario, and report mean + 95% CI.

Usage:
    python -m src.eval.aggregate_partials --partial-dir results/partial_results \
        --out-csv results/aggregated/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

GROUP_KEY = Tuple[str, int, int, float, int]  # graph_type, n_nodes, n_upgradeable, M, Hmax


def _mean_ci95(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    ci95 = 1.96 * std / math.sqrt(n)
    return mean, ci95


def load_partials(partial_dir: str) -> List[dict]:
    records = []
    for path in glob.glob(str(Path(partial_dir) / "**" / "partial_*.json"), recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            try:
                records.append(json.load(f))
            except json.JSONDecodeError:
                print(f"[warn] skipping malformed JSON: {path}")
    return records


def group_records(records: List[dict]) -> Dict[GROUP_KEY, List[dict]]:
    groups: Dict[GROUP_KEY, List[dict]] = {}
    for r in records:
        key = (
            r.get("graph_type"),
            int(r.get("n_nodes", 0)),
            int(r.get("n_upgradeable", 0)),
            float(r.get("M", 0.0)),
            int(r.get("Hmax", 0)),
        )
        groups.setdefault(key, []).append(r)
    return groups


def _col(recs: List[dict], name: str, source: List[dict] | None = None) -> Tuple[float, float]:
    """mean+CI95 of `name` over `source` (defaults to `recs`), skipping records
    where the field is absent -- lets fields added mid-experiment (e.g.
    num_constrs) coexist with older partials that predate them."""
    source = recs if source is None else source
    return _mean_ci95([r[name] for r in source if r.get(name) is not None])


def summarize(groups: Dict[GROUP_KEY, List[dict]]) -> List[dict]:
    rows = []
    for (graph_type, n_nodes, n_upg, M, hmax), recs in sorted(groups.items()):
        n = len(recs)
        feasible = [r for r in recs if r.get("feasible")]
        feasible_rate = len(feasible) / n if n else float("nan")

        makespan_mean, makespan_ci = _mean_ci95([r["makespan"] for r in feasible])
        obj_mean, obj_ci = _mean_ci95([r["objective"] for r in feasible])
        batches_mean, batches_ci = _mean_ci95([r["n_batches_emergent"] for r in feasible])
        reroutes_mean, reroutes_ci = _mean_ci95([r["total_reroutes"] for r in feasible])
        runtime_mean, runtime_ci = _col(recs, "runtime")
        gaps = [r["mip_gap"] for r in feasible if r.get("mip_gap") is not None]
        gap_mean, gap_ci = _mean_ci95(gaps)

        # absent on partials generated before network_load_* existed (see _network_load in milp_makespan.py)
        bottleneck_vals = [r["network_load_max"] for r in feasible if r.get("network_load_max") is not None]
        bottleneck_mean, bottleneck_ci = _mean_ci95(bottleneck_vals)
        avg_util_vals = [r["network_load_mean"] for r in feasible if r.get("network_load_mean") is not None]
        avg_util_mean, avg_util_ci = _mean_ci95(avg_util_vals)

        # model-size fields are present on every record regardless of feasibility
        node_count_mean, node_count_ci = _col(recs, "node_count")
        num_constrs_mean, num_constrs_ci = _col(recs, "num_constrs")
        num_vars_mean, num_vars_ci = _col(recs, "num_vars")

        # n_edges/n_commodities are constant within a group (same graph_type/
        # n_nodes/n_upgradeable/M/Hmax scenario), so any record's value works;
        # u_fraction is derived rather than stored directly on partials.
        n_edges = int(recs[0].get("n_edges") or 0)
        n_commodities = int(recs[0].get("n_commodities") or 0)
        u_fraction = (n_upg / n_edges) if n_edges else float("nan")

        # usually one seed per row: Hmax (in the group key) depends on each
        # seed's random L_e draw, so seeds rarely land in the same group
        instance_names = "|".join(sorted({r.get("instance_name", "?") for r in recs}))

        rows.append({
            "graph_type": graph_type,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_upgradeable": n_upg,
            "u_fraction": u_fraction,
            "n_commodities": n_commodities,
            "M": M,
            "Hmax": hmax,
            "n_samples": n,
            "instance_names": instance_names,
            "feasible_rate": feasible_rate,
            "makespan_mean": makespan_mean,
            "makespan_ci95": makespan_ci,
            "objective_mean": obj_mean,
            "objective_ci95": obj_ci,
            "n_batches_mean": batches_mean,
            "n_batches_ci95": batches_ci,
            "reroutes_mean": reroutes_mean,
            "reroutes_ci95": reroutes_ci,
            "runtime_mean": runtime_mean,
            "runtime_ci95": runtime_ci,
            "mip_gap_mean": gap_mean,
            "mip_gap_ci95": gap_ci,
            "bottleneck_util_mean": bottleneck_mean,
            "bottleneck_util_ci95": bottleneck_ci,
            "avg_util_mean": avg_util_mean,
            "avg_util_ci95": avg_util_ci,
            "node_count_mean": node_count_mean,
            "node_count_ci95": node_count_ci,
            "num_constrs_mean": num_constrs_mean,
            "num_constrs_ci95": num_constrs_ci,
            "num_vars_mean": num_vars_mean,
            "num_vars_ci95": num_vars_ci,
        })
    return rows


def write_csv(rows: List[dict], out_csv: str) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("No records found; nothing to write.")
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} scenario rows)")


def main():
    parser = argparse.ArgumentParser(description="Aggregate partial MILP results into a summary CSV")
    parser.add_argument("--partial-dir", type=str, default="results/partial_results")
    parser.add_argument("--out-csv", type=str, default="results/aggregated/summary.csv")
    args = parser.parse_args()

    records = load_partials(args.partial_dir)
    print(f"Loaded {len(records)} partial result files from {args.partial_dir}")
    groups = group_records(records)
    rows = summarize(groups)
    write_csv(rows, args.out_csv)


if __name__ == "__main__":
    main()
