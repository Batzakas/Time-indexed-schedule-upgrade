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
        runtime_mean, runtime_ci = _mean_ci95([r["runtime"] for r in recs])
        gaps = [r["mip_gap"] for r in feasible if r.get("mip_gap") is not None]
        gap_mean, gap_ci = _mean_ci95(gaps)

        rows.append({
            "graph_type": graph_type,
            "n_nodes": n_nodes,
            "n_upgradeable": n_upg,
            "M": M,
            "Hmax": hmax,
            "n_samples": n,
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
