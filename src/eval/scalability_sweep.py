"""
Scalability sweep for synthetic (erdos_renyi) instances: grow n_nodes over a
list of sizes, with |E| pinned to a function of n_nodes (default 2*n_nodes,
via --edge-factor), cross with a list of u_fraction values (default 0.3 and
0.5), generate several seeds per (n_nodes, u_fraction) cell, solve them all
at a single fixed M, and aggregate mean + 95% CI *across seeds, grouped by
(n_nodes, u_fraction)* (unlike aggregate_partials.py, which also groups by
Hmax and so almost never actually averages over seeds -- here Hmax is left
free to vary per seed on purpose, since it's an outcome of the random
instance, not a knob we're sweeping; u_fraction is recovered per-record from
n_upgradeable/n_edges rather than needing a schema change).

--parallel defaults to a conservative 4 workers -- an earlier unbounded run
(parallel=cpu_count()-1=11) let a single large instance's solve balloon past
available RAM and got OOM-killed by the kernel, silently taking the whole
sweep down with it.

Usage:
    python -m src.eval.scalability_sweep --n-nodes-list 6 8 10 12 16 20 24 \
        --edge-factor 2.0 --u-fraction-list 0.3 0.5 --n-seeds 5 \
        --gurobi-license gurobi.lic
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import src.configs.params as params
from src.core.instance import save
from src.core.runner import runner
from src.data.instance_generator import generate_instance

DEFAULT_INSTANCES_DIR = "data/instances/scalability"
DEFAULT_PARTIAL_DIR = "results/partial_results/scalability"
DEFAULT_OUT_CSV = "results/aggregated/scalability.csv"


def generate_sweep_instances(
    n_nodes_list: List[int],
    edge_factor: float,
    u_fraction_list: List[float],
    n_commodities: int,
    n_seeds: int,
    seed_start: int,
    out_dir: Path,
    m_weight: float,
) -> List[str]:
    paths = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for n_nodes in n_nodes_list:
        target_edges = max(n_nodes - 1, round(edge_factor * n_nodes))
        for u_fraction in u_fraction_list:
            for i in range(n_seeds):
                seed = seed_start + i
                inst = generate_instance(
                    graph_type="erdos_renyi",
                    n_nodes=n_nodes,
                    target_edges=target_edges,
                    u_fraction=u_fraction,
                    n_commodities=n_commodities,
                    seed=seed,
                    m_weight=m_weight,
                )
                # disambiguate from data/instances/ default erdos_renyi files
                inst.name = f"scal_{inst.name}"
                path = out_dir / f"{inst.name}.json"
                save(inst, path)
                paths.append(str(path))
                print(f"Wrote {path}  (|V|={inst.n_nodes}, |E|={len(inst.edges)}, "
                      f"|U|={len(inst.U)}, u_fraction={u_fraction}, Hmax={inst.Hmax})")
    return paths


def _mean_ci95(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    ci95 = 1.96 * math.sqrt(var) / math.sqrt(n)
    return mean, ci95


def _u_fraction_bucket(r: dict, candidates: List[float]) -> float:
    """Recover the intended u_fraction from the recorded n_upgradeable/n_edges
    ratio (both already written by model_tester.py into every partial JSON),
    snapped to the nearest value actually passed via --u-fraction-list.
    Snapping to the nearest *requested* value (rather than a fixed rounding
    grid, e.g. nearest 0.05) matters for small graphs: e.g. n_edges=12,
    u_fraction=0.3 -> round(0.3*12)=4 -> ratio=4/12=0.333, which is closer to
    a 0.05-grid bucket of 0.35 than to the 0.3 it actually came from."""
    n_edges = r.get("n_edges") or 1
    ratio = r.get("n_upgradeable", 0) / n_edges
    return min(candidates, key=lambda c: abs(c - ratio))


def aggregate_by_scenario(partial_dir: str, u_fraction_candidates: List[float]) -> List[dict]:
    records = []
    for path in glob.glob(str(Path(partial_dir) / "**" / "partial_*.json"), recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            try:
                records.append(json.load(f))
            except json.JSONDecodeError:
                print(f"[warn] skipping malformed JSON: {path}")

    groups: Dict[Tuple[int, float], List[dict]] = {}
    for r in records:
        key = (int(r["n_nodes"]), _u_fraction_bucket(r, u_fraction_candidates))
        groups.setdefault(key, []).append(r)

    rows = []
    for (n_nodes, u_fraction), recs in sorted(groups.items()):
        n = len(recs)
        feasible = [r for r in recs if r.get("feasible")]
        feasible_rate = len(feasible) / n if n else float("nan")

        def col(name, source=feasible):
            return _mean_ci95([r[name] for r in source if r.get(name) is not None])

        n_edges_mean, n_edges_ci = col("n_edges", recs)
        n_upg_mean, n_upg_ci = col("n_upgradeable", recs)
        hmax_mean, hmax_ci = col("Hmax", recs)
        runtime_mean, runtime_ci = col("runtime", recs)
        numvars_mean, numvars_ci = col("num_vars", recs)
        numconstrs_mean, numconstrs_ci = col("num_constrs", recs)
        makespan_mean, makespan_ci = col("makespan")
        reroutes_mean, reroutes_ci = col("total_reroutes")
        gap_mean, gap_ci = col("mip_gap", feasible)
        bottleneck_mean, bottleneck_ci = col("network_load_max")
        avg_util_mean, avg_util_ci = col("network_load_mean")

        rows.append({
            "n_nodes": n_nodes,
            "u_fraction": u_fraction,
            "n_samples": n,
            "instance_names": "|".join(sorted({r.get("instance_name", "?") for r in recs})),
            "n_edges_mean": n_edges_mean, "n_edges_ci95": n_edges_ci,
            "n_upgradeable_mean": n_upg_mean, "n_upgradeable_ci95": n_upg_ci,
            "Hmax_mean": hmax_mean, "Hmax_ci95": hmax_ci,
            "feasible_rate": feasible_rate,
            "makespan_mean": makespan_mean, "makespan_ci95": makespan_ci,
            "reroutes_mean": reroutes_mean, "reroutes_ci95": reroutes_ci,
            "runtime_mean": runtime_mean, "runtime_ci95": runtime_ci,
            "num_vars_mean": numvars_mean, "num_vars_ci95": numvars_ci,
            "num_constrs_mean": numconstrs_mean, "num_constrs_ci95": numconstrs_ci,
            "mip_gap_mean": gap_mean, "mip_gap_ci95": gap_ci,
            "bottleneck_util_mean": bottleneck_mean, "bottleneck_util_ci95": bottleneck_ci,
            "avg_util_mean": avg_util_mean, "avg_util_ci95": avg_util_ci,
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
    print(f"Wrote {out_path} ({len(rows)} n_nodes rows)")


def main():
    parser = argparse.ArgumentParser(description="Synthetic-topology scalability sweep (grow n_nodes, |E| = edge_factor * n_nodes)")
    parser.add_argument("--n-nodes-list", type=int, nargs="+", default=[6, 8, 10, 12, 16, 20, 24])
    parser.add_argument("--edge-factor", type=float, default=2.0, help="|E| ~= edge_factor * n_nodes")
    parser.add_argument("--u-fraction-list", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--n-commodities", type=int, default=params.DEFAULT_N_COMMODITIES)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--m-weight", type=float, default=1.0)
    parser.add_argument("--time-limit", type=float, default=180.0)
    parser.add_argument("--parallel", type=int, default=4,
                         help="worker processes; kept low by default to avoid a single large "
                              "instance's solve exhausting RAM and OOM-killing the whole sweep")
    parser.add_argument("--gurobi-license", type=str, default=params.DEFAULT_GUROBI_LICENSE)
    parser.add_argument("--instances-dir", type=str, default=DEFAULT_INSTANCES_DIR)
    parser.add_argument("--partial-dir", type=str, default=DEFAULT_PARTIAL_DIR)
    parser.add_argument("--out-csv", type=str, default=DEFAULT_OUT_CSV)
    parser.add_argument("--need-checkpoint", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    print(f"==> Generating instances for n_nodes={args.n_nodes_list}, "
          f"|E|~={args.edge_factor}*n_nodes, u_fraction in {args.u_fraction_list}, "
          f"{args.n_seeds} seeds each")
    instance_paths = generate_sweep_instances(
        n_nodes_list=args.n_nodes_list,
        edge_factor=args.edge_factor,
        u_fraction_list=args.u_fraction_list,
        n_commodities=args.n_commodities,
        n_seeds=args.n_seeds,
        seed_start=args.seed_start,
        out_dir=Path(args.instances_dir),
        m_weight=args.m_weight,
    )

    print(f"==> Solving {len(instance_paths)} instances at M={args.m_weight} "
          f"(parallel={args.parallel})")
    runner(
        instances=instance_paths,
        m_values=[args.m_weight],
        time_limit=args.time_limit,
        gurobi_license=args.gurobi_license,
        out_dir=args.partial_dir,
        need_checkpoint=bool(args.need_checkpoint),
        parallel=args.parallel,
    )

    print(f"==> Aggregating by (n_nodes, u_fraction) -> {args.out_csv}")
    rows = aggregate_by_scenario(args.partial_dir, args.u_fraction_list)
    write_csv(rows, args.out_csv)


if __name__ == "__main__":
    main()
