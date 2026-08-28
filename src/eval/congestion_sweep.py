"""
Congestion sweep: for a FIXED topology (either a synthetic erdos_renyi graph
of a given size, or a real topology `test5`/`dt12`), sweep over u_fraction
and n_commodities to see how network load (bottleneck/avg utilization, see
_network_load in milp_makespan.py) grows with traffic demand, at different
fractions of upgradeable links. Aggregates mean + 95% CI across seeds,
grouped by (u_fraction, n_commodities).

Unlike scalability_sweep.py (which grows n_nodes/|E| to test model size),
here the graph is held fixed and only demand (n_commodities) and upgrade
coverage (u_fraction) vary -- this tests actual network congestion, not
solver scale.

Pick --n-commodities-list per topology's capacity scale: synthetic defaults
to small absolute capacities (saturates at single-digit commodity counts)
unless --real-capacity-params is set, while real topologies use
BASE_CHANNEL_CAPACITY=80 (needs tens of commodities to saturate). A cell
where every seed fails to even generate an instance (no feasible initial
routing at h=0 -- the network can't carry that much traffic before any
upgrade) has n_samples=0 and is silently absent from the output CSV; that
absence is itself the "hit the wall" signal, also printed during generation.

Usage:
    # synthetic, fixed n_nodes/edge_factor
    python -m src.eval.congestion_sweep --topology synthetic --n-nodes 20 --edge-factor 2.0 \
        --u-fraction-list 0.3 0.5 0.7 0.9 1.0 --n-commodities-list 1 2 3 4 5 6 7 8 9 10 \
        --n-seeds 3 --gurobi-license gurobi.lic

    # real topology
    python -m src.eval.congestion_sweep --topology dt12 \
        --u-fraction-list 0.3 0.5 0.7 0.9 1.0 --n-commodities-list 10 20 30 45 60 75 90 \
        --n-seeds 3 --gurobi-license gurobi.lic
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import src.configs.params as params
from src.core.instance import save
from src.core.runner import runner
from src.data.instance_generator import generate_instance
from src.data.real_instance_generator import generate_real_instance
from src.eval.scalability_sweep import _mean_ci95, _u_fraction_bucket

REAL_TOPOLOGIES = params.REAL_TOPOLOGIES  # ("test5", "dt12")


def generate_sweep_instances(
    topology: str,
    n_nodes: int,
    edge_factor: float,
    u_fraction_list: List[float],
    n_commodities_list: List[int],
    n_seeds: int,
    seed_start: int,
    out_dir: Path,
    m_weight: float,
    real_capacity_params: bool = False,
) -> List[str]:
    paths = []
    out_dir.mkdir(parents=True, exist_ok=True)
    if topology == "synthetic":
        target_edges = max(n_nodes - 1, round(edge_factor * n_nodes))

    for u_fraction in u_fraction_list:
        for n_commodities in n_commodities_list:
            n_ok, n_fail = 0, 0
            for i in range(n_seeds):
                seed = seed_start + i
                try:
                    if topology == "synthetic":
                        inst = generate_instance(
                            graph_type="erdos_renyi", n_nodes=n_nodes, target_edges=target_edges,
                            u_fraction=u_fraction, n_commodities=n_commodities, seed=seed, m_weight=m_weight,
                            real_capacity_params=real_capacity_params,
                        )
                    else:
                        inst = generate_real_instance(
                            topology=topology, u_fraction=u_fraction, n_commodities=n_commodities,
                            seed=seed, m_weight=m_weight,
                        )
                except RuntimeError:
                    n_fail += 1
                    continue
                # inst.name (from generate_instance/generate_real_instance) does NOT
                # encode n_commodities -- at fixed (u_fraction, seed) it collides
                # across every n_commodities value, silently overwriting earlier
                # instance/partial files. Must disambiguate here.
                inst.name = f"cong_{topology}_{inst.name}_k{n_commodities}"
                path = out_dir / f"{inst.name}.json"
                save(inst, path)
                paths.append(str(path))
                n_ok += 1
            status = "" if n_fail == 0 else f"  ({n_fail}/{n_seeds} gen-failed: no feasible initial routing)"
            print(f"u_fraction={u_fraction:g} n_commodities={n_commodities:3d}: {n_ok}/{n_seeds} generated{status}")
    return paths


# A solve running under severe memory pressure can fall into OS-level swap
# thrashing: individual B&B nodes that would normally take milliseconds take
# minutes, and Gurobi's TimeLimit is only checked between such steps, so the
# reported runtime can overshoot the requested time_limit by many hours (one
# dt12 sweep cell had runtime=28755s and 56720s against a 3600s time_limit).
# That's a measurement artifact of the host running out of RAM, not the
# solver's actual behavior -- treat any runtime more than ~1.5x the largest
# sane budget used in this project (3600s) as contaminated and drop it from
# the mean/CI, same way a NaN/None reading would be dropped.
_RUNTIME_SANITY_CAP = 5400.0


def aggregate_by_scenario(partial_dir: str, u_fraction_candidates: List[float]) -> List[dict]:
    records = []
    for path in glob.glob(str(Path(partial_dir) / "**" / "partial_*.json"), recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            try:
                records.append(json.load(f))
            except json.JSONDecodeError:
                print(f"[warn] skipping malformed JSON: {path}")

    groups: Dict[Tuple[float, int], List[dict]] = {}
    for r in records:
        key = (_u_fraction_bucket(r, u_fraction_candidates), int(r["n_commodities"]))
        groups.setdefault(key, []).append(r)

    rows = []
    for (u_fraction, n_commodities), recs in sorted(groups.items()):
        n = len(recs)
        feasible = [r for r in recs if r.get("feasible")]
        feasible_rate = len(feasible) / n if n else float("nan")

        def col(name, source=feasible):
            return _mean_ci95([r[name] for r in source if r.get(name) is not None])

        makespan_mean, makespan_ci = col("makespan")
        runtime_mean, runtime_ci = _mean_ci95(
            [r["runtime"] for r in recs if r.get("runtime") is not None and r["runtime"] <= _RUNTIME_SANITY_CAP]
        )
        gap_mean, gap_ci = col("mip_gap", feasible)
        bottleneck_mean, bottleneck_ci = col("network_load_max")
        avg_util_mean, avg_util_ci = col("network_load_mean")
        # model-size fields are present on every record regardless of feasibility
        nodecount_mean, nodecount_ci = col("node_count", recs)
        numconstrs_mean, numconstrs_ci = col("num_constrs", recs)
        numvars_mean, numvars_ci = col("num_vars", recs)
        # obj_bound/work/iter_count were added mid-experiment: only present on a
        # subset of records (n_samples above still reflects the full group size)
        objbound_mean, objbound_ci = col("obj_bound", recs)
        work_mean, work_ci = col("work", recs)
        itercount_mean, itercount_ci = col("iter_count", recs)

        rows.append({
            "u_fraction": u_fraction,
            "n_commodities": n_commodities,
            "n_samples": n,
            "instance_names": "|".join(sorted({r.get("instance_name", "?") for r in recs})),
            "feasible_rate": feasible_rate,
            "makespan_mean": makespan_mean, "makespan_ci95": makespan_ci,
            "runtime_mean": runtime_mean, "runtime_ci95": runtime_ci,
            "mip_gap_mean": gap_mean, "mip_gap_ci95": gap_ci,
            "bottleneck_util_mean": bottleneck_mean, "bottleneck_util_ci95": bottleneck_ci,
            "avg_util_mean": avg_util_mean, "avg_util_ci95": avg_util_ci,
            "node_count_mean": nodecount_mean, "node_count_ci95": nodecount_ci,
            "num_constrs_mean": numconstrs_mean, "num_constrs_ci95": numconstrs_ci,
            "num_vars_mean": numvars_mean, "num_vars_ci95": numvars_ci,
            "obj_bound_mean": objbound_mean, "obj_bound_ci95": objbound_ci,
            "work_mean": work_mean, "work_ci95": work_ci,
            "iter_count_mean": itercount_mean, "iter_count_ci95": itercount_ci,
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
    print(f"Wrote {out_path} ({len(rows)} (u_fraction, n_commodities) rows)")


def main():
    parser = argparse.ArgumentParser(description="Congestion sweep: fixed topology, vary u_fraction x n_commodities")
    parser.add_argument("--topology", choices=["synthetic", *REAL_TOPOLOGIES], required=True)
    parser.add_argument("--n-nodes", type=int, default=20, help="synthetic only")
    parser.add_argument("--edge-factor", type=float, default=2.0, help="synthetic only: |E| ~= edge_factor * n_nodes")
    parser.add_argument("--real-capacity-params", action="store_true",
                         help="synthetic only: use BASE_CHANNEL_CAPACITY + UPGRADE_TYPES for "
                              "u0/u_fixed/u1/L, matching the real topology generator exactly "
                              "(same capacity per link and per upgrade), instead of the "
                              "synthetic generator's own randomized capacity ranges")
    parser.add_argument("--u-fraction-list", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9, 1.0])
    parser.add_argument("--n-commodities-list", type=int, nargs="+", required=True)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--m-weight", type=float, default=1.0)
    parser.add_argument("--time-limit", type=float, default=45.0)
    parser.add_argument("--max-memory", type=float, default=2.0)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--gurobi-license", type=str, default=params.DEFAULT_GUROBI_LICENSE)
    parser.add_argument("--instances-dir", type=str, default=None)
    parser.add_argument("--partial-dir", type=str, default=None)
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--need-checkpoint", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    instances_dir = args.instances_dir or f"data/instances/congestion_{args.topology}"
    partial_dir = args.partial_dir or f"results/partial_results/congestion_{args.topology}"
    out_csv = args.out_csv or f"results/aggregated/congestion_{args.topology}.csv"

    print(f"==> Generating instances: topology={args.topology}, u_fraction in {args.u_fraction_list}, "
          f"n_commodities in {args.n_commodities_list}, {args.n_seeds} seeds each")
    instance_paths = generate_sweep_instances(
        topology=args.topology,
        n_nodes=args.n_nodes,
        edge_factor=args.edge_factor,
        u_fraction_list=args.u_fraction_list,
        n_commodities_list=args.n_commodities_list,
        n_seeds=args.n_seeds,
        seed_start=args.seed_start,
        out_dir=Path(instances_dir),
        m_weight=args.m_weight,
        real_capacity_params=args.real_capacity_params,
    )

    print(f"==> Solving {len(instance_paths)} instances at M={args.m_weight} "
          f"(parallel={args.parallel}, max_memory={args.max_memory}GB/worker)")
    runner(
        instances=instance_paths,
        m_values=[args.m_weight],
        time_limit=args.time_limit,
        max_memory=args.max_memory,
        gurobi_license=args.gurobi_license,
        out_dir=partial_dir,
        need_checkpoint=bool(args.need_checkpoint),
        parallel=args.parallel,
    )

    print(f"==> Aggregating by (u_fraction, n_commodities) -> {out_csv}")
    rows = aggregate_by_scenario(partial_dir, args.u_fraction_list)
    write_csv(rows, out_csv)


if __name__ == "__main__":
    main()
