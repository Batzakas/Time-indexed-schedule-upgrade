"""
Instance generator built from real network topologies (test5, dt12 -- the
same topology files used by Network_Planner_Code's EON simulator, see
src/data/topology_loader.py) instead of synthetic random graphs.

Upgrade durations (L_e) and capacity jumps (u1_e / u0_e) are grounded in
Network_Planner_Code/sim/core/constants.py rather than sampled uniformly at
random -- see the comment block above the UPGRADE_* constants in
src/configs/params.py for exactly how each number was derived. Everything
else (which edges are upgradeable, which node pairs communicate, demand
sizes) is still randomized per seed, since the source project has no
public per-link "this one needs upgrading" ground truth to borrow.

Usage:
    python -m src.data.real_instance_generator --topology dt12 \
        --u-fraction 0.3 --n-commodities 5 --seed 0 --count 3 \
        --out-dir data/instances
"""
from __future__ import annotations

import argparse
import random
from math import ceil
from pathlib import Path
from typing import Tuple

import src.configs.params as params
from src.core.instance import Commodity, Instance, compute_initial_routing, save
from src.data.topology_loader import load_topology

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "instances"


def generate_real_instance(
    topology: str,
    u_fraction: float = params.DEFAULT_REAL_U_FRACTION,
    n_commodities: int = params.DEFAULT_REAL_N_COMMODITIES,
    demand_range: Tuple[float, float] = params.DEFAULT_DEMAND_CHANNELS_RANGE,
    seed: int = params.DEFAULT_SEED,
    hmax_override: int | None = None,
    m_weight: float | None = None,
    max_attempts: int = 25,
) -> Instance:
    """Generate one instance from a real topology. Retries with a perturbed
    seed (same knobs) up to max_attempts times if no feasible joint initial
    routing is found."""
    topo = load_topology(topology)
    n_nodes = topo["n_nodes"]
    edges = topo["edges"]
    length_km = topo["length_km"]
    all_eids = [eid for (_, _, eid) in edges]
    n_edges = len(edges)

    last_err = None
    for attempt in range(max_attempts):
        trial_seed = seed + attempt * 7919
        rng = random.Random(trial_seed)

        u_count = max(1, round(u_fraction * n_edges)) if n_edges > 0 else 0
        u_count = min(u_count, n_edges)
        U = sorted(rng.sample(all_eids, u_count)) if u_count > 0 else []
        U_set = set(U)

        u0, u1, L, upgrade_types = {}, {}, {}, {}
        u_fixed = {}
        for eid in all_eids:
            if eid in U_set:
                utype = rng.choice(params.UPGRADE_TYPES)
                upgrade_types[eid] = utype
                u0[eid] = params.BASE_CHANNEL_CAPACITY
                u1[eid] = params.BASE_CHANNEL_CAPACITY * params.UPGRADE_CAPACITY_MULTIPLIER[utype]
                L[eid] = max(1, ceil(params.UPGRADE_DURATION_DAYS[utype] / params.TIME_UNIT_DAYS))
            else:
                u_fixed[eid] = params.BASE_CHANNEL_CAPACITY

        commodities = []
        for _ in range(n_commodities):
            s, t = rng.sample(range(n_nodes), 2)
            d = rng.uniform(*demand_range)
            commodities.append(Commodity(s=s, t=t, d=d))

        routing = compute_initial_routing(
            n_nodes=n_nodes,
            edges=edges,
            U=U,
            u0=u0,
            u_fixed=u_fixed,
            commodities=commodities,
            rng=rng,
        )
        if routing is None:
            last_err = "no feasible joint initial routing"
            continue

        hmax = hmax_override if hmax_override is not None else (
            sum(L.values()) + params.HMAX_SLACK if L else 1
        )
        M = m_weight if m_weight is not None else params.M_DEFAULT

        u_pct = int(round(u_fraction * 100))
        name = f"{topology}_real_{u_pct}_{seed}"

        return Instance(
            name=name,
            seed=seed,
            graph_type=f"real:{topology}",
            n_nodes=n_nodes,
            edges=edges,
            U=U,
            u0=u0,
            u1=u1,
            u_fixed=u_fixed,
            L=L,
            commodities=commodities,
            Hmax=hmax,
            M=M,
            initial_routing=routing,
            upgrade_types=upgrade_types,
            link_length_km=length_km,
        )

    raise RuntimeError(
        f"Could not generate a feasible instance for topology={topology} after "
        f"{max_attempts} attempts (last error: {last_err}). Try lowering "
        f"n_commodities/demand or u_fraction."
    )


def main():
    parser = argparse.ArgumentParser(description="Generate batch-scheduling MILP instances from real topologies")
    parser.add_argument("--topology", choices=params.REAL_TOPOLOGIES, required=True)
    parser.add_argument("--u-fraction", type=float, default=params.DEFAULT_REAL_U_FRACTION)
    parser.add_argument("--n-commodities", type=int, default=params.DEFAULT_REAL_N_COMMODITIES)
    parser.add_argument("--demand-min", type=float, default=params.DEFAULT_DEMAND_CHANNELS_RANGE[0])
    parser.add_argument("--demand-max", type=float, default=params.DEFAULT_DEMAND_CHANNELS_RANGE[1])
    parser.add_argument("--hmax", type=int, default=None, help="Override the auto-computed Hmax")
    parser.add_argument("--m-weight", type=float, default=None, help="Override M_DEFAULT")
    parser.add_argument("--seed", type=int, default=params.DEFAULT_SEED)
    parser.add_argument("--count", type=int, default=1, help="Generate `count` instances with seeds seed..seed+count-1")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for i in range(args.count):
        seed = args.seed + i
        inst = generate_real_instance(
            topology=args.topology,
            u_fraction=args.u_fraction,
            n_commodities=args.n_commodities,
            demand_range=(args.demand_min, args.demand_max),
            seed=seed,
            hmax_override=args.hmax,
            m_weight=args.m_weight,
        )
        out_path = out_dir / f"{inst.name}.json"
        save(inst, out_path)
        print(f"Wrote {out_path}  (|V|={inst.n_nodes}, |E|={len(inst.edges)}, "
              f"|U|={len(inst.U)}, |K|={len(inst.commodities)}, Hmax={inst.Hmax})")


if __name__ == "__main__":
    main()
