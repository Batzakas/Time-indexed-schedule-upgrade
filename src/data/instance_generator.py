"""
Synthetic instance generator for the batch-scheduling MILP.

Usage:
    python -m src.data.instance_generator --graph-type ring --n-nodes 6 \
        --u-fraction 0.5 --n-commodities 3 --seed 0 --out-dir data/instances
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import src.configs.params as params
from src.core.instance import Commodity, Instance, compute_initial_routing, save

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "instances"


def _is_connected(n_nodes: int, edges: List[Tuple[int, int, int]]) -> bool:
    if n_nodes == 0:
        return True
    adj = {n: [] for n in range(n_nodes)}
    for u, v, _ in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nxt in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == n_nodes


def gen_erdos_renyi(n_nodes: int, edge_prob: float, rng: random.Random) -> List[Tuple[int, int, int]]:
    """Random spanning tree (guarantees connectivity) + extra random edges."""
    nodes = list(range(n_nodes))
    rng.shuffle(nodes)
    pairs = set()
    edges: List[Tuple[int, int, int]] = []
    eid = 0
    # random spanning tree: attach node i to a random earlier node
    for i in range(1, n_nodes):
        j = nodes[rng.randrange(i)]
        u, v = nodes[i], j
        a, b = min(u, v), max(u, v)
        pairs.add((a, b))
        edges.append((a, b, eid))
        eid += 1

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if (i, j) in pairs:
                continue
            if rng.random() < edge_prob:
                pairs.add((i, j))
                edges.append((i, j, eid))
                eid += 1
    return edges


def gen_ring(n_nodes: int, rng: random.Random) -> List[Tuple[int, int, int]]:
    edges = []
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        edges.append((min(i, j), max(i, j), i))
    return edges


def gen_grid(n_nodes: int, rng: random.Random) -> Tuple[int, List[Tuple[int, int, int]]]:
    """Roughly-square grid; returns the actual n_nodes used (rows*cols)."""
    rows = max(1, int(round(n_nodes ** 0.5)))
    cols = max(1, -(-n_nodes // rows))  # ceil division
    actual_n = rows * cols

    def idx(r, c):
        return r * cols + c

    edges = []
    eid = 0
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1), eid))
                eid += 1
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c), eid))
                eid += 1
    return actual_n, edges


def build_graph(graph_type: str, n_nodes: int, edge_prob: float, rng: random.Random):
    if graph_type == "erdos_renyi":
        edges = gen_erdos_renyi(n_nodes, edge_prob, rng)
        return n_nodes, edges
    elif graph_type == "ring":
        edges = gen_ring(n_nodes, rng)
        return n_nodes, edges
    elif graph_type == "grid":
        return gen_grid(n_nodes, rng)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")


def generate_instance(
    graph_type: str = params.DEFAULT_GRAPH_TYPE,
    n_nodes: int = params.DEFAULT_N_NODES,
    edge_prob: float = params.DEFAULT_EDGE_PROB,
    u_fraction: float = params.DEFAULT_U_FRACTION,
    n_commodities: int = params.DEFAULT_N_COMMODITIES,
    demand_range: Tuple[float, float] = params.DEFAULT_DEMAND_RANGE,
    l_range: Tuple[int, int] = params.DEFAULT_L_RANGE,
    u0_range: Tuple[float, float] = params.DEFAULT_U0_RANGE,
    upgrade_factor_range: Tuple[float, float] = params.DEFAULT_UPGRADE_FACTOR_RANGE,
    fixed_cap_range: Tuple[float, float] = params.DEFAULT_FIXED_CAP_RANGE,
    seed: int = params.DEFAULT_SEED,
    hmax_override: int | None = None,
    m_weight: float | None = None,
    max_attempts: int = 25,
) -> Instance:
    """Generate one instance. Retries with a perturbed seed (same knobs) up
    to max_attempts times if the graph is disconnected or no feasible joint
    initial routing is found."""
    last_err = None
    for attempt in range(max_attempts):
        trial_seed = seed + attempt * 7919  # large prime stride to decorrelate retries
        rng = random.Random(trial_seed)

        actual_n, edges = build_graph(graph_type, n_nodes, edge_prob, rng)
        if not _is_connected(actual_n, edges):
            last_err = "disconnected graph"
            continue

        n_edges = len(edges)
        u_count = max(1, round(u_fraction * n_edges)) if n_edges > 0 else 0
        u_count = min(u_count, n_edges)
        all_eids = [eid for (_, _, eid) in edges]
        U = sorted(rng.sample(all_eids, u_count)) if u_count > 0 else []
        U_set = set(U)

        u0, u1, L = {}, {}, {}
        u_fixed = {}
        for eid in all_eids:
            if eid in U_set:
                u0_val = rng.uniform(*u0_range)
                factor = rng.uniform(*upgrade_factor_range)
                u0[eid] = u0_val
                u1[eid] = u0_val * factor
                L[eid] = rng.randint(*l_range)
            else:
                u_fixed[eid] = rng.uniform(*fixed_cap_range)

        commodities = []
        for _ in range(n_commodities):
            s, t = rng.sample(range(actual_n), 2)
            d = rng.uniform(*demand_range)
            commodities.append(Commodity(s=s, t=t, d=d))

        routing = compute_initial_routing(
            n_nodes=actual_n,
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
        name = f"{graph_type}_{actual_n}_{u_pct}_{seed}"

        return Instance(
            name=name,
            seed=seed,
            graph_type=graph_type,
            n_nodes=actual_n,
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
        )

    raise RuntimeError(
        f"Could not generate a feasible instance after {max_attempts} attempts "
        f"(last error: {last_err}). Try lowering n_commodities/demand or "
        f"raising edge_prob/u0/fixed capacities."
    )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic batch-scheduling MILP instances")
    parser.add_argument("--graph-type", choices=["erdos_renyi", "ring", "grid"], default=params.DEFAULT_GRAPH_TYPE)
    parser.add_argument("--n-nodes", type=int, default=params.DEFAULT_N_NODES)
    parser.add_argument("--edge-prob", type=float, default=params.DEFAULT_EDGE_PROB)
    parser.add_argument("--u-fraction", type=float, default=params.DEFAULT_U_FRACTION)
    parser.add_argument("--n-commodities", type=int, default=params.DEFAULT_N_COMMODITIES)
    parser.add_argument("--demand-min", type=float, default=params.DEFAULT_DEMAND_RANGE[0])
    parser.add_argument("--demand-max", type=float, default=params.DEFAULT_DEMAND_RANGE[1])
    parser.add_argument("--l-min", type=int, default=params.DEFAULT_L_RANGE[0])
    parser.add_argument("--l-max", type=int, default=params.DEFAULT_L_RANGE[1])
    parser.add_argument("--hmax", type=int, default=None, help="Override the auto-computed Hmax")
    parser.add_argument("--m-weight", type=float, default=None, help="Override M_DEFAULT")
    parser.add_argument("--seed", type=int, default=params.DEFAULT_SEED)
    parser.add_argument("--count", type=int, default=1, help="Generate `count` instances with seeds seed..seed+count-1")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for i in range(args.count):
        seed = args.seed + i
        inst = generate_instance(
            graph_type=args.graph_type,
            n_nodes=args.n_nodes,
            edge_prob=args.edge_prob,
            u_fraction=args.u_fraction,
            n_commodities=args.n_commodities,
            demand_range=(args.demand_min, args.demand_max),
            l_range=(args.l_min, args.l_max),
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
