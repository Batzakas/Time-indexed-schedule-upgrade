"""
Makespan-minimizing MILP for the link-upgrade batch-scheduling problem: min C + M * sum(rho).
Free parallelism, no cap on simultaneous upgrades beyond what routing
feasibility (Eqs. 4-6) allows.

"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import gurobipy as gp
from gurobipy import GRB

from src.core.instance import Instance
import src.configs.params as params


def _capval(u0: float, u1: float, L: int, tau: int, h: int) -> float:
    if h < tau:
        return u0
    if h < tau + L:
        return 0.0
    return u1


def run_ilp(
    instance: Instance,
    M: Optional[float] = None,
    Hmax: Optional[int] = None,
    time_limit: float = params.TIME_LIMIT,
    max_memory: Optional[float] = None,
    gurobi_license: Optional[str] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = False,
) -> dict:
    """Build and solve the makespan MILP for one instance.

    Never raises on infeasibility/time limit -- callers should check
    result["status"] / result["feasible"].
    """
    if gurobi_license is not None:
        os.environ["GRB_LICENSE_FILE"] = gurobi_license
    elif os.path.exists(params.DEFAULT_GUROBI_LICENSE):
        os.environ["GRB_LICENSE_FILE"] = params.DEFAULT_GUROBI_LICENSE

    Hmax = Hmax if Hmax is not None else instance.Hmax
    M = M if M is not None else instance.M
    H = range(Hmax)

    for e in instance.U:
        if instance.L[e] > Hmax:
            raise ValueError(
                f"Edge {e} has L_e={instance.L[e]} > Hmax={Hmax}; no valid "
                f"start time tau exists (Eq. 1's domain would be empty)."
            )

    env = gp.Env(empty=True)
    if not verbose:
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
    env.start()

    model = gp.Model("batch_upgrade_makespan", env=env)
    if max_memory is not None:
        model.setParam("MemLimit", max_memory)
    model.setParam("TimeLimit", time_limit)
    if mip_gap is not None:
        model.setParam("MIPGap", mip_gap)

    edges = instance.edges
    edge_ids = [eid for (_, _, eid) in edges]
    U = instance.U
    U_set = set(U)

    # two directed arcs per undirected edge, sharing the same edge_id/upgrade state
    arcs = []
    for (u, v, eid) in edges:
        arcs.append((u, v, eid))
        arcs.append((v, u, eid))
    n_arcs = len(arcs)

    out_arcs = {n: [] for n in range(instance.n_nodes)}
    in_arcs = {n: [] for n in range(instance.n_nodes)}
    for idx, (a, b, _eid) in enumerate(arcs):
        out_arcs[a].append(idx)
        in_arcs[b].append(idx)

    arcs_of_edge: Dict[int, List[int]] = {eid: [] for eid in edge_ids}
    for idx, (_a, _b, eid) in enumerate(arcs):
        arcs_of_edge[eid].append(idx)

    K = instance.commodities
    n_k = len(K)

    # y[e, tau]: upgrade start-time variables (Eq. 1)
    valid_taus = {e: list(range(0, Hmax - instance.L[e] + 1)) for e in U}
    y = {}
    for e in U:
        for tau in valid_taus[e]:
            y[e, tau] = model.addVar(vtype=GRB.BINARY, name=f"y[{e},{tau}]")

    for e in U:
        model.addConstr(
            gp.quicksum(y[e, tau] for tau in valid_taus[e]) == 1,
            name=f"start_once[{e}]",
        )  # Eq. (1)

    def d_expr(e: int, h: int):  # Eq. (2)
        return gp.quicksum(
            y[e, tau] for tau in valid_taus[e] if tau <= h < tau + instance.L[e]
        )

    def cap_expr(e: int, h: int):  # linear replacement for Eq. (6)
        if e in U_set:
            u0e, u1e, Le = instance.u0[e], instance.u1[e], instance.L[e]
            return gp.quicksum(
                y[e, tau] * _capval(u0e, u1e, Le, tau, h) for tau in valid_taus[e]
            )
        return instance.u_fixed[e]

    def F_expr(e: int):  # Eq. (3)
        Le = instance.L[e]
        return gp.quicksum((tau + Le) * y[e, tau] for tau in valid_taus[e])

    C = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="C")
    for e in U:
        model.addConstr(C >= F_expr(e), name=f"makespan[{e}]")  # Eq. (3)

    f = model.addVars(n_k, n_arcs, Hmax, vtype=GRB.BINARY, name="f")

    for k, c in enumerate(K):  # Eq. (4)
        for h in H:
            for v in range(instance.n_nodes):
                rhs = 1.0 if v == c.s else (-1.0 if v == c.t else 0.0)
                model.addConstr(
                    gp.quicksum(f[k, idx, h] for idx in out_arcs[v])
                    - gp.quicksum(f[k, idx, h] for idx in in_arcs[v])
                    == rhs,
                    name=f"flow[{k},{v},{h}]",
                )

    def use_expr(k: int, e: int, h: int):
        # aggregate flow of commodity k through edge e, either direction
        return gp.quicksum(f[k, idx, h] for idx in arcs_of_edge[e])

    for e in edge_ids:  # Eq. (5) (undirected edge)
        for h in H:
            model.addConstr(
                gp.quicksum(c.d * use_expr(k, e, h) for k, c in enumerate(K))
                <= cap_expr(e, h),
                name=f"cap[{e},{h}]",
            )

    for k, c in enumerate(K):  # Eq. (7)
        node_seq = _edge_path_to_nodes(instance, c.s, instance.initial_routing[k])
        used_arcs = set()
        for (a, b) in zip(node_seq[:-1], node_seq[1:]):
            for idx in out_arcs[a]:
                if arcs[idx][1] == b:
                    used_arcs.add(idx)
                    break
        for idx in range(n_arcs):
            val = 1.0 if idx in used_arcs else 0.0
            f[k, idx, 0].LB = val
            f[k, idx, 0].UB = val

    g = {}  # Eq. (8)
    for k in range(n_k):
        for e in U:
            for h in range(1, Hmax):
                g[k, e, h] = model.addVar(vtype=GRB.BINARY, name=f"g[{k},{e},{h}]")
                use_prev = use_expr(k, e, h - 1)
                d_h = d_expr(e, h)
                model.addConstr(g[k, e, h] <= use_prev, name=f"g_ub1[{k},{e},{h}]")
                model.addConstr(g[k, e, h] <= d_h, name=f"g_ub2[{k},{e},{h}]")
                model.addConstr(
                    g[k, e, h] >= use_prev + d_h - 1, name=f"g_lb[{k},{e},{h}]"
                )

    a = model.addVars(n_k, range(1, Hmax), vtype=GRB.BINARY, name="a")  # Eq. (9)
    for k in range(n_k):
        for h in range(1, Hmax):
            for e in U:
                model.addConstr(a[k, h] >= g[k, e, h], name=f"a_lb[{k},{e},{h}]")
            model.addConstr(
                a[k, h] <= gp.quicksum(g[k, e, h] for e in U), name=f"a_ub[{k},{h}]"
            )

    for k in range(n_k):  # Eq. (10)
        for h in range(1, Hmax):
            for e in edge_ids:
                use_now = use_expr(k, e, h)
                use_prev = use_expr(k, e, h - 1)
                model.addConstr(use_now - use_prev <= a[k, h], name=f"stick1[{k},{e},{h}]")
                model.addConstr(use_prev - use_now <= a[k, h], name=f"stick2[{k},{e},{h}]")

    reroute_term = gp.quicksum(a[k, h] for k in range(n_k) for h in range(1, Hmax))
    model.setObjective(C + M * reroute_term, GRB.MINIMIZE)  # Eq. (13)

    model.optimize()

    return _extract_result(model, instance, Hmax, K, y, valid_taus, a, C)


def _edge_path_to_nodes(instance: Instance, start: int, edge_path: List[int]) -> List[int]:
    """Reconstruct the node sequence of a path given as edge_ids, starting at
    `start`. Each edge_id has a unique pair of endpoints, so from `current`
    there is exactly one neighbor reachable via that edge_id."""
    adj = instance.adjacency()
    nodes = [start]
    current = start
    for eid in edge_path:
        nxt = next(n for (n, e) in adj[current] if e == eid)
        nodes.append(nxt)
        current = nxt
    return nodes


def _extract_result(model, instance, Hmax, K, y, valid_taus, a, C) -> dict:
    metrics = {
        "status": model.Status,
        "runtime": model.Runtime,
        "node_count": model.NodeCount,
        "num_vars": model.NumVars,
        "num_bin_vars": model.NumBinVars,
        "num_constrs": model.NumConstrs,
        "mip_gap": model.MIPGap if model.SolCount > 0 else None,
    }

    if model.Status == GRB.INFEASIBLE:
        model.computeIIS()
        iis_dir = Path("results") / "infeasible"
        iis_dir.mkdir(parents=True, exist_ok=True)
        iis_path = iis_dir / f"{instance.name}.ilp"
        model.write(str(iis_path))
        metrics["iis_file"] = str(iis_path)
        metrics["feasible"] = False
        return metrics

    if model.SolCount == 0:
        metrics["feasible"] = False
        return metrics

    start_time = {
        e: tau
        for e in instance.U
        for tau in valid_taus[e]
        if y[e, tau].X > 0.5
    }
    batches_emergent = sorted(set(start_time.values()))
    reroute_by_commodity = [
        sum(1 for h in range(1, Hmax) if a[k, h].X > 0.5) for k in range(len(K))
    ]

    metrics.update({
        "feasible": True,
        "makespan": C.X,
        "objective": model.ObjVal,
        "start_time": start_time,
        "n_batches_emergent": len(batches_emergent),
        "batches_emergent": batches_emergent,
        "total_reroutes": int(sum(reroute_by_commodity)),
        "reroutes_by_commodity": reroute_by_commodity,
    })
    return metrics


def solve_ilp(instance_path: str, **kwargs) -> dict:
    """Load an instance from disk, solve it, and tag the result with its name."""
    from src.core.instance import load

    instance = load(instance_path)
    result = run_ilp(instance, **kwargs)
    result["instance_name"] = instance.name
    return result
