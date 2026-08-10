"""
Instance schema and I/O for the batch-scheduling MILP.

An instance is small enough to fit entirely as a JSON document. Nodes are
plain ints 0..n_nodes-1. Every undirected physical link is stored once (as
an "edge") and expanded into two directed arcs sharing the same edge_id /
upgrade state when the MILP is built (see algorithms/milp_makespan.py)
upgrading a physical link takes both directions down together.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Commodity:
    s: int
    t: int
    d: float


@dataclass
class Instance:
    name: str
    seed: int
    graph_type: str
    n_nodes: int
    edges: List[Tuple[int, int, int]]
    U: List[int]                           
    u0: Dict[int, float]                  
    u1: Dict[int, float]                   
    u_fixed: Dict[int, float]             
    L: Dict[int, int]                      
    commodities: List[Commodity]
    Hmax: int
    M: float
    initial_routing: Dict[int, List[int]] 

    def capacity_before(self, edge_id: int) -> float:
        """Capacity of edge_id at h=0, i.e. u0_e if upgradeable, u_fixed otherwise."""
        if edge_id in self.u0:
            return self.u0[edge_id]
        return self.u_fixed[edge_id]

    def adjacency(self) -> Dict[int, List[Tuple[int, int]]]:
        """node -> list of (neighbor, edge_id)."""
        adj: Dict[int, List[Tuple[int, int]]] = {n: [] for n in range(self.n_nodes)}
        for u, v, eid in self.edges:
            adj[u].append((v, eid))
            adj[v].append((u, eid))
        return adj

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON object keys must be strings; our dict keys are edge/commodity ints.
        d["u0"] = {str(k): v for k, v in self.u0.items()}
        d["u1"] = {str(k): v for k, v in self.u1.items()}
        d["u_fixed"] = {str(k): v for k, v in self.u_fixed.items()}
        d["L"] = {str(k): v for k, v in self.L.items()}
        d["initial_routing"] = {str(k): v for k, v in self.initial_routing.items()}
        return d

    @staticmethod
    def from_dict(d: dict) -> "Instance":
        return Instance(
            name=d["name"],
            seed=d["seed"],
            graph_type=d["graph_type"],
            n_nodes=d["n_nodes"],
            edges=[tuple(e) for e in d["edges"]],
            U=list(d["U"]),
            u0={int(k): v for k, v in d["u0"].items()},
            u1={int(k): v for k, v in d["u1"].items()},
            u_fixed={int(k): v for k, v in d["u_fixed"].items()},
            L={int(k): v for k, v in d["L"].items()},
            commodities=[Commodity(**c) for c in d["commodities"]],
            Hmax=d["Hmax"],
            M=d["M"],
            initial_routing={int(k): v for k, v in d["initial_routing"].items()},
        )


def save(instance: Instance, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(instance.to_dict(), f, indent=2)


def load(path: str | Path) -> Instance:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return Instance.from_dict(d)


# compute_initial_routing greedily assigns each commodity a simple path
# (shortest first, random order) respecting remaining edge capacity at h=0.
# Returns None if no feasible joint routing was found; caller should
# regenerate the instance in that case.

def _simple_paths(adj: Dict[int, List[Tuple[int, int]]], s: int, t: int,
                   max_len: int) -> List[List[int]]:
    """All simple node-paths from s to t with at most max_len edges (DFS)."""
    paths: List[List[int]] = []
    visited = {s}
    path = [s]

    def dfs(node: int):
        if len(path) - 1 >= max_len:
            return
        for nxt, _eid in adj[node]:
            if nxt in visited:
                continue
            if nxt == t:
                paths.append(path + [nxt])
                continue
            visited.add(nxt)
            path.append(nxt)
            dfs(nxt)
            path.pop()
            visited.remove(nxt)

    dfs(s)
    paths.sort(key=len)
    return paths


def _path_edges(path_nodes: List[int], adj: Dict[int, List[Tuple[int, int]]]) -> List[int]:
    edge_ids = []
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        eid = next(e for (n, e) in adj[a] if n == b)
        edge_ids.append(eid)
    return edge_ids


def compute_initial_routing(
    n_nodes: int,
    edges: List[Tuple[int, int, int]],
    U: List[int],
    u0: Dict[int, float],
    u_fixed: Dict[int, float],
    commodities: List[Commodity],
    rng: Optional[random.Random] = None,
    max_path_len: Optional[int] = None,
) -> Optional[Dict[int, List[int]]]:
    rng = rng or random.Random()
    adj: Dict[int, List[Tuple[int, int]]] = {n: [] for n in range(n_nodes)}
    for u, v, eid in edges:
        adj[u].append((v, eid))
        adj[v].append((u, eid))

    max_path_len = max_path_len or n_nodes

    remaining = {eid: (u0[eid] if eid in U else u_fixed[eid]) for (_, _, eid) in edges}

    order = list(range(len(commodities)))
    rng.shuffle(order)

    routing: Dict[int, List[int]] = {}
    for idx in order:
        c = commodities[idx]
        candidates = _simple_paths(adj, c.s, c.t, max_path_len)
        chosen = None
        for node_path in candidates:
            eids = _path_edges(node_path, adj)
            if all(remaining[eid] >= c.d for eid in eids):
                chosen = eids
                break
        if chosen is None:
            return None
        for eid in chosen:
            remaining[eid] -= c.d
        routing[idx] = chosen

    return routing
