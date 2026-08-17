"""
Loader for the real network topologies (test5, dt12).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

TOPOLOGIES_DIR = Path(__file__).resolve().parent / "topologies"

AVAILABLE_TOPOLOGIES = {
    "test5": "test5.txt",
    "dt12": "dt12.txt",
}


class Topology(TypedDict):
    name: str
    n_nodes: int
    edges: List[Tuple[int, int, int]]      # (u, v, edge_id), one entry per undirected link
    length_km: Dict[int, float]            # edge_id -> physical link length (km)


def _parse_matrix(lines: List[str]) -> List[List[int]]:
    return [[int(x.strip()) for x in line.split(",")] for line in lines if line.strip()]


def load_topology(name: str) -> Topology:
    if name not in AVAILABLE_TOPOLOGIES:
        available = ", ".join(AVAILABLE_TOPOLOGIES)
        raise ValueError(f"Unknown topology '{name}'. Available: {available}")

    path = TOPOLOGIES_DIR / AVAILABLE_TOPOLOGIES[name]
    sections: Dict[str, List[str]] = {}
    current = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line in ("TOPOLOGY", "TOPOLOGY_LINK_LENGTHS", "LINK_INDEX"):
                current = line
                sections[current] = []
                continue
            if "=" in line and current is None:
                continue  
            if current is not None:
                sections[current].append(line)

    adjacency = _parse_matrix(sections["TOPOLOGY"])
    lengths = _parse_matrix(sections["TOPOLOGY_LINK_LENGTHS"])
    link_index = _parse_matrix(sections["LINK_INDEX"])
    n_nodes = len(adjacency)

    edges: List[Tuple[int, int, int]] = []
    length_km: Dict[int, float] = {}
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adjacency[i][j] == 1:
                eid = link_index[i][j]
                edges.append((i, j, eid))
                length_km[eid] = float(lengths[i][j])
    edges.sort(key=lambda e: e[2])

    return {"name": name, "n_nodes": n_nodes, "edges": edges, "length_km": length_km}
