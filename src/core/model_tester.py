"""
Single-instance CLI worker: solve one instance (one M / Hmax override
combination) and write ONE partial JSON result.

Usage:
    python -m src.core.model_tester --instance data/instances/foo.json \
        --M 1.0 --need-checkpoint 1 --gurobi-license gurobi.lic
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import src.configs.params as params
from src.algorithms.milp_makespan import run_ilp
from src.core.instance import load

DEFAULT_PARTIAL_DIR = Path(__file__).resolve().parents[2] / "results" / "partial_results"


def _json_safe(obj: Any) -> Any:
    """Recursively convert dict/tuple keys and values into JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def partial_result_path(instance_name: str, M: float, hmax: int, out_dir: Path) -> Path:
    tag_dir = out_dir / instance_name.rsplit("_", 1)[0]  # group by graph_type_n_upct
    fname = f"partial_{instance_name}_M{M:g}_H{hmax}.json"
    return tag_dir / fname


def solve_one(
    instance_path: str,
    M: Optional[float] = None,
    hmax: Optional[int] = None,
    time_limit: float = params.TIME_LIMIT,
    max_memory: Optional[float] = None,
    gurobi_license: Optional[str] = None,
    mip_gap: Optional[float] = None,
    out_dir: Path = DEFAULT_PARTIAL_DIR,
    need_checkpoint: bool = True,
) -> Path:
    instance = load(instance_path)
    eff_M = M if M is not None else instance.M
    eff_hmax = hmax if hmax is not None else instance.Hmax

    out_path = partial_result_path(instance.name, eff_M, eff_hmax, out_dir)
    if need_checkpoint and out_path.exists():
        print(f"[skip] {out_path} already exists")
        return out_path

    t0 = time.time()
    result = run_ilp(
        instance,
        M=M,
        Hmax=hmax,
        time_limit=time_limit,
        max_memory=max_memory,
        gurobi_license=gurobi_license,
        mip_gap=mip_gap,
    )
    wall_time = time.time() - t0

    record = {
        "instance_name": instance.name,
        "instance_path": str(instance_path),
        "graph_type": instance.graph_type,
        "n_nodes": instance.n_nodes,
        "n_edges": len(instance.edges),
        "n_upgradeable": len(instance.U),
        "n_commodities": len(instance.commodities),
        "M": eff_M,
        "Hmax": eff_hmax,
        "wall_time": wall_time,
        **_json_safe(result),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    status_str = "OPTIMAL" if record.get("feasible") else f"NOT-SOLVED(status={record.get('status')})"
    print(f"[done] {instance.name} M={eff_M} Hmax={eff_hmax} -> {status_str} -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Solve one batch-scheduling MILP instance")
    parser.add_argument("--instance", type=str, required=True, help="Path to instance JSON")
    parser.add_argument("--M", type=float, default=None, help="Override instance.M")
    parser.add_argument("--hmax", type=int, default=None, help="Override instance.Hmax")
    parser.add_argument("--time-limit", type=float, default=params.TIME_LIMIT)
    parser.add_argument("--max-memory", type=float, default=params.DEFAULT_MAX_MEMORY)
    parser.add_argument("--mip-gap", type=float, default=params.MIP_GAP)
    parser.add_argument("--gurobi-license", type=str, default=params.DEFAULT_GUROBI_LICENSE)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_PARTIAL_DIR))
    parser.add_argument("--need-checkpoint", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    solve_one(
        instance_path=args.instance,
        M=args.M,
        hmax=args.hmax,
        time_limit=args.time_limit,
        max_memory=args.max_memory,
        gurobi_license=args.gurobi_license,
        mip_gap=args.mip_gap,
        out_dir=Path(args.out_dir),
        need_checkpoint=bool(args.need_checkpoint),
    )


if __name__ == "__main__":
    main()
