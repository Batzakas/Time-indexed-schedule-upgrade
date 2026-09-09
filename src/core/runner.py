"""
Parallel dispatcher: launches src.core.model_tester as a subprocess for
every (instance, M) combination, so each solve gets its own process (and
its own Gurobi environment) and a crash/timeout in one instance cannot take
down the others.

Usage:
    python -m src.core.runner --instances "data/instances/*.json" \
        --m-values 0.5 1.0 2.0 --parallel 4 --gurobi-license gurobi.lic
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_TESTER_MODULE = "src.core.model_tester"
LOG_DIR = REPO_ROOT / "src" / "logs"


def build_model_tester_cmd(
    instance_path: str,
    M: Optional[float],
    hmax: Optional[int],
    time_limit: float,
    gurobi_license: Optional[str],
    out_dir: str,
    need_checkpoint: bool,
) -> List[str]:
    cmd = [sys.executable, "-u", "-m", MODEL_TESTER_MODULE, "--instance", instance_path]
    if M is not None:
        cmd += ["--M", str(M)]
    if hmax is not None:
        cmd += ["--hmax", str(hmax)]
    cmd += ["--time-limit", str(time_limit)]
    if gurobi_license is not None:
        cmd += ["--gurobi-license", gurobi_license]
    cmd += ["--out-dir", out_dir]
    cmd += ["--need-checkpoint", "1" if need_checkpoint else "0"]
    return cmd


def _run_one(args_tuple) -> dict:
    (instance_path, M, hmax, time_limit, gurobi_license,
     out_dir, need_checkpoint) = args_tuple

    cmd = build_model_tester_cmd(
        instance_path, M, hmax, time_limit, gurobi_license,
        out_dir, need_checkpoint,
    )

    inst_stem = Path(instance_path).stem
    m_tag = "default" if M is None else f"{M:g}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"log_{inst_stem}_M{m_tag}.txt"

    with open(log_file, "w") as lf:
        print(f"Starting: {' '.join(shlex.quote(c) for c in cmd)} -> {log_file}")
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=lf, stderr=lf)
        ret = proc.wait()

    return {"instance": instance_path, "M": M, "success": ret == 0, "returncode": ret, "log": str(log_file)}


def runner(
    instances: List[str],
    m_values: List[Optional[float]],
    hmax: Optional[int] = None,
    time_limit: float = 7200.0,
    gurobi_license: Optional[str] = "gurobi.lic",
    out_dir: str = "results/partial_results",
    need_checkpoint: bool = True,
    parallel: Optional[int] = None,
):
    tasks = [
        (inst, M, hmax, time_limit, gurobi_license, out_dir, need_checkpoint)
        for inst in instances
        for M in m_values
    ]

    if parallel is None:
        parallel = max(1, mp.cpu_count() - 1)

    print(f"Launching {len(tasks)} solve tasks with parallel={parallel} ...")
    if not tasks:
        print("No tasks to run.")
        return []

    mp.set_start_method("spawn", force=False)
    with mp.Pool(processes=min(parallel, len(tasks))) as pool:
        results = pool.map(_run_one, tasks)

    n_ok = sum(1 for r in results if r["success"])
    print(f"Completed {len(results)} tasks: {n_ok} succeeded, {len(results) - n_ok} failed")
    for r in results:
        print(r)
    return results


def main():
    parser = argparse.ArgumentParser(description="Runner for batch-scheduling MILP experiments")
    parser.add_argument("--instances", nargs="+", required=True,
                         help="Instance JSON paths or glob patterns")
    parser.add_argument("--m-values", nargs="+", type=float, default=[None],
                         help="Rerouting-penalty weights to sweep (default: use each instance's own M)")
    parser.add_argument("--hmax", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=7200.0)
    parser.add_argument("--gurobi-license", type=str, default="gurobi.lic")
    parser.add_argument("--out-dir", type=str, default="results/partial_results")
    parser.add_argument("--need-checkpoint", type=int, choices=[0, 1], default=1)
    parser.add_argument("--parallel", type=int, default=None)
    args = parser.parse_args()

    expanded: List[str] = []
    for pattern in args.instances:
        matches = sorted(glob.glob(pattern))
        expanded.extend(matches if matches else [pattern])

    runner(
        instances=expanded,
        m_values=args.m_values,
        hmax=args.hmax,
        time_limit=args.time_limit,
        gurobi_license=args.gurobi_license,
        out_dir=args.out_dir,
        need_checkpoint=bool(args.need_checkpoint),
        parallel=args.parallel,
    )


if __name__ == "__main__":
    main()
