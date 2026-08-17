## Repository overview

MILP solver (Gurobi) for the **link-upgrade batch-scheduling problem**: given a network graph,
a subset of upgradeable edges `U`, and a set of commodities with fixed
demand, decide when each edge starts its upgrade so that the network can
route all commodities feasibly at every point in time, **minimizing the
makespan** 

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install gurobipy numpy

# 1) Generate synthetic instances
python -m src.data.instance_generator --graph-type erdos_renyi --n-nodes 8 \
    --u-fraction 0.3 --n-commodities 3 --seed 0 --count 5 --out-dir data/instances

# 2) Solve one instance directly
python -m src.core.model_tester --instance data/instances/erdos_renyi_8_30_0.json \
    --M 1.0 --gurobi-license gurobi.lic

# 3) Or sweep a whole directory x several M values in parallel
python -m src.core.runner --instances "data/instances/*.json" \
    --m-values 0.5 1.0 2.0 --parallel 4 --gurobi-license gurobi.lic

# 4) Aggregate partial results into a summary CSV
python -m src.eval.aggregate_partials --partial-dir results/partial_results \
    --out-csv results/aggregated/summary.csv
```

### Real topologies

Instead of synthetic graphs, instances can be built from the same topology
files used by `Network_Planner_Code`'s EON simulator (`test5`: 5 nodes/7
links, `dt12`: 12 nodes/20 links -- copied into `src/data/topologies/`).
Upgrade duration (`L_e`) and the before/after capacity jump (`u0_e`/`u1_e`)
are derived from real per-upgrade-type constants in
`Network_Planner_Code/sim/core/constants.py` (`t_C1`/`t_CL1`/`t_b`/`t_3C`
days, and fiber/band/core counts), not sampled uniformly at random -- see
the comment above `UPGRADE_TYPES` in `src/configs/params.py` for exactly
how each number was derived. Which edges are upgradeable, which node pairs
communicate, and demand sizes are still randomized per seed.

```bash
python -m src.data.real_instance_generator --topology dt12 \
    --u-fraction 0.3 --n-commodities 5 --seed 0 --count 5 --out-dir data/instances

python -m src.core.runner --instances "data/instances/dt12_real_*.json" \
    --m-values 0.5 1.0 2.0 --parallel 4 --gurobi-license gurobi.lic

python -m src.eval.aggregate_partials --partial-dir results/partial_results \
    --out-csv results/aggregated/summary.csv

# 5) Render charts (results/figures/tradeoff.png, scalability.png, diagnostics.png)
python -m src.eval.graficos --summary-csv results/aggregated/summary.csv \
    --out-dir results/figures
```

`run_pipeline.sh --topology dt12` (or `test5`) runs all three steps above
end to end; omit `--topology` (or pass `synthetic`) to keep using the
random-graph generator.

`--topology dt12` defaults to `--u-fraction 0.5 --n-commodities 10` instead
of the synthetic/test5 default (`0.3`/`3`) -- a scale sweep (fixing
`u_fraction` vs `n_commodities` one at a time) found `u_fraction` is what
actually drives MILP difficulty: at `0.3` dt12 solves in <1s regardless of
`n_commodities`, at `1.0` it climbs into tens of seconds to minutes (and can
fail to even find a feasible solution within a few minutes once combined
with more commodities). `0.5`/`10` sits in the middle: still solves to
proven optimality in about a second, but exercises a meaningfully larger
model than the trivial default.

Always invoke with the venv's Python (or an activated venv) -- `runner.py`
re-launches `model_tester.py` via `sys.executable`, so the same interpreter
(and its `gurobipy`) is used in the subprocesses too.

## Directory layout

```text
batch_milp/
├── gurobi.lic                  #copied from PWOF-Fronthaul-Optimization-hpc
├── src/
│   ├── configs/params.py       # solver + generator defaults
│   ├── data/instance_generator.py   # synthetic instance generator (CLI)
│   ├── data/real_instance_generator.py  # real-topology instance generator (CLI)
│   ├── data/topology_loader.py      # parses src/data/topologies/*.txt (test5, dt12)
│   ├── core/
│   │   ├── instance.py         # instance schema, JSON I/O, initial-routing solver
│   │   ├── model_tester.py     # single-instance CLI worker (writes 1 partial JSON, resumable)
│   │   └── runner.py           # parallel dispatcher over instances x M values
│   ├── algorithms/milp_makespan.py  # the Gurobi model (run_ilp / solve_ilp)
│   └── eval/
│       ├── aggregate_partials.py    # merges partials -> results/aggregated/summary.csv
│       └── graficos.py              # summary.csv -> results/figures/*.png
└── results/
    ├── partial_results/<graph_type>_<n_nodes>_<u_pct>/partial_<name>_M<M>_H<Hmax>.json
    ├── aggregated/summary.csv
    └── figures/{tradeoff,scalability,diagnostics}.png
```

## Instance format

See the dataclasses in `src/core/instance.py`. Each undirected physical
link is stored once (`edges: [[u, v, edge_id], ...]`); `U` lists the
upgradeable `edge_id`s with before/after capacities `u0`/`u1` and upgrade
duration `L`; `u_fixed` holds capacities for non-upgradeable edges;
`commodities` is a list of `{s, t, d}`; `initial_routing` is a single
feasible path (list of `edge_id`s) per commodity at `h=0`, computed
greedily by `compute_initial_routing`.
