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
│   ├── core/
│   │   ├── instance.py         # instance schema, JSON I/O, initial-routing solver
│   │   ├── model_tester.py     # single-instance CLI worker (writes 1 partial JSON, resumable)
│   │   └── runner.py           # parallel dispatcher over instances x M values
│   ├── algorithms/milp_makespan.py  # the Gurobi model (run_ilp / solve_ilp)
│   └── eval/aggregate_partials.py   # merges partials -> results/aggregated/summary.csv
└── results/
    ├── partial_results/<graph_type>_<n_nodes>_<u_pct>/partial_<name>_M<M>_H<Hmax>.json
    └── aggregated/summary.csv
```

## Instance format

See the dataclasses in `src/core/instance.py`. Each undirected physical
link is stored once (`edges: [[u, v, edge_id], ...]`); `U` lists the
upgradeable `edge_id`s with before/after capacities `u0`/`u1` and upgrade
duration `L`; `u_fixed` holds capacities for non-upgradeable edges;
`commodities` is a list of `{s, t, d}`; `initial_routing` is a single
feasible path (list of `edge_id`s) per commodity at `h=0`, computed
greedily by `compute_initial_routing`.
