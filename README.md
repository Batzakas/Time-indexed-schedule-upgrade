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


```bash
python -m src.data.real_instance_generator --topology dt12 \
    --u-fraction 0.3 --n-commodities 5 --seed 0 --count 5 --out-dir data/instances

python -m src.core.runner --instances "data/instances/dt12_real_*.json" \
    --m-values 0.5 1.0 2.0 --parallel 4 --gurobi-license gurobi.lic

python -m src.eval.aggregate_partials --partial-dir results/partial_results \
    --out-csv results/aggregated/summary.csv

# 5) Render charts: network load/feasibility and solver metrics vs M
#    (results/figures/network_load_all.png, solver_all.png)
python -m src.eval.graficos --summary-csv results/aggregated/summary.csv \
    --topology-label all --out-dir results/figures \
    --x-col M --x-label "M (reroute penalty weight)" --series-col graph_type

# Congestion sweeps (src/eval/congestion_sweep.py) produce a CSV shaped for
# the same charts' default axes (x=n_commodities, series=u_fraction):
python -m src.eval.graficos --summary-csv results/aggregated/congestion_dt12.csv \
    --topology-label dt12 --out-dir results/figures
```

`run_pipeline.sh --topology dt12` (or `test5`) runs all three steps above
end to end; omit `--topology` (or pass `synthetic`) to keep using the
random-graph generator.

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
│       ├── congestion_sweep.py      # u_fraction x n_commodities sweep -> congestion_<topology>.csv
│       └── graficos.py              # summary/congestion csv -> results/figures/{network_load,solver}_<label>.png
└── results/
    ├── partial_results/<graph_type>_<n_nodes>_<u_pct>/partial_<name>_M<M>_H<Hmax>.json
    ├── aggregated/{summary.csv,congestion_<topology>.csv}
    └── figures/{network_load,solver}_<label>.png
```

## Instance format

See the dataclasses in `src/core/instance.py`. Each undirected physical
link is stored once (`edges: [[u, v, edge_id], ...]`); `U` lists the
upgradeable `edge_id`s with before/after capacities `u0`/`u1` and upgrade
duration `L`; `u_fixed` holds capacities for non-upgradeable edges;
`commodities` is a list of `{s, t, d}`; `initial_routing` is a single
feasible path (list of `edge_id`s) per commodity at `h=0`, computed
greedily by `compute_initial_routing`.
