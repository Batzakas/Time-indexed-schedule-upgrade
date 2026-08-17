#!/usr/bin/env bash
# End-to-end pipeline: generate synthetic instances -> solve them -> aggregate
# results into a summary CSV. Uses `uv` to manage the venv/deps (no manual
# `source .venv/bin/activate` needed).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- defaults (mirror src/configs/params.py) ----
TOPOLOGY="synthetic"    # synthetic | test5 | dt12
GRAPH_TYPE="erdos_renyi"
N_NODES=8
EDGE_PROB=0.5
U_FRACTION=""
N_COMMODITIES=""
SEED=0
COUNT=5
INSTANCES_DIR="data/instances"

M_VALUES="0.5 1.0 2.0 10.0 40.0 100.0 1000.0 100000.0"
PARALLEL=""
GUROBI_LICENSE="gurobi.lic"
PARTIAL_DIR="results/partial_results"
SUMMARY_CSV="results/aggregated/summary.csv"
FIGURES_DIR="results/figures"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Instance generation:
  --topology T           synthetic|test5|dt12 (default: $TOPOLOGY)
                          test5/dt12 use real topologies (src/data/real_instance_generator.py);
                          --graph-type/--n-nodes/--edge-prob are ignored in that case.
  --graph-type TYPE     erdos_renyi|ring|grid (default: $GRAPH_TYPE)
  --n-nodes N            (default: $N_NODES)
  --edge-prob P          (default: $EDGE_PROB)
  --u-fraction F         (default: 0.3 for synthetic/test5, 0.5 for dt12)
  --n-commodities N      (default: 3 for synthetic/test5, 10 for dt12)
  --seed N               (default: $SEED)
  --count N              number of instances, seeds SEED..SEED+N-1 (default: $COUNT)
  --instances-dir DIR    (default: $INSTANCES_DIR)

Solving:
  --m-values "V1 V2 .."  rerouting-penalty weights to sweep (default: "$M_VALUES")
  --parallel N           worker processes (default: cpu_count - 1)
  --gurobi-license PATH  (default: $GUROBI_LICENSE)
  --partial-dir DIR      (default: $PARTIAL_DIR)

Aggregation:
  --summary-csv PATH     (default: $SUMMARY_CSV)
  --figures-dir DIR      (default: $FIGURES_DIR)

  -h, --help             show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --topology) TOPOLOGY="$2"; shift 2 ;;
        --graph-type) GRAPH_TYPE="$2"; shift 2 ;;
        --n-nodes) N_NODES="$2"; shift 2 ;;
        --edge-prob) EDGE_PROB="$2"; shift 2 ;;
        --u-fraction) U_FRACTION="$2"; shift 2 ;;
        --n-commodities) N_COMMODITIES="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --count) COUNT="$2"; shift 2 ;;
      --instances-dir) INSTANCES_DIR="$2"; shift 2 ;;
    --m-values) M_VALUES="$2"; shift 2 ;;
      --parallel) PARALLEL="$2"; shift 2 ;;
        --gurobi-license) GUROBI_LICENSE="$2"; shift 2 ;;
        --partial-dir) PARTIAL_DIR="$2"; shift 2 ;;
        --summary-csv) SUMMARY_CSV="$2"; shift 2 ;;
        --figures-dir) FIGURES_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$U_FRACTION" ]]; then
    U_FRACTION=$([[ "$TOPOLOGY" == "dt12" ]] && echo "0.5" || echo "0.3")
fi
if [[ -z "$N_COMMODITIES" ]]; then
    N_COMMODITIES=$([[ "$TOPOLOGY" == "dt12" ]] && echo "10" || echo "3")
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' not found on PATH. Install it: https://docs.astral.sh/uv/" >&2
    exit 1
fi

if [[ ! -f "$GUROBI_LICENSE" ]]; then
    echo "error: Gurobi license file not found at '$GUROBI_LICENSE'." >&2
    exit 1
fi

echo "==> [1/5] uv sync"
uv sync

if [[ "$TOPOLOGY" == "synthetic" ]]; then
    echo "==> [2/5] Generating $COUNT instance(s) ($GRAPH_TYPE, n_nodes=$N_NODES, seed=$SEED..$((SEED + COUNT - 1))) -> $INSTANCES_DIR"
    uv run python -m src.data.instance_generator \
        --graph-type "$GRAPH_TYPE" \
        --n-nodes "$N_NODES" \
        --edge-prob "$EDGE_PROB" \
        --u-fraction "$U_FRACTION" \
        --n-commodities "$N_COMMODITIES" \
        --seed "$SEED" \
        --count "$COUNT" \
        --out-dir "$INSTANCES_DIR"
    instance_glob="$INSTANCES_DIR"/"${GRAPH_TYPE}"_*.json
else
    echo "==> [2/5] Generating $COUNT instance(s) (real topology=$TOPOLOGY, seed=$SEED..$((SEED + COUNT - 1))) -> $INSTANCES_DIR"
    uv run python -m src.data.real_instance_generator \
        --topology "$TOPOLOGY" \
        --u-fraction "$U_FRACTION" \
        --n-commodities "$N_COMMODITIES" \
        --seed "$SEED" \
        --count "$COUNT" \
        --out-dir "$INSTANCES_DIR"
    instance_glob="$INSTANCES_DIR"/"${TOPOLOGY}"_real_*.json
fi

echo "==> [3/5] Solving instances (M in {$M_VALUES})"
runner_args=(
    --instances "$instance_glob"
    --m-values $M_VALUES
    --gurobi-license "$GUROBI_LICENSE"
    --out-dir "$PARTIAL_DIR"
)
if [[ -n "$PARALLEL" ]]; then
    runner_args+=(--parallel "$PARALLEL")
fi
uv run python -m src.core.runner "${runner_args[@]}"

echo "==> [4/5] Aggregating -> $SUMMARY_CSV"
uv run python -m src.eval.aggregate_partials \
    --partial-dir "$PARTIAL_DIR" \
    --out-csv "$SUMMARY_CSV"

echo "==> [5/5] Plotting -> $FIGURES_DIR"
uv run python -m src.eval.graficos \
    --summary-csv "$SUMMARY_CSV" \
    --out-dir "$FIGURES_DIR"

echo "Done. Summary: $SUMMARY_CSV -- Figures: $FIGURES_DIR"
