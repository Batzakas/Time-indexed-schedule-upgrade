#!/usr/bin/env bash
# End-to-end pipeline: generate synthetic instances -> solve them -> aggregate
# results into a summary CSV. Uses `uv` to manage the venv/deps (no manual
# `source .venv/bin/activate` needed).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- defaults (mirror src/configs/params.py) ----
GRAPH_TYPE="erdos_renyi"
N_NODES=8
EDGE_PROB=0.5
U_FRACTION=0.3
N_COMMODITIES=3
SEED=0
COUNT=5
INSTANCES_DIR="data/instances"

M_VALUES="0.5 1.0 2.0"
PARALLEL=""
GUROBI_LICENSE="gurobi.lic"
PARTIAL_DIR="results/partial_results"
SUMMARY_CSV="results/aggregated/summary.csv"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Instance generation:
  --graph-type TYPE     erdos_renyi|ring|grid (default: $GRAPH_TYPE)
  --n-nodes N            (default: $N_NODES)
  --edge-prob P          (default: $EDGE_PROB)
  --u-fraction F         (default: $U_FRACTION)
  --n-commodities N      (default: $N_COMMODITIES)
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

  -h, --help             show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' not found on PATH. Install it: https://docs.astral.sh/uv/" >&2
    exit 1
fi

if [[ ! -f "$GUROBI_LICENSE" ]]; then
    echo "error: Gurobi license file not found at '$GUROBI_LICENSE'." >&2
    exit 1
fi

echo "==> [1/4] uv sync"
uv sync

echo "==> [2/4] Generating $COUNT instance(s) ($GRAPH_TYPE, n_nodes=$N_NODES, seed=$SEED..$((SEED + COUNT - 1))) -> $INSTANCES_DIR"
uv run python -m src.data.instance_generator \
    --graph-type "$GRAPH_TYPE" \
    --n-nodes "$N_NODES" \
    --edge-prob "$EDGE_PROB" \
    --u-fraction "$U_FRACTION" \
    --n-commodities "$N_COMMODITIES" \
    --seed "$SEED" \
    --count "$COUNT" \
    --out-dir "$INSTANCES_DIR"

echo "==> [3/4] Solving instances (M in {$M_VALUES})"
runner_args=(
    --instances "$INSTANCES_DIR"/*.json
    --m-values $M_VALUES
    --gurobi-license "$GUROBI_LICENSE"
    --out-dir "$PARTIAL_DIR"
)
if [[ -n "$PARALLEL" ]]; then
    runner_args+=(--parallel "$PARALLEL")
fi
uv run python -m src.core.runner "${runner_args[@]}"

echo "==> [4/4] Aggregating -> $SUMMARY_CSV"
uv run python -m src.eval.aggregate_partials \
    --partial-dir "$PARTIAL_DIR" \
    --out-csv "$SUMMARY_CSV"

echo "Done. Summary: $SUMMARY_CSV"
