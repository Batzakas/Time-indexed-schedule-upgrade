#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Survive an SSH disconnect: on the first (attended) invocation, re-launch
# ourselves detached from the terminal (setsid) and immune to SIGHUP (nohup),
# with all output captured to a timestamped log, then return control to the
# caller immediately. RUN_DETACHED=1 marks the re-exec so it actually runs
# instead of detaching again.
_want_help=0
for _arg in "$@"; do
    [[ "$_arg" == "-h" || "$_arg" == "--help" ]] && _want_help=1
done

if [[ "${RUN_DETACHED:-0}" != "1" && "$_want_help" != "1" ]]; then
    mkdir -p logs
    LOG_FILE="logs/congestion_sweep_dt12_$(date +%Y%m%d_%H%M%S).log"
    echo "==> Detaching: safe to close this SSH session now."
    echo "    Log:  $LOG_FILE"
    RUN_DETACHED=1 nohup setsid "$0" "$@" > "$LOG_FILE" 2>&1 < /dev/null &
    echo "    PID:  $!"
    echo "    Watch progress with: tail -f $LOG_FILE"
    exit 0
fi

# ---- defaults ----
TOPOLOGY="dt12"
N_COMMODITIES_LIST="5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90"
U_FRACTION_LIST="0.2 0.4 0.5 0.7 0.8 1.0"
N_SEEDS=10
SEED_START=0
M_WEIGHT=1.0
TIME_LIMIT=7200
PARALLEL=1
THREADS=""
MEM_LIMIT=""
GUROBI_LICENSE="gurobi.lic"
INSTANCES_DIR="data/instances/congestion_dt12"
PARTIAL_DIR="results/partial_results/congestion_dt12"
OUT_CSV="results/aggregated/congestion_dt12.csv"
NEED_CHECKPOINT=1        

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Sweep knobs:
  --topology T             synthetic|test5|dt12 (default: $TOPOLOGY)
  --n-commodities-list "L" space-separated list (default: "$N_COMMODITIES_LIST")
  --u-fraction-list "L"    space-separated list (default: "$U_FRACTION_LIST")
  --n-seeds N               seeds per (u_fraction, n_commodities) cell (default: $N_SEEDS)
  --seed-start N            (default: $SEED_START)
  --m-weight F              rerouting-penalty weight (default: $M_WEIGHT)

Resource limits:
  --time-limit SECONDS      Gurobi TimeLimit per instance (default: $TIME_LIMIT = 2h)

Execution:
  --parallel N               worker processes (default: $PARALLEL, no parallelism)
  --threads N                Gurobi Threads per solve (default: 20)
  --mem-limit GB             Gurobi MemLimit per solve, in GB (default: 32)
  --gurobi-license PATH      (default: $GUROBI_LICENSE)
  --instances-dir DIR        (default: $INSTANCES_DIR)
  --partial-dir DIR          (default: $PARTIAL_DIR)
  --out-csv PATH             (default: $OUT_CSV)
  --need-checkpoint 0|1      skip instances that already have a result (default: $NEED_CHECKPOINT)

  -h, --help                 show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --topology) TOPOLOGY="$2"; shift 2 ;;
        --n-commodities-list) N_COMMODITIES_LIST="$2"; shift 2 ;;
        --u-fraction-list) U_FRACTION_LIST="$2"; shift 2 ;;
        --n-seeds) N_SEEDS="$2"; shift 2 ;;
        --seed-start) SEED_START="$2"; shift 2 ;;
        --m-weight) M_WEIGHT="$2"; shift 2 ;;
        --time-limit) TIME_LIMIT="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --mem-limit) MEM_LIMIT="$2"; shift 2 ;;
        --gurobi-license) GUROBI_LICENSE="$2"; shift 2 ;;
        --instances-dir) INSTANCES_DIR="$2"; shift 2 ;;
        --partial-dir) PARTIAL_DIR="$2"; shift 2 ;;
        --out-csv) OUT_CSV="$2"; shift 2 ;;
        --need-checkpoint) NEED_CHECKPOINT="$2"; shift 2 ;;
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

echo "==> uv sync"
uv sync

echo "==> Congestion sweep: topology=$TOPOLOGY"
echo "    n_commodities: $N_COMMODITIES_LIST"
echo "    u_fraction:    $U_FRACTION_LIST"
echo "    n_seeds=$N_SEEDS seed_start=$SEED_START M=$M_WEIGHT"
echo "    time_limit=${TIME_LIMIT}s parallel=${PARALLEL}"

sweep_args=(
    --topology "$TOPOLOGY"
    --u-fraction-list $U_FRACTION_LIST
    --n-commodities-list $N_COMMODITIES_LIST
    --n-seeds "$N_SEEDS"
    --seed-start "$SEED_START"
    --m-weight "$M_WEIGHT"
    --time-limit "$TIME_LIMIT"
    --gurobi-license "$GUROBI_LICENSE"
    --instances-dir "$INSTANCES_DIR"
    --partial-dir "$PARTIAL_DIR"
    --out-csv "$OUT_CSV"
    --need-checkpoint "$NEED_CHECKPOINT"
)
if [[ -n "$PARALLEL" ]]; then
    sweep_args+=(--parallel "$PARALLEL")
fi
if [[ -n "$THREADS" ]]; then
    sweep_args+=(--threads "$THREADS")
fi
if [[ -n "$MEM_LIMIT" ]]; then
    sweep_args+=(--mem-limit "$MEM_LIMIT")
fi

uv run python -m src.eval.congestion_sweep "${sweep_args[@]}"

echo "Done. Partial results: $PARTIAL_DIR -- Aggregated CSV: $OUT_CSV"
