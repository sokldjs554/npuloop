#!/bin/bash
# Runs every experiment repeatedly (each script skips finished configs) until all baselines are trained
# and a final full pass completes. Safe to re-run after an interruption.
cd "$(dirname "$0")/.."
export NPULOOP_THREADS=${NPULOOP_THREADS:-2}
export OMP_WAIT_POLICY=PASSIVE
export NPULOOP_INT_EVAL=${NPULOOP_INT_EVAL:-10000}
mkdir -p results/logs
pass() {
  for s in e1_baselines e2_ptq_grid e7_requant_ablation e5_calibration e4_surgery e6_pruning; do
    echo "=== $s $(date -u +%H:%M:%S)" >> results/logs/run_all.log
    python3 experiments/$s.py >> results/logs/$s.log 2>&1 || echo "FAILED $s" >> results/logs/run_all.log
  done
  python3 experiments/e3_lint_vs_drop.py >> results/logs/e3_lint_vs_drop.log 2>&1 || true
}
# Default: one pass over every experiment (each script skips configs that already have a record).
# Set NPULOOP_WAIT_FOR_QUEUE=1 to keep re-running while a separate training queue fills runs/ ,
# stopping once $NPULOOP_QUEUE_DONE appears.
QUEUE_DONE=${NPULOOP_QUEUE_DONE:-${NPULOOP_RUNS:-runs}/QUEUE_DONE}
if [ -n "${NPULOOP_WAIT_FOR_QUEUE:-}" ]; then
  while true; do
    pass
    if [ -f "$QUEUE_DONE" ]; then break; fi
    sleep 120
  done
fi
pass
echo ALL_DONE >> results/logs/run_all.log; touch results/ALL_DONE
