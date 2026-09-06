#!/bin/bash
# Runs every experiment repeatedly (each script skips finished configs) until all baselines are trained
# and a final full pass completes. Safe to re-run after an interruption.
cd "$(dirname "$0")/.."
export NPULOOP_THREADS=${NPULOOP_THREADS:-2}
export NPULOOP_INT_EVAL=${NPULOOP_INT_EVAL:-10000}
mkdir -p results/logs
pass() {
  for s in e1_baselines e2_ptq_grid e7_requant_ablation e5_calibration e4_surgery e6_pruning; do
    echo "=== $s $(date -u +%H:%M:%S)" >> results/logs/run_all.log
    python3 experiments/$s.py >> results/logs/$s.log 2>&1 || echo "FAILED $s" >> results/logs/run_all.log
  done
  python3 experiments/e3_lint_vs_drop.py >> results/logs/e3_lint_vs_drop.log 2>&1 || true
}
while true; do
  pass
  if [ -f /home/user/work/runs/QUEUE_DONE ]; then pass; echo ALL_DONE >> results/logs/run_all.log; touch results/ALL_DONE; break; fi
  sleep 120
done
