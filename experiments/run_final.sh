#!/bin/bash
# Two non-overlapping runners (disjoint result files), then E3. Safe to re-run: every script skips finished configs.
cd "$(dirname "$0")/.."
export OMP_WAIT_POLICY=PASSIVE NPULOOP_THREADS=2 NPULOOP_INT_EVAL=10000
mkdir -p results/logs
runA() { for s in e4_surgery e6_pruning e4_surgery; do echo "=== A $s $(date -u +%H:%M:%S)" >> results/logs/run_all.log; python3 experiments/$s.py >> results/logs/$s.log 2>&1 || echo "FAILED A $s" >> results/logs/run_all.log; done; echo "=== A done $(date -u +%H:%M:%S)" >> results/logs/run_all.log; }
runB() { for s in e2_ptq_grid e7_requant_ablation e5_calibration e1_baselines; do echo "=== B $s $(date -u +%H:%M:%S)" >> results/logs/run_all.log; NPULOOP_MODELS=mnv2_050_relu6 python3 experiments/$s.py >> results/logs/$s.log 2>&1 || echo "FAILED B $s" >> results/logs/run_all.log; done; echo "=== B done $(date -u +%H:%M:%S)" >> results/logs/run_all.log; }
runA & runB & wait
python3 experiments/e3_lint_vs_drop.py >> results/logs/e3_lint_vs_drop.log 2>&1 || true
echo ALL_DONE >> results/logs/run_all.log; touch results/ALL_DONE
