#!/bin/bash
cd "$(dirname "$0")/.."
export OMP_WAIT_POLICY=PASSIVE NPULOOP_THREADS=2 NPULOOP_INT_EVAL=10000 NPULOOP_MODELS=mnv2_050_relu6
while [ "$(ps -eo args | grep -c '^python3 experiments/e7_requant_ablation.py')" -gt 0 ]; do sleep 30; done
for s in e7_requant_ablation e5_calibration e1_baselines; do echo "=== B2 $s $(date -u +%H:%M:%S)" >> results/logs/run_all.log; python3 experiments/$s.py >> results/logs/$s.log 2>&1 || echo "FAILED B2 $s" >> results/logs/run_all.log; done
echo "=== B2 done $(date -u +%H:%M:%S)" >> results/logs/run_all.log
