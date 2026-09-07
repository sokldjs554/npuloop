#!/bin/bash
# Side runner for the MobileNetV2 baseline (E2 -> E7 -> E5); the main runner handles E4/E6 meanwhile.
cd "$(dirname "$0")/.."
export NPULOOP_THREADS=2 OMP_WAIT_POLICY=PASSIVE NPULOOP_INT_EVAL=10000 NPULOOP_MODELS=mnv2_050_relu6
for s in e2_ptq_grid e7_requant_ablation e5_calibration; do
  echo "=== side $s $(date -u +%H:%M:%S)" >> results/logs/run_all.log
  python3 experiments/$s.py >> results/logs/$s.log 2>&1 || echo "FAILED side $s" >> results/logs/run_all.log
done
echo "=== side done $(date -u +%H:%M:%S)" >> results/logs/run_all.log
