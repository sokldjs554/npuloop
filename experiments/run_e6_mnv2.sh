#!/bin/bash
cd "$(dirname "$0")/.."
export OMP_WAIT_POLICY=PASSIVE NPULOOP_THREADS=2 NPULOOP_MODELS=mnv2_050_relu6 NPULOOP_E6_NAME=e6_pruning_mnv2
echo "=== C e6_pruning(mnv2) $(date -u +%H:%M:%S)" >> results/logs/run_all.log
python3 experiments/e6_pruning.py >> results/logs/e6_pruning_mnv2.log 2>&1 || echo "FAILED C e6 mnv2" >> results/logs/run_all.log
echo "=== C done $(date -u +%H:%M:%S)" >> results/logs/run_all.log
