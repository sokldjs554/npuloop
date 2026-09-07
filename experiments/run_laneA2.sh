#!/bin/bash
cd "$(dirname "$0")/.."
export OMP_WAIT_POLICY=PASSIVE NPULOOP_THREADS=2 NPULOOP_INT_EVAL=10000
for s in e4_surgery e6_pruning e2_ptq_grid; do echo "=== A2 $s $(date -u +%H:%M:%S)" >> results/logs/run_all.log; python3 experiments/$s.py >> results/logs/$s.log 2>&1 || echo "FAILED A2 $s" >> results/logs/run_all.log; done
echo "=== A2 done $(date -u +%H:%M:%S)" >> results/logs/run_all.log
