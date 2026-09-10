#!/bin/bash
# Full reproduction, v2 (validation-split checkpoints): train the five baselines sequentially while the
# experiment scripts sweep runs/ (each skips finished configs), then the customer intake, E3 and the
# timing run (E10) alone on an idle box. Safe to re-run after an interruption: training resumes from
# state.pt and every experiment script skips records it already has.
cd "$(dirname "$0")/.."
export OMP_WAIT_POLICY=PASSIVE NPULOOP_INT_EVAL=${NPULOOP_INT_EVAL:-10000}
# The integer engines allocate and free large int64 temporaries per node; with glibc's default mmap threshold
# every one is a fresh mmap + page-fault storm (measured: 64% of wall-clock in the kernel on a 500-image batch).
# Keep freed blocks on the heap instead: 2.4x faster evaluation, identical results.
export MALLOC_MMAP_THRESHOLD_=2147483647 MALLOC_TRIM_THRESHOLD_=2147483647 MALLOC_TOP_PAD_=268435456
mkdir -p results/logs runs
RUNS=${NPULOOP_RUNS:-runs}
QUEUE_DONE=$RUNS/QUEUE_DONE
rm -f "$QUEUE_DONE" results/ALL_DONE

done_run() { [ -f "$RUNS/$1/log.json" ] && python3 -c "import json,sys; sys.exit(0 if 'test_acc' in json.load(open('$RUNS/$1/log.json')) else 1)" 2>/dev/null; }
train() { name=$1; shift
  if done_run "$name"; then echo "skip $name" >> results/logs/train_queue.log; return; fi
  echo "=== train $name $(date -u +%H:%M:%S)" >> results/logs/train_queue.log
  python3 -m npuloop.zoo.train --threads ${NPULOOP_TRAIN_THREADS:-4} --resume --out "$RUNS/$name" "$@" >> "results/logs/train_$name.log" 2>&1 \
    || echo "FAILED train $name" >> results/logs/train_queue.log
}
queue() {
  train resnet20_relu   --arch resnet --depth 20 --width 16 --act relu --epochs 30
  train resnet20_silu   --arch resnet --depth 20 --width 16 --act silu --epochs 30
  train cust_inception  --arch inception --width 32 --act relu --epochs 30
  train cust_vit        --arch vit --dim 128 --depth 6 --heads 4 --patch 4 --mlp-ratio 2 --act gelu --epochs 40 --lr 0.001 --wd 0.05 --optimizer adamw
  train mnv2_050_relu6  --arch mobilenetv2 --width-mult 0.5 --act relu6 --epochs 30
  echo QUEUE_DONE > "$QUEUE_DONE"
  echo "=== queue done $(date -u +%H:%M:%S)" >> results/logs/train_queue.log
}
run() { s=$1; echo "=== $s $(date -u +%H:%M:%S)" >> results/logs/run_all.log
  NPULOOP_THREADS=${NPULOOP_EXP_THREADS:-2} python3 experiments/$s.py >> results/logs/$s.log 2>&1 || echo "FAILED $s $(date -u +%H:%M:%S)" >> results/logs/run_all.log; }
pass() { for s in e1_baselines e2_ptq_grid e7_requant_ablation e5_calibration e4_surgery e6_pruning e9_customer_intake; do run $s; done; }
experiments() {
  while true; do
    pass
    if [ -f "$QUEUE_DONE" ]; then break; fi
    sleep 120
  done
  pass                                  # one more full pass once every checkpoint exists
  run e3_lint_vs_drop
}
queue & experiments & wait
run e8_scalesim                         # shape-only; skips models already recorded
rm -f results/e10_engine_timing.json    # timing is re-measured on the idle box, single thread
run e10_engine_timing
echo ALL_DONE >> results/logs/run_all.log; touch results/ALL_DONE
