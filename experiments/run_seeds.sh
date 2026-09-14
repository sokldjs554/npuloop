#!/usr/bin/env bash
# Seed replicates of the three CNN baselines (seed 0 lives in runs/<name>). Same hyper-parameters as
# experiments/run_v2.sh; only --seed differs, so the validation split, epochs and schedule are identical.
# Resumable: re-running skips finished runs and resumes an interrupted one from state.pt.
set -u
cd "$(dirname "$0")/.."
RUNS=${NPULOOP_RUNS:-runs}
mkdir -p results/logs
SEEDS=${NPULOOP_SEEDS:-"1 2"}
train() { name=$1; shift
  if [ -f "$RUNS/$name/log.json" ] && grep -q final_test_acc "$RUNS/$name/log.json"; then echo "skip $name" >> results/logs/train_seeds.log; return; fi
  echo "=== train $name $(date -u +%H:%M:%S)" >> results/logs/train_seeds.log
  python3 -m npuloop.zoo.train --threads "${NPULOOP_TRAIN_THREADS:-4}" --resume --out "$RUNS/$name" "$@" >> "results/logs/train_$name.log" 2>&1 \
    || echo "FAILED train $name" >> results/logs/train_seeds.log
}
for s in $SEEDS; do
  train "resnet20_relu_s$s"  --arch resnet --depth 20 --width 16 --act relu --epochs 30 --seed "$s"
  train "resnet20_silu_s$s"  --arch resnet --depth 20 --width 16 --act silu --epochs 30 --seed "$s"
done
for s in $SEEDS; do
  train "mnv2_050_relu6_s$s" --arch mobilenetv2 --width-mult 0.5 --act relu6 --epochs 30 --seed "$s"
done
echo "=== queue done $(date -u +%H:%M:%S)" >> results/logs/train_seeds.log
