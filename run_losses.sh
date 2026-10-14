#!/usr/bin/env bash
# run_losses.sh -- physics-loss ablation: data loss + ONE physics term,
# three weights each, at the best bin resolution of each architecture.
#
#   nohup bash run_losses.sh > logs_losses.txt 2>&1 &
#   cat logs_losses.txt                 # progress
#
# Waits until no other train_global.py is running, then runs the jobs one at a
# time. Survives a lost connection (nohup). Results: results/global/<run>/,
# e.g. deeponet_predicted_b2_lm0.1. Data-only baselines already exist.
# Stop everything: pkill -f run_losses.sh; pkill -f train_global.py

cd "$(dirname "$0")"
mkdir -p logs

# activate the conda env here, so it does not matter which env the terminal had
eval "$(conda shell.bash hook)" && conda activate TEP4541
if ! python -c "import numpy, pandas, torch, cantera" 2>/dev/null; then
  echo "$(date '+%F %T')  ERROR: TEP4541 env not usable (numpy/pandas/torch/cantera missing). Stopping."
  exit 1
fi
echo "$(date '+%F %T')  using $(which python)"

DEEPONET_BINS=2      # best DeepONet resolution -- change if the seed runs say otherwise
CRKPINN_BINS=1       # best CRK-PINN resolution
EPOCHS=30
WEIGHTS="0.1 1 10"

JOBS=()
# CRK-PINN first: ~12 min per run, so these results are ready early
for term in mole element enthalpy; do
  for w in $WEIGHTS; do
    JOBS+=("--arch crkpinn --tend predicted --n_bins $CRKPINN_BINS --lambda_$term $w")
  done
done
# DeepONet: ~1.2 h per run
for term in mole element enthalpy; do
  for w in $WEIGHTS; do
    JOBS+=("--arch deeponet --tend predicted --n_bins $DEEPONET_BINS --lambda_$term $w")
  done
done

while pgrep -f "python.*train_global.py" > /dev/null; do
  echo "$(date '+%F %T')  waiting for running train_global.py to finish..."
  sleep 300
done

echo "$(date '+%F %T')  ${#JOBS[@]} jobs queued"
for args in "${JOBS[@]}"; do
  name=$(echo $args | sed 's/--//g; s/ /_/g')
  log="logs/${name}.txt"
  echo "$(date '+%F %T')  START  $args  -> $log"
  if python -u train_global.py $args --epochs $EPOCHS > "$log" 2>&1; then
    echo "$(date '+%F %T')  DONE   $args"
  else
    echo "$(date '+%F %T')  FAILED $args  (see $log)"
  fi
done
echo "$(date '+%F %T')  queue finished"