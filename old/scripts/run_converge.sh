#!/usr/bin/env bash
# run_converge.sh -- bin-resolution study, every model trained until converged.
#
#   nohup bash run_converge.sh > logs_converge.txt 2>&1 &
#   cat logs_converge.txt            # progress
#
# Every bin model gets the same step budget per epoch and stops when its
# validation loss has not improved by 0.1 % for 10 epochs (max 100 epochs).
# Results: results/global/<arch>_predicted_b<N>_conv/
# Waits until no other training is running. Survives a lost connection.

cd "$(dirname "$0")"
mkdir -p logs

# use the TEP4541 environment no matter what the terminal had active
eval "$(conda shell.bash hook)" && conda activate TEP4541
if ! python -c "import numpy, pandas, torch, cantera" 2>/dev/null; then
  echo "$(date '+%F %T')  ERROR: TEP4541 environment not usable. Stopping."
  exit 1
fi

MAX_EPOCHS=100

# CRK-PINN first (~20 s per epoch), then DeepONet (~135 s per epoch)
JOBS=(
  "--arch crkpinn  --n_bins 1"
  "--arch crkpinn  --n_bins 2"
  "--arch crkpinn  --n_bins 5"
  "--arch crkpinn  --n_bins 10"
  "--arch deeponet --n_bins 1"
  "--arch deeponet --n_bins 2"
  "--arch deeponet --n_bins 5"
)

# wait for other training (e.g. the loss queue) to finish
while pgrep -f "python.*train_global.py" > /dev/null || pgrep -f "run_losses.sh" > /dev/null; do
  echo "$(date '+%F %T')  waiting for other training to finish..."
  sleep 300
done

for args in "${JOBS[@]}"; do
  name=$(echo $args | sed 's/--//g; s/ \+/_/g')_conv
  echo "$(date '+%F %T')  START  $args"
  if python -u train_global.py $args --converge --epochs $MAX_EPOCHS > "logs/${name}.txt" 2>&1; then
    echo "$(date '+%F %T')  DONE   $args"
  else
    echo "$(date '+%F %T')  FAILED $args  (see logs/${name}.txt)"
  fi
done
echo "$(date '+%F %T')  finished"