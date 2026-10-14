#!/usr/bin/env bash
cd "$(dirname "$0")"
mkdir -p logs
for args in "--arch deeponet --tend predicted --n_bins 2 --seed 1" \
            "--arch deeponet --tend predicted --n_bins 5 --seed 1"; do
  name=$(echo $args | sed 's/--//g; s/ /_/g')
  echo "$(date '+%F %T')  START  $args"
  if python -u train_global.py $args --epochs 30 > "logs/${name}.txt" 2>&1; then
    echo "$(date '+%F %T')  DONE   $args"
  else
    echo "$(date '+%F %T')  FAILED $args  (see logs/${name}.txt)"
  fi
done
echo "$(date '+%F %T')  finished"
