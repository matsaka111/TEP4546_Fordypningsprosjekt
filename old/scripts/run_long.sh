#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate TEP4541
python -c "import numpy, torch, pandas" || { echo "env not working"; exit 1; }
while pgrep -f train_global.py > /dev/null; do sleep 60; done
mkdir -p logs

for JOB in "crkpinn 10" "deeponet 10" "deeponet 5"; do
    set -- $JOB
    echo "$(date)  start $1 b$2" >> logs/queue_long.txt
    python train_global.py --arch $1 --n_bins $2 --seed 2 --converge --epochs 500 \
        > logs/${1}_b${2}_s2_conv.txt 2>&1
    echo "$(date)  done  $1 b$2 (exit $?)" >> logs/queue_long.txt
done
    