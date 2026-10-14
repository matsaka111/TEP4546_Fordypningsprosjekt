#!/bin/bash
# DeepONet, 1 / 2 / 5 bins, seed 1, train until converged (max 1000 epochs)
eval "$(conda shell.bash hook)"
conda activate TEP4541
python -c "import numpy, torch, pandas" || { echo "env not working"; exit 1; }

# wait for any other training to finish
while pgrep -f train_global.py > /dev/null; do sleep 60; done

mkdir -p logs
for B in 1 2 5; do
    echo "$(date)  start DeepONet b$B" >> logs/queue_deeponet_long.txt
    python train_global.py --arch deeponet --n_bins $B --seed 1 --converge --epochs 1000 \
        > logs/deeponet_b${B}_s1_conv.txt 2>&1
    echo "$(date)  done  DeepONet b$B (exit $?)" >> logs/queue_deeponet_long.txt
done
