#!/usr/bin/env bash
# run_queue.sh -- runs the training plan one job at a time (one GPU).
#
#   nohup bash run_queue.sh > logs/queue.txt 2>&1 &
#   tail -f logs/queue.txt              # progress of the queue
#   tail -f logs/<job>.txt              # progress of the current job
#
# Waits until no other train_global.py is running, then runs JOBS in
# order (most important first). A failed job is logged and the queue
# continues. Comment out lines in JOBS to skip them. Kill everything with:
#   pkill -f run_queue.sh; pkill -f train_global.py

cd "$(dirname "$0")"
mkdir -p logs

# epochs per (bin) model. Every job takes roughly the same time, since the
# total number of gradient steps does not depend on the number of bins:
# job time ~= EPOCHS * (seconds per epoch of the 1-bin run).
EPOCHS=30

JOBS=(
  # ---- phase 1: 1 bin, data loss only
  "--arch deeponet --tend predicted --n_bins 1"
  "--arch crkpinn  --tend predicted --n_bins 1"
  # ---- phase 2: bin resolution, data loss only
  "--arch deeponet --tend predicted --n_bins 2"
  "--arch crkpinn  --tend predicted --n_bins 2"
  "--arch deeponet --tend predicted --n_bins 5"
  "--arch crkpinn  --tend predicted --n_bins 5"
  "--arch deeponet --tend predicted --n_bins 10"
  "--arch crkpinn  --tend predicted --n_bins 10"
  # ---- extras if there is time left: second seed (run-to-run noise)
  "--arch deeponet --tend predicted --n_bins 1  --seed 1"
  "--arch crkpinn  --tend predicted --n_bins 1  --seed 1"
  "--arch deeponet --tend predicted --n_bins 10 --seed 1"
  "--arch crkpinn  --tend predicted --n_bins 10 --seed 1"
)

while pgrep -f "python.*train_global.py" > /dev/null; do
  echo "$(date '+%F %T')  waiting for running train_global.py to finish..."
  sleep 300
done

summary() {
python - <<'EOF'
import glob, json, pandas as pd
rows = []
for f in sorted(glob.glob("results/global/*/metrics.csv")):
    d = pd.read_csv(f)
    r = d[d["T0 range"] == "all"].iloc[0].to_dict()
    run = f.split("/")[-2]
    r["run"] = run
    try:
        c = json.load(open(f"results/global/{run}/config.json"))
        r["train h"] = sum(c["train_seconds"].values()) / 3600
        r["test inference s"] = c.get("test_inference_seconds")
    except Exception:
        pass
    rows.append(r)
if rows:
    s = pd.DataFrame(rows).set_index("run").drop(columns="T0 range")
    s.to_csv("results/global/summary.csv")
    pd.set_option("display.width", 250)
    print(s.to_string(float_format=lambda x: f"{x:.3g}"))
EOF
}

for args in "${JOBS[@]}"; do
  name=$(echo $args | sed 's/--//g; s/ /_/g')
  log="logs/${name}.txt"
  echo "$(date '+%F %T')  START  $args  -> $log"
  if python -u train_global.py $args --epochs $EPOCHS > "$log" 2>&1; then
    echo "$(date '+%F %T')  DONE   $args"
  else
    echo "$(date '+%F %T')  FAILED $args  (see $log)"
  fi
  summary > /dev/null 2>&1      # keep results/global/summary.csv up to date
done

summary
echo "$(date '+%F %T')  queue finished"