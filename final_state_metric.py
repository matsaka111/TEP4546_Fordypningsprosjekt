# final_state_metric.py
"""
Final-state percentage error E_final, per trajectory, then aggregated over the test set.

For trajectory j, at the last time point:
    S_j = tracked species (CH2O, H2O2, OH, C2H6, NO) with final Y_true > Y_FLOOR
          + every other species with final Y_true > Y_THR  (N2 and AR excluded)
    E_j = 100 * mean over {T} + S_j of |pred - true| / true          [%]

Aggregated over trajectories: mean, median, p95, max of E_j (the mean and p95
keep the large errors visible). Each component is also reported on its own.

  python final_state_metric.py
  python final_state_metric.py --runs crkpinn_predicted_b10_s2_conv,deeponet_predicted_b10_s2_conv --y_thr 1e-3

Note: runs saved before 2026-10-11 store log10(Y) in pred_test.npz as float16.
That rounding gives up to ~0.1 % relative error in Y for Y ~ 0.1, ~0.2 % for Y ~ 1e-3,
~0.45 % for Y ~ 1e-6 and ~0.9 % for Y ~ 1e-8. Re-test such runs with
train_global.py --eval_only (saves float32) before using this script.
"""
import argparse
import numpy as np
import pandas as pd
import cantera as ct

TRACKED = ["CH2O", "H2O2", "OH", "C2H6", "NO"]
EXCLUDE = ["N2", "AR"]

p = argparse.ArgumentParser()
p.add_argument("--runs", default="crkpinn_predicted_b10_s2_conv,deeponet_predicted_b10_s2_conv")
p.add_argument("--results", default="results/global")
p.add_argument("--test_csv", default="data/raw/data_test.csv")
p.add_argument("--y_thr", type=float, default=1e-3, help="mass fraction threshold for 'large' species")
p.add_argument("--y_floor", type=float, default=1e-8, help="tracked species only counted above this")
p.add_argument("--out", default="results/diagnostics/final_state_metric.csv")
args = p.parse_args()

species = ct.Solution("gri30.yaml").species_names
y_cols = [f"Y_{s}" for s in species]

# ---- truth at the last time point of each trajectory
print("loading test data (last time point per trajectory)...", flush=True)
parts = []
for ch in pd.read_csv(args.test_csv, usecols=["traj_idx", "t", "T"] + y_cols, chunksize=2_000_000):
    parts.append(ch.sort_values(["traj_idx", "t"]).groupby("traj_idx").tail(1))
truth = pd.concat(parts).sort_values(["traj_idx", "t"]).groupby("traj_idx").tail(1).set_index("traj_idx")
T_true = truth["T"].to_numpy(np.float64)
Y_true = truth[y_cols].to_numpy(np.float64)
NT = len(truth)

# ---- which species count, per trajectory
tracked_k = [species.index(s) for s in TRACKED]
other_k = [k for k, s in enumerate(species) if s not in TRACKED + EXCLUDE]
mask = np.zeros_like(Y_true, dtype=bool)
mask[:, tracked_k] = Y_true[:, tracked_k] > args.y_floor
mask[:, other_k] = Y_true[:, other_k] > args.y_thr
print(f"{NT} trajectories; species counted per trajectory: "
      f"median {np.median(mask.sum(1)):.0f}, min {mask.sum(1).min()}, max {mask.sum(1).max()}")

rows, comp_rows = [], []
for run in args.runs.split(","):
    d = np.load(f"{args.results}/{run}/pred_test.npz")
    idx = d["traj_idx"]
    last = np.r_[idx[1:] != idx[:-1], True]
    order = pd.Index(idx[last]).get_indexer(truth.index)
    assert (order >= 0).all(), f"{run}: trajectories missing in pred_test.npz"
    T_pred = d["T_pred"][last][order].astype(np.float64)
    Y_pred = np.power(10.0, d["log10Y_pred"][last][order].astype(np.float64))

    eT = np.abs(T_pred - T_true) / T_true
    eY = np.abs(Y_pred - Y_true) / np.where(mask, Y_true, 1.0)
    E = 100 * (eT + (eY * mask).sum(1)) / (1 + mask.sum(1))

    rows.append({"run": run, "E_mean_%": E.mean(), "E_median_%": np.median(E),
                 "E_p95_%": np.percentile(E, 95), "E_max_%": E.max(),
                 "T_relerr_mean_%": 100 * eT.mean(), "T_relerr_p95_%": 100 * np.percentile(eT, 95)})
    for k in tracked_k + other_k:
        m = mask[:, k]
        if m.sum() == 0:
            continue
        e = 100 * eY[m, k]
        comp_rows.append({"run": run, "species": species[k], "n_traj": int(m.sum()),
                          "mean_%": e.mean(), "median_%": np.median(e), "p95_%": np.percentile(e, 95)})

R = pd.DataFrame(rows)
C = pd.DataFrame(comp_rows)
pd.set_option("display.width", 250)
print("\n=== E_final per trajectory, aggregated over the test set [%] ===")
print(R.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
print("\n=== per component, mean / p95 over trajectories where it is counted [%] ===")
piv = C.pivot(index="species", columns="run", values=["n_traj", "mean_%", "p95_%"])
piv = piv.sort_values(("n_traj", C.run.iloc[0]), ascending=False)
print(piv.to_string(float_format=lambda x: f"{x:.3g}"))
R.to_csv(args.out, index=False)
C.to_csv(args.out.replace(".csv", "_species.csv"), index=False)
print(f"\nsaved {args.out}")