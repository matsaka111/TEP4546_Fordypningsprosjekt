# crkpinn_onestep.py
"""
CRK-PINN: one-step error vs rollout error on the test set, per T0 range.

One-step: every step starts from the TRUE state (T_i, Y_i) and predicts
(T_{i+1}, Y_{i+1}) -- this is what the model is trained on.
Rollout:  the model is fed its own predictions for ~100 steps (from
pred_test.npz, written by train_global.py).

If one-step errors are similar across bin resolutions but rollout errors
differ by 10-100x, the difference comes from error amplification in the
rollout, not from how well each model fits single steps.

  python crkpinn_onestep.py
  python crkpinn_onestep.py --runs crkpinn_predicted_b1,crkpinn_predicted_b10

Outputs (results/diagnostics/):
  crkpinn_onestep_vs_rollout.csv   run x T0 range: one-step / rollout |dT| (mean,
                                   median) and amplification = rollout / one-step
  plots/crkpinn_onestep_vs_rollout.png
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fordypning.dataset import get_species_names
from fordypning.models import CRKPINN
from fordypning.idt_predictor import load_idt_predictor

SIM_TIME, N_TEND = 0.5, 5
T0_LO, T0_HI, N_RANGES = 500.0, 2000.0, 10
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_ONE, C_ROLL = "#2a78d6", "#eb6834"

p = argparse.ArgumentParser()
p.add_argument("--results", default="results/global")
p.add_argument("--test_csv", default="data/raw/data_test.csv")
p.add_argument("--idt_model", default="model_weights/idt_predictor.pt")
p.add_argument("--out", default="results/diagnostics")
p.add_argument("--runs", default=None, help="comma-separated; default: all seed-0 data-only CRK-PINN runs")
args = p.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(args.out)
(out / "plots").mkdir(parents=True, exist_ok=True)

# ---- test data, same ordering as train_global.py
species = get_species_names()
y_cols = [f"Y_{s}" for s in species]
cols = ["traj_idx", "t", "T", "P", "T0", "phi", "t_ign", "ignited"] + y_cols
dt = {c: np.float32 for c in ["T", "P", "T0", "phi"] + y_cols}
dt.update({"t": np.float64, "t_ign": np.float64})
print("loading test data...", flush=True)
df = pd.concat(pd.read_csv(args.test_csv, usecols=cols, dtype=dt, chunksize=2_000_000), ignore_index=True)
df = df.sort_values(["traj_idx", "t"], kind="stable").reset_index(drop=True)
code = pd.factorize(df["traj_idx"])[0].astype(np.int64)
first = np.r_[0, np.flatnonzero(np.diff(code)) + 1]
t = df["t"].to_numpy()
T = df["T"].to_numpy(np.float32)
Y = df[y_cols].to_numpy(np.float32)
T0 = df["T0"].to_numpy(np.float32)[first]
phi = df["phi"].to_numpy(np.float32)[first]
P = df["P"].to_numpy(np.float32)[first]
ign = df["ignited"].to_numpy().astype(bool)[first]
t_ign = df["t_ign"].to_numpy()[first]
del df
NT = len(first)

pair = np.flatnonzero(code[:-1] == code[1:])          # row i -> i+1, same trajectory
pc = code[pair]
dtp = (t[pair + 1] - t[pair]).astype(np.float32)

rng_edges = np.linspace(T0_LO, T0_HI, N_RANGES + 1)
labels = [f"[{a:.0f},{b:.0f})" for a, b in zip(rng_edges[:-1], rng_edges[1:])]
rng_of = np.clip(np.digitize(T0, rng_edges) - 1, 0, N_RANGES - 1)

idt_model = load_idt_predictor(args.idt_model, device) if Path(args.idt_model).exists() else None


def t_end_for(mode):
    idt = np.full(NT, SIM_TIME)
    if mode == "cantera":
        idt[ign] = t_ign[ign]
    else:
        idt[ign] = idt_model.predict_idt(T0[ign], phi[ign]).cpu().numpy()
    return (N_TEND * idt).astype(np.float32)


if args.runs:
    runs = args.runs.split(",")
else:
    runs = sorted(Path(f).parent.name for f in glob.glob(f"{args.results}/crkpinn_*/pred_test.npz"))
    runs = [r for r in runs if "_s" not in r.split("predicted")[-1] and "_l" not in r.split("predicted")[-1]]
import re
runs = sorted(runs, key=lambda r: int(re.search(r"_b(\d+)", r).group(1)) if re.search(r"_b(\d+)", r) else 0)
print(f"runs: {runs}")

dev = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
rec = []
for run in runs:
    mfiles = sorted(glob.glob(f"{args.results}/{run}/model_bin*.pt"),
                    key=lambda f: int(Path(f).stem.replace("model_bin", "")))
    ck = [torch.load(f, map_location=device) for f in mfiles]
    edges = [c["bin_edges"] for c in ck]
    t_end = t_end_for(ck[0].get("tend", "predicted"))
    traj_model = np.full(NT, -1)
    for k, (lo, hi) in enumerate(edges):
        traj_model[(T0 >= lo) & ((T0 < hi) | (k == len(edges) - 1))] = k

    # one-step predictions from the true state
    T_one = np.full(len(pair), np.nan, np.float32)
    with torch.no_grad():
        for k, c in enumerate(ck):
            m = CRKPINN(**c["kwargs"]).to(device)
            m.load_state_dict(c["state_dict"])
            m.eval()
            sel = np.flatnonzero(traj_model[pc] == k)
            for i in range(0, len(sel), 65536):
                s = sel[i:i + 65536]
                r, cc = pair[s], pc[s]
                Tp, _, _ = m(dev(T[r]), dev(Y[r]), dev(T0[cc]), dev(P[cc]), dev(dtp[s]), dev(t_end[cc]))
                T_one[s] = Tp.cpu().numpy()
    one_err = np.abs(T_one - T[pair + 1])

    # rollout error from the saved predictions (same rows, excluding each first point)
    d = np.load(f"{args.results}/{run}/pred_test.npz")
    roll_err = np.abs(d["T_pred"][pair + 1] - T[pair + 1])

    for r in range(N_RANGES):
        msk = rng_of[pc] == r
        if not msk.any():
            continue
        o, ro = one_err[msk], roll_err[msk]
        rec.append({"run": run, "n_bins": len(ck), "T0 range": labels[r],
                    "onestep_mean_K": o.mean(), "onestep_med_K": np.median(o),
                    "rollout_mean_K": ro.mean(), "rollout_med_K": np.median(ro),
                    "amplification_mean": ro.mean() / max(o.mean(), 1e-12)})
    o, ro = one_err, roll_err
    rec.append({"run": run, "n_bins": len(ck), "T0 range": "all",
                "onestep_mean_K": o.mean(), "onestep_med_K": np.median(o),
                "rollout_mean_K": ro.mean(), "rollout_med_K": np.median(ro),
                "amplification_mean": ro.mean() / max(o.mean(), 1e-12)})
    print(f"{run}: one-step mean |dT| {o.mean():.3g} K, rollout mean |dT| {ro.mean():.3g} K", flush=True)

R = pd.DataFrame(rec)
R.to_csv(out / "crkpinn_onestep_vs_rollout.csv", index=False)
pd.set_option("display.width", 200)
print("\n" + R[R["T0 range"] == "all"].to_string(index=False, float_format=lambda x: f"{x:.3g}"))

# ---- plot: one panel per run, one-step vs rollout per T0 range
plt.rcParams.update({"axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "legend.frameon": False})
fig, axs = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.8), sharey=True, squeeze=False)
x = np.arange(N_RANGES)
for ax, run in zip(axs[0], runs):
    rr = R[(R["run"] == run) & (R["T0 range"] != "all")].set_index("T0 range").reindex(labels)
    ax.semilogy(x, rr["onestep_mean_K"], color=C_ONE, marker="o", ms=5, ls="--", label="one step (true input)")
    ax.semilogy(x, rr["rollout_mean_K"], color=C_ROLL, marker="s", ms=5, ls="-", label="rollout (own predictions)")
    nb = int(R[R["run"] == run]["n_bins"].iloc[0])
    ax.set_title(f"CRK-PINN, {nb} bin" + ("s" if nb > 1 else ""), color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("T0 range [K]")
axs[0, 0].set_ylabel("mean |ΔT| [K]")
axs[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "plots" / "crkpinn_onestep_vs_rollout.png", dpi=150)
print(f"\nsaved {out}/crkpinn_onestep_vs_rollout.csv and plots/crkpinn_onestep_vs_rollout.png")