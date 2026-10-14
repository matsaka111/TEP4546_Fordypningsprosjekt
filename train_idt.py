# train_idt.py
"""
Train the ignition delay time predictor (one sample per trajectory).

  python train_idt.py
  python train_idt.py --n_train 10000        # learning-curve point
  python train_idt.py --rel_weight 1.0       # add CRK-PINN relative-error term

Uses the output of data/generate_data.py:
  train/val: data/raw/data_train.csv  (split 90/10 by trajectory)
  test:      data/raw/data_test.csv   (separate seed)

Outputs (model_weights/):
  idt_predictor.pt                  state_dict + use_P, n_in
  idt_table_{train,test}.csv        per-trajectory cache (raw CSVs are huge)
  idt_parity.png, idt_arrhenius.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fordypning.idt_predictor import (extract_idt_table, igniting_only, make_features,
                                      fit_linear_baseline, train_idt_predictor,
                                      rel_err, error_table)

p = argparse.ArgumentParser()
p.add_argument("--train_csv", default="data/raw/data_train.csv")
p.add_argument("--test_csv", default="data/raw/data_test.csv")
p.add_argument("--n_bins", type=int, default=10, help="only for the per-bin error report")
p.add_argument("--T0_lo", type=float, default=500.0)
p.add_argument("--T0_hi", type=float, default=2000.0)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--n_train", type=int, default=None, help="subsample train set")
p.add_argument("--rel_weight", type=float, default=0.0)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
out = Path("model_weights")
out.mkdir(exist_ok=True)


def load_table(csv_path, name):
    cache = out / f"idt_table_{name}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    print(f"extracting per-trajectory table from {csv_path} (one-off, slow)...")
    df = extract_idt_table(csv_path)
    df.to_csv(cache, index=False)
    return df


# ---- 1. per-trajectory tables
df_tr_all = load_table(args.train_csv, "train")
df_te_all = load_table(args.test_csv, "test")
for name, d in [("train", df_tr_all), ("test", df_te_all)]:
    msg = f"{name}: {len(d)} traj, {int(d['ignited'].sum())} ignited"
    if "reached_end" in d:
        n_unsure = int((~d["ignited"] & ~d["reached_end"]).sum())
        msg += f", {n_unsure} non-ignited with scout NOT reaching SIM_TIME (label uncertain)"
    print(msg)

# ---- 2. regression only on igniting trajectories (non-igniting = censored)
df_tr, df_te = igniting_only(df_tr_all), igniting_only(df_te_all)
use_P = bool(df_tr["P"].std() > 1.0)
print(f"features: 1000/T0, ln(phi){', ln(P)' if use_P else ''}   (P varies: {use_P})")

def xy(d):
    X = make_features(d["T0"], d["phi"], d["P"], use_P)
    return X, np.log(d["t_ign"].to_numpy()).astype(np.float32), d["T0"].to_numpy()

X_all, y_all, _ = xy(df_tr)
X_te, y_te, T0_te = xy(df_te)

rng = np.random.default_rng(args.seed)
perm = rng.permutation(len(X_all))
n_va = len(perm) // 10
va, tr = perm[:n_va], perm[n_va:]
if args.n_train:
    tr = tr[:args.n_train]
print(f"train {len(tr)}  val {len(va)}  test {len(X_te)}")

bin_edges = np.linspace(args.T0_lo, args.T0_hi, args.n_bins + 1)
idt_te = np.exp(y_te)

# ---- 3. linear Arrhenius baseline
base = fit_linear_baseline(X_all[tr], y_all[tr])
print("\nlinear baseline (test):")
print(error_table(T0_te, idt_te, np.exp(base(X_te)), bin_edges).to_string(index=False))

# ---- 4. MLP
model = train_idt_predictor(X_all[tr], y_all[tr], X_all[va], y_all[va], use_P=use_P,
                            device=device, n_epochs=args.epochs, rel_weight=args.rel_weight)
model.eval()
with torch.no_grad():
    idt_hat = torch.exp(model.ln_idt(torch.tensor(X_te, device=device))).cpu().numpy()
print("\nMLP (test):")
print(error_table(T0_te, idt_te, idt_hat, bin_edges).to_string(index=False))

tag = f"_n{args.n_train}" if args.n_train else ""
torch.save({"state_dict": model.state_dict(), "use_P": use_P, "n_in": int(X_te.shape[1])},
           out / f"idt_predictor{tag}.pt")

# ---- 5. plots
fig, ax = plt.subplots(figsize=(5, 5))
ax.loglog(idt_te, idt_hat, ".", ms=2, alpha=0.4)
lim = [idt_te.min(), idt_te.max()]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("true IDT [s]"); ax.set_ylabel("predicted IDT [s]")
ax.set_title(f"test, median rel. err {np.median(rel_err(idt_hat, idt_te)):.2%}")
fig.tight_layout(); fig.savefig(out / f"idt_parity{tag}.png", dpi=150)

fig, ax = plt.subplots(figsize=(6, 4))
sc = ax.scatter(1000 / T0_te, y_te, c=np.log(df_te["phi"]), s=2, cmap="viridis")
ax.scatter(1000 / T0_te, np.log(idt_hat), s=1, c="r", alpha=0.3, label="MLP")
ax.set_xlabel("1000 / T0 [1/K]"); ax.set_ylabel("ln IDT [ln s]")
fig.colorbar(sc, label="ln(phi)"); ax.legend(markerscale=5)
fig.tight_layout(); fig.savefig(out / f"idt_arrhenius{tag}.png", dpi=150)
print(f"\nsaved model + plots to {out}/")