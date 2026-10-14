# bo_search.py
"""
Bayesian optimization of the physics-loss weights (one bin, current data).

  python bo_search.py                          # DeepONet, 10 bins, bin 7
  python bo_search.py --n_iter 30 --epochs 40  # longer search
  Cost: --epochs 40 x --steps_per_epoch 1000 = at most 40k steps per evaluation
  (~35 min for DeepONet on the M4000, ~45 min with the entropy term), so the
  default 6 + 20 evaluations take roughly 15-20 h.
  (stop with Ctrl+C or kill at any time; running it again continues where it stopped)

What is optimized
  Weights of the element, enthalpy and entropy loss terms. The mole loss is not
  searched (lambda_mole = 0): sum_e Z_e = sum_k Y_k for any Y, so element
  conservation implies mass conservation, L_mole <= N_e^2 * L_element.
  Each weight is searched as
      lambda_t = r_t * L_data / L_t ,   log10 r_t in [-3, 0]
  where L_data / L_t are the validation losses of the reference run's saved model
  for this bin (computed once with train_global.py --val_losses_only), so
  r_t = 0.1 means "term t is about 10 % of the data loss". The learning rate and batch size are fixed (train_global.py defaults).

Objective (lower is better)
  val_mape_final from train_global.py: mean over the bin's VALIDATION
  trajectories of the final-state percentage error of T + largest species
  (CO, CH4, O2, C2H2, H2O, CO2) + tracked species (CH2O, H2O2, OH, C2H6, NO),
  species only where Y_true > 1e-8. The test set is never used.
  The validation split is fixed (--val_seed); every evaluation trains with a new
  seed, so the objective is noisy.

Method
  Gaussian process on log(objective) with a Matern 5/2 kernel (one length scale
  per weight) + a white-noise kernel (seed noise), inputs scaled to [0, 1].
  Start: --n_init points from a Latin hypercube. Then each new point maximizes
  Expected Improvement (minimization, xi = --xi) over the box; the incumbent is
  the lowest GP mean at the evaluated points (standard choice for noisy objectives).

Each evaluation = one train_global.py run:
  --arch A --n_bins N --only_bin K --converge --epochs E --val_seed V --seed S
  --no_test --lambda_element ... --lambda_enthalpy ... --lambda_entropy ...
  results in results/global/bo/<tag>/eval_XXX/, log in <out>/logs/eval_XXX.txt

Outputs (<out>, default results/bo/<arch>_b<N>_bin<K>/):
  reference.json   L_data / L_t ratios used to scale the weights
  evals.csv        one row per evaluation (weights, seed, objective, GP prediction, EI)
  summary.json     best evaluated point + point with the lowest GP mean
  bo_convergence.png, bo_slices.png
"""
import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

p = argparse.ArgumentParser()
p.add_argument("--arch", default="deeponet", choices=["deeponet", "crkpinn"])
p.add_argument("--n_bins", type=int, default=10)
p.add_argument("--bin", type=int, default=7)
p.add_argument("--ref_run", default="results/global/deeponet_predicted_b10_s2_conv",
               help="converged run with the same arch/bins; its saved model gives L_data/L_t")
p.add_argument("--terms", default="element,enthalpy,entropy")
p.add_argument("--log_r_min", type=float, default=-3.0)
p.add_argument("--log_r_max", type=float, default=0.0)
p.add_argument("--n_init", type=int, default=6, help="Latin hypercube start points")
p.add_argument("--n_iter", type=int, default=20, help="Bayesian optimization steps after the start points")
p.add_argument("--epochs", type=int, default=40, help="max epochs per evaluation (with --converge)")
p.add_argument("--steps_per_epoch", type=int, default=1000,
               help="passed to train_global.py (1000 steps; a full pass over all data is ~2600)")
p.add_argument("--val_seed", type=int, default=0, help="fixed train/validation split")
p.add_argument("--seed_start", type=int, default=100, help="evaluation i trains with seed seed_start + i")
p.add_argument("--xi", type=float, default=0.01, help="EI exploration parameter (on log objective)")
p.add_argument("--out", default=None)
p.add_argument("--fake", action="store_true", help="test the loop with a synthetic objective (no training)")
args = p.parse_args()

TERMS = args.terms.split(",")
D = len(TERMS)
LO, HI = args.log_r_min, args.log_r_max
tag = f"{args.arch}_b{args.n_bins}_bin{args.bin}" + ("_fake" if args.fake else "")
out = Path(args.out or f"results/bo/{tag}")
(out / "logs").mkdir(parents=True, exist_ok=True)
csv_path = out / "evals.csv"
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ======================================================================= reference ratios
def reference_ratios():
    """L_data / L_t of the reference run's saved model for this bin, on the validation split.
    Computed with train_global.py --val_losses_only (all terms, also those the reference
    run was not trained or logged with)."""
    path = out / "reference.json"
    if path.exists():
        return json.load(open(path))["ratio"]
    if args.fake:
        ratio, losses = {t: 1.0 for t in TERMS}, None
    else:
        ref_name = Path(args.ref_run).name
        cmd = [sys.executable, "train_global.py", "--arch", args.arch, "--n_bins", str(args.n_bins),
               "--only_bin", str(args.bin), "--val_seed", str(args.val_seed), "--run_name", ref_name,
               "--val_losses_only"]
        print("reference losses:", " ".join(cmd), flush=True)
        with open(out / "logs" / "reference.txt", "w") as log:
            ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        lpath = Path("results/global") / ref_name / "val_losses.json"
        if ret.returncode != 0 or not lpath.exists():
            raise SystemExit(f"reference losses failed, see {out}/logs/reference.txt")
        losses = json.load(open(lpath))["bins"][str(args.bin)]
        ratio = {}
        for t in TERMS:
            L_t = losses[t]
            if not L_t > 0:
                print(f"WARNING: L_{t} = {L_t} for the reference model; using 1e-12 for the ratio")
                L_t = 1e-12
            ratio[t] = losses["data"] / L_t
        print(f"reference {args.ref_run}, bin {args.bin}: " + ", ".join(f"L_{n} {v:.3e}" for n, v in losses.items()))
    json.dump({"ref_run": args.ref_run, "bin": args.bin, "losses": losses, "ratio": ratio},
              open(path, "w"), indent=2)
    return ratio


RATIO = reference_ratios()
print("L_data / L_t:", {t: f"{v:.3e}" for t, v in RATIO.items()}, flush=True)


# ======================================================================= one evaluation
def evaluate(i, log_r):
    """Train one model with these weights, return (objective, details)."""
    lam = {t: 10 ** log_r[d] * RATIO[t] for d, t in enumerate(TERMS)}
    seed = args.seed_start + i
    t0 = time.time()
    if args.fake:
        x = (np.asarray(log_r) - LO) / (HI - LO)
        val = float(np.exp(1.0 + 2.0 * np.sum((x - 0.35) ** 2) + 0.05 * np.random.default_rng(seed).normal()))
        return val, {"lambda": lam, "seed": seed, "minutes": 0.0, "median": val, "p95": val}
    run_name = f"bo/{tag}/eval_{i:03d}"
    cmd = [sys.executable, "train_global.py", "--arch", args.arch, "--n_bins", str(args.n_bins),
           "--only_bin", str(args.bin), "--converge", "--epochs", str(args.epochs),
           "--val_seed", str(args.val_seed), "--seed", str(seed), "--no_test", "--run_name", run_name,
           "--lambda_mole", "0"]
    if args.steps_per_epoch:
        cmd += ["--steps_per_epoch", str(args.steps_per_epoch)]
    for t in TERMS:
        cmd += [f"--lambda_{t}", f"{lam[t]:.6g}"]
    with open(out / "logs" / f"eval_{i:03d}.txt", "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    mpath = Path("results/global") / run_name / "val_metric.json"
    if ret.returncode != 0 or not mpath.exists():
        raise SystemExit(f"evaluation {i} failed (exit {ret.returncode}), see {out}/logs/eval_{i:03d}.txt")
    vm = json.load(open(mpath))
    return vm["val_mape_final"], {"lambda": lam, "seed": seed, "minutes": (time.time() - t0) / 60,
                                  "median": vm["val_mape_final_median"], "p95": vm["val_mape_final_p95"]}


# ======================================================================= GP + EI
def to_unit(log_r):
    return (np.asarray(log_r) - LO) / (HI - LO)


def fit_gp(X, y):
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.full(D, 0.5), length_scale_bounds=(1e-2, 1e1), nu=2.5)
              + WhiteKernel(1e-2, (1e-6, 1e0)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10,
                                  random_state=0)
    gp.fit(X, y)
    return gp


def expected_improvement(gp, X, best):
    mu, sd = gp.predict(np.atleast_2d(X), return_std=True)
    sd = np.maximum(sd, 1e-9)
    imp = best - mu - args.xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def next_point(gp, X_obs, seed):
    best = gp.predict(X_obs).min()                       # incumbent: lowest GP mean at evaluated points
    cand = qmc.Sobol(D, seed=seed).random(2048)
    ei = expected_improvement(gp, cand, best)
    starts = cand[np.argsort(-ei)[:5]]
    best_x, best_ei = cand[np.argmax(ei)], ei.max()
    for s in starts:
        r = minimize(lambda x: -expected_improvement(gp, x, best)[0], s, bounds=[(0, 1)] * D,
                     method="L-BFGS-B")
        if -r.fun > best_ei:
            best_x, best_ei = r.x, -r.fun
    return best_x, float(best_ei)


# ======================================================================= loop
rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
if rows:
    print(f"continuing: {len(rows)} evaluations already in {csv_path}", flush=True)
init = qmc.LatinHypercube(D, seed=args.val_seed).random(args.n_init) * (HI - LO) + LO
n_total = args.n_init + args.n_iter

for i in range(len(rows), n_total):
    X = np.array([[r[f"log_r_{t}"] for t in TERMS] for r in rows]).reshape(-1, D)
    y = np.log([r["val_mape_final"] for r in rows])
    if i < args.n_init:
        phase, log_r, ei, mu, sd = "init", init[i], np.nan, np.nan, np.nan
    else:
        gp = fit_gp(to_unit(X), y)
        x_unit, ei = next_point(gp, to_unit(X), seed=i)
        mu, sd = gp.predict(x_unit[None], return_std=True)
        mu, sd = float(np.exp(mu[0])), float(sd[0])
        phase, log_r = "bo", x_unit * (HI - LO) + LO
    print(f"\neval {i:3d} ({phase}): log10 r = " + ", ".join(f"{t} {v:+.2f}" for t, v in zip(TERMS, log_r))
          + (f" | GP predicts {mu:.3g} %, EI {ei:.3g}" if phase == "bo" else ""), flush=True)
    val, info = evaluate(i, log_r)
    row = {"i": i, "phase": phase, "seed": info["seed"],
           **{f"log_r_{t}": float(v) for t, v in zip(TERMS, log_r)},
           **{f"lambda_{t}": info["lambda"][t] for t in TERMS},
           "val_mape_final": val, "val_mape_median": info["median"], "val_mape_p95": info["p95"],
           "gp_pred": mu, "gp_log_sd": sd, "ei": ei, "minutes": info["minutes"]}
    rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    best_so_far = min(r["val_mape_final"] for r in rows)
    print(f"   -> val MAPE {val:.4g} % (best so far {best_so_far:.4g} %), {info['minutes']:.1f} min", flush=True)

# ======================================================================= summary + plots
E = pd.DataFrame(rows)
X = to_unit(E[[f"log_r_{t}" for t in TERMS]].to_numpy())
y = np.log(E["val_mape_final"].to_numpy())
gp = fit_gp(X, y)
mu_obs = gp.predict(X)
i_obs = int(np.argmin(E["val_mape_final"]))
i_gp = int(np.argmin(mu_obs))
summary = {
    "n_evals": len(E), "kernel": str(gp.kernel_),
    "best_evaluated": {"i": int(E["i"][i_obs]), "val_mape_final": float(E["val_mape_final"][i_obs]),
                       **{f"log_r_{t}": float(E[f"log_r_{t}"][i_obs]) for t in TERMS},
                       **{f"lambda_{t}": float(E[f"lambda_{t}"][i_obs]) for t in TERMS}},
    "best_gp_mean": {"i": int(E["i"][i_gp]), "gp_mean_val_mape": float(np.exp(mu_obs[i_gp])),
                     **{f"log_r_{t}": float(E[f"log_r_{t}"][i_gp]) for t in TERMS},
                     **{f"lambda_{t}": float(E[f"lambda_{t}"][i_gp]) for t in TERMS}},
    "noise_note": "best_gp_mean is less affected by seed noise than best_evaluated",
}
json.dump(summary, open(out / "summary.json", "w"), indent=2)
print("\n" + json.dumps(summary, indent=2))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(E["i"], E["val_mape_final"], "o", color="#2a78d6", label="evaluation")
ax.plot(E["i"], E["val_mape_final"].cummin(), "-", color="#0b0b0b", label="best so far")
ax.axvline(args.n_init - 0.5, color="#52514e", ls=":", lw=1)
ax.set_xlabel("evaluation"), ax.set_ylabel("validation final-state MAPE [%]"), ax.set_yscale("log")
ax.legend(frameon=False)
fig.tight_layout(), fig.savefig(out / "bo_convergence.png", dpi=150)

xb = X[i_gp]
fig, axs = plt.subplots(1, D, figsize=(4 * D, 3.2), sharey=True)
for d, ax in enumerate(np.atleast_1d(axs)):
    grid = np.tile(xb, (200, 1))
    grid[:, d] = np.linspace(0, 1, 200)
    m, s = gp.predict(grid, return_std=True)
    lr = grid[:, d] * (HI - LO) + LO
    ax.plot(lr, np.exp(m), color="#2a78d6")
    ax.fill_between(lr, np.exp(m - s), np.exp(m + s), color="#2a78d6", alpha=0.2)
    ax.plot(E[f"log_r_{TERMS[d]}"], E["val_mape_final"], "o", ms=3, color="#52514e")
    ax.set_xlabel(f"log10 r, {TERMS[d]}"), ax.set_yscale("log")
np.atleast_1d(axs)[0].set_ylabel("val. MAPE [%] (GP mean ± 1 sd)")
fig.suptitle("GP slices through the point with the lowest GP mean (dots: all evaluations)")
fig.tight_layout(), fig.savefig(out / "bo_slices.png", dpi=150)
print(f"saved {out}/")