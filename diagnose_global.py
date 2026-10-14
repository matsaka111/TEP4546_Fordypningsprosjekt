# diagnose_global.py
"""
Diagnostics over all runs in results/global/ (output of train_global.py).
Reads each run's pred_test.npz + per_traj.csv + config.json + history.csv
and compares them against data/raw/data_test.csv.

  python diagnose_global.py
  python diagnose_global.py --plot_runs deeponet_predicted_b2,crkpinn_predicted_b1

Outputs (results/diagnostics/):
  summary.csv          one row per run, all scalar metrics (definitions below)
  summary_seeds.csv    same, averaged over seeds per configuration (+ min/max)
  per_T0_range.csv     run x 150 K T0 range
  species.csv          run x species (all 53)
  plots/               see bottom of file

Metric definitions (all on the 20k-trajectory test set):
  T_MAE_K              mean |T_pred - T_true| over all test points [K]
  T_MAE_K_ign/noign    same, igniting / non-igniting trajectories only
  final_dT_med/p95/max |T_pred - T_true| at the last time point [K]
  ign_time_err_med/p95 relative error of the time where T crosses
                       T0 + 0.5*(T_final_true - T0), igniting trajectories
  ign_missed           igniting trajectories where T_pred never crosses it
  false_ign            non-igniting trajectories where max T_pred exceeds
                       max T_true by more than 200 K
  logY_MAE_tracked     mean over the tracked species of their logY_MAE
  logY_MAE_all         mean |log10 Y_pred - log10 Y_true| over all species and
                       points with Y_true > 1e-8
  major_relerr_med     median |Y_pred - Y_true| / Y_true over the largest
                       species, points with Y_true > 1e-3
  final_comp_err_med   median over trajectories of sum_k |Y_pred - Y_true| at
                       the last time point (mass fraction misallocated)
  mass_sum_err_med/p95 |sum_k Y_pred - 1| over all points
  element_err_med/p95  max over C, H, O, N of |Z_e(Y_pred) - Z_e(Y_0)| / Z_e(Y_0)
                       over all points (Z_e = element mass fraction; constant
                       in the true solution)
  entropy_viol_frac    mean over trajectories of the fraction of steps where the
                       predicted specific entropy decreases (> 1e-5 kJ/kg/K)
  entropy_viol_K_p95   95th pct over trajectories of the largest decrease,
                       as T*|ds|/1.5 kJ/kg/K, in K
  enthalpy_err_K_med/p95  |h(T_pred, Y_pred) - h_0| / cp_true at the last time
                       point, in K (h constant: adiabatic, constant pressure).
                       h = sum_k Y_k h_k(T), NOT renormalized, so mass-sum
                       errors show up here too.
  per species:
    logY_MAE           mean |log10 Y_pred - log10 Y_true|, points Y_true > 1e-8
    relerr_med         median relative error, points Y_true > 1e-3 (NaN if
                       fewer than 100 such points)
    final_relerr_med   median relative error at the last point, final Y_true > 1e-8
    peak_relerr_med    median relative error of the trajectory maximum
    peak_time_relerr_med  median relative error of the time of that maximum
                       (peak metrics: tracked species + CO, igniting
                       trajectories, true peak > 1e-8)
Note: runs saved before 2026-10-11 store log10Y_pred as float16, which alone gives
~0.1-0.9 % relative error in Y (larger for smaller Y). Re-test such runs with
train_global.py --eval_only to get float32 predictions before quoting species
errors below ~1 %.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cantera as ct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACKED = ["CH2O", "H2O2", "OH", "C2H6", "NO"]
PEAK_SPECIES = TRACKED + ["CO"]
N_LARGEST = 6                     # largest species by peak mass fraction, excl. N2/AR
Y_LOG_THR, Y_REL_THR, PEAK_THR = 1e-8, 1e-3, 1e-8
T0_LO, T0_HI, N_RANGES = 500.0, 2000.0, 10

# colours: truth = ink, runs = categorical slots in fixed order (validated)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
STYLES = ["--", "-.", ":", (0, (5, 1))]
MARKERS = ["o", "s", "^", "D"]

p = argparse.ArgumentParser()
p.add_argument("--results", default="results/global")
p.add_argument("--test_csv", default="data/raw/data_test.csv")
p.add_argument("--out", default="results/diagnostics")
p.add_argument("--suffix", default=None,
               help="which bin-study runs the bin plots/tables use: '_conv' (trained until "
                    "converged) or '' (old 30-epoch runs). Default: '_conv' if such runs exist")
p.add_argument("--plot_runs", default=None,
               help="comma-separated runs to overlay in trajectory/species plots (max 3); "
                    "default: best DeepONet, best CRK-PINN, DeepONet 1 bin")
args = p.parse_args()

out = Path(args.out)
(out / "plots").mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "legend.frameon": False, "lines.linewidth": 1.6,
})

# ======================================================================= truth
gas = ct.Solution("gri30.yaml")
species = gas.species_names
NS = len(species)
sidx = {s: i for i, s in enumerate(species)}
y_cols = [f"Y_{s}" for s in species]

print("loading test data...", flush=True)
cols = ["traj_idx", "t", "T", "P", "T0", "phi", "ignited"] + y_cols
dt = {c: np.float32 for c in ["T", "P", "T0", "phi"] + y_cols}
dt["t"] = np.float64
df = pd.concat(pd.read_csv(args.test_csv, usecols=cols, dtype=dt, chunksize=2_000_000),
               ignore_index=True)
df = df.sort_values(["traj_idx", "t"], kind="stable").reset_index(drop=True)
code = pd.factorize(df["traj_idx"])[0].astype(np.int64)
first = np.r_[0, np.flatnonzero(np.diff(code)) + 1]
last = np.r_[first[1:], len(code)] - 1
NT = len(first)
t = df["t"].to_numpy()
T_true = df["T"].to_numpy(np.float32)
Y_true = df[y_cols].to_numpy(np.float32)
P = df["P"].to_numpy(np.float32)
traj_idx_rows = df["traj_idx"].to_numpy()
T0 = df["T0"].to_numpy(np.float32)[first]
phi = df["phi"].to_numpy(np.float32)[first]
ignited = df["ignited"].to_numpy().astype(bool)[first]
del df
ign_rows = ignited[code]
rng_of = np.clip(np.digitize(T0, np.linspace(T0_LO, T0_HI, N_RANGES + 1)) - 1, 0, N_RANGES - 1)
range_labels = [f"[{a:.0f},{b:.0f})" for a, b in
                zip(np.linspace(T0_LO, T0_HI, N_RANGES + 1)[:-1],
                    np.linspace(T0_LO, T0_HI, N_RANGES + 1)[1:])]

# largest species (data-driven): highest peak mass fraction over the test set
peakmax = Y_true.max(0)
LARGEST = [species[i] for i in np.argsort(-peakmax) if species[i] not in ("N2", "AR")][:N_LARGEST]
print(f"largest species (by max mass fraction, excl. N2/AR): {LARGEST}")
print(f"tracked species: {TRACKED}")

# element mass fractions: Z = Y @ M,  M[k, e] = n_ke * aw_e / mw_k
ELEMENTS = ["C", "H", "O", "N"]
M = np.array([[gas.n_atoms(s, e) * gas.atomic_weight(e) / gas.molecular_weights[k]
               for e in ELEMENTS] for k, s in enumerate(species)], dtype=np.float64)
Z0 = Y_true[first].astype(np.float64) @ M                      # per trajectory


def h_mass_species(T):
    """Per-species specific enthalpy h_k(T) [J/kg], shape (len(T), NS)."""
    H = np.empty((len(T), NS))
    for i, Ti in enumerate(T):
        gas.TP = float(Ti), ct.one_atm
        H[i] = gas.standard_enthalpies_RT * ct.gas_constant * Ti / gas.molecular_weights
    return H


print("computing reference enthalpy / cp...", flush=True)
h0 = (Y_true[first] * h_mass_species(T_true[first])).sum(1)
cp_final = np.empty(NT)
for j in range(NT):
    gas.TPY = float(T_true[last[j]]), float(P[last[j]]), Y_true[last[j]].astype(np.float64)
    cp_final[j] = gas.cp_mass
h_true_final = (Y_true[last] * h_mass_species(T_true[last])).sum(1)
print(f"  sanity check, true data: enthalpy drift median "
      f"{np.median(np.abs(h_true_final - h0) / cp_final):.3g} K (should be ~0)")

# true peaks
pk_cols = [sidx[s] for s in PEAK_SPECIES]
grp = pd.DataFrame(Y_true[:, pk_cols], columns=PEAK_SPECIES).groupby(code)
peak_true = grp.max().to_numpy()
peak_t_true = t[grp.idxmax().to_numpy()]


def t_cross(tt, T, level):
    k = np.flatnonzero(T >= level)
    if len(k) == 0:
        return np.nan
    k = k[0]
    if k == 0:
        return tt[0]
    return tt[k - 1] + (level - T[k - 1]) * (tt[k] - tt[k - 1]) / (T[k] - T[k - 1])


cross_lvl = T0 + 0.5 * (T_true[last] - T0)
cross_true = np.array([t_cross(t[first[j]:last[j] + 1], T_true[first[j]:last[j] + 1], cross_lvl[j])
                       if ignited[j] and T_true[last[j]] - T0[j] >= 50 else np.nan
                       for j in range(NT)])
maxT_true = np.maximum.reduceat(T_true, first)


# ======================================================================= runs
def parse(run):
    parts = run.split("_")
    seed = int([q[1:] for q in parts if q.startswith("s") and q[1:].isdigit()][0]) \
        if any(q.startswith("s") and q[1:].isdigit() for q in parts) else 0
    nb = int([q[1:] for q in parts if q.startswith("b") and q[1:].isdigit()][0])
    config = "_".join(q for q in parts if not (q.startswith("s") and q[1:].isdigit()))
    return parts[0], nb, seed, config


runs = sorted(Path(f).parent.name for f in glob.glob(f"{args.results}/*/pred_test.npz"))
SUFFIX = args.suffix if args.suffix is not None else ("_conv" if any(r.endswith("_conv") for r in runs) else "")
BIN_RE = r"_predicted_b\d+" + SUFFIX          # data-only bin-study runs used in the bin plots/tables
print(f"bin-study runs: names matching '*{BIN_RE}'")
print(f"\n{len(runs)} runs: {runs}\n")
summary, per_range, per_species, preds_for_plot = [], [], [], {}

for run in runs:
    d = np.load(f"{args.results}/{run}/pred_test.npz")
    assert np.array_equal(d["traj_idx"], traj_idx_rows) and np.allclose(d["t"], t), \
        f"{run}: prediction rows do not match {args.test_csv}"
    T_pred = d["T_pred"].astype(np.float32)
    Y_pred = np.power(10.0, d["log10Y_pred"].astype(np.float32))
    cfg = json.load(open(f"{args.results}/{run}/config.json"))
    pt = pd.read_csv(f"{args.results}/{run}/per_traj.csv")
    arch, nb, seed, config = parse(run)
    print(f"{run} ...", flush=True)

    # --- temperature
    aT = np.abs(T_pred - T_true)
    fT = aT[last]
    cross_pred = np.array([t_cross(t[first[j]:last[j] + 1], T_pred[first[j]:last[j] + 1], cross_lvl[j])
                           if np.isfinite(cross_true[j]) else np.nan for j in range(NT)])
    has_ct = np.isfinite(cross_true)
    missed = has_ct & np.isnan(cross_pred)
    ign_err = np.abs(cross_pred - cross_true) / cross_true
    false_ign = (~ignited) & (np.maximum.reduceat(T_pred, first) - maxT_true > 200.0)

    # --- species
    logerr = np.abs(np.log10(np.clip(Y_pred, 1e-30, None)) - np.log10(np.clip(Y_true, 1e-30, None)))
    mlog = Y_true > Y_LOG_THR
    sp_logmae = np.where(mlog.sum(0) > 0, (logerr * mlog).sum(0) / np.maximum(mlog.sum(0), 1), np.nan)
    relerr = np.abs(Y_pred - Y_true) / np.maximum(Y_true, 1e-30)
    mrel = Y_true > Y_REL_THR
    sp_rel = np.array([np.median(relerr[mrel[:, k], k]) if mrel[:, k].sum() >= 100 else np.nan
                       for k in range(NS)])
    fY_true, fY_pred = Y_true[last], Y_pred[last]
    sp_final = np.array([np.median(np.abs(fY_pred[m, k] - fY_true[m, k]) / fY_true[m, k])
                         if (m := fY_true[:, k] > Y_LOG_THR).any() else np.nan for k in range(NS)])
    gp = pd.DataFrame(Y_pred[:, pk_cols], columns=PEAK_SPECIES).groupby(code)
    peak_pred, peak_t_pred = gp.max().to_numpy(), t[gp.idxmax().to_numpy()]
    pk_ok = ignited[:, None] & (peak_true > PEAK_THR)
    peak_rel = np.abs(peak_pred - peak_true) / np.maximum(peak_true, 1e-30)
    peak_t_rel = np.abs(peak_t_pred - peak_t_true) / peak_t_true

    major_idx = [sidx[s] for s in LARGEST]
    mm = mrel[:, major_idx]

    # --- conservation
    mass_err = np.abs(Y_pred.sum(1) - 1.0)
    Zp = Y_pred.astype(np.float64) @ M
    elem_err = (np.abs(Zp - Z0[code]) / np.maximum(Z0[code], 1e-30)).max(1)
    h_pred_final = (fY_pred * h_mass_species(T_pred[last])).sum(1)
    h_err_K = np.abs(h_pred_final - h0) / cp_final

    tracked_logmae = np.nanmean([sp_logmae[sidx[s]] for s in TRACKED])
    row = {
        "run": run, "config": config, "arch": arch, "n_bins": nb, "seed": seed,
        "T_MAE_K": aT.mean(), "T_MAE_K_ign": aT[ign_rows].mean(), "T_MAE_K_noign": aT[~ign_rows].mean(),
        "final_dT_med": np.median(fT), "final_dT_p95": np.percentile(fT, 95), "final_dT_max": fT.max(),
        "ign_time_err_med": np.nanmedian(ign_err), "ign_time_err_p95": np.nanpercentile(ign_err, 95),
        "ign_missed": int(missed.sum()), "false_ign": int(false_ign.sum()),
        "logY_MAE_tracked": tracked_logmae, "logY_MAE_all": (logerr * mlog).sum() / mlog.sum(),
        "major_relerr_med": np.median(relerr[:, major_idx][mm]),
        "final_comp_err_med": np.median(np.abs(fY_pred - fY_true).sum(1)),
        "mass_sum_err_med": np.median(mass_err), "mass_sum_err_p95": np.percentile(mass_err, 95),
        "element_err_med": np.median(elem_err), "element_err_p95": np.percentile(elem_err, 95),
        "enthalpy_err_K_med": np.median(h_err_K), "enthalpy_err_K_p95": np.percentile(h_err_K, 95),
        # entropy (second law), computed in train_global.py from full-precision predictions
        "entropy_viol_frac": pt["entropy_viol_frac"].mean() if "entropy_viol_frac" in pt else np.nan,
        "entropy_viol_K_p95": np.percentile(pt["entropy_viol_K_max"], 95) if "entropy_viol_K_max" in pt else np.nan,
        "train_h": sum(cfg.get("train_seconds", {}).values()) / 3600,
        "inference_s": cfg.get("test_inference_seconds"),
        "inference_ms_per_traj": 1e3 * cfg.get("test_inference_seconds", np.nan) / NT,
    }
    for s in TRACKED:
        row[f"logY_MAE_{s}"] = sp_logmae[sidx[s]]
    summary.append(row)

    # --- per T0 range
    traj_TMAE = np.bincount(code, aT, NT) / np.bincount(code, None, NT)
    traj_logerr_tr = {}
    for s in TRACKED:
        k = sidx[s]
        num = np.bincount(code, logerr[:, k] * mlog[:, k], NT)
        den = np.bincount(code, mlog[:, k], NT)
        traj_logerr_tr[s] = (num, den)
    for r in range(N_RANGES):
        m = rng_of == r
        if not m.any():
            continue
        rec = {"run": run, "config": config, "arch": arch, "n_bins": nb, "seed": seed,
               "T0 range": range_labels[r], "n": int(m.sum()), "n_ign": int((m & ignited).sum()),
               "T_MAE_K": traj_TMAE[m].mean(),
               "final_dT_med": np.median(fT[m]), "final_dT_p95": np.percentile(fT[m], 95),
               "ign_time_err_med": np.nanmedian(ign_err[m]) if np.isfinite(ign_err[m]).any() else np.nan,
               "ign_missed": int(missed[m].sum()), "false_ign": int(false_ign[m].sum())}
        for s in TRACKED:
            num, den = traj_logerr_tr[s]
            rec[f"logY_MAE_{s}"] = num[m].sum() / den[m].sum() if den[m].sum() else np.nan
        per_range.append(rec)

    # --- per species
    for k, s in enumerate(species):
        rec = {"run": run, "config": config, "arch": arch, "n_bins": nb, "seed": seed,
               "species": s, "tracked": s in TRACKED, "largest": s in LARGEST,
               "logY_MAE": sp_logmae[k], "relerr_med": sp_rel[k], "final_relerr_med": sp_final[k]}
        if s in PEAK_SPECIES:
            c = PEAK_SPECIES.index(s)
            ok = pk_ok[:, c]
            rec["peak_relerr_med"] = np.median(peak_rel[ok, c]) if ok.any() else np.nan
            rec["peak_time_relerr_med"] = np.median(peak_t_rel[ok, c]) if ok.any() else np.nan
        per_species.append(rec)

    preds_for_plot[run] = {"T": T_pred, "Y": Y_pred, "peak": peak_pred}
    if len(preds_for_plot) > 3:          # keep memory bounded; re-selected below
        preds_for_plot.pop(next(iter(preds_for_plot)))
    del Y_pred, logerr, relerr, Zp

S = pd.DataFrame(summary)
R = pd.DataFrame(per_range)
SP = pd.DataFrame(per_species)
S.to_csv(out / "summary.csv", index=False)
R.to_csv(out / "per_T0_range.csv", index=False)
SP.to_csv(out / "species.csv", index=False)

num_cols = [c for c in S.columns if c not in ("run", "config", "arch", "n_bins", "seed")]
G = S.groupby(["config", "arch", "n_bins"])
SS = G[num_cols].mean()
SS.insert(0, "n_seeds", G.size())
for c in ["T_MAE_K", "final_dT_p95", "logY_MAE_tracked"]:
    SS[f"{c}_min"], SS[f"{c}_max"] = G[c].min(), G[c].max()
SS = SS.reset_index().sort_values(["arch", "n_bins"])
SS.to_csv(out / "summary_seeds.csv", index=False)

pd.set_option("display.width", 250)
show = ["config", "n_seeds", "T_MAE_K", "final_dT_p95", "ign_time_err_med", "ign_missed",
        "false_ign", "logY_MAE_tracked", "major_relerr_med", "mass_sum_err_med",
        "element_err_med", "enthalpy_err_K_med", "entropy_viol_frac"]
print("\n=== summary (mean over seeds) ===")
print(SS[show].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
print("\n=== tracked species, logY_MAE (mean over seeds) ===")
print(SS[["config"] + [f"logY_MAE_{s}" for s in TRACKED]].to_string(
    index=False, float_format=lambda x: f"{x:.3g}"))

# ======================================================================= plots
P_DIR = out / "plots"

# choose runs to overlay
if args.plot_runs:
    plot_runs = args.plot_runs.split(",")[:3]
else:
    s0 = S[(S["seed"] == 0) & S["config"].str.fullmatch(r"(deeponet|crkpinn)" + BIN_RE)]
    plot_runs = []
    for a in ["deeponet", "crkpinn"]:
        sa = s0[s0["arch"] == a]
        if len(sa):
            plot_runs.append(sa.sort_values("T_MAE_K")["run"].iloc[0])
    d1 = "deeponet_predicted_b1" + SUFFIX
    if d1 in runs and d1 not in plot_runs:
        plot_runs.append(d1)
print(f"\nplot runs: {plot_runs}")
for run in plot_runs:          # reload the selected ones
    if run not in preds_for_plot:
        d = np.load(f"{args.results}/{run}/pred_test.npz")
        Yp = np.power(10.0, d["log10Y_pred"].astype(np.float32))
        gp = pd.DataFrame(Yp[:, pk_cols], columns=PEAK_SPECIES).groupby(code)
        preds_for_plot[run] = {"T": d["T_pred"], "Y": Yp, "peak": gp.max().to_numpy()}
pr = {r: preds_for_plot[r] for r in plot_runs}


def plain_log_ticks(ax):
    """Log axis with plain numbers (0.5, 2, 300) instead of 5x10^-1 etc."""
    f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(f)
    ax.yaxis.set_minor_formatter(f)


def save(fig, name):
    for ax in fig.axes:
        if ax.get_yscale() == "log" and ax.get_ylim()[1] / max(ax.get_ylim()[0], 1e-300) < 1e3:
            plain_log_ticks(ax)
    fig.tight_layout()
    fig.savefig(P_DIR / name, dpi=150)
    plt.close(fig)


# 1. bin resolution: metric vs number of bins, per architecture (seeds as points)
metrics = [("T_MAE_K", "mean |ΔT| [K]"), ("final_dT_p95", "final |ΔT|, 95th pct [K]"),
           ("logY_MAE_tracked", "log10 Y error, tracked species"),
           ("ign_time_err_med", "ignition time rel. error (median)")]
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (m, lab) in zip(axs, metrics):
    for i, a in enumerate(["deeponet", "crkpinn"]):
        sa = S[(S["arch"] == a) & (S["config"].str.fullmatch(a + BIN_RE))]
        if sa.empty:
            continue
        mean = sa.groupby("n_bins")[m].mean()
        ax.plot(mean.index, mean.values, color=SERIES[i], ls="-", marker=MARKERS[i], ms=7,
                label="DeepONet" if a == "deeponet" else "CRK-PINN")
        ax.plot(sa["n_bins"], sa[m], ls="none", marker=MARKERS[i], ms=4, color=SERIES[i], alpha=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1, 2, 5, 10]); ax.set_xticklabels(["1", "2", "5", "10"])
    ax.set_xlabel("number of T0 bins"); ax.set_title(lab, fontsize=9, color=INK)
axs[0].legend()
fig.suptitle("Bin resolution (line: mean over seeds, small points: individual seeds)", color=INK)
save(fig, "bins_comparison.png")

# 2. error vs T0 range, one panel per architecture, one line per bin resolution (seed 0)
xr = np.arange(N_RANGES)
for m, lab in [("T_MAE_K", "mean |ΔT| [K]"), ("final_dT_p95", "final |ΔT|, 95th pct [K]")] + \
              [(f"logY_MAE_{s}", f"log10 Y error, {s}") for s in TRACKED]:
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, a in zip(axs, ["deeponet", "crkpinn"]):
        ra = R[(R["arch"] == a) & (R["seed"] == 0) & R["config"].str.fullmatch(a + BIN_RE)]
        for i, nb in enumerate(sorted(ra["n_bins"].unique())):
            rr = ra[ra["n_bins"] == nb].set_index("T0 range").reindex(range_labels)
            ax.plot(xr, rr[m].values, color=SERIES[i % 4], ls="-", marker=MARKERS[i % 4], ms=5,
                    label=f"{nb} bin" + ("s" if nb > 1 else ""))
        ax.set_yscale("log"); ax.set_xticks(xr)
        ax.set_xticklabels(range_labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("T0 range [K]"); ax.set_title("DeepONet" if a == "deeponet" else "CRK-PINN",
                                                   color=INK)
        ax.legend()
    axs[0].set_ylabel(lab)
    save(fig, f"vs_T0_{m}.png")

# 3. per-species error for the plot runs (tracked + largest)
sp_list = TRACKED + [s for s in LARGEST if s not in TRACKED]
fig, ax = plt.subplots(figsize=(10, 3.8))
w = 0.8 / len(plot_runs)
for i, run in enumerate(plot_runs):
    v = SP[SP["run"] == run].set_index("species").loc[sp_list, "logY_MAE"]
    ax.bar(np.arange(len(sp_list)) + (i - (len(plot_runs) - 1) / 2) * w, v.values, w * 0.92,
           color=SERIES[i], label=run)
ax.set_xticks(np.arange(len(sp_list))); ax.set_xticklabels(sp_list)
ax.axvline(len(TRACKED) - 0.5, color=MUTED, lw=0.8)
ax.text(len(TRACKED) / 2 - 0.5, ax.get_ylim()[1] * 0.95, "tracked", ha="center", va="top", color=MUTED)
ax.text(len(TRACKED) + (len(sp_list) - len(TRACKED)) / 2 - 0.5, ax.get_ylim()[1] * 0.95,
        "largest", ha="center", va="top", color=MUTED)
ax.set_ylabel("mean |log10 Y_pred − log10 Y_true|  (Y_true > 1e-8)")
ax.legend()
save(fig, "species_error.png")

# 4. trajectories: T + largest + tracked species, truth vs plot runs
targets = [(800, 1.0), (1050, 1.0), (1200, 3.0), (1400, 1.0), (1650, 6.0), (1900, 2.0)]
picks = []
for T0t, phit in targets:
    dist = ((T0 - T0t) / 100.0) ** 2 + ((phi - phit) / 1.0) ** 2
    picks.append(int(np.argmin(dist)))
panels = ["T"] + LARGEST + TRACKED
for j in picks:
    sl = slice(first[j], last[j] + 1)
    nc = 4
    nr = int(np.ceil(len(panels) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(14, 2.8 * nr), sharex=True)
    for ax, name in zip(axs.ravel(), panels):
        if name == "T":
            ax.plot(t[sl], T_true[sl], color=INK, lw=2.0, label="Cantera")
            for i, run in enumerate(plot_runs):
                ax.plot(t[sl], pr[run]["T"][sl], color=SERIES[i], ls=STYLES[i], label=run)
            ax.set_ylabel("T [K]")
        else:
            k = sidx[name]
            ax.plot(t[sl], Y_true[sl, k], color=INK, lw=2.0)
            for i, run in enumerate(plot_runs):
                ax.plot(t[sl], pr[run]["Y"][sl, k], color=SERIES[i], ls=STYLES[i])
            if name in TRACKED:
                ax.set_yscale("log")
                lo = max(Y_true[sl, k].max() * 1e-6, 1e-14)
                ax.set_ylim(bottom=lo)
        ax.set_xscale("log")
        ax.set_title(name + ("  (tracked)" if name in TRACKED else ""), fontsize=9, color=INK)
    for ax in axs.ravel()[len(panels):]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("t [s]")
    axs.ravel()[0].legend(fontsize=7)
    fig.suptitle(f"test trajectory {traj_idx_rows[first[j]]}: T0 = {T0[j]:.0f} K, phi = {phi[j]:.2f}, "
                 f"{'ignites' if ignited[j] else 'no ignition'}", color=INK)
    save(fig, f"traj_T0_{T0[j]:.0f}_phi_{phi[j]:.2f}.png")

# 5. peak parity for tracked species (igniting trajectories)
fig, axs = plt.subplots(1, len(TRACKED), figsize=(3.2 * len(TRACKED), 3.4))
for ax, s in zip(axs, TRACKED):
    c = PEAK_SPECIES.index(s)
    ok = ignited & (peak_true[:, c] > PEAK_THR)
    if not ok.any():
        ax.set_title(f"{s}: no peaks > {PEAK_THR:g}", fontsize=9)
        continue
    x = peak_true[ok, c]
    for i, run in enumerate(plot_runs):
        ax.loglog(x, pr[run]["peak"][ok, c], ls="none", marker=MARKERS[i], ms=2, alpha=0.35,
                  color=SERIES[i], label=run)
    lim = [x.min(), x.max()]
    ax.plot(lim, lim, color=INK, lw=1)
    ax.set_title(f"peak {s}", fontsize=9, color=INK); ax.set_xlabel("true")
axs[0].set_ylabel("predicted"); axs[0].legend(fontsize=7, markerscale=4)
save(fig, "peak_parity_tracked.png")

# 6. loss curves (val, mean over bins), one panel per architecture
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, a in zip(axs, ["deeponet", "crkpinn"]):
    sa = S[(S["arch"] == a) & (S["seed"] == 0)].sort_values("n_bins")
    for i, run in enumerate(sa["run"]):
        hf = Path(f"{args.results}/{run}/history.csv")
        if not hf.exists():
            continue
        h = pd.read_csv(hf).groupby("epoch")["val"].mean()
        ax.semilogy(h.index, h.values, color=SERIES[i % 4], ls=["-", "--", "-.", ":"][i % 4],
                    label=run.replace(f"{a}_predicted_", ""))
    ax.set_title("DeepONet" if a == "deeponet" else "CRK-PINN", color=INK)
    ax.set_xlabel("epoch"); ax.legend()
axs[0].set_ylabel("validation loss (mean over bin models)")
save(fig, "loss_curves.png")

print(f"\nsaved tables to {out}/ and plots to {P_DIR}/")

# ======================================================================= LaTeX tables
# results/diagnostics/latex/*.tex -- each a complete table environment for
# \input{} in the report. Numbers are means over seeds; ranges are min-max.
TEX = out / "latex"
TEX.mkdir(exist_ok=True)
ARCH_NAME = {"deeponet": "DeepONet", "crkpinn": "CRK-PINN"}


def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if pct:
        v = 100 * v
    if isinstance(v, (int, np.integer)) or (float(v).is_integer() and abs(v) >= 10) or abs(v) >= 1000:
        return f"{int(round(v))}"
    if v == 0:
        return "0"
    if abs(v) < 1e-3:
        m, e = f"{v:.2e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{v:.3g}"


import re


def tex_species(s):
    return r"$\mathrm{" + re.sub(r"(\d+)", r"_{\1}", s) + "}$"


def tex_config(c):
    a, nb = parse(c)[:2]
    extra = {"lm": "mole", "le": "element", "lh": "enthalpy", "ls": "entropy"}
    losses = [f"{extra[m.group(1)]} {m.group(2)}" for m in re.finditer(r"_(lm|le|lh|ls)([\d.e+-]+?)(?=_|$)", c)]
    return f"{ARCH_NAME[a]}, {nb} bin" + ("s" if nb > 1 else "") + (" + " + ", ".join(losses) if losses else "")


def write_table(name, header, rows, caption, label, align=None):
    align = align or "l" + "r" * (len(header) - 1)
    lines = [r"\begin{table}[htbp]", r"    \centering", r"    \small",
             rf"    \begin{{tabular}}{{{align}}}", r"        \hline",
             "        " + " & ".join(header) + r" \\", r"        \hline"]
    lines += ["        " + " & ".join(r) + r" \\" for r in rows]
    lines += [r"        \hline", r"    \end{tabular}", rf"    \caption{{{caption}}}",
              rf"    \label{{{label}}}", r"\end{table}", ""]
    (TEX / f"{name}.tex").write_text("\n".join(lines))


def rng_txt(row, c):
    if row["n_seeds"] > 1:
        return f"{fmt(row[c])} ({fmt(row[c + '_min'])}--{fmt(row[c + '_max'])})"
    return fmt(row[c])


# 1. bin resolution, both architectures
main_cfg = SS[SS["config"].str.fullmatch(r"(deeponet|crkpinn)" + BIN_RE)]
rows = []
for _, r in main_cfg.sort_values(["arch", "n_bins"], ascending=[False, True]).iterrows():
    rows.append([ARCH_NAME[r["arch"]], str(int(r["n_bins"])), str(int(r["n_seeds"])),
                 rng_txt(r, "T_MAE_K"), rng_txt(r, "final_dT_p95"),
                 fmt(r["ign_time_err_med"], pct=True), fmt(r["ign_missed"]), fmt(r["false_ign"]),
                 rng_txt(r, "logY_MAE_tracked"), fmt(r["inference_ms_per_traj"])])
write_table(
    "tab_bins",
    ["Model", "Bins", "Seeds", r"$\overline{|\Delta T|}$ [K]", r"$|\Delta T_\mathrm{f}|_{95}$ [K]",
     r"$\epsilon_{t_\mathrm{ign}}$ [\%]", "Missed", "False", r"$\overline{\Delta\log_{10}Y}_\mathrm{tr}$",
     r"ms/traj"],
    rows,
    r"Test-set errors against number of $T_0$ bins (20\,000 test trajectories). Values are means "
    r"over seeds, with min--max in parentheses. $\overline{|\Delta T|}$: mean absolute temperature "
    r"error over all points; $|\Delta T_\mathrm{f}|_{95}$: 95th percentile of the final-state "
    r"temperature error; $\epsilon_{t_\mathrm{ign}}$: median relative error of the ignition time; "
    r"Missed/False: missed and false ignitions; $\overline{\Delta\log_{10}Y}_\mathrm{tr}$: mean "
    r"$|\log_{10}Y_\mathrm{pred}-\log_{10}Y_\mathrm{true}|$ of the tracked species "
    r"(" + ", ".join(tex_species(x) for x in TRACKED) + r", points with $Y>10^{-8}$); ms/traj: test inference time per "
    r"trajectory on the GPU.",
    "tab:bins")

# chosen configurations = configs of the plot runs (seed stripped)
chosen = []
for run in plot_runs:
    c = parse(run)[3]
    if c not in chosen:
        chosen.append(c)

# 2. species, per chosen configuration (mean over seeds)
SPm = SP.groupby(["config", "species"]).mean(numeric_only=True)
for c in chosen:
    c_tex = tex_config(c)
    rows = []
    for s in TRACKED:
        r = SPm.loc[(c, s)]
        rows.append([tex_species(s), "tracked", fmt(r["logY_MAE"]), fmt(r.get("peak_relerr_med"), pct=True),
                     fmt(r.get("peak_time_relerr_med"), pct=True), fmt(r["final_relerr_med"], pct=True)])
    for s in LARGEST:
        r = SPm.loc[(c, s)]
        rows.append([tex_species(s), "largest", fmt(r["logY_MAE"]), fmt(r.get("peak_relerr_med"), pct=True),
                     fmt(r.get("peak_time_relerr_med"), pct=True), fmt(r["final_relerr_med"], pct=True)])
    write_table(
        f"tab_species_{c}",
        ["Species", "Group", r"$\overline{\Delta\log_{10}Y}$", r"Peak err. [\%]",
         r"Peak time err. [\%]", r"Final err. [\%]"],
        rows,
        rf"Species errors on the test set, {c_tex}. "
        r"$\overline{\Delta\log_{10}Y}$: mean $|\log_{10}Y_\mathrm{pred}-\log_{10}Y_\mathrm{true}|$ "
        r"over points with $Y>10^{-8}$. Peak and peak-time errors: median relative error of the "
        r"trajectory maximum and of its time, igniting trajectories (tracked species and CO only). "
        r"Final err.: median relative error at the last time point.",
        f"tab:species_{c}")

# 3. conservation, chosen configurations
rows = []
for c in chosen:
    r = SS[SS["config"] == c].iloc[0]
    rows.append([tex_config(c), fmt(r["mass_sum_err_med"]), fmt(r["mass_sum_err_p95"]),
                 fmt(r["element_err_med"]), fmt(r["element_err_p95"]),
                 fmt(r["enthalpy_err_K_med"]), fmt(r["enthalpy_err_K_p95"])])
write_table(
    "tab_conservation",
    ["Model", r"$|\sum Y-1|_{50}$", r"$|\sum Y-1|_{95}$", r"$\epsilon_{Z,50}$", r"$\epsilon_{Z,95}$",
     r"$\Delta h_{50}$ [K]", r"$\Delta h_{95}$ [K]"],
    rows,
    r"Conservation errors of the predictions (median and 95th percentile). $|\sum Y-1|$: mass "
    r"fraction sum, all points; $\epsilon_Z$: largest relative error in the C, H, O, N element "
    r"mass fractions, all points; $\Delta h$: enthalpy error at the final state, "
    r"$|h(T_\mathrm{pred},Y_\mathrm{pred})-h_0|/c_p$, expressed in K.",
    "tab:conservation")

# 4. per T0 range for each chosen configuration
Rm = R.groupby(["config", "T0 range"]).mean(numeric_only=True)
for c in chosen:
    c_tex = tex_config(c)
    rows = []
    for lab in range_labels:
        if (c, lab) not in Rm.index:
            continue
        r = Rm.loc[(c, lab)]
        rows.append([lab.replace("[", "").replace(")", "").replace(",", "--"), fmt(r["n"]),
                     fmt(r["n_ign"]), fmt(r["T_MAE_K"]), fmt(r["final_dT_med"]), fmt(r["final_dT_p95"]),
                     fmt(r["ign_time_err_med"], pct=True), fmt(r["ign_missed"])])
    write_table(
        f"tab_T0_{c}",
        [r"$T_0$ [K]", "$n$", r"$n_\mathrm{ign}$", r"$\overline{|\Delta T|}$ [K]",
         r"$|\Delta T_\mathrm{f}|_{50}$ [K]", r"$|\Delta T_\mathrm{f}|_{95}$ [K]",
         r"$\epsilon_{t_\mathrm{ign}}$ [\%]", "Missed"],
        rows,
        rf"Test-set errors per initial-temperature range, {c_tex} (mean over seeds).",
        f"tab:T0_{c}")

print(f"LaTeX tables: {sorted(p.name for p in TEX.glob('*.tex'))}")