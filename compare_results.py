# compare_results.py
"""
Post-hoc analysis of results/sweep_results_full.csv (no models loaded).
Metrics: T abs/% error, composition error (total mass fraction misallocated),
% error on major species (only where true mass fraction > MASS_FLOOR).
"""
import numpy as np
import pandas as pd

RESULTS_CSV = "results/sweep_results_full.csv"
MAJORS = ["CH4", "O2", "CO2", "H2O", "CO", "N2"]
MASS_FLOOR = 1e-3      # species get a % error only if true final mass fraction > this
IGNITION_DT = 200.0    # K; final T - T0 above this counts as "ignited"
N_BOOT = 2000
rng = np.random.default_rng(0)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")

df = pd.read_csv(RESULTS_CSV)
df = df.copy()
species = [c[:-5] for c in df.columns if c.endswith("_true") and c != "T_true"]

Y_true = df[[f"{s}_true" for s in species]].to_numpy()
Y_pred = df[[f"{s}_pred" for s in species]].to_numpy()

# ---------------- per-row metrics ----------------
df["T_rel_pct"] = 100 * df["T_abs_err"] / df["T_true"]
df["comp_err"] = np.abs(Y_pred - Y_true).sum(axis=1)     # total mass fraction misallocated
for m in MAJORS:
    t = df[f"{m}_true"]
    df[f"{m}_rel_pct"] = np.where(t > MASS_FLOOR, 100 * df[f"{m}_abs_err"] / t, np.nan)
df["major_rel_pct"] = df[[f"{m}_rel_pct" for m in MAJORS]].mean(axis=1)   # mean over majors present
df["regime"] = np.where(df["T_true"] - df["T0"] > IGNITION_DT, "ignited", "not_ignited")
df["T_persist_err"] = (df["T_true"] - df["T0"]).abs()    # error of predicting "no change"


def q90(x):
    return x.quantile(0.9)


def boot_ci_median(x, n_boot=N_BOOT):
    x = x.to_numpy()
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    meds = np.median(x[idx], axis=1)
    return np.percentile(meds, [2.5, 97.5])


# ---------------- Table 1: headline per (arch, n_bins) ----------------
t1 = df.groupby(["arch", "n_bins"]).agg(
    n=("test_traj_id", "count"),
    T_med_K=("T_abs_err", "median"), T_mean_K=("T_abs_err", "mean"),
    T_p90_K=("T_abs_err", q90), T_max_K=("T_abs_err", "max"),
    T_med_pct=("T_rel_pct", "median"),
    comp_med=("comp_err", "median"), comp_p90=("comp_err", q90),
    major_med_pct=("major_rel_pct", "median"),
)
ci = df.groupby(["arch", "n_bins"])["T_abs_err"].apply(boot_ci_median)
t1["T_med_CI_lo"] = [c[0] for c in ci]
t1["T_med_CI_hi"] = [c[1] for c in ci]
print("=== Table 1: headline metrics per (arch, n_bins) ===")
print(t1.to_string())

# ---------------- Table 2: % error per major species (median) ----------------
t2 = df.groupby(["arch", "n_bins"])[[f"{m}_rel_pct" for m in MAJORS]].median()
t2.columns = MAJORS
print(f"\n=== Table 2: median % error per major species (true mass fraction > {MASS_FLOOR}) ===")
print(t2.to_string())

# ---------------- Table 3: by regime ----------------
t3 = df.groupby(["arch", "n_bins", "regime"]).agg(
    n=("test_traj_id", "count"),
    T_med_K=("T_abs_err", "median"), T_mean_K=("T_abs_err", "mean"),
    persist_mean_K=("T_persist_err", "mean"),
    comp_med=("comp_err", "median"),
)
t3["T_err_vs_persistence"] = t3["T_mean_K"] / t3["persist_mean_K"]   # <1 beats "predict no change"
print(f"\n=== Table 3: by true regime (ignited = final T - T0 > {IGNITION_DT:.0f} K) ===")
print(t3.to_string())

# ---------------- Table 4: per-trajectory T error, sorted by T0 ----------------
t4 = df.pivot_table(index=["T0", "test_traj_id", "regime"], columns=["arch", "n_bins"],
                    values="T_abs_err").sort_index(level="T0")
print("\n=== Table 4: |T error| (K) per test trajectory ===")
print(t4.to_string())

# ---------------- Table 5: paired DeepONet vs CRKPINN, same trajectories ----------------
p = df.pivot_table(index=["n_bins", "test_traj_id"], columns="arch", values="T_abs_err")
t5 = pd.DataFrame({
    "deeponet_win_rate": (p["deeponet"] < p["crkpinn"]).groupby("n_bins").mean(),
    "median_ratio_crk_over_deep": (p["crkpinn"] / p["deeponet"]).groupby("n_bins").median(),
})
print("\n=== Table 5: paired comparison per n_bins (T abs error) ===")
print(t5.to_string())

# ---------------- save ----------------
t1.to_csv("results/table1_headline.csv")
t2.to_csv("results/table2_major_species_pct.csv")
t3.to_csv("results/table3_by_regime.csv")
t4.to_csv("results/table4_per_trajectory_T.csv")
t5.to_csv("results/table5_paired.csv")
print("\nSaved results/table1..5 CSVs")