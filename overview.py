# overview.py
"""
One-page overview of all results. Run AFTER diagnose_global.py:

  python diagnose_global.py
  python overview.py

Prints three tables (and saves them to results/diagnostics/overview.txt):
  1. bin resolution   -- data loss only, both architectures, mean over seeds
  2. physics losses   -- per architecture: data-only baseline vs each loss run,
                         with % change of the main errors vs the baseline
  3. best run per architecture (lowest mean |dT|)
Lower is better for every column.
"""
import re
import pandas as pd

S = pd.read_csv("results/diagnostics/summary_seeds.csv")
NAME = {"deeponet": "DeepONet", "crkpinn": "CRK-PINN"}
out_lines = []


def show(title, df, cols, fmt="{:.3g}"):
    txt = df[cols].to_string(index=False, float_format=lambda x: fmt.format(x))
    block = f"\n=== {title} ===\n{txt}\n"
    print(block)
    out_lines.append(block)


# ---- 1. bin resolution
bins = S[S["config"].str.fullmatch(r"(deeponet|crkpinn)_predicted_b\d+(_e\d+|_conv)?")].copy()
bins["epochs"] = bins["config"].str.extract(r"_e(\d+)$")[0].fillna("30")
bins.loc[bins["config"].str.endswith("_conv"), "epochs"] = "conv"
bins["model"] = bins["arch"].map(NAME)
bins = bins.sort_values(["arch", "n_bins"], ascending=[False, True])
show("1. Bin resolution (data loss only, mean over seeds)", bins,
     ["model", "n_bins", "epochs", "n_seeds", "T_MAE_K", "final_dT_p95", "ign_time_err_med",
      "ign_missed", "logY_MAE_tracked"])

# ---- 2. physics losses
loss_re = re.compile(r"_(lm|le|lh|ls)([\d.e+-]+)")
TERM = {"lm": "mole", "le": "element", "lh": "enthalpy", "ls": "entropy"}
main = ["T_MAE_K", "final_dT_p95", "ign_time_err_med", "ign_missed", "logY_MAE_tracked"]
cons = [c for c in ["mass_sum_err_med", "element_err_med", "enthalpy_err_K_med",
                    "entropy_viol_frac", "entropy_viol_K_p95"] if c in S.columns]
for arch in ["deeponet", "crkpinn"]:
    L = S[(S["arch"] == arch) & S["config"].str.contains(r"_l[mehs]")].copy()
    if L.empty:
        continue
    nb = int(L["n_bins"].iloc[0])
    conv = L["config"].str.endswith("_conv").any()
    if conv:                                   # compare converged with converged
        L = L[L["config"].str.endswith("_conv")]
    base = S[S["config"] == f"{arch}_predicted_b{nb}" + ("_conv" if conv else "")]
    if base.empty:
        continue
    L["loss"] = L["config"].apply(lambda c: " + ".join(
        f"{TERM[m.group(1)]} {m.group(2)}" for m in loss_re.finditer(c)))
    b = base.copy()
    b["loss"] = "data only (baseline)"
    T = pd.concat([b, L.sort_values("config")])
    for c in ["T_MAE_K", "logY_MAE_tracked"]:
        T[f"{c} vs base %"] = 100 * (T[c] / base[c].iloc[0] - 1)
    show(f"2. Physics losses, {NAME[arch]} {nb} bin(s): test errors", T,
         ["loss", "n_seeds"] + main + ["T_MAE_K vs base %", "logY_MAE_tracked vs base %"])
    show(f"2. Physics losses, {NAME[arch]} {nb} bin(s): conservation residuals", T, ["loss"] + cons)

# ---- 3. best per architecture
best = S.loc[S.groupby("arch")["T_MAE_K"].idxmin()].copy()
best["model"] = best["arch"].map(NAME)
show("3. Best configuration per architecture (lowest mean |dT|)", best,
     ["model", "config", "n_seeds"] + main)

open("results/diagnostics/overview.txt", "w").write("".join(out_lines))
print("saved to results/diagnostics/overview.txt")