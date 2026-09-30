"""
File for generating data for TEP4541 project. 
Creates datapoints for trajectories used to train and 
validate machine learning models.
"""

import os
import numpy as np
import pandas as pd
import cantera as ct

# ---------------- Config ----------------
MECH = "gri30.yaml"
FUEL = "CH4"
OXIDIZER = "O2:1, N2:3.76"          # air

T0_RANGE = (500.0, 2000.0)          # K
PHI_RANGE = (0.5, 8.0)              # outside GRI-Mech's stated validity (phi 0.1-5, T 1000-2500 K)
P_FIXED = 101325.0                  # Pa, constant-pressure reactor
SIM_TIME = 0.5                      # s

N_BINS_T0 = 10
N_TRAIN_PER_BIN = 12000
N_TEST_PER_BIN = 2000

N_PRE_IGNITION = 15
N_IGNITION = 70
N_POST_IGNITION = 15
IGNITION_WINDOW_LO = 0.3
IGNITION_WINDOW_HI = 3.0

DTDT_THRESHOLD = 1e5

TRAIN_CSV = "raw/data_train.csv"
TEST_CSV = "raw/data_test.csv"

TRAIN_SEED = 42
TEST_SEED = 1337
# -----------------------------------------


def sample_initial_conditions(n_per_bin, seed):
    rng = np.random.default_rng(seed)
    edges = np.linspace(T0_RANGE[0], T0_RANGE[1], N_BINS_T0 + 1)
    T0s, phis, bins = [], [], []
    for b in range(N_BINS_T0):
        T0s.append(rng.uniform(edges[b], edges[b + 1], n_per_bin))
        phis.append(rng.uniform(*PHI_RANGE, n_per_bin))
        bins.append(np.full(n_per_bin, b))
    return pd.DataFrame({
        "T0": np.concatenate(T0s),
        "phi": np.concatenate(phis),
        "bin_idx": np.concatenate(bins),
    })


def scout_ignition(gas, T0, phi, P0, n_scout_steps=5000):
    gas.TP = T0, P0
    gas.set_equivalence_ratio(phi, FUEL, OXIDIZER)
    gas.TP = T0, P0

    reactor = ct.IdealGasConstPressureReactor(gas)
    net = ct.ReactorNet([reactor])

    t_prev, T_prev = 0.0, gas.T
    t_peak, max_dTdt = np.nan, 0.0
    reached_end = False

    for _ in range(n_scout_steps):
        t = net.step()
        dt = t - t_prev
        if dt > 0:
            dTdt = (gas.T - T_prev) / dt
            if dTdt > max_dTdt:
                max_dTdt = dTdt
                t_peak = t_prev + 0.5 * dt
        t_prev, T_prev = t, gas.T
        if t >= SIM_TIME:
            reached_end = True
            break

    ignited = max_dTdt > DTDT_THRESHOLD
    t_ign = t_peak if ignited else np.nan
    return t_ign, ignited, reached_end


def build_sample_schedule(t_ign, ignited):
    n_total = N_PRE_IGNITION + N_IGNITION + N_POST_IGNITION
    t_min = SIM_TIME / 1e6

    if not ignited:
        return np.logspace(np.log10(t_min), np.log10(SIM_TIME), n_total)

    t_lo = max(t_ign * IGNITION_WINDOW_LO, t_min)
    t_hi = min(t_ign * IGNITION_WINDOW_HI, SIM_TIME)

    pre = np.logspace(np.log10(t_min), np.log10(t_lo), N_PRE_IGNITION, endpoint=False)
    ign = np.linspace(t_lo, t_hi, N_IGNITION)
    post = np.logspace(np.log10(t_hi), np.log10(SIM_TIME), N_POST_IGNITION + 1)[1:]

    return np.unique(np.concatenate([pre, ign, post]))


def run_trajectory(gas, T0, phi, P0, traj_idx, bin_idx):
    t_ign, ignited, reached_end = scout_ignition(gas, T0, phi, P0)
    t_points = build_sample_schedule(t_ign, ignited)

    gas.TP = T0, P0
    gas.set_equivalence_ratio(phi, FUEL, OXIDIZER)
    gas.TP = T0, P0

    reactor = ct.IdealGasConstPressureReactor(gas)
    net = ct.ReactorNet([reactor])

    species_cols = gas.species_names
    rows = []

    for t in t_points:
        try:
            net.advance(t)
        except ct.CanteraError:
            break

        record = {
            "traj_idx": traj_idx, "bin_idx": bin_idx, "t": t, "T": gas.T, "P": gas.P,
            "rho": gas.density, "T0": T0, "phi": phi,
            "t_ign": t_ign, "ignited": ignited, "reached_end": reached_end,
        }
        record.update({f"Y_{s}": y for s, y in zip(species_cols, gas.Y)})
        rows.append(record)

    return rows


def generate(ic_df, seed_label):
    gas = ct.Solution(MECH)
    all_rows = []
    n_ignited = 0
    n_not_reached = 0

    for i, row in ic_df.iterrows():
        try:
            rows = run_trajectory(gas, row["T0"], row["phi"], P_FIXED, i, int(row["bin_idx"]))
        except ct.CanteraError:
            continue
        if rows:
            if rows[0]["ignited"]:
                n_ignited += 1
            if not rows[0]["reached_end"]:
                n_not_reached += 1
        all_rows.extend(rows)

        if (i + 1) % 500 == 0:
            print(f"[{seed_label}] ...{i + 1}/{len(ic_df)} trajectories done")

    out_df = pd.DataFrame(all_rows)
    n_traj_total = out_df["traj_idx"].nunique()
    print(f"[{seed_label}] Generated {len(out_df)} datapoints across {n_traj_total} trajectories")
    print(f"[{seed_label}] {n_ignited}/{n_traj_total} ignited")
    print(f"[{seed_label}] {n_not_reached}/{n_traj_total} scouting pass did NOT reach SIM_TIME "
          f"(possible missed ignition beyond scout step budget)")

    print(f"[{seed_label}] Per-bin trajectory / ignition counts:")
    for b in range(N_BINS_T0):
        sub = out_df[out_df["bin_idx"] == b]
        n_traj = sub["traj_idx"].nunique()
        n_ign = sub.groupby("traj_idx")["ignited"].first().sum()
        print(f"  bin {b}: {n_traj} traj, {n_ign} ignited")

    return out_df


def main():
    os.makedirs("raw", exist_ok=True)

    train_ic = sample_initial_conditions(N_TRAIN_PER_BIN, TRAIN_SEED)
    train_df = generate(train_ic, "train")
    train_df.to_csv(TRAIN_CSV, index=False)
    print(f"Saved train to {TRAIN_CSV}")

    test_ic = sample_initial_conditions(N_TEST_PER_BIN, TEST_SEED)
    test_df = generate(test_ic, "test")
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Saved test to {TEST_CSV}")


if __name__ == "__main__":
    main()