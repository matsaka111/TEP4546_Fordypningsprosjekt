"""
Trajectory-based training data generator for GRI-Mech 3.0 CH4/air combustion.

Same ignition-aware sampling strategy as generate_data_simple.py's v2 (H2/O2)
version: a cheap scouting pass locates each trajectory's ignition time
(peak dT/dt), then most samples are concentrated in a window around it,
with only a few points in the flat pre-/post-ignition regions.

Updated for multi-bin training (router by T0): samples are drawn evenly
across N_BINS_T0 fixed-width temperature bins spanning T0_LO to T0_HI, so
each bin gets a comparable number of trajectories regardless of how likely
ignition is in that range. Non-igniting trajectories are now KEPT (not
dropped) -- low-T0 bins are expected to be mostly non-igniting, and the
low-T0 model needs "nothing happens" examples to learn that correctly
rather than being trained on an artificially ignition-only distribution.
"""

import os
import numpy as np
import pandas as pd
import cantera as ct

# ---------------- Config ----------------
OUTPUT_CSV = "data/raw/gri30_ch4_air_trajectories_cbins.csv"
MECH = "gri30.yaml"
FUEL = "CH4"
OXIDIZER = "O2:1, N2:3.76"  # air

T0_LO = 500.0
T0_HI = 2000.0
N_BINS_T0 = 5                  # bin width = (T0_HI - T0_LO) / N_BINS_T0 = 300 K
N_TRAJECTORIES_PER_BIN = 600   # ~3000 trajectories total across all bins

SIM_TIME = 0.5                 # seconds, integration horizon
PHI_RANGE = (0.5, 2.0)      # Increase to 6,9 if we want to capture C2H6 Sot
P_FIXED = 101325.0

# sampling budget, split across three regions of a trajectory
N_PRE_IGNITION = 15
N_IGNITION = 70
N_POST_IGNITION = 15

IGNITION_WINDOW_LO = 0.3
IGNITION_WINDOW_HI = 3.0

SEED = 42
# -----------------------------------------

rng = np.random.default_rng(SEED)

BIN_EDGES = np.linspace(T0_LO, T0_HI, N_BINS_T0 + 1)

def scout_ignition_time(gas, T0, phi, P0, n_scout_steps=5000):
    gas.TP = T0, P0
    gas.set_equivalence_ratio(phi, FUEL, OXIDIZER)
    gas.TP = T0, P0

    reactor = ct.IdealGasConstPressureReactor(gas)
    net = ct.ReactorNet([reactor])

    t_prev, T_prev = 0.0, gas.T
    t_ign, max_dTdt = 0.0, 0.0

    for _ in range(n_scout_steps):
        t = net.step()
        dT = gas.T - T_prev
        dt = t - t_prev
        if dt > 0:
            dTdt = dT / dt
            if dTdt > max_dTdt:
                max_dTdt = dTdt
                t_ign = t_prev + 0.5 * dt
        t_prev, T_prev = t, gas.T
        if t >= SIM_TIME:
            break

    ignited = max_dTdt > 1e5
    if not ignited:
        t_ign = SIM_TIME   
    return t_ign, ignited

def build_sample_schedule(t_ign, ignited):
    if not ignited or t_ign <= 0:
        return np.logspace(np.log10(SIM_TIME / 1e6), np.log10(SIM_TIME),
                            N_PRE_IGNITION + N_IGNITION + N_POST_IGNITION)

    t_lo = max(t_ign * IGNITION_WINDOW_LO, SIM_TIME / 1e6)
    t_hi = min(t_ign * IGNITION_WINDOW_HI, SIM_TIME)

    pre = np.logspace(np.log10(SIM_TIME / 1e6), np.log10(t_lo), N_PRE_IGNITION, endpoint=False)
    ign = np.linspace(t_lo, t_hi, N_IGNITION)
    post = np.logspace(np.log10(t_hi), np.log10(SIM_TIME), N_POST_IGNITION + 1)[1:]

    return np.concatenate([pre, ign, post])


def run_trajectory(gas, T0, phi, P0, traj_idx, bin_idx):
    t_ign, ignited = scout_ignition_time(gas, T0, phi, P0)
    t_points = np.sort(build_sample_schedule(t_ign, ignited))

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

        rho = gas.density
        wdot = gas.net_production_rates
        mw = gas.molecular_weights
        Ydot = wdot * mw / rho

        u_partial = gas.partial_molar_int_energies
        cv = gas.cv_mass
        Tdot = -np.sum(u_partial * wdot) / (rho * cv)

        record = {
            "traj_idx": traj_idx, "bin_idx": bin_idx, "t": t, "T": gas.T, "P": gas.P,
            "rho": rho, "Tdot": Tdot, "T0": T0, "phi": phi,
            "t_ign": t_ign, "ignited": ignited,
        }
        record.update({f"Y_{s}": y for s, y in zip(species_cols, gas.Y)})
        record.update({f"Ydot_{s}": yd for s, yd in zip(species_cols, Ydot)})
        rows.append(record)

    return rows


def main():
    gas = ct.Solution(MECH)
    all_rows = []
    n_ignited = 0
    traj_idx = 0

    for bin_idx in range(N_BINS_T0):
        lo, hi = BIN_EDGES[bin_idx], BIN_EDGES[bin_idx + 1]
        print(f"--- bin {bin_idx}: T0 in [{lo:.0f}, {hi:.0f}) ---")

        for _ in range(N_TRAJECTORIES_PER_BIN):
            T0 = rng.uniform(lo, hi)
            phi = rng.uniform(*PHI_RANGE)
            try:
                rows = run_trajectory(gas, T0, phi, P_FIXED, traj_idx, bin_idx)
            except ct.CanteraError:
                traj_idx += 1
                continue
            if rows and rows[0]["ignited"]:
                n_ignited += 1
            all_rows.extend(rows)
            traj_idx += 1

            if traj_idx % 100 == 0:
                print(f"...{traj_idx} trajectories done")

    out_df = pd.DataFrame(all_rows)
    n_traj_total = out_df["traj_idx"].nunique()
    print(f"Generated {len(out_df)} datapoints across {n_traj_total} trajectories")
    print(f"{n_ignited}/{n_traj_total} trajectories ignited within SIM_TIME")
    print(f"Temperature range: {out_df['T'].min():.1f} K to {out_df['T'].max():.1f} K")

    print("Per-bin trajectory / ignition counts:")
    for bin_idx in range(N_BINS_T0):
        sub = out_df[out_df["bin_idx"] == bin_idx]
        n_traj = sub["traj_idx"].nunique()
        n_ign = sub.groupby("traj_idx")["ignited"].first().sum()
        lo, hi = BIN_EDGES[bin_idx], BIN_EDGES[bin_idx + 1]
        print(f"  bin {bin_idx} [{lo:.0f},{hi:.0f}): {n_traj} traj, {n_ign} ignited")

    # NOTE: non-igniting trajectories are NOT dropped anymore -- the low-T0
    # bins are expected to be mostly non-igniting, and that's the signal
    # those bins' models need to learn.

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()