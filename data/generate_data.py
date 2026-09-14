"""
Trajectory-based training data generator for GRI-Mech 3.0 CH4/air combustion.

Same ignition-aware sampling strategy as generate_data_simple.py's v2 (H2/O2)
version: a cheap scouting pass locates each trajectory's ignition time
(peak dT/dt), then most samples are concentrated in a window around it,
with only a few points in the flat pre-/post-ignition regions. This avoids
the data-imbalance problem that made earlier rate-based models collapse to
predicting ~0 everywhere.

Differences from the H2/O2 version:
  - MECH = gri30.yaml (53 species, 325 reactions) instead of h2o2.yaml
  - Fuel/oxidizer: CH4 vs air (O2:1, N2:3.76), not H2 vs pure O2
  - T0_RANGE shifted higher and SIM_TIME extended -- CH4/air autoignition
    delay times are generally longer than H2/O2's, especially at the lower
    end of the temperature range, and low-T0 conditions may not ignite
    within SIM_TIME at all (check the printed ignited/total ratio)
  - N_TRAJECTORIES increased for a larger dataset

Output: single CSV, one row per (trajectory, time) sample.
"""

import os
import numpy as np
import pandas as pd
import cantera as ct

# ---------------- Config ----------------
OUTPUT_CSV = "raw/gri30_ch4_air_trajectories.csv"
MECH = "gri30.yaml"
FUEL = "CH4"
OXIDIZER = "O2:1, N2:3.76"  # air

N_TRAJECTORIES = 5000
SIM_TIME = 1.0                 # seconds

T0_RANGE = (300.0, 2000.0)    
PHI_RANGE = (0.5, 2.0)
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


def scout_ignition_time(gas, T0, phi, P0, n_scout_steps=5000):
    """Same approach as the H2/O2 version -- track dT/dt over the solver's
    own adaptive steps to cheaply locate ignition. n_scout_steps raised vs.
    the H2/O2 version since GRI30 trajectories can need more internal steps
    to resolve the stiffer, longer-duration chemistry."""
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

    ignited = max_dTdt > 1e5   # lower threshold than the H2/O2 version -- CH4/air ignition is generally less explosive
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


def run_trajectory(gas, T0, phi, P0, traj_idx):
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
            "traj_idx": traj_idx, "t": t, "T": gas.T, "P": gas.P, "rho": rho,
            "Tdot": Tdot, "T0": T0, "phi": phi, "t_ign": t_ign, "ignited": ignited,
        }
        record.update({f"Y_{s}": y for s, y in zip(species_cols, gas.Y)})
        record.update({f"Ydot_{s}": yd for s, yd in zip(species_cols, Ydot)})
        rows.append(record)

    return rows


def main():
    gas = ct.Solution(MECH)
    all_rows = []
    n_ignited = 0

    for traj_idx in range(N_TRAJECTORIES):
        T0 = rng.uniform(*T0_RANGE)
        phi = rng.uniform(*PHI_RANGE)
        try:
            rows = run_trajectory(gas, T0, phi, P_FIXED, traj_idx)
        except ct.CanteraError:
            continue
        if rows and rows[0]["ignited"]:
            n_ignited += 1
        all_rows.extend(rows)

        if (traj_idx + 1) % 50 == 0:
            print(f"...{traj_idx + 1}/{N_TRAJECTORIES} trajectories done")

    out_df = pd.DataFrame(all_rows)
    print(f"Generated {len(out_df)} datapoints across {out_df['traj_idx'].nunique()} trajectories")
    print(f"{n_ignited}/{out_df['traj_idx'].nunique()} trajectories ignited within SIM_TIME")
    print(f"Temperature range: {out_df['T'].min():.1f} K to {out_df['T'].max():.1f} K")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()