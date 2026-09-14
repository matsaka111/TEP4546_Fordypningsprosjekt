"""
Simplified trajectory-based training data generator for the H2/O2 system (h2o2.yaml).

v2: ignition-aware sampling. The first version log-spaced samples across the
whole fixed SIM_TIME, which meant the vast majority of rows landed in the
flat pre-/post-ignition region and only ~2-5% of rows captured the actual
ignition spike (where Ydot is largest and most informative). Training on that
distribution let the model collapse to predicting ~0 everywhere and still get
low average loss.

Fix: for each trajectory, first do a cheap scouting pass to find the ignition
time t_ign (where dT/dt peaks), then build a sampling schedule that
concentrates most points in a window around t_ign, with only a few points in
the flat pre-ignition and post-ignition regions.

Output: single CSV, one row per (trajectory, time) sample.
"""

import os
import numpy as np
import pandas as pd
import cantera as ct

# Config 
OUTPUT_CSV = "raw/h2o2_trajectories.csv"
MECH = "h2o2.yaml"

N_TRAJECTORIES = 20000
SIM_TIME = 1e-2                # seconds, integration horizon

T0_RANGE = (900.0, 1400.0)
PHI_RANGE = (0.5, 2.0)
P_FIXED = 101325.0

# sampling budget, split across three regions of a trajectory
N_PRE_IGNITION = 15            # flat region before ignition
N_IGNITION = 70                # more dense samples around the ignition spike
N_POST_IGNITION = 15           # flat region after burnout

# how wide ignition window is, as a multiple of the scouted t_ign
IGNITION_WINDOW_LO = 0.3       # window starts at 0.3 * t_ign
IGNITION_WINDOW_HI = 3.0       # window ends at 3.0 * t_ign

SEED = 42
# -----------------------------------------

rng = np.random.default_rng(SEED)


def scout_ignition_time(gas, T0, phi, P0, n_scout_steps=2000):
    """Cheap first pass: integrate with the solver's own adaptive steps and
    track dT/dt to locate the ignition time (peak dT/dt). Uses net.step()
    (internal adaptive stepping) rather than fixed advance() calls, since the
    solver's own step size already shrinks around the stiff ignition
    transient -- convenient for spotting it cheaply."""
    gas.TP = T0, P0
    gas.set_equivalence_ratio(phi, "H2", "O2:1.0")
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
                t_ign = t_prev + 0.5 * dt   # midpoint of the steepest step
        t_prev, T_prev = t, gas.T
        if t >= SIM_TIME:
            break

    ignited = max_dTdt > 1e6   # dT/dt threshold -- adjust if your T0/phi range changes
    return t_ign, ignited


def build_sample_schedule(t_ign, ignited):
    """Log-spaced pre-ignition points, dense points around the ignition
    window, log-spaced post-ignition points. Falls back to plain log-spacing
    across SIM_TIME if the trajectory never ignited (avoids a window around
    a meaningless t_ign=0)."""
    if not ignited or t_ign <= 0:
        return np.logspace(np.log10(SIM_TIME / 1e6), np.log10(SIM_TIME),
                            N_PRE_IGNITION + N_IGNITION + N_POST_IGNITION)

    t_lo = max(t_ign * IGNITION_WINDOW_LO, SIM_TIME / 1e6)
    t_hi = min(t_ign * IGNITION_WINDOW_HI, SIM_TIME)

    pre = np.logspace(np.log10(SIM_TIME / 1e6), np.log10(t_lo), N_PRE_IGNITION, endpoint=False)
    ign = np.linspace(t_lo, t_hi, N_IGNITION)                       # linear: window is narrow, don't need log here
    post = np.logspace(np.log10(t_hi), np.log10(SIM_TIME), N_POST_IGNITION + 1)[1:]

    return np.concatenate([pre, ign, post])


def run_trajectory(gas, T0, phi, P0, traj_idx):
    t_ign, ignited = scout_ignition_time(gas, T0, phi, P0)
    t_points = np.sort(build_sample_schedule(t_ign, ignited))

    gas.TP = T0, P0
    gas.set_equivalence_ratio(phi, "H2", "O2:1.0")
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
        wdot = gas.net_production_rates          # kmol/m3/s
        mw = gas.molecular_weights                # kg/kmol
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

    out_df = pd.DataFrame(all_rows)
    print(f"Generated {len(out_df)} datapoints across {out_df['traj_idx'].nunique()} trajectories")
    print(f"{n_ignited}/{out_df['traj_idx'].nunique()} trajectories ignited within SIM_TIME")
    print(f"Temperature range: {out_df['T'].min():.1f} K to {out_df['T'].max():.1f} K")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()