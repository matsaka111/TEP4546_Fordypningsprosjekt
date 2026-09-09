"""
Generate training data as a mix of two single-step (no time integration) sampling
strategies:
  1. Anchor-perturbation: perturb Y around the GRI30 flame example's unburned
     inlet composition (phi=1.0), T swept over T_RANGE.
  2. Phi/dilution sweep: randomize equivalence ratio and N2 dilution level with
     a CH4/O2/N2 base, T swept over T_RANGE. Broader physical coverage than (1),
     still chemically plausible (NOT raw Dirichlet over all species).

Output: single CSV with a 'source' column marking which strategy produced each row.
"""

import numpy as np
import pandas as pd
import cantera as ct

# ---------------- Config ----------------
INPUT_CSV = "GRI30_CH4-air_PHI1.0_T298-P1.csv"
OUTPUT_CSV = "raw/mixed_data.csv"
MECH = "gri30.yaml"

N_TOTAL = 100_000
FRAC_ANCHOR = 0.70              # fraction from anchor-perturbation
N_ANCHOR = int(N_TOTAL * FRAC_ANCHOR)
N_PHI_SWEEP = N_TOTAL - N_ANCHOR

T_RANGE = (500.0, 2500.0)       # feasible initial-temperature range
P_FIXED = 101325.0              # 1 atm, matches anchor's pressure

Y_PERTURB_STD = 0.02            # relative std for anchor-perturbation noise
PHI_RANGE = (0.5, 1.5)          # equivalence ratio range for phi-sweep
DILUTION_RANGE = (2.5, 5.0)     # N2:O2 molar ratio range (3.76 = standard air)

SEED = 42
# -----------------------------------------

rng = np.random.default_rng(SEED)


def load_anchor(path, idx=0):
    df = pd.read_csv(path)
    species_cols = [c for c in df.columns if c not in ("z (m)", "u (m/s)", "V (1/s)", "T (K)", "P (Pa)", "rho (kg/m3)")]
    row = df.iloc[idx]
    Y0 = row[species_cols].to_numpy(dtype=float)
    return Y0, species_cols


def sample_anchor_perturb(Y0, species_cols):
    """Strategy 1: perturb composition around the stoichiometric anchor."""
    T = rng.uniform(*T_RANGE)
    noise = rng.normal(1.0, Y_PERTURB_STD, size=len(Y0))
    Y = np.clip(Y0 * noise, 0.0, None)
    Y_sum = Y.sum()
    if Y_sum <= 0:
        return None, None
    Y = Y / Y_sum
    return T, Y


def sample_phi_sweep(gas, species_cols):
    """Strategy 2: random equivalence ratio + dilution, CH4/O2/N2 base."""
    T = rng.uniform(*T_RANGE)
    phi = rng.uniform(*PHI_RANGE)
    dilution = rng.uniform(*DILUTION_RANGE)  # N2:O2 molar ratio
    gas.set_equivalence_ratio(phi, "CH4", f"O2:1, N2:{dilution}")
    Y_full = np.zeros(len(species_cols))
    for i, s in enumerate(species_cols):
        Y_full[i] = gas.Y[gas.species_index(s)] if s in gas.species_names else 0.0
    return T, Y_full


def compute_ydot(gas, T, P, Y, species_cols):
    gas.TPY = T, P, dict(zip(species_cols, Y))
    rho = gas.density
    wdot = gas.net_production_rates          # kmol/m3/s
    mw = gas.molecular_weights                # kg/kmol
    Ydot = wdot * mw / rho

    u_partial = gas.partial_molar_int_energies  # J/kmol
    cv = gas.cv_mass
    Tdot = -np.sum(u_partial * wdot) / (rho * cv)

    return rho, Tdot, Ydot


def main():
    Y0, species_cols = load_anchor(INPUT_CSV, idx=0)
    gas = ct.Solution(MECH)

    rows = []

    # Strategy 1: anchor-perturbation
    for i in range(N_ANCHOR):
        T, Y = sample_anchor_perturb(Y0, species_cols)
        if Y is None:
            continue
        try:
            rho, Tdot, Ydot = compute_ydot(gas, T, P_FIXED, Y, species_cols)
        except ct.CanteraError:
            continue
        record = {"T": T, "P": P_FIXED, "rho": rho, "Tdot": Tdot, "source": "anchor_perturb", "sample_idx": i}
        record.update({f"Y_{s}": y for s, y in zip(species_cols, Y)})
        record.update({f"Ydot_{s}": yd for s, yd in zip(species_cols, Ydot)})
        rows.append(record)

    # Strategy 2: phi/dilution sweep
    for i in range(N_PHI_SWEEP):
        T, Y = sample_phi_sweep(gas, species_cols)
        try:
            rho, Tdot, Ydot = compute_ydot(gas, T, P_FIXED, Y, species_cols)
        except ct.CanteraError:
            continue
        record = {"T": T, "P": P_FIXED, "rho": rho, "Tdot": Tdot, "source": "phi_sweep", "sample_idx": i}
        record.update({f"Y_{s}": y for s, y in zip(species_cols, Y)})
        record.update({f"Ydot_{s}": yd for s, yd in zip(species_cols, Ydot)})
        rows.append(record)

    out_df = pd.DataFrame(rows)
    print(f"Generated {len(out_df)} datapoints")
    print(out_df["source"].value_counts())
    print(f"Temperature range: {out_df['T'].min():.1f} K to {out_df['T'].max():.1f} K")

    import os
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()