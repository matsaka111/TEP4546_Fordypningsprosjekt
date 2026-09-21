# fordypning/dataset.py
import numpy as np
import pandas as pd
import cantera as ct

N_TEND = 5  # t_end = N_TEND * t_ign, per CRK-PINN Eq. 16


def get_species_names(mech: str = "gri30.yaml") -> list[str]:
    """Species order matching the mechanism used to generate the data."""
    return ct.Solution(mech).species_names


def load_raw_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_pairs(df: pd.DataFrame, species: list[str]):
    """
    Build consecutive same-trajectory pairs, vectorized via groupby + shift.
    Returns a dict of numpy arrays, one row per (t_i, t_{i+1}) pair:
      T, Y, T0, P, dt, t_end, rho   -- current-state inputs to CRKPINN.forward
      T_next, Y_next                -- targets
      traj_id                       -- for trajectory-level splitting
    """
    y_cols = [f"Y_{s}" for s in species]

    df = df.sort_values(["traj_idx", "t"]).reset_index(drop=True)
    g = df.groupby("traj_idx")

    next_t = g["t"].shift(-1)
    next_T = g["T"].shift(-1)
    next_Y = g[y_cols].shift(-1)

    valid = next_t.notna()

    dt = (next_t - df["t"])[valid].to_numpy(dtype=np.float32)
    t_ign = df.loc[valid, "t_ign"].to_numpy(dtype=np.float32)

    return {
        "T": df.loc[valid, "T"].to_numpy(dtype=np.float32),
        "Y": df.loc[valid, y_cols].to_numpy(dtype=np.float32),
        "T0": df.loc[valid, "T0"].to_numpy(dtype=np.float32),
        "P": df.loc[valid, "P"].to_numpy(dtype=np.float32),
        "dt": dt,
        "t_end": N_TEND * t_ign,
        "rho": df.loc[valid, "rho"].to_numpy(dtype=np.float32),
        "T_next": next_T[valid].to_numpy(dtype=np.float32),
        "Y_next": next_Y[valid].to_numpy(dtype=np.float32),
        "traj_id": df.loc[valid, "traj_idx"].to_numpy(),
    }

def build_sequences(df: pd.DataFrame, species: list[str], k_steps: int):
    """
    Build K-step consecutive sequences per trajectory, for rollout training.
    Each sequence has K+1 points: the starting state plus K true next-states,
    so the model can be unrolled K times and compared against the true
    trajectory at every intermediate step, not just one step ahead.

    Returns arrays shaped (n_sequences, K+1, ...) for T/Y, (n_sequences, K)
    for dt, and (n_sequences,) for T0/P/t_end/traj_id (constant per sequence
    since these don't change within one trajectory).
    """
    y_cols = [f"Y_{s}" for s in species]
    df = df.sort_values(["traj_idx", "t"]).reset_index(drop=True)

    T_seqs, Y_seqs, dt_seqs = [], [], []
    T0_list, P_list, tend_list, traj_list = [], [], [], []

    for traj_idx, g in df.groupby("traj_idx"):
        g = g.sort_values("t").reset_index(drop=True)
        n = len(g)
        if n < k_steps + 1:
            continue

        T0 = g["T0"].iloc[0]
        P = g["P"].iloc[0]
        t_end = 5 * g["t_ign"].iloc[0]  # N_TEND=5, matches build_pairs

        t_vals = g["t"].to_numpy()
        T_vals = g["T"].to_numpy(dtype=np.float32)
        Y_vals = g[y_cols].to_numpy(dtype=np.float32)

        for start in range(n - k_steps):
            T_seqs.append(T_vals[start:start + k_steps + 1])
            Y_seqs.append(Y_vals[start:start + k_steps + 1])
            dt_seqs.append(np.diff(t_vals[start:start + k_steps + 1]))
            T0_list.append(T0)
            P_list.append(P)
            tend_list.append(t_end)
            traj_list.append(traj_idx)

    return {
        "T_seq": np.stack(T_seqs).astype(np.float32),       # (n_seq, K+1)
        "Y_seq": np.stack(Y_seqs).astype(np.float32),       # (n_seq, K+1, n_species)
        "dt_seq": np.stack(dt_seqs).astype(np.float32),     # (n_seq, K)
        "T0": np.array(T0_list, dtype=np.float32),
        "P": np.array(P_list, dtype=np.float32),
        "t_end": np.array(tend_list, dtype=np.float32),
        "traj_id": np.array(traj_list),
    }

def build_deeponet_data(df: pd.DataFrame, species: list[str]):
    """Every ROW of every trajectory becomes one (IC, query_t) -> state(t)
    example. IC = (Y0, T0, phi, P) from t=0 of that trajectory."""
    y_cols = [f"Y_{s}" for s in species]
    df = df.sort_values(["traj_idx", "t"]).reset_index(drop=True)

    Y0_list, T0_list, phi_list, P_list = [], [], [], []
    t_list, tend_list, T_list, Y_list, traj_list = [], [], [], [], []

    for traj_idx, g in df.groupby("traj_idx"):
        g = g.sort_values("t").reset_index(drop=True)
        Y0 = g[y_cols].iloc[0].to_numpy(dtype=np.float32)
        T0 = g["T0"].iloc[0]
        phi = g["phi"].iloc[0]
        P = g["P"].iloc[0]
        t_end = 5 * g["t_ign"].iloc[0]

        n = len(g)
        Y0_list.append(np.tile(Y0, (n, 1)))
        T0_list.append(np.full(n, T0, dtype=np.float32))
        phi_list.append(np.full(n, phi, dtype=np.float32))
        P_list.append(np.full(n, P, dtype=np.float32))
        t_list.append(g["t"].to_numpy(dtype=np.float32))
        tend_list.append(np.full(n, t_end, dtype=np.float32))
        T_list.append(g["T"].to_numpy(dtype=np.float32))
        Y_list.append(g[y_cols].to_numpy(dtype=np.float32))
        traj_list.append(np.full(n, traj_idx))

    return {
        "Y0": np.concatenate(Y0_list), "T0": np.concatenate(T0_list),
        "phi": np.concatenate(phi_list), "P": np.concatenate(P_list),
        "t": np.concatenate(t_list), "t_end": np.concatenate(tend_list),
        "T": np.concatenate(T_list), "Y": np.concatenate(Y_list),
        "traj_id": np.concatenate(traj_list),
    }

def trajectory_split(traj_ids: np.ndarray, val_fraction: float, seed: int):
    """Split by whole trajectory, not by point, to avoid leakage."""
    unique_traj = np.unique(traj_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_traj)

    n_val = int(len(unique_traj) * val_fraction)
    val_traj = set(unique_traj[:n_val])

    val_mask = np.isin(traj_ids, list(val_traj))
    return ~val_mask, val_mask  # train_mask, val_mask


if __name__ == "__main__":
    df = load_raw_csv("data/raw/gri30_ch4_air_trajectories.csv")
    species = get_species_names()
    data = build_pairs(df, species)

    for k, v in data.items():
        print(f"{k:10s} shape={v.shape}")

    train_mask, val_mask = trajectory_split(data["traj_id"], val_fraction=0.2, seed=0)
    print(f"train pairs: {train_mask.sum()}, val pairs: {val_mask.sum()}")