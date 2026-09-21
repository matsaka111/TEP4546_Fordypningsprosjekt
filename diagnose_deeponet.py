# diagnose_deeponet.py
import pandas as pd
import torch

from fordypning.config import Config
from fordypning.dataset import get_species_names
from fordypning.deeponet import DeepONetCRK
from fordypning.diagnostics import plot_trajectory_rollout, compute_metrics, print_metrics

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
species = get_species_names()
n_species = len(species)

df = pd.read_csv("data/raw/gri30_ch4_air_trajectories_cbins.csv")
PLOT_SPECIES = ["CH4", "O2", "CO2", "H2O", "CO", "OH"]


def load_deeponet(path):
    model = DeepONetCRK(n_species=n_species).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def deeponet_predict_trajectory(model, traj_df, species, device):
    import numpy as np
    y_cols = [f"Y_{s}" for s in species]
    traj_df = traj_df.sort_values("t").reset_index(drop=True)

    Y0 = traj_df[y_cols].iloc[0].to_numpy(dtype=np.float32)
    T0 = float(traj_df["T0"].iloc[0])
    phi = float(traj_df["phi"].iloc[0])
    P = float(traj_df["P"].iloc[0])
    t_end = 5 * float(traj_df["t_ign"].iloc[0])
    t = traj_df["t"].to_numpy(dtype=np.float32)
    n = len(t)

    Y0_t = torch.tensor(np.tile(Y0, (n, 1)), dtype=torch.float32, device=device)
    T0_t = torch.full((n,), T0, dtype=torch.float32, device=device)
    phi_t = torch.full((n,), phi, dtype=torch.float32, device=device)
    P_t = torch.full((n,), P, dtype=torch.float32, device=device)
    t_t = torch.tensor(t, dtype=torch.float32, device=device)
    tend_t = torch.full((n,), t_end, dtype=torch.float32, device=device)

    with torch.no_grad():
        T_pred, Y_pred, _ = model(Y0_t, T0_t, phi_t, P_t, t_t, tend_t)

    return {
        "t": t,
        "T_true": traj_df["T"].to_numpy(dtype=np.float32),
        "Y_true": traj_df[y_cols].to_numpy(dtype=np.float32),
        "T_pred": T_pred.cpu().numpy(),
        "Y_pred": Y_pred.cpu().numpy(),
    }

def main():
    df_bin1 = df[df["bin_idx"] == 1]
    ign_traj = df_bin1.groupby("traj_idx")["ignited"].first()
    df_ign = df_bin1[df_bin1["traj_idx"].isin(ign_traj[ign_traj].index)]
    df_noign = df_bin1[df_bin1["traj_idx"].isin(ign_traj[~ign_traj].index)]

    cases = [
        ("bin1_ignite", df_ign, 611),
        ("bin1_noignite", df_noign, 600),
    ]

    for label, df_sub, traj_idx in cases:
        model = load_deeponet(cfg.weights_dir / f"deeponet_{label}.pt")
        traj_df = df_sub[df_sub["traj_idx"] == traj_idx]

        rollout = deeponet_predict_trajectory(model, traj_df, species, device)
        plot_trajectory_rollout(rollout, species, PLOT_SPECIES,
                                 title=f"DeepONet {label}  traj_idx={traj_idx}",
                                 save_path=f"model_weights/deeponet_rollout_{label}.png")

        labels = ["T"] + species
        y_true = pd.concat([pd.DataFrame({"T": rollout["T_true"]}),
                             pd.DataFrame(rollout["Y_true"], columns=species)], axis=1).to_numpy()
        y_pred = pd.concat([pd.DataFrame({"T": rollout["T_pred"]}),
                             pd.DataFrame(rollout["Y_pred"], columns=species)], axis=1).to_numpy()

        metrics = compute_metrics(y_true, y_pred, labels)
        print(f"\n--- DeepONet {label}  traj_idx={traj_idx} error table ---")
        print_metrics(metrics, top_n=15)

if __name__ == "__main__":
    main()