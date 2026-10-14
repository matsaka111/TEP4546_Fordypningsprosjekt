# eval_deeponet_all.py
import numpy as np
import pandas as pd
import torch

from fordypning.config import Config
from fordypning.dataset import get_species_names, build_deeponet_data, trajectory_split
from fordypning.deeponet import DeepONetCRK

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
species = get_species_names()
n_species = len(species)

df = pd.read_csv("data/raw/gri30_ch4_air_trajectories_cbins.csv")


def load_deeponet(path):
    model = DeepONetCRK(n_species=n_species).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def eval_bin(df_bin, label):
    data = build_deeponet_data(df_bin, species)
    train_mask, val_mask = trajectory_split(data["traj_id"], cfg.train.val_fraction, cfg.train.seed)

    def to_tensor(arr):
        return torch.tensor(arr[val_mask], dtype=torch.float32, device=device)

    Y0, T0, phi, P = to_tensor(data["Y0"]), to_tensor(data["T0"]), to_tensor(data["phi"]), to_tensor(data["P"])
    t, tend = to_tensor(data["t"]), to_tensor(data["t_end"])
    T_true, Y_true = to_tensor(data["T"]), to_tensor(data["Y"])

    model = load_deeponet(cfg.weights_dir / f"deeponet_{label}.pt")
    with torch.no_grad():
        T_pred, Y_pred, _ = model(Y0, T0, phi, P, t, tend)

    T_err = (T_pred - T_true).abs().cpu().numpy()
    Y_err = (Y_pred - Y_true).abs().cpu().numpy()  # (n, n_species)

    row = {"T": T_err.mean()}
    for i, s in enumerate(species):
        row[s] = Y_err[:, i].mean()
    return row


labels_and_dfs = [
    ("bin0", df[df.bin_idx == 0]),
    ("bin1_ignite", None),
    ("bin1_noignite", None),
    ("bin2", df[df.bin_idx == 2]),
    ("bin3", df[df.bin_idx == 3]),
    ("bin4", df[df.bin_idx == 4]),
]

df_bin1 = df[df.bin_idx == 1]
ign_traj = df_bin1.groupby("traj_idx")["ignited"].first()
labels_and_dfs[1] = ("bin1_ignite", df_bin1[df_bin1["traj_idx"].isin(ign_traj[ign_traj].index)])
labels_and_dfs[2] = ("bin1_noignite", df_bin1[df_bin1["traj_idx"].isin(ign_traj[~ign_traj].index)])

results = {}
for label, df_sub in labels_and_dfs:
    if len(df_sub) == 0:
        continue
    results[label] = eval_bin(df_sub, label)

summary = pd.DataFrame(results)  # rows = species+T, columns = bins
summary["max_across_bins"] = summary.max(axis=1)
summary = summary.sort_values("max_across_bins", ascending=False)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)
print(summary)

summary.to_csv("model_weights/deeponet_all_bins_error_summary.csv")
print("\nsaved to model_weights/deeponet_all_bins_error_summary.csv")