# compare_final_state.py
import numpy as np
import pandas as pd
import torch

from fordypning.config import Config
from fordypning.dataset import get_species_names
from fordypning.models import CRKPINN
from fordypning.deeponet import DeepONetCRK

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
species = get_species_names()
n_species = len(species)
y_cols = [f"Y_{s}" for s in species]

df = pd.read_csv("data/raw/gri30_ch4_air_trajectories_cbins.csv")


def load_crkpinn(path, residual=False):
    model = CRKPINN(n_species=n_species, hidden=cfg.model.hidden_dim,
                     n_hidden_layers=cfg.model.num_layers, residual=residual).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_deeponet(path):
    model = DeepONetCRK(n_species=n_species).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def crkpinn_final_state(model, traj_df):
    """Chain through every step of the trajectory, keep only the last prediction."""
    traj_df = traj_df.sort_values("t").reset_index(drop=True)
    T0 = float(traj_df["T0"].iloc[0])
    P = float(traj_df["P"].iloc[0])
    t_end = 5 * float(traj_df["t_ign"].iloc[0])
    t = traj_df["t"].to_numpy()

    T_cur = torch.tensor([traj_df["T"].iloc[0]], dtype=torch.float32, device=device)
    Y_cur = torch.tensor(traj_df[y_cols].iloc[0:1].to_numpy(dtype=np.float32), device=device)
    T0_t = torch.tensor([T0], dtype=torch.float32, device=device)
    P_t = torch.tensor([P], dtype=torch.float32, device=device)
    tend_t = torch.tensor([t_end], dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in range(1, len(t)):
            dt_t = torch.tensor([t[i] - t[i - 1]], dtype=torch.float32, device=device)
            T_next, Y_next, _ = model(T_cur, Y_cur, T0_t, P_t, dt_t, tend_t)
            T_cur, Y_cur = T_next, Y_next

    return T_cur.item(), Y_cur.cpu().numpy()[0]


def deeponet_final_state(model, traj_df):
    """One direct query at the trajectory's final time -- no chaining."""
    traj_df = traj_df.sort_values("t").reset_index(drop=True)
    Y0 = traj_df[y_cols].iloc[0].to_numpy(dtype=np.float32)
    T0 = float(traj_df["T0"].iloc[0])
    phi = float(traj_df["phi"].iloc[0])
    P = float(traj_df["P"].iloc[0])
    t_end = 5 * float(traj_df["t_ign"].iloc[0])
    t_final = float(traj_df["t"].iloc[-1])

    Y0_t = torch.tensor(Y0[None, :], dtype=torch.float32, device=device)
    T0_t = torch.tensor([T0], dtype=torch.float32, device=device)
    phi_t = torch.tensor([phi], dtype=torch.float32, device=device)
    P_t = torch.tensor([P], dtype=torch.float32, device=device)
    t_t = torch.tensor([t_final], dtype=torch.float32, device=device)
    tend_t = torch.tensor([t_end], dtype=torch.float32, device=device)

    with torch.no_grad():
        T_pred, Y_pred, _ = model(Y0_t, T0_t, phi_t, P_t, t_t, tend_t)

    return T_pred.item(), Y_pred.cpu().numpy()[0]


def evaluate_bin_final_state(df_bin, crk_label, deep_label, residual, n_traj_sample=50, seed=0):
    rng = np.random.default_rng(seed)
    traj_ids = df_bin["traj_idx"].unique()
    if len(traj_ids) > n_traj_sample:
        traj_ids = rng.choice(traj_ids, size=n_traj_sample, replace=False)

    crk_model = load_crkpinn(cfg.weights_dir / f"crkpinn_{crk_label}.pt", residual=residual)
    deep_model = load_deeponet(cfg.weights_dir / f"deeponet_{deep_label}.pt")

    crk_T_err, crk_Y_err = [], []
    deep_T_err, deep_Y_err = [], []

    for traj_idx in traj_ids:
        traj_df = df_bin[df_bin["traj_idx"] == traj_idx]
        if len(traj_df) < 2:
            continue
        traj_df_sorted = traj_df.sort_values("t")
        T_true_final = traj_df_sorted["T"].iloc[-1]
        Y_true_final = traj_df_sorted[y_cols].iloc[-1].to_numpy(dtype=np.float32)

        T_crk, Y_crk = crkpinn_final_state(crk_model, traj_df)
        crk_T_err.append(abs(T_crk - T_true_final))
        crk_Y_err.append(np.abs(Y_crk - Y_true_final))

        T_deep, Y_deep = deeponet_final_state(deep_model, traj_df)
        deep_T_err.append(abs(T_deep - T_true_final))
        deep_Y_err.append(np.abs(Y_deep - Y_true_final))

    crk_row = {"T": np.mean(crk_T_err)}
    deep_row = {"T": np.mean(deep_T_err)}
    crk_Y_err, deep_Y_err = np.array(crk_Y_err), np.array(deep_Y_err)
    for i, s in enumerate(species):
        crk_row[s] = crk_Y_err[:, i].mean()
        deep_row[s] = deep_Y_err[:, i].mean()

    return crk_row, deep_row


bins = [
    ("bin0", "bin0", "bin0", False),
    ("bin1_ignite", "bin1_ignite", "bin1_ignite", False),
    ("bin1_noignite", "bin1_noignite", "bin1_noignite", True),
    ("bin2", "bin2", "bin2", False),
    ("bin3", "bin3", "bin3", False),
    ("bin4", "bin4", "bin4", False),
]

crk_all, deep_all = {}, {}

for name, crk_label, deep_label, residual in bins:
    if name in ("bin1_ignite", "bin1_noignite"):
        df_bin1 = df[df.bin_idx == 1]
        ign_traj = df_bin1.groupby("traj_idx")["ignited"].first()
        df_sub = df_bin1[df_bin1["traj_idx"].isin(ign_traj[ign_traj].index)] if "no" not in name \
            else df_bin1[df_bin1["traj_idx"].isin(ign_traj[~ign_traj].index)]
    else:
        bin_idx = int(name.replace("bin", ""))
        df_sub = df[df.bin_idx == bin_idx]

    print(f"evaluating final-state error: {name}...")
    crk_row, deep_row = evaluate_bin_final_state(df_sub, crk_label, deep_label, residual)
    crk_all[name] = crk_row
    deep_all[name] = deep_row

crk_df = pd.DataFrame(crk_all)
deep_df = pd.DataFrame(deep_all)

comparison = pd.DataFrame({
    **{f"{c}_crk": crk_df[c] for c in crk_df.columns},
    **{f"{c}_deep": deep_df[c] for c in deep_df.columns},
})
comparison = comparison[sorted(comparison.columns)]

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 250)
print("\n=== FINAL-STATE mean|abs error|: CRKPINN (chained) vs DeepONet (direct) ===")
print(comparison)

comparison.to_csv("model_weights/final_state_comparison.csv")
print("\nsaved to model_weights/final_state_comparison.csv")