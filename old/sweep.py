# sweep.py
import os
import sys
import numpy as np
import pandas as pd
import torch
import cantera as ct

from fordypning.config import Config
from fordypning.dataset import get_species_names, build_pairs, build_deeponet_data, trajectory_split
from fordypning.models import CRKPINN
from fordypning.deeponet import DeepONetCRK
from fordypning.chemistry import GlobalArrheniusRate, build_physics_utils
from fordypning.losses import loss_data, loss_mole, loss_element, loss_enthalpy, loss_arrhenius
from fordypning.classifier import IgnitionClassifier

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

CSV_PATH = "data/raw/gri30_ch4_air_trajectories_cbins.csv"
T0_LO, T0_HI = 500.0, 2000.0
RESOLUTIONS = [1, 2, 5, 10]
N_TEST_TRAJ = 10
TEST_SEED = 0

LAMBDA_MOLE = 0.1
LAMBDA_ELEMENT = 0.0004


def to_tensor(arr, dev=device):
    return torch.tensor(arr, dtype=torch.float32, device=dev)


# ---------------- bin assignment ----------------

def assign_bins(df, n_bins):
    edges = np.linspace(T0_LO, T0_HI, n_bins + 1)
    per_traj_T0 = df.groupby("traj_idx")["T0"].first()
    bin_idx = np.clip(np.digitize(per_traj_T0.values, edges) - 1, 0, n_bins - 1)
    traj_to_bin = dict(zip(per_traj_T0.index, bin_idx))
    return df["traj_idx"].map(traj_to_bin).to_numpy(), edges


def split_ignite_groups(df_bin):
    """Returns list of (suffix, df_sub) -- one entry if single-class,
    two entries ('ignite'/'noignite') if the bin has both."""
    per_traj = df_bin.groupby("traj_idx")["ignited"].first()
    ign_ids = per_traj[per_traj].index
    noign_ids = per_traj[~per_traj].index

    if len(ign_ids) > 0 and len(noign_ids) > 0:
        return [
            ("ignite", df_bin[df_bin["traj_idx"].isin(ign_ids)], len(ign_ids)),
            ("noignite", df_bin[df_bin["traj_idx"].isin(noign_ids)], len(noign_ids)),
        ]
    else:
        return [("all", df_bin, len(per_traj))]


# ---------------- fixed test trajectories (same across every config) ----------------

def select_test_trajectories(df, n=N_TEST_TRAJ, seed=TEST_SEED):
    """Evenly spans T0 range: sort unique trajectories by T0, take n
    roughly-evenly-spaced ones. Deterministic given seed."""
    per_traj_T0 = df.groupby("traj_idx")["T0"].first().sort_values()
    idx = np.linspace(0, len(per_traj_T0) - 1, n).astype(int)
    return per_traj_T0.index[idx].to_numpy()


# ---------------- CRKPINN training ----------------

def train_crkpinn(df_sub, save_path, species, gas, n_traj):
    data = build_pairs(df_sub, species)
    train_mask, val_mask = trajectory_split(data["traj_id"], cfg.train.val_fraction, cfg.train.seed)
    if val_mask.sum() == 0:  # too few trajectories to split -- train on everything, no val
        train_mask, val_mask = np.ones_like(train_mask, dtype=bool), train_mask.copy()

    T_tr, Y_tr = to_tensor(data["T"][train_mask]), to_tensor(data["Y"][train_mask])
    T0_tr, P_tr = to_tensor(data["T0"][train_mask]), to_tensor(data["P"][train_mask])
    dt_tr, tend_tr = to_tensor(data["dt"][train_mask]), to_tensor(data["t_end"][train_mask])
    T_next_tr, Y_next_tr = to_tensor(data["T_next"][train_mask]), to_tensor(data["Y_next"][train_mask])

    T_va, Y_va = to_tensor(data["T"][val_mask]), to_tensor(data["Y"][val_mask])
    T0_va, P_va = to_tensor(data["T0"][val_mask]), to_tensor(data["P"][val_mask])
    dt_va, tend_va = to_tensor(data["dt"][val_mask]), to_tensor(data["t_end"][val_mask])
    T_next_va, Y_next_va = to_tensor(data["T_next"][val_mask]), to_tensor(data["Y_next"][val_mask])

    model = CRKPINN(n_species=len(species), hidden=cfg.model.hidden_dim,
                     n_hidden_layers=cfg.model.num_layers).to(device)

    Y_MIN = model.Y_MIN
    logY_all = torch.log(torch.clamp(torch.cat([Y_tr, Y_next_tr], dim=0), min=Y_MIN))
    logP_all = torch.log(torch.clamp(P_tr, min=1e-6))
    tprime_raw = dt_tr / torch.clamp(tend_tr, min=1e-12)

    model.set_norm_stats(
        logY_mean=logY_all.mean(dim=0), logY_std=logY_all.std(dim=0),
        T_mean=torch.cat([T_tr, T_next_tr]).mean(), T_std=torch.cat([T_tr, T_next_tr]).std(),
        T0_mean=T0_tr.mean(), T0_std=T0_tr.std(),
        logP_mean=logP_all.mean(), logP_std=logP_all.std(),
        tprime_mean=tprime_raw.mean(), tprime_std=tprime_raw.std(),
    )

    physics = build_physics_utils(gas, species, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    n_train = T_tr.shape[0]
    for epoch in range(cfg.train.num_epochs):
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, cfg.train.batch_size):
            idx = perm[i:i + cfg.train.batch_size]
            T_next_pred, Y_next_pred, info = model(
                T_tr[idx], Y_tr[idx], T0_tr[idx], P_tr[idx], dt_tr[idx], tend_tr[idx])

            L_data = loss_data(model, info, T_next_tr[idx], Y_next_tr[idx])
            L_mole = loss_mole(Y_next_pred)
            L_element = loss_element(physics.element_mass_fractions, Y_tr[idx], Y_next_pred)
            loss = L_data + LAMBDA_MOLE * L_mole + LAMBDA_ELEMENT * L_element

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    torch.save(model.state_dict(), save_path)
    return model


# ---------------- DeepONet training ----------------

def train_deeponet(df_sub, save_path, species, n_traj):
    data = build_deeponet_data(df_sub, species)
    train_mask, val_mask = trajectory_split(data["traj_id"], cfg.train.val_fraction, cfg.train.seed)
    if val_mask.sum() == 0:
        train_mask, val_mask = np.ones_like(train_mask, dtype=bool), train_mask.copy()

    Y0_tr, T0_tr = to_tensor(data["Y0"][train_mask]), to_tensor(data["T0"][train_mask])
    phi_tr, P_tr = to_tensor(data["phi"][train_mask]), to_tensor(data["P"][train_mask])
    t_tr, tend_tr = to_tensor(data["t"][train_mask]), to_tensor(data["t_end"][train_mask])
    T_tr, Y_tr = to_tensor(data["T"][train_mask]), to_tensor(data["Y"][train_mask])

    model = DeepONetCRK(n_species=len(species)).to(device)

    Y_MIN = model.Y_MIN
    logY0_all = torch.log(torch.clamp(Y0_tr, min=Y_MIN))
    logY_all = torch.log(torch.clamp(Y_tr, min=Y_MIN))
    logP_all = torch.log(torch.clamp(P_tr, min=1e-6))
    tprime_raw = t_tr / torch.clamp(tend_tr, min=1e-12)

    model.set_norm_stats(
        logY0_mean=logY0_all.mean(dim=0), logY0_std=logY0_all.std(dim=0),
        T0_mean=T0_tr.mean(), T0_std=T0_tr.std(),
        phi_mean=phi_tr.mean(), phi_std=phi_tr.std(),
        logP_mean=logP_all.mean(), logP_std=logP_all.std(),
        tprime_mean=tprime_raw.mean(), tprime_std=tprime_raw.std(),
        logY_mean=logY_all.mean(dim=0), logY_std=logY_all.std(dim=0),
        T_mean=T_tr.mean(), T_std=T_tr.std(),
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    loss_fn = torch.nn.functional.huber_loss

    n_train = Y0_tr.shape[0]
    for epoch in range(cfg.train.num_epochs):
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, cfg.train.batch_size):
            idx = perm[i:i + cfg.train.batch_size]
            T_pred, Y_pred, info = model(Y0_tr[idx], T0_tr[idx], phi_tr[idx], P_tr[idx],
                                          t_tr[idx], tend_tr[idx])
            Tn_true = (T_tr[idx] - model.T_mean) / model.T_std
            Yn_true = (torch.log(torch.clamp(Y_tr[idx], min=Y_MIN)) - model.logY_mean) / model.logY_std
            loss = loss_fn(info["Tn_pred"], Tn_true, delta=5.0) + loss_fn(info["Yn_pred"], Yn_true, delta=5.0)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    torch.save(model.state_dict(), save_path)
    return model


# ---------------- evaluation on fixed test trajectories ----------------

def crkpinn_final_state(model, traj_df, y_cols):
    traj_df = traj_df.sort_values("t").reset_index(drop=True)
    T0, P = float(traj_df["T0"].iloc[0]), float(traj_df["P"].iloc[0])
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


def deeponet_final_state(model, traj_df, y_cols):
    traj_df = traj_df.sort_values("t").reset_index(drop=True)
    Y0 = traj_df[y_cols].iloc[0].to_numpy(dtype=np.float32)
    T0, phi, P = float(traj_df["T0"].iloc[0]), float(traj_df["phi"].iloc[0]), float(traj_df["P"].iloc[0])
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


def route_test_traj(test_traj_id, df, bin_idx_arr, model_groups):
    """model_groups: dict[bin_idx] -> list of (suffix, model, n_traj, ign_ids)
    Returns the right model for this test trajectory."""
    row = df[df["traj_idx"] == test_traj_id].iloc[0]
    T0 = row["T0"]
    edges = model_groups["edges"]
    b = np.clip(np.digitize([T0], edges)[0] - 1, 0, len(edges) - 2)
    groups = model_groups["bins"][b]
    if len(groups) == 1:
        return groups[0][1], groups[0][0]  # model, suffix
    is_ignited = bool(df[(df["traj_idx"] == test_traj_id)]["ignited"].iloc[0])
    for suffix, model, n_traj in groups:
        if (suffix == "ignite") == is_ignited:
            return model, suffix
    return groups[0][1], groups[0][0]


# ---------------- main sweep ----------------

def run_sweep():
    df = pd.read_csv(CSV_PATH)
    species = get_species_names()
    y_cols = [f"Y_{s}" for s in species]
    gas = ct.Solution("gri30.yaml")

    test_traj_ids = select_test_trajectories(df)
    print(f"fixed test trajectories: {test_traj_ids.tolist()}")

    results_rows = []

    for arch in ["crkpinn", "deeponet"]:
        for n_bins in RESOLUTIONS:
            tag = "no_bin" if n_bins == 1 else f"{n_bins}_bins"
            out_dir = f"results/{arch}/{tag}"
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n########## {arch} / {tag} ##########")

            bin_idx_arr, edges = assign_bins(df, n_bins)
            df_local = df.copy()
            df_local["_sweep_bin"] = bin_idx_arr

            bins_models = {}
            for b in range(n_bins):
                df_bin = df_local[df_local["_sweep_bin"] == b]
                if len(df_bin) == 0:
                    bins_models[b] = []
                    continue

                groups = split_ignite_groups(df_bin)
                trained = []
                for suffix, df_sub, n_traj in groups:
                    label = f"bin{b}_{suffix}" if len(groups) > 1 else f"bin{b}"
                    save_path = f"{out_dir}/{arch}_{label}.pt"
                    print(f"  training {label}  (n_traj={n_traj})")

                    if arch == "crkpinn":
                        model = train_crkpinn(df_sub, save_path, species, gas, n_traj)
                    else:
                        model = train_deeponet(df_sub, save_path, species, n_traj)

                    trained.append((suffix, model, n_traj))
                bins_models[b] = trained

            model_groups = {"edges": edges, "bins": bins_models}

            # evaluate on fixed test trajectories
            for test_id in test_traj_ids:
                traj_df = df[df["traj_idx"] == test_id]
                if len(traj_df) < 2:
                    continue
                model, suffix = route_test_traj(test_id, df, bin_idx_arr, model_groups)
                T0 = float(traj_df["T0"].iloc[0])

                traj_sorted = traj_df.sort_values("t")
                T_true_final = traj_sorted["T"].iloc[-1]
                Y_true_final = traj_sorted[y_cols].iloc[-1].to_numpy(dtype=np.float32)

                if arch == "crkpinn":
                    T_pred, Y_pred = crkpinn_final_state(model, traj_df, y_cols)
                else:
                    T_pred, Y_pred = deeponet_final_state(model, traj_df, y_cols)

                row = {
                    "arch": arch, "n_bins": n_bins, "test_traj_id": test_id,
                    "T0": T0, "routed_group": suffix,
                    "T_true": T_true_final, "T_pred": T_pred, "T_abs_err": abs(T_pred - T_true_final),
                }
                for i, s in enumerate(species):
                    row[f"{s}_true"] = Y_true_final[i]
                    row[f"{s}_pred"] = Y_pred[i]
                    row[f"{s}_abs_err"] = abs(Y_pred[i] - Y_true_final[i])
                results_rows.append(row)

            pd.DataFrame(results_rows).to_csv("results/sweep_results_partial.csv", index=False)

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv("results/sweep_results_full.csv", index=False)
    print(f"\nsaved full results to results/sweep_results_full.csv  ({len(results_df)} rows)")
    return results_df


if __name__ == "__main__":
    run_sweep()