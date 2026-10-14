# train_deeponet.py
import sys
import numpy as np
import pandas as pd
import torch

from fordypning.config import Config
from fordypning.dataset import get_species_names, build_deeponet_data, trajectory_split
from fordypning.deeponet import DeepONetCRK

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")
cfg.weights_dir.mkdir(exist_ok=True)

CSV_PATH = "data/raw/gri30_ch4_air_trajectories_cbins.csv"


def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=device)


def train_deeponet(df_bin: pd.DataFrame, label: str, species: list[str]):
    print(f"\n=== training DeepONet {label}  (n_rows={len(df_bin)}, "
          f"n_traj={df_bin['traj_idx'].nunique()}) ===")

    data = build_deeponet_data(df_bin, species)
    train_mask, val_mask = trajectory_split(data["traj_id"], cfg.train.val_fraction, cfg.train.seed)

    Y0_tr, T0_tr = to_tensor(data["Y0"][train_mask]), to_tensor(data["T0"][train_mask])
    phi_tr, P_tr = to_tensor(data["phi"][train_mask]), to_tensor(data["P"][train_mask])
    t_tr, tend_tr = to_tensor(data["t"][train_mask]), to_tensor(data["t_end"][train_mask])
    T_tr, Y_tr = to_tensor(data["T"][train_mask]), to_tensor(data["Y"][train_mask])

    Y0_va, T0_va = to_tensor(data["Y0"][val_mask]), to_tensor(data["T0"][val_mask])
    phi_va, P_va = to_tensor(data["phi"][val_mask]), to_tensor(data["P"][val_mask])
    t_va, tend_va = to_tensor(data["t"][val_mask]), to_tensor(data["t_end"][val_mask])
    T_va, Y_va = to_tensor(data["T"][val_mask]), to_tensor(data["Y"][val_mask])

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

    save_path = cfg.weights_dir / f"deeponet_{label}.pt"
    best_val_loss = float("inf")

    n_train = Y0_tr.shape[0]
    for epoch in range(cfg.train.num_epochs):
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for i in range(0, n_train, cfg.train.batch_size):
            idx = perm[i:i + cfg.train.batch_size]

            T_pred, Y_pred, info = model(Y0_tr[idx], T0_tr[idx], phi_tr[idx], P_tr[idx],
                                          t_tr[idx], tend_tr[idx])

            Tn_true = (T_tr[idx] - model.T_mean) / model.T_std
            Yn_true = (torch.log(torch.clamp(Y_tr[idx], min=Y_MIN)) - model.logY_mean) / model.logY_std

            loss = loss_fn(info["Tn_pred"], Tn_true, delta=5.0) \
                 + loss_fn(info["Yn_pred"], Yn_true, delta=5.0)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item() * len(idx)

        with torch.no_grad():
            _, _, info_v = model(Y0_va, T0_va, phi_va, P_va, t_va, tend_va)
            Tn_true_v = (T_va - model.T_mean) / model.T_std
            Yn_true_v = (torch.log(torch.clamp(Y_va, min=Y_MIN)) - model.logY_mean) / model.logY_std
            val_loss = (loss_fn(info_v["Tn_pred"], Tn_true_v, delta=5.0)
                        + loss_fn(info_v["Yn_pred"], Yn_true_v, delta=5.0)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == cfg.train.num_epochs - 1:
            print(f"  {label}  epoch {epoch:4d}  train {total_loss/n_train:.4f}  "
                  f"val {val_loss:.4f}  best {best_val_loss:.4f}")

    print(f"  {label}: best val_loss={best_val_loss:.4f}, saved to {save_path}")
    return model


def main():
    df = pd.read_csv(CSV_PATH)
    species = get_species_names()

    bins_to_run = [int(b) for b in sys.argv[1:]] if len(sys.argv) > 1 else [0, 1, 2, 3, 4]
    print(f"training bins: {bins_to_run}")

    for bin_idx in bins_to_run:
        df_bin = df[df["bin_idx"] == bin_idx]
        if len(df_bin) == 0:
            print(f"bin {bin_idx}: no data, skipping")
            continue

        if bin_idx == 1:
            ign_traj = df_bin.groupby("traj_idx")["ignited"].first()
            df_ign = df_bin[df_bin["traj_idx"].isin(ign_traj[ign_traj].index)]
            df_noign = df_bin[df_bin["traj_idx"].isin(ign_traj[~ign_traj].index)]
            train_deeponet(df_ign, "bin1_ignite", species)
            train_deeponet(df_noign, "bin1_noignite", species)
        else:
            train_deeponet(df_bin, f"bin{bin_idx}", species)


if __name__ == "__main__":
    main()