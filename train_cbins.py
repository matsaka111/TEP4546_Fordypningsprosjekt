# train_cbins.py
import sys
import numpy as np
import pandas as pd
import torch
import cantera as ct

from fordypning.config import Config
from fordypning.dataset import get_species_names, build_pairs, trajectory_split
from fordypning.models import CRKPINN
from fordypning.chemistry import GlobalArrheniusRate, build_physics_utils
from fordypning.losses import loss_data, loss_data_huber, loss_mole, loss_element, loss_enthalpy, loss_arrhenius
from fordypning.classifier import IgnitionClassifier

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")
cfg.weights_dir.mkdir(exist_ok=True)

CSV_PATH = "data/raw/gri30_ch4_air_trajectories_cbins.csv"
N_BINS = 5

LAMBDA_DATA = 1.0
LAMBDA_MOLE = 0.1
LAMBDA_ELEMENT = 0.0004
LAMBDA_ENTHALPY = 0.0
LAMBDA_ARRHENIUS = 0.0  # slow (batched Jacobian) -- keep off unless specifically testing it


def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=device)


def train_ignition_classifier(df_bin, name, device, n_epochs=200):
    per_traj = df_bin.groupby("traj_idx").first()[["T0", "phi", "P", "ignited"]]
    T0 = torch.tensor(per_traj["T0"].to_numpy(), dtype=torch.float32, device=device)
    phi = torch.tensor(per_traj["phi"].to_numpy(), dtype=torch.float32, device=device)
    P = torch.tensor(per_traj["P"].to_numpy(), dtype=torch.float32, device=device)
    y = torch.tensor(per_traj["ignited"].to_numpy(), dtype=torch.float32, device=device)

    clf = IgnitionClassifier().to(device)
    X = torch.stack([T0, phi, P], dim=-1)
    clf.set_norm_stats(X.mean(dim=0), X.std(dim=0))

    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        logits = clf(T0, phi, P)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = (torch.sigmoid(clf(T0, phi, P)) > 0.5).float()
        acc = (pred == y).float().mean().item()
    print(f"  classifier train accuracy: {acc:.3f}  (n={len(y)}, "
          f"{int(y.sum())} ignite / {int((1-y).sum())} no-ignite)")

    save_path = cfg.weights_dir / f"classifier_{name}.pt"
    torch.save(clf.state_dict(), save_path)
    print(f"  saved classifier to {save_path}")
    return clf


def train_one_bin(df_bin: pd.DataFrame, bin_idx, species: list[str], gas,
                   residual: bool = False, use_huber: bool = False):
    print(f"\n=== training bin {bin_idx}  (n_rows={len(df_bin)}, "
          f"n_traj={df_bin['traj_idx'].nunique()}, residual={residual}, huber={use_huber}) ===")

    data_loss_fn = loss_data_huber if use_huber else loss_data

    data = build_pairs(df_bin, species)
    train_mask, val_mask = trajectory_split(data["traj_id"], cfg.train.val_fraction, cfg.train.seed)

    T_tr, Y_tr = to_tensor(data["T"][train_mask]), to_tensor(data["Y"][train_mask])
    T0_tr, P_tr = to_tensor(data["T0"][train_mask]), to_tensor(data["P"][train_mask])
    dt_tr, tend_tr = to_tensor(data["dt"][train_mask]), to_tensor(data["t_end"][train_mask])
    rho_tr = to_tensor(data["rho"][train_mask])
    T_next_tr, Y_next_tr = to_tensor(data["T_next"][train_mask]), to_tensor(data["Y_next"][train_mask])

    T_va, Y_va = to_tensor(data["T"][val_mask]), to_tensor(data["Y"][val_mask])
    T0_va, P_va = to_tensor(data["T0"][val_mask]), to_tensor(data["P"][val_mask])
    dt_va, tend_va = to_tensor(data["dt"][val_mask]), to_tensor(data["t_end"][val_mask])
    T_next_va, Y_next_va = to_tensor(data["T_next"][val_mask]), to_tensor(data["Y_next"][val_mask])

    model = CRKPINN(n_species=len(species), hidden=cfg.model.hidden_dim,
                     n_hidden_layers=cfg.model.num_layers, residual=residual).to(device)

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

    mw = np.array([gas.molecular_weights[gas.species_index(s)] for s in species])
    physics = build_physics_utils(gas, species, device)
    arrhenius_module = GlobalArrheniusRate(n_species=len(species), molecular_weights=mw).to(device)

    opt = torch.optim.Adam(list(model.parameters()) + list(arrhenius_module.parameters()),
                            lr=cfg.train.learning_rate)

    save_path = cfg.weights_dir / f"crkpinn_bin{bin_idx}.pt"
    best_val_loss = float("inf")

    n_train = T_tr.shape[0]
    for epoch in range(cfg.train.num_epochs):
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for i in range(0, n_train, cfg.train.batch_size):
            idx = perm[i:i + cfg.train.batch_size]

            T_next_pred, Y_next_pred, info = model(
                T_tr[idx], Y_tr[idx], T0_tr[idx], P_tr[idx], dt_tr[idx], tend_tr[idx]
            )

            L_data = data_loss_fn(model, info, T_next_tr[idx], Y_next_tr[idx])
            L_mole = loss_mole(Y_next_pred) if LAMBDA_MOLE > 0 else torch.tensor(0.0, device=device)
            L_element = loss_element(physics.element_mass_fractions, Y_tr[idx], Y_next_pred) if LAMBDA_ELEMENT > 0 else torch.tensor(0.0, device=device)
            L_enthalpy = loss_enthalpy(physics.mixture_specific_enthalpy, T_tr[idx], Y_tr[idx], T_next_pred, Y_next_pred) if LAMBDA_ENTHALPY > 0 else torch.tensor(0.0, device=device)
            L_arrhenius = loss_arrhenius(model, arrhenius_module, T_tr[idx], Y_tr[idx], T0_tr[idx], P_tr[idx], dt_tr[idx], tend_tr[idx], rho_tr[idx]) if LAMBDA_ARRHENIUS > 0 else torch.tensor(0.0, device=device)

            loss = (LAMBDA_DATA * L_data + LAMBDA_MOLE * L_mole
                    + LAMBDA_ELEMENT * L_element + LAMBDA_ENTHALPY * L_enthalpy
                    + LAMBDA_ARRHENIUS * L_arrhenius)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item() * len(idx)

        with torch.no_grad():
            _, _, info_v = model(T_va, Y_va, T0_va, P_va, dt_va, tend_va)
            val_loss = data_loss_fn(model, info_v, T_next_va, Y_next_va).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == cfg.train.num_epochs - 1:
            print(f"  bin {bin_idx}  epoch {epoch:4d}  train {total_loss/n_train:.4f}  "
                  f"val {val_loss:.4f}  best {best_val_loss:.4f}")

    print(f"  bin {bin_idx}: best val_loss={best_val_loss:.4f}, saved to {save_path}")
    return model

from fordypning.dataset import get_species_names, build_pairs, build_sequences, trajectory_split
from fordypning.losses import loss_data, loss_data_huber, loss_kstep, loss_mole, loss_element, loss_enthalpy, loss_arrhenius


def train_one_bin_kstep(df_bin: pd.DataFrame, bin_idx, species: list[str], gas,
                         residual: bool = False, use_huber: bool = False, k_steps: int = 20):
    print(f"\n=== training bin {bin_idx} (K-STEP, K={k_steps})  (n_rows={len(df_bin)}, "
          f"n_traj={df_bin['traj_idx'].nunique()}, residual={residual}, huber={use_huber}) ===")

    seqs = build_sequences(df_bin, species, k_steps)
    train_mask, val_mask = trajectory_split(seqs["traj_id"], cfg.train.val_fraction, cfg.train.seed)

    T_seq_tr = to_tensor(seqs["T_seq"][train_mask])
    Y_seq_tr = to_tensor(seqs["Y_seq"][train_mask])
    dt_seq_tr = to_tensor(seqs["dt_seq"][train_mask])
    T0_tr, P_tr, tend_tr = to_tensor(seqs["T0"][train_mask]), to_tensor(seqs["P"][train_mask]), to_tensor(seqs["t_end"][train_mask])

    T_seq_va = to_tensor(seqs["T_seq"][val_mask])
    Y_seq_va = to_tensor(seqs["Y_seq"][val_mask])
    dt_seq_va = to_tensor(seqs["dt_seq"][val_mask])
    T0_va, P_va, tend_va = to_tensor(seqs["T0"][val_mask]), to_tensor(seqs["P"][val_mask]), to_tensor(seqs["t_end"][val_mask])

    model = CRKPINN(n_species=len(species), hidden=cfg.model.hidden_dim,
                     n_hidden_layers=cfg.model.num_layers, residual=residual).to(device)

    Y_MIN = model.Y_MIN
    Y_flat_tr = Y_seq_tr.reshape(-1, len(species))
    T_flat_tr = T_seq_tr.reshape(-1)
    logY_all = torch.log(torch.clamp(Y_flat_tr, min=Y_MIN))
    logP_all = torch.log(torch.clamp(P_tr, min=1e-6))
    tprime_raw = dt_seq_tr.reshape(-1) / torch.clamp(tend_tr.repeat_interleave(k_steps), min=1e-12)

    model.set_norm_stats(
        logY_mean=logY_all.mean(dim=0), logY_std=logY_all.std(dim=0),
        T_mean=T_flat_tr.mean(), T_std=T_flat_tr.std(),
        T0_mean=T0_tr.mean(), T0_std=T0_tr.std(),
        logP_mean=logP_all.mean(), logP_std=logP_all.std(),
        tprime_mean=tprime_raw.mean(), tprime_std=tprime_raw.std(),
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    save_path = cfg.weights_dir / f"crkpinn_bin{bin_idx}.pt"
    best_val_loss = float("inf")

    n_train = T_seq_tr.shape[0]
    for epoch in range(cfg.train.num_epochs):
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for i in range(0, n_train, cfg.train.batch_size):
            idx = perm[i:i + cfg.train.batch_size]

            loss = loss_kstep(model, T_seq_tr[idx], Y_seq_tr[idx], T0_tr[idx], P_tr[idx],
                               dt_seq_tr[idx], tend_tr[idx], k_steps, use_huber=use_huber)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item() * len(idx)

        with torch.no_grad():
            val_loss = loss_kstep(model, T_seq_va, Y_seq_va, T0_va, P_va,
                                   dt_seq_va, tend_va, k_steps, use_huber=use_huber).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == cfg.train.num_epochs - 1:
            print(f"  bin {bin_idx}  epoch {epoch:4d}  train {total_loss/n_train:.4f}  "
                  f"val {val_loss:.4f}  best {best_val_loss:.4f}")

    print(f"  bin {bin_idx}: best val_loss={best_val_loss:.4f}, saved to {save_path}")
    return model

def train_bin1_split(df_bin1: pd.DataFrame, species: list[str], gas):
    print(f"\n=== bin 1: splitting by ignition ===")
    train_ignition_classifier(df_bin1, "bin1", device)

    ign_traj = df_bin1.groupby("traj_idx")["ignited"].first()
    ign_ids = ign_traj[ign_traj].index
    noign_ids = ign_traj[~ign_traj].index

    df_ign = df_bin1[df_bin1["traj_idx"].isin(ign_ids)]
    df_noign = df_bin1[df_bin1["traj_idx"].isin(noign_ids)]

    print(f"  bin 1 split: {len(ign_ids)} ignite traj, {len(noign_ids)} no-ignite traj")

    if len(ign_ids) > 0:
        train_one_bin_kstep(df_ign, "1_ignite", species, gas, residual=False, use_huber=False, k_steps=20)
    if len(noign_ids) > 0:
        train_one_bin_kstep(df_noign, "1_noignite", species, gas, residual=True, use_huber=True, k_steps=20)

def main():
    df = pd.read_csv(CSV_PATH)
    species = get_species_names()
    gas = ct.Solution("gri30.yaml")

    bins_to_run = [int(b) for b in sys.argv[1:]] if len(sys.argv) > 1 else list(range(N_BINS))
    print(f"training bins: {bins_to_run}")

    for bin_idx in bins_to_run:
        df_bin = df[df["bin_idx"] == bin_idx]
        if len(df_bin) == 0:
            print(f"bin {bin_idx}: no data, skipping")
            continue

        if bin_idx == 1:
            train_bin1_split(df_bin, species, gas)
        elif bin_idx == 0:
            train_one_bin_kstep(df_bin, 0, species, gas, residual=True, use_huber=True, k_steps=20)
        else:
            train_one_bin(df_bin, bin_idx, species, gas)


if __name__ == "__main__":
    main()