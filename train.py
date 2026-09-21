# train.py
import numpy as np
import torch
import cantera as ct

from fordypning.config import Config
from fordypning.dataset import load_raw_csv, get_species_names, build_pairs, trajectory_split
from fordypning.models import CRKPINN
from fordypning.chemistry import GlobalArrheniusRate, build_physics_utils
from fordypning.losses import loss_data, loss_mole, loss_element, loss_enthalpy, loss_arrhenius

import sys
run_name = sys.argv[1] if len(sys.argv) > 1 else "default"

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using devide: {device}")

# --- data ---
df = load_raw_csv("data/raw/gri30_ch4_air_trajectories.csv")
species = get_species_names()
data = build_pairs(df, species)

train_mask, val_mask = trajectory_split(data["traj_id"], cfg.train.val_fraction, cfg.train.seed)

def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=device)

T_tr, Y_tr = to_tensor(data["T"][train_mask]), to_tensor(data["Y"][train_mask])
T0_tr, P_tr = to_tensor(data["T0"][train_mask]), to_tensor(data["P"][train_mask])
dt_tr, tend_tr = to_tensor(data["dt"][train_mask]), to_tensor(data["t_end"][train_mask])
rho_tr = to_tensor(data["rho"][train_mask])
T_next_tr, Y_next_tr = to_tensor(data["T_next"][train_mask]), to_tensor(data["Y_next"][train_mask])

T_va, Y_va = to_tensor(data["T"][val_mask]), to_tensor(data["Y"][val_mask])
T0_va, P_va = to_tensor(data["T0"][val_mask]), to_tensor(data["P"][val_mask])
dt_va, tend_va = to_tensor(data["dt"][val_mask]), to_tensor(data["t_end"][val_mask])
rho_va = to_tensor(data["rho"][val_mask])
T_next_va, Y_next_va = to_tensor(data["T_next"][val_mask]), to_tensor(data["Y_next"][val_mask])

# --- model + norm stats (fit on TRAIN split only) ---
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

# --- physics utils + auxiliary Arrhenius module ---
gas = ct.Solution("gri30.yaml")
mw = np.array([gas.molecular_weights[gas.species_index(s)] for s in species])
physics = build_physics_utils(gas, species, device)
arrhenius_module = GlobalArrheniusRate(n_species=len(species), molecular_weights=mw).to(device)

opt = torch.optim.Adam(list(model.parameters()) + list(arrhenius_module.parameters()),
                        lr=cfg.train.learning_rate)

# --- loss weights -- start conservative, raise gradually once stable ---
LAMBDA_DATA = 1.0
LAMBDA_MOLE = 0.1
LAMBDA_ELEMENT = 0.0004
LAMBDA_ENTHALPY = 0.0
LAMBDA_ARRHENIUS = 0.00001

n_train = T_tr.shape[0]
train_losses, val_losses, val_epochs = [], [], []

for epoch in range(cfg.train.num_epochs):
    perm = torch.randperm(n_train, device=device)
    total_loss = 0.0
    for i in range(0, n_train, cfg.train.batch_size):
        idx = perm[i:i + cfg.train.batch_size]

        T_next_pred, Y_next_pred, info = model(
            T_tr[idx], Y_tr[idx], T0_tr[idx], P_tr[idx], dt_tr[idx], tend_tr[idx]
        )

        L_data = loss_data(model, info, T_next_tr[idx], Y_next_tr[idx])
        L_mole = loss_mole(Y_next_pred) if LAMBDA_MOLE > 0 else torch.tensor(0.0, device=device)
        L_element = loss_element(physics.element_mass_fractions, Y_tr[idx], Y_next_pred) if LAMBDA_ELEMENT > 0 else torch.tensor(0.0, device=device)
        L_enthalpy = loss_enthalpy(physics.mixture_specific_enthalpy, T_tr[idx], Y_tr[idx], T_next_pred, Y_next_pred) if LAMBDA_ENTHALPY > 0 else torch.tensor(0.0, device=device)
        L_arrhenius = loss_arrhenius(model, arrhenius_module, T_tr[idx], Y_tr[idx], T0_tr[idx], P_tr[idx], dt_tr[idx], tend_tr[idx], rho_tr[idx]) if LAMBDA_ARRHENIUS > 0 else torch.tensor(0.0, device=device)

        loss = (LAMBDA_DATA * L_data
                + LAMBDA_MOLE * L_mole
                + LAMBDA_ELEMENT * L_element
                + LAMBDA_ARRHENIUS * L_arrhenius)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total_loss += loss.item() * len(idx)

    train_losses.append(total_loss / n_train)

    if epoch % 10 == 0:
        with torch.no_grad():
            T_next_pred_v, Y_next_pred_v, info_v = model(T_va, Y_va, T0_va, P_va, dt_va, tend_va)
            val_loss = loss_data(model, info_v, T_next_va, Y_next_va)
        val_losses.append(val_loss.item())
        val_epochs.append(len(train_losses) - 1)
        print(f"epoch {len(train_losses)-1:4d}  train {total_loss/n_train:.4f}  val(data only) {val_loss.item():.4f}")

# --- save ---
cfg.weights_dir.mkdir(exist_ok=True)
torch.save(model.state_dict(), cfg.weights_dir / "crkpinn.pt")
print(f"saved to {cfg.weights_dir / 'crkpinn.pt'}")

# train.py
import numpy as np
import torch
import cantera as ct

from fordypning.config import Config
from fordypning.dataset import load_raw_csv, get_species_names, build_pairs, trajectory_split
from fordypning.models import CRKPINN
from fordypning.chemistry import GlobalArrheniusRate, build_physics_utils
from fordypning.losses import loss_data, loss_mole, loss_element, loss_enthalpy, loss_arrhenius

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- data ---
df = load_raw_csv("data/raw/gri30_ch4_air_trajectories.csv")
species = get_species_names()
data = build_pairs(df, species)

train_mask, val_mask = trajectory_split(data["traj_id"], cfg.train.val_fraction, cfg.train.seed)

def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=device)

T_tr, Y_tr = to_tensor(data["T"][train_mask]), to_tensor(data["Y"][train_mask])
T0_tr, P_tr = to_tensor(data["T0"][train_mask]), to_tensor(data["P"][train_mask])
dt_tr, tend_tr = to_tensor(data["dt"][train_mask]), to_tensor(data["t_end"][train_mask])
rho_tr = to_tensor(data["rho"][train_mask])
T_next_tr, Y_next_tr = to_tensor(data["T_next"][train_mask]), to_tensor(data["Y_next"][train_mask])

T_va, Y_va = to_tensor(data["T"][val_mask]), to_tensor(data["Y"][val_mask])
T0_va, P_va = to_tensor(data["T0"][val_mask]), to_tensor(data["P"][val_mask])
dt_va, tend_va = to_tensor(data["dt"][val_mask]), to_tensor(data["t_end"][val_mask])
rho_va = to_tensor(data["rho"][val_mask])
T_next_va, Y_next_va = to_tensor(data["T_next"][val_mask]), to_tensor(data["Y_next"][val_mask])

# --- model + norm stats (fit on TRAIN split only) ---
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

# --- physics utils + auxiliary Arrhenius module ---
gas = ct.Solution("gri30.yaml")
mw = np.array([gas.molecular_weights[gas.species_index(s)] for s in species])
physics = build_physics_utils(gas, species, device)
arrhenius_module = GlobalArrheniusRate(n_species=len(species), molecular_weights=mw).to(device)

opt = torch.optim.Adam(list(model.parameters()) + list(arrhenius_module.parameters()),
                        lr=cfg.train.learning_rate)

# --- loss weights -- start conservative, raise gradually once stable ---
LAMBDA_DATA = 1.0
LAMBDA_MOLE = 0.1
LAMBDA_ELEMENT = 0.0
LAMBDA_ENTHALPY = 0.0
LAMBDA_ARRHENIUS = 0.0

n_train = T_tr.shape[0]
train_losses, val_losses, val_epochs = [], [], []

for epoch in range(cfg.train.num_epochs):
    perm = torch.randperm(n_train, device=device)
    total_loss = 0.0
    for i in range(0, n_train, cfg.train.batch_size):
        idx = perm[i:i + cfg.train.batch_size]

        T_next_pred, Y_next_pred, info = model(
            T_tr[idx], Y_tr[idx], T0_tr[idx], P_tr[idx], dt_tr[idx], tend_tr[idx]
        )

        L_data = loss_data(model, info, T_next_tr[idx], Y_next_tr[idx])
        L_mole = loss_mole(Y_next_pred) if LAMBDA_MOLE > 0 else torch.tensor(0.0, device=device)
        L_element = loss_element(physics.element_mass_fractions, Y_tr[idx], Y_next_pred) if LAMBDA_ELEMENT > 0 else torch.tensor(0.0, device=device)
        L_enthalpy = loss_enthalpy(physics.mixture_specific_enthalpy, T_tr[idx], Y_tr[idx], T_next_pred, Y_next_pred) if LAMBDA_ENTHALPY > 0 else torch.tensor(0.0, device=device)
        L_arrhenius = loss_arrhenius(model, arrhenius_module, T_tr[idx], Y_tr[idx], T0_tr[idx], P_tr[idx], dt_tr[idx], tend_tr[idx], rho_tr[idx]) if LAMBDA_ARRHENIUS > 0 else torch.tensor(0.0, device=device)

        loss = (LAMBDA_DATA * L_data
                + LAMBDA_MOLE * L_mole
                + LAMBDA_ELEMENT * L_element
                + LAMBDA_ARRHENIUS * L_arrhenius)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total_loss += loss.item() * len(idx)

    train_losses.append(total_loss / n_train)

    if epoch % 1 == 10:
        with torch.no_grad():
            T_next_pred_v, Y_next_pred_v, info_v = model(T_va, Y_va, T0_va, P_va, dt_va, tend_va)
            val_loss = loss_data(model, info_v, T_next_va, Y_next_va)
        val_losses.append(val_loss.item())
        val_epochs.append(len(train_losses) - 1)
        print(f"epoch {len(train_losses)-1:4d}  train {total_loss/n_train:.4f}  val(data only) {val_loss.item():.4f}")


# --- trajectory rollouts across T0 ranges ---
from fordypning.diagnostics import select_trajectory_by_T0_range, autoregressive_rollout, plot_trajectory_rollout

PLOT_SPECIES = ["CH4", "O2", "CO2", "H2O", "CO", "OH"]  # adjust as you like
T0_RANGES = [(500, 900), (900, 1300), (1300, 1600)]

for lo, hi in T0_RANGES:
    traj_idx = select_trajectory_by_T0_range(df, lo, hi, seed=cfg.train.seed)
    if traj_idx is None:
        print(f"no trajectory found with T0 in [{lo}, {hi}) -- your data only covers "
              f"T0_RANGE=(1000, 1600) per generate_data.py, so [500,900) will be empty")
        continue
    traj_df = df[df["traj_idx"] == traj_idx]
    rollout = autoregressive_rollout(model, traj_df, species, n_tend=5, device=device)
    plot_trajectory_rollout(rollout, species, PLOT_SPECIES,
                             title=f"T0 in [{lo},{hi}), traj_idx={traj_idx}",
                             save_path=f"model_weights/rollout_T0_{lo}_{hi}.png")


# --- save ---
cfg.weights_dir.mkdir(exist_ok=True)
torch.save(model.state_dict(), cfg.weights_dir / f"crkpinn_{run_name}.pt")
print(f"saved to {cfg.weights_dir / 'crkpinn.pt'}")