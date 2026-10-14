# diagnose_cbins.py
import pandas as pd
import torch

from fordypning.config import Config
from fordypning.dataset import get_species_names
from fordypning.models import CRKPINN
from fordypning.classifier import IgnitionClassifier
from fordypning.diagnostics import (
    autoregressive_rollout, plot_trajectory_rollout,
    compute_metrics, print_metrics,
)

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
species = get_species_names()
n_species = len(species)

df = pd.read_csv("data/raw/gri30_ch4_air_trajectories_cbins.csv")
PLOT_SPECIES = ["CH4", "O2", "CO2", "H2O", "CO", "OH"]
N_TEND = 5


def load_crkpinn(path, residual=False):
    model = CRKPINN(n_species=n_species, hidden=cfg.model.hidden_dim,
                     n_hidden_layers=cfg.model.num_layers, residual=residual).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_classifier(path):
    clf = IgnitionClassifier().to(device)
    clf.load_state_dict(torch.load(path, map_location=device))
    clf.eval()
    return clf


def diagnose_one(model, traj_df, tag, save_prefix):
    rollout = autoregressive_rollout(model, traj_df, species, n_tend=N_TEND, device=device)

    plot_trajectory_rollout(rollout, species, PLOT_SPECIES,
                             title=tag,
                             save_path=f"model_weights/{save_prefix}.png")

    labels = ["T"] + species
    y_true = pd.concat([pd.DataFrame({"T": rollout["T_true"]}),
                         pd.DataFrame(rollout["Y_true"], columns=species)], axis=1).to_numpy()
    y_pred = pd.concat([pd.DataFrame({"T": rollout["T_pred"]}),
                         pd.DataFrame(rollout["Y_pred"], columns=species)], axis=1).to_numpy()

    metrics = compute_metrics(y_true, y_pred, labels)
    print(f"\n--- {tag} error table ---")
    print_metrics(metrics, top_n=15)


# --- bins 0, 2, 3, 4: single model per bin, as before ---
for bin_idx in [0, 2, 3, 4]:
    df_bin = df[df["bin_idx"] == bin_idx]
    if len(df_bin) == 0:
        print(f"bin {bin_idx}: no data")
        continue

    traj_idx = df_bin["traj_idx"].unique()[0]
    traj_df = df_bin[df_bin["traj_idx"] == traj_idx]

    model = load_crkpinn(cfg.weights_dir / f"crkpinn_bin{bin_idx}.pt", residual=False)
    lo, hi = 500 + bin_idx * 300, 500 + (bin_idx + 1) * 300
    diagnose_one(model, traj_df, f"bin {bin_idx}  T0 in [{lo},{hi})  traj_idx={traj_idx}",
                 f"rollout_bin{bin_idx}")


# --- bin 1: classifier routes to ignite/no-ignite sub-model ---
df_bin1 = df[df["bin_idx"] == 1]
clf = load_classifier(cfg.weights_dir / "classifier_bin1.pt")
model_ignite = load_crkpinn(cfg.weights_dir / "crkpinn_bin1_ignite.pt", residual=False)
model_noignite = load_crkpinn(cfg.weights_dir / "crkpinn_bin1_noignite.pt", residual=True)

for want_ignite, label in [(True, "ignite"), (False, "noignite")]:
    per_traj = df_bin1.groupby("traj_idx")["ignited"].first()
    candidates = per_traj[per_traj == want_ignite].index
    if len(candidates) == 0:
        print(f"bin 1 ({label}): no trajectories")
        continue
    traj_idx = candidates[0]
    traj_df = df_bin1[df_bin1["traj_idx"] == traj_idx]

    # route via classifier, using this trajectory's own (T0, phi, P)
    T0 = torch.tensor([traj_df["T0"].iloc[0]], dtype=torch.float32, device=device)
    phi = torch.tensor([traj_df["phi"].iloc[0]], dtype=torch.float32, device=device)
    P = torch.tensor([traj_df["P"].iloc[0]], dtype=torch.float32, device=device)
    pred_ignite, prob = clf.predict(T0, phi, P)
    pred_ignite = bool(pred_ignite.item())

    model = model_ignite if pred_ignite else model_noignite
    print(f"\nbin 1 traj_idx={traj_idx}  true ignited={want_ignite}  "
          f"classifier predicted={pred_ignite} (p={prob.item():.3f})")

    diagnose_one(model, traj_df, f"bin 1 ({label})  traj_idx={traj_idx}",
                 f"rollout_bin1_{label}")