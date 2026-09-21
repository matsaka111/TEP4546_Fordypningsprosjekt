# compare_bins.py
import pandas as pd
import torch

from fordypning.config import Config
from fordypning.dataset import get_species_names
from fordypning.models import CRKPINN
from fordypning.classifier import IgnitionClassifier
from fordypning.diagnostics import evaluate_bin_normalized, summarize_normalized_errors, compare_bins_summary

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
species = get_species_names()
n_species = len(species)

df = pd.read_csv("data/raw/gri30_ch4_air_trajectories_cbins.csv")


def load_crkpinn(path, residual=False):
    model = CRKPINN(n_species=n_species, hidden=cfg.model.hidden_dim,
                     n_hidden_layers=cfg.model.num_layers, residual=residual).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


bin_errors = {}

for bin_idx in [0, 2, 3, 4]:
    df_bin = df[df["bin_idx"] == bin_idx]
    if len(df_bin) == 0:
        continue
    model = load_crkpinn(cfg.weights_dir / f"crkpinn_bin{bin_idx}.pt", residual=False)
    err = evaluate_bin_normalized(model, df_bin, species, device)
    bin_errors[f"bin{bin_idx}"] = err

    detail = summarize_normalized_errors(err, species)
    print(f"\n--- bin {bin_idx}: worst 10 by mean|norm err| ---")
    print(detail.head(10).to_string(index=False))

df_bin1 = df[df["bin_idx"] == 1]
ign_traj = df_bin1.groupby("traj_idx")["ignited"].first()
df_ign = df_bin1[df_bin1["traj_idx"].isin(ign_traj[ign_traj].index)]
df_noign = df_bin1[df_bin1["traj_idx"].isin(ign_traj[~ign_traj].index)]

model_ign = load_crkpinn(cfg.weights_dir / "crkpinn_bin1_ignite.pt", residual=False)
model_noign = load_crkpinn(cfg.weights_dir / "crkpinn_bin1_noignite.pt", residual=True)

err_ign = evaluate_bin_normalized(model_ign, df_ign, species, device)
err_noign = evaluate_bin_normalized(model_noign, df_noign, species, device)
bin_errors["bin1_ignite"] = err_ign
bin_errors["bin1_noignite"] = err_noign

print(f"\n--- bin1_ignite: worst 10 ---")
print(summarize_normalized_errors(err_ign, species).head(10).to_string(index=False))
print(f"\n--- bin1_noignite: worst 10 ---")
print(summarize_normalized_errors(err_noign, species).head(10).to_string(index=False))

print(f"\n=== cross-bin summary (all outputs pooled per bin) ===")
summary = compare_bins_summary(bin_errors)
print(summary.to_string(index=False))