T0_LO = 500.0
T0_HI = 2000.0
N_BINS = 5
BIN_WIDTH = (T0_HI - T0_LO) / N_BINS  # 300

def bin_index(T0: float) -> int:
    idx = int((T0 - T0_LO) // BIN_WIDTH)
    return max(0, min(idx, N_BINS - 1))

def model_path(T0: float, weights_dir) -> str:
    idx = bin_index(T0)
    return weights_dir / f"crkpinn_bin{idx}.pt"

import torch
from fordypning.models import CRKPINN

_model_cache = {}

def load_model(bin_idx: int, n_species: int, hidden: int, n_layers: int,
                weights_dir, device: str = "cpu") -> CRKPINN:
    if bin_idx in _model_cache:
        return _model_cache[bin_idx]
    model = CRKPINN(n_species=n_species, hidden=hidden, n_hidden_layers=n_layers).to(device)
    path = weights_dir / f"crkpinn_bin{bin_idx}.pt"
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    _model_cache[bin_idx] = model
    return model

def predict(T0: float, n_species: int, hidden: int, n_layers: int,
            weights_dir, device: str = "cpu") -> CRKPINN:
    idx = bin_index(T0)
    return load_model(idx, n_species, hidden, n_layers, weights_dir, device)