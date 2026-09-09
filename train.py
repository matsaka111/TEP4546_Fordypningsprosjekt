"""
train.py

Trains the surrogate model h_theta(y) ~= f(y) on the dataset produced by
data/generate_data.py.
"""

import numpy as np
import torch
import torch.nn as nn
from models.PhysicsNN import PhysicsNN


def load_dataset(path="data/raw/training_data.npz"):
    data = np.load(path)
    return data["Y"], data["Ydot"], data["traj_id"], data["t"]


def split_by_trajectory(Y, Ydot, traj_id, val_fraction=0.2, seed=0):
    """
    Hold out entire trajectories for validation, not individual points.
    """
    rng = np.random.default_rng(seed)
    unique_trajs = np.unique(traj_id)
    rng.shuffle(unique_trajs)

    n_val = int(len(unique_trajs) * val_fraction)
    val_trajs = set(unique_trajs[:n_val])

    is_val = np.array([t in val_trajs for t in traj_id])

    return (
        Y[~is_val], Ydot[~is_val],
        Y[is_val], Ydot[is_val],
    )


def log_transform(Y):
    """Log-transform species mass fractions, leave temperature untouched."""
    Y_species = np.log(Y[:, :-1] + 1e-12)
    T = Y[:, -1:]
    return np.concatenate([Y_species, T], axis=1)


def train(Y_train, Ydot_train, Y_val, Ydot_val, epochs=100, lr=1e-3, batch_size=256):
    Y_train_log = log_transform(Y_train)
    Y_val_log = log_transform(Y_val)

    Y_mean, Y_std = Y_train_log.mean(0), Y_train_log.std(0) + 1e-8
    Ydot_mean, Ydot_std = Ydot_train.mean(0), Ydot_train.std(0) + 1e-8

    def normalize(x, mean, std):
        return (x - mean) / std

    Y_train_t = torch.tensor(normalize(Y_train_log, Y_mean, Y_std), dtype=torch.float32)
    Ydot_train_t = torch.tensor(normalize(Ydot_train, Ydot_mean, Ydot_std), dtype=torch.float32)
    Y_val_t = torch.tensor(normalize(Y_val_log, Y_mean, Y_std), dtype=torch.float32)
    Ydot_val_t = torch.tensor(normalize(Ydot_val, Ydot_mean, Ydot_std), dtype=torch.float32)

    # PhysicsNN's Arrhenius part needs REAL, physical temperature -- not normalized,
    # not log-transformed -- separate from the usual normalized input
    T_train_t = torch.tensor(Y_train[:, -1:], dtype=torch.float32)
    T_val_t = torch.tensor(Y_val[:, -1:], dtype=torch.float32)

    model = PhysicsNN(dim=Y_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = len(Y_train_t)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(Y_train_t[idx], T_train_t[idx])
            loss = loss_fn(pred, Ydot_train_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                val_loss = loss_fn(model(Y_val_t, T_val_t), Ydot_val_t).item()
            print(f"Epoch {epoch+1}: train_loss={total_loss/n:.5f}, val_loss={val_loss:.5f}")

    return model, (Y_mean, Y_std, Ydot_mean, Ydot_std)


def run_training(Y, Ydot, traj_id, subsample_fraction=1.0, output_path="models/saved/simple_nn.pt"):
    Y_train, Ydot_train, Y_val, Ydot_val = split_by_trajectory(Y, Ydot, traj_id)

    if subsample_fraction < 1.0:
        rng = np.random.default_rng(0)
        n_sub = int(subsample_fraction * len(Y_train))
        idx = rng.choice(len(Y_train), size=n_sub, replace=False)
        Y_train = Y_train[idx]
        Ydot_train = Ydot_train[idx]

    print(f"Training on {len(Y_train)} points")

    model, norm_stats = train(Y_train, Ydot_train, Y_val, Ydot_val)

    torch.save({"model_state": model.state_dict(), "norm_stats": norm_stats}, output_path)
    print(f"Saved to {output_path}")

    return model, norm_stats

def resample_trajectory(y, t, t_grid):
    y_resampled = np.zeros((len(t_grid), y.shape[1]))
    for dim in range(y.shape[1]):
        y_resampled[:, dim] = np.interp(t_grid, t, y[:, dim])
    return y_resampled


def build_sequence_dataset(Y, traj_id, t_all, t_grid):
    y0_list, y_seq_list = [], []
    for tid in np.unique(traj_id):
        mask = traj_id == tid
        y, t = Y[mask], t_all[mask]
        if t.max() < t_grid[-1]:
            continue
        y_resampled = resample_trajectory(y, t, t_grid)
        y0_list.append(y_resampled[0])
        y_seq_list.append(y_resampled[1:])
    return np.array(y0_list), np.array(y_seq_list)


if __name__ == "__main__":
    Y, Ydot, traj_id, t = load_dataset()
    run_training(Y, Ydot, traj_id, subsample_fraction=0.1)
