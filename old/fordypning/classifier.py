import torch
import torch.nn as nn


class IgnitionClassifier(nn.Module):
    """Predicts whether a trajectory will ignite, from (T0, phi, P) alone.
    Used only for bin 1, whose T0 range contains a genuine mix of igniting
    and non-igniting trajectories -- a single CRKPINN struggles to fit both
    regimes at once, so this routes to the correct sub-model instead."""

    def __init__(self, hidden=32, n_hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(3, hidden), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

        self.register_buffer("mean", torch.zeros(3))
        self.register_buffer("std", torch.ones(3))

    def set_norm_stats(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6))

    def forward(self, T0, phi, P):
        x = torch.stack([T0, phi, P], dim=-1)
        x = (x - self.mean) / self.std
        return self.net(x).squeeze(-1)  # logits, use BCEWithLogitsLoss

    def predict(self, T0, phi, P, threshold=0.5):
        with torch.no_grad():
            prob = torch.sigmoid(self.forward(T0, phi, P))
        return (prob > threshold), prob


def train_ignition_classifier(df_bin, save_path, device, n_epochs=200):
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

    torch.save(clf.state_dict(), save_path)
    return clf