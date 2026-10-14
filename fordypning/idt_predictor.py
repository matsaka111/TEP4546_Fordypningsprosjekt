# fordypning/idt_predictor.py
"""
Ignition delay time (IDT) predictor: (T0, phi[, P]) -> ln(IDT), no simulation.

Used to get t_end = n * IDT for time normalization at deployment, when no
Cantera-scouted t_ign exists for a new initial condition. Follows the idea of
CRK-PINN Appendix B (Zhang, Zhang & Wang 2024): small MLP on ln(IDT).

Choices:
- Inputs 1000/T0 (ln IDT is ~linear in 1/T0, Arrhenius) and ln(phi)
  (phi spans 0.5-8). ln(P) only if P actually varies -- with P_FIXED its
  std is 0, which is the same trap that broke CRKPINN normalization.
- Target ln(IDT), z-scored. Plain MSE in normalized log space. The
  CRK-PINN relative-error term is optional (`rel_weight`): with raw outputs
  it explodes early in training (exp of an untrained log prediction).
- Regression uses only igniting trajectories. Non-igniting ones are
  censored (IDT > SIM_TIME) and belong to the ignition classifier.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PER_TRAJ_COLS = ["traj_idx", "T0", "phi", "P", "t_ign", "ignited", "reached_end"]


# ---------------------------------------------------------------- data
def extract_idt_table(csv_path: str, chunksize: int = 2_000_000) -> pd.DataFrame:
    """One row per trajectory. Reads only the needed columns, in chunks,
    so the many-GB trajectory CSV never has to fit in memory."""
    header = pd.read_csv(csv_path, nrows=0).columns
    cols = [c for c in PER_TRAJ_COLS if c in header]
    parts = [chunk.drop_duplicates("traj_idx")
             for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=chunksize)]
    df = pd.concat(parts).drop_duplicates("traj_idx").reset_index(drop=True)
    df["ignited"] = df["ignited"].astype(bool)
    if "reached_end" in df:
        df["reached_end"] = df["reached_end"].astype(bool)
    return df


def igniting_only(df: pd.DataFrame) -> pd.DataFrame:
    m = df["ignited"] & df["t_ign"].notna() & (df["t_ign"] > 0)
    return df[m].reset_index(drop=True)


def make_features(T0, phi, P=None, use_P=False):
    cols = [1000.0 / np.asarray(T0, dtype=np.float64),
            np.log(np.asarray(phi, dtype=np.float64))]
    if use_P:
        cols.append(np.log(np.asarray(P, dtype=np.float64)))
    return np.stack(cols, axis=-1).astype(np.float32)


# ---------------------------------------------------------------- model
class IDTPredictor(nn.Module):
    """MLP: normalized features -> normalized ln(IDT).
    Norm stats are buffers, so they are saved with state_dict."""

    def __init__(self, n_in=2, hidden=64, n_layers=3, use_P=False):
        super().__init__()
        self.use_P = use_P
        layers, d = [], n_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

        self.register_buffer("x_mean", torch.zeros(n_in))
        self.register_buffer("x_std", torch.ones(n_in))
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def set_norm_stats(self, X, lnidt):
        X, lnidt = torch.as_tensor(X), torch.as_tensor(lnidt)
        self.x_mean.copy_(X.mean(0))
        self.x_std.copy_(X.std(0).clamp_min(1e-3))
        self.y_mean.copy_(lnidt.mean().reshape(1))
        self.y_std.copy_(lnidt.std().clamp_min(1e-3).reshape(1))

    def forward(self, X):
        """X: features from make_features. Returns normalized ln(IDT)."""
        return self.net((X - self.x_mean) / self.x_std).squeeze(-1)

    def ln_idt(self, X):
        return self.forward(X) * self.y_std + self.y_mean

    @torch.no_grad()
    def predict_idt(self, T0, phi, P=None):
        """Convenience for deployment: T0, phi (, P) as arrays/tensors -> IDT [s]."""
        to_np = lambda a: a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
        X = make_features(to_np(T0), to_np(phi), None if P is None else to_np(P), self.use_P)
        X = torch.tensor(X, device=self.x_mean.device)
        return torch.exp(self.ln_idt(X))


def load_idt_predictor(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = IDTPredictor(ck["n_in"], use_P=ck["use_P"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval()


def loss_idt(model, X, lnidt_true, rel_weight=0.0):
    """MSE on normalized ln(IDT) (+ optional CRK-PINN Eq. B.2 relative-error
    term, clamped so it can't blow up while the model is still poor)."""
    pred_n = model(X)
    true_n = (lnidt_true - model.y_mean) / model.y_std
    loss = ((pred_n - true_n) ** 2).mean()
    if rel_weight > 0:
        d = (model.ln_idt(X) - lnidt_true).clamp(-3, 3)   # |ln ratio| <= 3
        loss = loss + rel_weight * ((torch.exp(d) - 1) ** 2).mean()
    return loss


# ---------------------------------------------------------------- baseline
def fit_linear_baseline(X, lnidt):
    """Least squares: ln IDT = a + b*(1000/T0) + c*ln(phi) + d*ln(phi)^2 (+ e*lnP)."""
    design = lambda Z: np.column_stack([np.ones(len(Z)), Z, Z[:, 1] ** 2])
    coef, *_ = np.linalg.lstsq(design(X), lnidt, rcond=None)
    return lambda Xq: design(Xq) @ coef


# ---------------------------------------------------------------- train / eval
def rel_err(idt_hat, idt):
    return np.abs(idt_hat - idt) / idt


def train_idt_predictor(X_tr, y_tr, X_va, y_va, use_P=False, device="cpu",
                        n_epochs=300, batch_size=1024, lr=1e-3,
                        hidden=64, n_layers=3, rel_weight=0.0):
    """y_* are ln(IDT). Returns the best-on-validation model."""
    model = IDTPredictor(X_tr.shape[1], hidden, n_layers, use_P).to(device)
    model.set_norm_stats(X_tr, y_tr)

    Xt, yt = torch.tensor(X_tr, device=device), torch.tensor(y_tr, device=device)
    Xv, yv = torch.tensor(X_va, device=device), torch.tensor(y_va, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    best, best_state = float("inf"), None

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), batch_size):
            b = perm[i:i + batch_size]
            loss = loss_idt(model, Xt[b], yt[b], rel_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val = loss_idt(model, Xv, yv, rel_weight).item()
        if val < best:
            best = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:4d}  train {loss.item():.2e}  val {val:.2e}")

    model.load_state_dict(best_state)
    return model


def error_table(T0, idt, idt_hat, bin_edges):
    """Median / 95th-percentile relative error, overall and per T0 bin."""
    e = rel_err(idt_hat, idt)
    rows = [("all", len(e), np.median(e), np.percentile(e, 95))]
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (T0 >= lo) & (T0 < hi)
        if m.sum():
            rows.append((f"[{lo:.0f},{hi:.0f})", int(m.sum()),
                         np.median(e[m]), np.percentile(e[m], 95)))
    return pd.DataFrame(rows, columns=["T0 range", "n", "median rel err", "p95 rel err"])