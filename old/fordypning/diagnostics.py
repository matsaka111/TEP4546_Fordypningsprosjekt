# fordypning/diagnostics.py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ---------------- Error table (PPMCC + absolute error) ----------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    """
    Per-column mean absolute error and PPMCC (Pearson correlation), on
    physical (un-normalized) values. MAE instead of relative L2 -- many
    species are near-zero for most of a trajectory, so dividing by the
    true value blows up (see the AR/T collapse earlier); absolute error
    stays well-behaved regardless of scale.
    """
    metrics = {}
    for i, name in enumerate(labels):
        true_col, pred_col = y_true[:, i], y_pred[:, i]

        mae = np.mean(np.abs(pred_col - true_col))
        max_abs_err = np.max(np.abs(pred_col - true_col))

        if np.std(true_col) < 1e-30 or np.std(pred_col) < 1e-30:
            ppmcc = np.nan
        else:
            ppmcc = np.corrcoef(true_col, pred_col)[0, 1]

        metrics[name] = {"mae": mae, "max_abs_err": max_abs_err, "ppmcc": ppmcc}
    return metrics


def print_metrics(metrics: dict, top_n: int = None):
    """Print sorted by worst PPMCC first (weakest correlation = biggest concern)."""
    items = sorted(metrics.items(),
                    key=lambda kv: (np.isnan(kv[1]["ppmcc"]), -abs(kv[1]["ppmcc"] if not np.isnan(kv[1]["ppmcc"]) else 0)))
    if top_n:
        items = items[:top_n]
    print(f"{'name':<12} {'MAE':>12} {'max|err|':>12} {'PPMCC':>10}")
    for name, m in items:
        print(f"{name:<12} {m['mae']:>12.4e} {m['max_abs_err']:>12.4e} {m['ppmcc']:>10.4f}")


def metrics_to_dataframe(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(metrics).T.sort_values("ppmcc")


# ---------------- Scatter plot (true vs predicted, all species) ----------------

def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str],
                  save_path: str = None, n_cols: int = 6):
    n = len(labels)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for i, name in enumerate(labels):
        ax = axes[i]
        true_col, pred_col = y_true[:, i], y_pred[:, i]
        ax.scatter(true_col, pred_col, s=2, alpha=0.3)
        lims = [min(true_col.min(), pred_col.min()), max(true_col.max(), pred_col.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"saved scatter plot to {save_path}")
    return fig


# ---------------- Autoregressive trajectory rollout ----------------

def select_trajectory_by_T0_range(df: pd.DataFrame, lo: float, hi: float, seed: int = 0):
    """Pick one trajectory whose T0 falls in [lo, hi). Returns traj_idx or None."""
    candidates = df.loc[(df["T0"] >= lo) & (df["T0"] < hi), "traj_idx"].unique()
    if len(candidates) == 0:
        return None
    rng = np.random.default_rng(seed)
    return rng.choice(candidates)


def autoregressive_rollout(model, traj_df: pd.DataFrame, species: list[str],
                            n_tend: int = 5, device: str = "cpu"):
    """
    Roll the model forward along one trajectory's actual (non-uniform) time
    points, feeding each prediction back in as the next input -- CRK-PINN's
    "time-advanced" scenario, not time-continuous.
    """
    traj_df = traj_df.sort_values("t").reset_index(drop=True)
    y_cols = [f"Y_{s}" for s in species]

    t = traj_df["t"].to_numpy()
    T0 = float(traj_df["T0"].iloc[0])
    P = float(traj_df["P"].iloc[0])
    t_ign = float(traj_df["t_ign"].iloc[0])
    t_end = n_tend * t_ign

    T_true = traj_df["T"].to_numpy(dtype=np.float32)
    Y_true = traj_df[y_cols].to_numpy(dtype=np.float32)

    T_pred = np.zeros_like(T_true)
    Y_pred = np.zeros_like(Y_true)
    T_pred[0], Y_pred[0] = T_true[0], Y_true[0]

    T_cur = torch.tensor([T_true[0]], dtype=torch.float32, device=device)
    Y_cur = torch.tensor(Y_true[0:1], dtype=torch.float32, device=device)
    T0_t = torch.tensor([T0], dtype=torch.float32, device=device)
    P_t = torch.tensor([P], dtype=torch.float32, device=device)
    tend_t = torch.tensor([t_end], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        for i in range(1, len(t)):
            dt_t = torch.tensor([t[i] - t[i - 1]], dtype=torch.float32, device=device)
            T_next, Y_next, _ = model(T_cur, Y_cur, T0_t, P_t, dt_t, tend_t)
            T_pred[i] = T_next.item()
            Y_pred[i] = Y_next.cpu().numpy()[0]
            T_cur, Y_cur = T_next, Y_next

    return {"t": t, "T_true": T_true, "Y_true": Y_true, "T_pred": T_pred, "Y_pred": Y_pred}


def plot_trajectory_rollout(rollout: dict, species: list[str], plot_species: list[str],
                             title: str = "", save_path: str = None):
    """Panels: T, plus one per name in plot_species. True solid, predicted dashed."""
    t = rollout["t"]
    species_idx = {s: i for i, s in enumerate(species)}
    panels = ["T"] + plot_species
    n_cols = 4
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for i, name in enumerate(panels):
        ax = axes[i]
        if name == "T":
            true_col, pred_col = rollout["T_true"], rollout["T_pred"]
        else:
            j = species_idx[name]
            true_col, pred_col = rollout["Y_true"][:, j], rollout["Y_pred"][:, j]
        ax.plot(t, true_col, "-", label="true", linewidth=1.5)
        ax.plot(t, pred_col, "--", label="pred", linewidth=1.5)
        ax.set_title(name, fontsize=10)
        ax.set_xscale("log")
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"saved trajectory plot to {save_path}")
    return fig



def evaluate_bin_normalized(model, df_bin: pd.DataFrame, species: list[str],
                             device: str, n_traj_sample: int = 30, seed: int = 0):
    """
    Runs autoregressive rollout over a sample of trajectories in this bin,
    collects the NORMALIZED prediction error (i.e. in the model's own
    z-scored space) at every step of every trajectory, and returns summary
    statistics. Normalized error is used instead of physical-space percent
    error because physical percent error blows up whenever the true value
    is near zero (which is common for trace species) -- normalized error
    is scale-free and comparable across species and bins without that
    failure mode: an error of 1.0 means "one training-set standard
    deviation off", regardless of the species' physical magnitude.
    """
    rng = np.random.default_rng(seed)
    traj_ids = df_bin["traj_idx"].unique()
    if len(traj_ids) > n_traj_sample:
        traj_ids = rng.choice(traj_ids, size=n_traj_sample, replace=False)

    y_cols = [f"Y_{s}" for s in species]
    all_Yn_err, all_Tn_err = [], []

    model.eval()
    for traj_idx in traj_ids:
        traj_df = df_bin[df_bin["traj_idx"] == traj_idx].sort_values("t").reset_index(drop=True)
        if len(traj_df) < 2:
            continue

        t = traj_df["t"].to_numpy()
        T0 = float(traj_df["T0"].iloc[0])
        P = float(traj_df["P"].iloc[0])
        t_end = 5 * float(traj_df["t_ign"].iloc[0])

        T_true = traj_df["T"].to_numpy(dtype=np.float32)
        Y_true = traj_df[y_cols].to_numpy(dtype=np.float32)

        T_cur = torch.tensor([T_true[0]], dtype=torch.float32, device=device)
        Y_cur = torch.tensor(Y_true[0:1], dtype=torch.float32, device=device)
        T0_t = torch.tensor([T0], dtype=torch.float32, device=device)
        P_t = torch.tensor([P], dtype=torch.float32, device=device)
        tend_t = torch.tensor([t_end], dtype=torch.float32, device=device)

        with torch.no_grad():
            for i in range(1, len(t)):
                dt_t = torch.tensor([t[i] - t[i - 1]], dtype=torch.float32, device=device)
                T_next, Y_next, info = model(T_cur, Y_cur, T0_t, P_t, dt_t, tend_t)

                Tn_true_i = model.normalize_T(torch.tensor([T_true[i]], device=device))
                Yn_true_i = model.normalize_Y(torch.tensor(Y_true[i:i+1], device=device))

                all_Tn_err.append((info["Tn_next"] - Tn_true_i).abs().cpu().numpy())
                all_Yn_err.append((info["Yn_next"] - Yn_true_i).abs().cpu().numpy())

                T_cur, Y_cur = T_next, Y_next

    Tn_err = np.concatenate(all_Tn_err).ravel()
    Yn_err = np.concatenate(all_Yn_err)  # (n_points, n_species)

    return {
        "T": Tn_err,
        "Y": Yn_err,  # per-species, shape (n_points, n_species)
    }


def summarize_normalized_errors(err: dict, species: list[str]) -> pd.DataFrame:
    """One row per output (T + each species): mean, median, std of |normalized error|."""
    rows = []
    rows.append({
        "name": "T", "mean_abs": err["T"].mean(), "median_abs": np.median(err["T"]),
        "std": err["T"].std(),
    })
    for i, s in enumerate(species):
        col = err["Y"][:, i]
        rows.append({
            "name": s, "mean_abs": col.mean(), "median_abs": np.median(col),
            "std": col.std(),
        })
    return pd.DataFrame(rows).sort_values("mean_abs", ascending=False)


def compare_bins_summary(bin_errors: dict) -> pd.DataFrame:
    """
    bin_errors: dict[bin_label] -> err dict from evaluate_bin_normalized.
    Produces ONE overall row per bin, aggregating across T + all species
    together, for a quick cross-bin comparison.
    """
    rows = []
    for label, err in bin_errors.items():
        all_err = np.concatenate([err["T"], err["Y"].ravel()])
        rows.append({
            "bin": label,
            "n_points": len(all_err),
            "mean_abs_norm_err": all_err.mean(),
            "median_abs_norm_err": np.median(all_err),
            "std_norm_err": all_err.std(),
            "p90_abs_norm_err": np.percentile(all_err, 90),
        })
    return pd.DataFrame(rows)