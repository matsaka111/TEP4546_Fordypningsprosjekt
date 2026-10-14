# train_global.py
#
# Trains DeepONet or CRK-PINN on the GRI-3.0 CH4/air data and tests it.
#
# The T0 range 500-2000 K is split into --n_bins equal bins. One model is
# trained per bin (--n_bins 1 = one model for everything). Each test
# trajectory is predicted by the model of its own bin.
#
# Time is normalized with t_end = 5 * IDT. IDT comes from the IDT predictor
# (--tend predicted) or from Cantera (--tend cantera). Trajectories that do
# not ignite get IDT = 0.5 s (the simulation time). NOTE: whether a trajectory
# ignites is still taken from the data (true label), not from a classifier.
#
# Examples:
#   python train_global.py --arch deeponet --n_bins 2 --converge --epochs 100
#   python train_global.py --arch crkpinn  --n_bins 1 --lambda_mole 0.1
#
# Bayesian optimization (bo_search.py) uses: --only_bin, --val_seed, --steps_per_epoch,
# --run_name, --no_test. It reads val_metric.json (validation, never the test set).
#
# Re-test an existing run without training (loads its model_bin{k}.pt and
# rewrites metrics.csv, per_traj.csv, pred_test.npz, traj.png). Give the same
# arguments as the original run so the run name matches, plus --eval_only:
#   python train_global.py --arch deeponet --n_bins 10 --seed 2 --converge --epochs 500 --eval_only
#
# Steps in this file:
#   1. settings (command-line arguments)
#   2. load training data
#   3. train/validation split and training samples
#   4. physics terms (mole, element, enthalpy)
#   5. model + loss
#   6. training loop (one model per bin)
#   7. test: predict all test trajectories
#   8. test metrics + save everything
#
# Everything is saved in results/global/<run name>/:
#   config.json      settings, training time, why each model stopped
#   history.csv      losses per epoch (train and validation, every loss term)
#   model_bin{k}.pt  best model (lowest validation loss) for bin k
#   metrics.csv      test errors per 150 K T0 range
#   per_traj.csv     test errors per test trajectory
#   pred_test.npz    all test predictions (for diagnose_global.py), float32
#                    (runs saved before 2026-10-11 stored log10 Y as float16)
#   traj.png         T(t) for 8 test trajectories
#   val_metric.json  final-state percentage error on the validation trajectories
#                    (T + largest + tracked species), see step 6b

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cantera as ct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fordypning.dataset import get_species_names
from fordypning.deeponet import DeepONetCRK
from fordypning.models import CRKPINN
from fordypning.idt_predictor import load_idt_predictor
from fordypning.chemistry import build_physics_utils
from fordypning.losses import loss_data

SIM_TIME = 0.5             # [s] simulation time of the data
N_TEND = 5                 # t_end = N_TEND * IDT
T0_LO, T0_HI = 500.0, 2000.0
Y_MIN = 1e-10              # smallest mass fraction before taking log
CP_REF = 1.5               # [kJ/(kg K)] turns an enthalpy/entropy error into a temperature error
DT_REL = 0.05              # DeepONet entropy check: compare the state at t and at t*(1+DT_REL)
S_TOL = 1e-5               # [kJ/(kg K)] entropy decreases smaller than this count as round-off


# =============================================================================
# 1. Settings
# =============================================================================
p = argparse.ArgumentParser()
p.add_argument("--arch", choices=["deeponet", "crkpinn"], required=True)
p.add_argument("--n_bins", type=int, default=1, help="number of T0 bins (one model per bin)")
p.add_argument("--tend", choices=["predicted", "cantera"], default="predicted",
               help="where IDT for t_end comes from")
p.add_argument("--seed", type=int, default=0)

# training length
p.add_argument("--epochs", type=int, default=30,
               help="without --converge: epochs per model. With --converge: maximum number of epochs")
p.add_argument("--converge", action="store_true",
               help="train each model until the validation loss stops improving (see step 6)")
p.add_argument("--lr_patience", type=int, default=3,
               help="--converge: halve the learning rate after this many epochs without improvement")
p.add_argument("--stop_patience", type=int, default=10,
               help="--converge: stop after this many epochs without improvement")
p.add_argument("--min_delta", type=float, default=1e-3,
               help="--converge: a validation loss must be this much lower (relative) to count as improvement")
p.add_argument("--batch_size", type=int, default=4096)
p.add_argument("--lr", type=float, default=1e-3)

# physics loss weights (0 = term is only logged, not trained on)
p.add_argument("--lambda_mole", type=float, default=0.0)
p.add_argument("--lambda_element", type=float, default=0.0)
p.add_argument("--lambda_enthalpy", type=float, default=0.0)
p.add_argument("--lambda_entropy", type=float, default=0.0)

# network sizes
p.add_argument("--p", type=int, default=128, help="DeepONet latent size")
p.add_argument("--branch_hidden", type=int, default=256, help="DeepONet branch width")
p.add_argument("--trunk_hidden", type=int, default=128, help="DeepONet trunk width")
p.add_argument("--hidden", type=int, default=256, help="CRK-PINN width")
p.add_argument("--layers", type=int, default=4, help="CRK-PINN hidden layers")
p.add_argument("--residual", action="store_true", help="CRK-PINN predicts a change instead of the state")

# other
p.add_argument("--no_log_time", action="store_true", help="time input t/t_end instead of ln(t/t_end)")
p.add_argument("--train_csv", default="data/raw/data_train.csv")
p.add_argument("--test_csv", default="data/raw/data_test.csv")
p.add_argument("--idt_model", default="model_weights/idt_predictor.pt")
p.add_argument("--n_train_traj", type=int, default=None, help="use only this many trajectories (quick tests)")
p.add_argument("--only_bin", type=int, default=None,
               help="train only this bin (the other bins get no model)")
p.add_argument("--val_seed", type=int, default=None,
               help="seed of the train/validation split; default = --seed. Fix it to keep the "
                    "same validation trajectories while --seed changes")
p.add_argument("--steps_per_epoch", type=int, default=None,
               help="--converge: steps per epoch (default: one pass over all training data)")
p.add_argument("--run_name", default=None, help="results folder name (default: built from the arguments)")
p.add_argument("--no_test", action="store_true", help="stop after training + validation metric (no test set)")
p.add_argument("--val_losses_only", action="store_true",
               help="no training: load this run's saved models (--run_name), compute every loss term "
                    "(unweighted) on the validation split, write val_losses.json and stop")
p.add_argument("--eval_only", action="store_true",
               help="no training: load this run's saved model_bin{k}.pt and redo the test (steps 7-8)")
args = p.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
log_time = not args.no_log_time

# run name, e.g. deeponet_predicted_b2_conv or crkpinn_predicted_b1_lm0.1
run_name = f"{args.arch}_{args.tend}_b{args.n_bins}"
if args.residual:
    run_name += "_res"
if not log_time:
    run_name += "_lin"
if args.seed:
    run_name += f"_s{args.seed}"
if args.lambda_mole:
    run_name += f"_lm{args.lambda_mole:g}"
if args.lambda_element:
    run_name += f"_le{args.lambda_element:g}"
if args.lambda_enthalpy:
    run_name += f"_lh{args.lambda_enthalpy:g}"
if args.lambda_entropy:
    run_name += f"_ls{args.lambda_entropy:g}"
if args.converge:
    run_name += "_conv"
elif args.epochs != 30:
    run_name += f"_e{args.epochs}"

if args.run_name:
    run_name = args.run_name
run_dir = Path("results/global") / run_name
run_dir.mkdir(parents=True, exist_ok=True)

config = {"run": run_name, "args": vars(args), "device": device,
          "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
          "started": time.strftime("%Y-%m-%d %H:%M:%S"),
          "train_seconds": {}, "stop": {}}
try:
    config["git_commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                                   text=True).strip()
except Exception:
    pass

if args.eval_only:
    # keep the original run's config (training time, stop reasons) and only add a note
    if not list(run_dir.glob("model_bin*.pt")):
        raise SystemExit(f"--eval_only: no model_bin*.pt in {run_dir}. "
                         f"Use the same arguments as the original run.")
    commit_now = config.get("git_commit")
    if (run_dir / "config.json").exists():
        with open(run_dir / "config.json") as f:
            config = json.load(f)
    config.setdefault("eval_only", []).append(
        {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": commit_now})


def save_config():
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)


bin_edges = np.linspace(T0_LO, T0_HI, args.n_bins + 1)


def bin_of(T0):
    """Bin number (0 .. n_bins-1) for each T0."""
    return np.clip(np.digitize(T0, bin_edges) - 1, 0, args.n_bins - 1)


print(f"run {run_name} | device {device}", flush=True)
species = get_species_names()
y_cols = [f"Y_{s}" for s in species]


# =============================================================================
# 2. Load data
# =============================================================================
def load(csv_path, max_traj=None):
    """Read a data CSV.

    Returns two dicts:
      rows: one entry per data point, sorted by trajectory and time
            code = trajectory number 0..N-1, t, T, Y
      traj: one entry per trajectory
            T0, phi, P, ignited, t_ign, Y0 (first composition),
            first (row of its first point), traj_idx (id in the CSV)
    """
    cols = ["traj_idx", "t", "T", "P", "T0", "phi", "t_ign", "ignited"] + y_cols
    dtypes = {c: np.float32 for c in ["T", "P", "T0", "phi"] + y_cols}
    dtypes["t"] = np.float64
    dtypes["t_ign"] = np.float64
    t_start = time.time()

    # read in chunks so the large CSV does not need double memory
    df = pd.concat(pd.read_csv(csv_path, usecols=cols, dtype=dtypes, chunksize=2_000_000),
                   ignore_index=True)
    if max_traj is not None:
        keep = df["traj_idx"].drop_duplicates().iloc[:max_traj]
        df = df[df["traj_idx"].isin(keep)]
    df = df.sort_values(["traj_idx", "t"], kind="stable").reset_index(drop=True)

    code = pd.factorize(df["traj_idx"])[0].astype(np.int64)    # 0, 0, ..., 1, 1, ...
    first = np.r_[0, np.flatnonzero(np.diff(code)) + 1]         # first row of each trajectory
    Y = df[y_cols].to_numpy(np.float32)

    rows = {"code": code, "t": df["t"].to_numpy(), "T": df["T"].to_numpy(np.float32), "Y": Y}
    traj = {
        "T0": df["T0"].to_numpy(np.float32)[first],
        "phi": df["phi"].to_numpy(np.float32)[first],
        "P": df["P"].to_numpy(np.float32)[first],
        "ignited": df["ignited"].to_numpy().astype(bool)[first],
        "t_ign": df["t_ign"].to_numpy()[first],
        "Y0": Y[first],
        "first": first,
        "traj_idx": df["traj_idx"].to_numpy()[first],
    }
    print(f"loaded {csv_path}: {len(first)} trajectories, {len(code)} rows "
          f"({time.time() - t_start:.0f} s)", flush=True)
    return rows, traj


def compute_t_end(traj):
    """t_end = 5 * IDT per trajectory. Non-igniting: IDT = SIM_TIME."""
    idt = np.full(len(traj["T0"]), SIM_TIME)
    ign = traj["ignited"]
    if args.tend == "cantera":
        idt[ign] = traj["t_ign"][ign]
    elif ign.any():
        idt[ign] = idt_model.predict_idt(traj["T0"][ign], traj["phi"][ign]).cpu().numpy()
    return (N_TEND * idt).astype(np.float32)


idt_model = load_idt_predictor(args.idt_model) if args.tend == "predicted" else None

if not args.eval_only:
    rows, traj = load(args.train_csv, args.n_train_traj)
    traj["t_end"] = compute_t_end(traj)
    traj["bin"] = bin_of(traj["T0"])
    n_traj = len(traj["T0"])


# =============================================================================
# 3. Train/validation split and training samples
# =============================================================================
if not args.eval_only:
    # 10 % of the trajectories (whole trajectories) are used for validation
    # (--val_seed: own generator, so the split does not change with --seed)
    split_rng = rng if args.val_seed is None else np.random.default_rng(args.val_seed)
    is_val_traj = np.zeros(n_traj, bool)
    is_val_traj[split_rng.permutation(n_traj)[: n_traj // 10]] = True

    # One training sample:
    #   DeepONet : one data point (initial condition, t) -> state at t
    #   CRK-PINN : one step, row i -> row i+1 of the same trajectory
    if args.arch == "deeponet":
        sample_rows = np.arange(len(rows["code"]))
    else:
        sample_rows = np.flatnonzero(rows["code"][:-1] == rows["code"][1:])
    sample_traj = rows["code"][sample_rows]          # trajectory of each sample

    # number of steps in one epoch over ALL training data (used by --converge,
    # so every bin model gets evaluated after the same number of steps)
    STEPS_FULL_EPOCH = int(np.ceil((~is_val_traj[sample_traj]).sum() / args.batch_size))


    # All data as torch tensors, on the GPU if it fits.
    def to_tensor(a):
        return torch.from_numpy(np.require(a, requirements=["C", "W"]))


    data = {
        # per row
        "code": to_tensor(rows["code"]),
        "t": to_tensor(rows["t"].astype(np.float32)),
        "T": to_tensor(rows["T"]),
        "Y": to_tensor(rows["Y"]),
        # per trajectory
        "T0": to_tensor(traj["T0"]),
        "phi": to_tensor(traj["phi"]),
        "P": to_tensor(traj["P"]),
        "Y0": to_tensor(traj["Y0"]),
        "t_end": to_tensor(traj["t_end"]),
    }
    if args.arch == "crkpinn":
        dt = np.zeros(len(rows["code"]), np.float32)          # dt from row i to row i+1
        dt[sample_rows] = rows["t"][sample_rows + 1] - rows["t"][sample_rows]
        data["dt"] = to_tensor(dt)

    data_device = device
    try:
        data = {k: v.to(device) for k, v in data.items()}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        data_device = "cpu"
        print("data does not fit on the GPU, keeping it on the CPU", flush=True)


def get_batch(idx):
    """Collect the inputs/targets for the samples `idx` and move them to the GPU."""
    c = data["code"][idx]                  # trajectory of each sample
    b = {"t": data["t"][idx], "T": data["T"][idx], "Y": data["Y"][idx],
         "T0": data["T0"][c], "phi": data["phi"][c], "P": data["P"][c],
         "Y0": data["Y0"][c], "t_end": data["t_end"][c],
         "Z0": data["Z0"][c], "h0": data["h0"][c]}
    if args.arch == "crkpinn":
        b["dt"] = data["dt"][idx]
        b["T_next"] = data["T"][idx + 1]
        b["Y_next"] = data["Y"][idx + 1]
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


# =============================================================================
# 4. Physics terms
# =============================================================================
# Computed for every batch and logged (unweighted) in history.csv. They are
# only added to the training loss if their lambda > 0.
#
#   L_mole     = mean( (sum_k Y_k - 1)^2 )
#   L_element  = mean( (Z_e(Y_pred) - Z_e,ref)^2 )        Z_e = element mass fractions (C,H,O,N,Ar)
#   L_enthalpy = mean( ((h_pred - h_ref) / (CP_REF * T_std))^2 )
#                (enthalpy error turned into a temperature error, scaled like T in the data loss)
#   L_entropy  = mean( (T * max(0, s_before - s_after) / (CP_REF * T_std))^2 )
#                second law: in this adiabatic, constant-pressure reactor the specific
#                entropy s can only increase. Only DEcreases are penalized (one-sided),
#                converted to an equivalent temperature error like the enthalpy term.
#                CRK-PINN: s(input state) vs s(predicted next state).
#                DeepONet: s(predicted state at t) vs s(predicted state at t*(1+DT_REL)).
#
# Reference: DeepONet uses the initial state of the trajectory (element mass
# fractions and enthalpy stay constant in this reactor). CRK-PINN uses the
# input state of the step, as in fordypning/losses.py.
physics = build_physics_utils(ct.Solution("gri30.yaml"), species, device)
LAMBDAS = {"mole": args.lambda_mole, "element": args.lambda_element,
           "enthalpy": args.lambda_enthalpy, "entropy": args.lambda_entropy}


def make_entropy_fn():
    """Mixture specific entropy s(T, Y, P) in kJ/(kg K), from the NASA-7 polynomials:
        s = sum_k (Y_k / W_k) * ( s0_k(T) - R ln(X_k P / P_atm) )
    Y is renormalized to sum to 1 first. Checked against Cantera's entropy_mass."""
    gas = ct.Solution("gri30.yaml")
    idx = [gas.species_index(sp) for sp in species]
    coeffs = np.array([gas.species(i).thermo.coeffs for i in idx])
    W = torch.tensor(gas.molecular_weights[idx], dtype=torch.float32, device=device)   # kg/kmol
    T_mid = torch.tensor(coeffs[:, 0], dtype=torch.float32, device=device)
    a_high = torch.tensor(coeffs[:, 1:8], dtype=torch.float32, device=device)
    a_low = torch.tensor(coeffs[:, 8:15], dtype=torch.float32, device=device)
    R = 8.314462618                                                                    # kJ/(kmol K)

    def s0_over_R(T, a):     # standard-state entropy of each species / R
        T = T.unsqueeze(-1)
        return (a[:, 0] * torch.log(T) + a[:, 1] * T + a[:, 2] / 2 * T**2
                + a[:, 3] / 3 * T**3 + a[:, 4] / 4 * T**4 + a[:, 6])

    def entropy(T, Y, P):
        # Y is renormalized to sum 1: otherwise a mass-sum error (sum Y != 1, which is
        # L_mole's job) would change s and hide or create entropy changes
        Y = torch.clamp(Y, min=0.0)
        Y = Y / torch.clamp(Y.sum(-1, keepdim=True), min=1e-12)
        s0 = torch.where(T.unsqueeze(-1) > T_mid, s0_over_R(T, a_high), s0_over_R(T, a_low)) * R
        n = Y / W                                     # kmol/kg
        X = n / n.sum(-1, keepdim=True)               # mole fractions
        mixing = R * torch.log(torch.clamp(X, min=1e-30) * (P.unsqueeze(-1) / ct.one_atm))
        return (n * (s0 - mixing)).sum(-1)            # kJ/(kg K)
    return entropy


entropy = make_entropy_fn()

# reference values per trajectory, computed once (training data only)
if not args.eval_only:
    with torch.no_grad():
        Y0 = torch.as_tensor(traj["Y0"], device=device)
        T_first = torch.as_tensor(rows["T"][traj["first"]], device=device)
        data["Z0"] = physics.element_mass_fractions(Y0).to(data_device)
        data["h0"] = physics.mixture_specific_enthalpy(T_first, Y0).to(data_device)
        del Y0, T_first


def physics_terms(model, T_pred, Y_pred, Z_ref, h_ref, s_before, s_after):
    dh = (physics.mixture_specific_enthalpy(T_pred, Y_pred) - h_ref) / (CP_REF * model.T_std)
    # entropy decrease (0 if s increased), as an equivalent temperature error
    ds_bad = torch.relu(s_before - s_after) * T_pred / (CP_REF * model.T_std)
    return {
        "mole": ((Y_pred.sum(-1) - 1.0) ** 2).mean(),
        "element": ((physics.element_mass_fractions(Y_pred) - Z_ref) ** 2).mean(),
        "enthalpy": (dh ** 2).mean(),
        "entropy": (ds_bad ** 2).mean(),
    }


# =============================================================================
# 5. Model and loss
# =============================================================================
def time_input(x, t_end):
    """What the model sees as time: ln(x / t_end) (or x / t_end with --no_log_time)."""
    r = x / torch.clamp(t_end, min=1e-12)
    return torch.log(torch.clamp(r, min=1e-12)) if log_time else r


def build_model(train_idx, train_trajs):
    """New model with normalization statistics (mean/std) from this bin's training data."""
    # up to 2 million random training samples are enough for mean/std
    s = get_batch(train_idx[torch.randperm(len(train_idx), device=data_device)[:2_000_000]])
    logY = torch.log(torch.clamp(s["Y"], min=Y_MIN))
    logP = torch.log(torch.clamp(s["P"], min=1e-6))

    if args.arch == "deeponet":
        kwargs = dict(n_species=len(species), p=args.p, branch_hidden=args.branch_hidden,
                      trunk_hidden=args.trunk_hidden, log_time=log_time)
        model = DeepONetCRK(**kwargs).to(device)
        logY0 = torch.log(torch.clamp(torch.as_tensor(traj["Y0"][train_trajs], device=device), min=Y_MIN))
        tin = time_input(s["t"], s["t_end"])
        model.set_norm_stats(
            logY0_mean=logY0.mean(0), logY0_std=logY0.std(0),
            T0_mean=s["T0"].mean(), T0_std=s["T0"].std(),
            phi_mean=s["phi"].mean(), phi_std=s["phi"].std(),
            logP_mean=logP.mean(), logP_std=logP.std(),
            tprime_mean=tin.mean(), tprime_std=tin.std(),
            logY_mean=logY.mean(0), logY_std=logY.std(0),
            T_mean=s["T"].mean(), T_std=s["T"].std())
    else:
        kwargs = dict(n_species=len(species), hidden=args.hidden, n_hidden_layers=args.layers,
                      residual=args.residual, log_time=log_time)
        model = CRKPINN(**kwargs).to(device)
        tin = time_input(s["dt"], s["t_end"])
        model.set_norm_stats(
            logY_mean=logY.mean(0), logY_std=logY.std(0),
            T_mean=s["T"].mean(), T_std=s["T"].std(),
            T0_mean=s["T0"].mean(), T0_std=s["T0"].std(),
            logP_mean=logP.mean(), logP_std=logP.std(),
            tprime_mean=tin.mean(), tprime_std=tin.std())
    return model, kwargs


def compute_loss(model, b):
    """Returns (loss to train on, dict with every term unweighted)."""
    if args.arch == "deeponet":
        # predict the state at time t directly from the initial condition
        T_pred, Y_pred, info = model(b["Y0"], b["T0"], b["phi"], b["P"], b["t"], b["t_end"])
        # data loss on normalized T and normalized log Y (Huber, delta 5)
        T_true_n = (b["T"] - model.T_mean) / model.T_std
        Y_true_n = (torch.log(torch.clamp(b["Y"], min=Y_MIN)) - model.logY_mean) / model.logY_std
        huber = torch.nn.functional.huber_loss
        data_loss = huber(info["Tn_pred"], T_true_n, delta=5.0) + huber(info["Yn_pred"], Y_true_n, delta=5.0)
        Z_ref, h_ref = b["Z0"], b["h0"]
        # entropy: second prediction slightly later in time (only when it is needed:
        # training with lambda_entropy > 0, or validation)
        s_before = entropy(T_pred, Y_pred, b["P"])
        if args.lambda_entropy > 0 or not model.training:
            T_later, Y_later, _ = model(b["Y0"], b["T0"], b["phi"], b["P"],
                                        b["t"] * (1 + DT_REL), b["t_end"])
            s_after = entropy(T_later, Y_later, b["P"])
        else:
            s_after = s_before
    else:
        # predict the next state from the current (true) state
        T_pred, Y_pred, info = model(b["T"], b["Y"], b["T0"], b["P"], b["dt"], b["t_end"])
        data_loss = loss_data(model, info, b["T_next"], b["Y_next"])     # MSE, normalized
        Z_ref = physics.element_mass_fractions(b["Y"])
        h_ref = physics.mixture_specific_enthalpy(b["T"], b["Y"])
        s_before = entropy(b["T"], b["Y"], b["P"])      # true input state
        s_after = entropy(T_pred, Y_pred, b["P"])       # predicted next state

    terms = physics_terms(model, T_pred, Y_pred, Z_ref, h_ref, s_before, s_after)
    loss = data_loss
    for name, weight in LAMBDAS.items():
        if weight > 0:
            loss = loss + weight * terms[name]
    terms["data"] = data_loss
    return loss, terms


# =============================================================================
# 6. Training (one model per bin)
# =============================================================================
# Two modes:
#
#  default     : --epochs passes over this bin's training data. Learning rate
#                goes smoothly from --lr to 0 (cosine). NOTE: a bin with less
#                data gets fewer optimizer steps.
#
#  --converge  : an "epoch" is a fixed number of steps (STEPS_FULL_EPOCH, = one
#                pass over ALL training data), the same for every bin model.
#                After each epoch the validation loss is checked:
#                  - no improvement for --lr_patience epochs -> learning rate / 2
#                  - no improvement for --stop_patience epochs -> stop (converged)
#                  - at most --epochs epochs
#                "Improvement" = at least --min_delta (0.1 %) lower than the best.
#
# In both modes the model with the lowest validation data loss is saved
# (model_bin{k}.pt) and used for testing, not the model of the last epoch.
history = []


def train_one_bin(k):
    # ---- this bin's samples
    in_bin = traj["bin"] == k
    train_idx = sample_rows[in_bin[sample_traj] & ~is_val_traj[sample_traj]]
    val_idx = sample_rows[in_bin[sample_traj] & is_val_traj[sample_traj]]
    if len(val_idx) > 500_000:                       # validation on at most 500k samples
        val_idx = rng.choice(val_idx, 500_000, replace=False)
    lo, hi = bin_edges[k], bin_edges[k + 1]
    print(f"\n=== bin {k}: T0 {lo:.0f}-{hi:.0f} K, {int(in_bin.sum())} trajectories "
          f"({int((in_bin & traj['ignited']).sum())} ignite), "
          f"{len(train_idx)} train / {len(val_idx)} val samples ===", flush=True)
    if len(train_idx) == 0:
        return None
    if len(val_idx) == 0:                             # only in tiny test runs
        val_idx = train_idx
    train_idx = to_tensor(train_idx).to(data_device)
    val_idx = to_tensor(val_idx).to(data_device)

    # ---- model, optimizer, learning-rate schedule
    model, kwargs = build_model(train_idx, np.flatnonzero(in_bin & ~is_val_traj))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.converge:
        steps_per_epoch = args.steps_per_epoch or STEPS_FULL_EPOCH
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.lr_patience,
            threshold=args.min_delta, threshold_mode="rel", min_lr=1e-6)
    else:
        steps_per_epoch = int(np.ceil(len(train_idx) / args.batch_size))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * steps_per_epoch)

    # endless stream of shuffled batches from this bin
    def batches():
        while True:
            shuffled = train_idx[torch.randperm(len(train_idx), device=data_device)]
            for i in range(0, len(shuffled), args.batch_size):
                yield shuffled[i:i + args.batch_size]
    batch_stream = batches()

    save_path = run_dir / f"model_bin{k}.pt"
    best_val = float("inf")
    last_improvement = 0          # epoch of the last improvement by more than min_delta
    steps_done = 0
    stop_reason = f"reached {args.epochs} epochs"
    t_bin_start = time.time()

    for epoch in range(args.epochs):
        t_epoch = time.time()

        # ---- train for one epoch
        model.train()
        sum_loss = 0.0
        sum_terms = {name: 0.0 for name in ["data", *LAMBDAS]}
        for _ in range(steps_per_epoch):
            loss, terms = compute_loss(model, get_batch(next(batch_stream)))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if not args.converge:
                scheduler.step()                     # cosine: update every step
            sum_loss += loss.item()
            for name in sum_terms:
                sum_terms[name] += terms[name].item()
        steps_done += steps_per_epoch

        # ---- validation
        model.eval()
        with torch.no_grad():
            val_parts = [compute_loss(model, get_batch(val_idx[i:i + 65536]))[1]
                         for i in range(0, len(val_idx), 65536)]
        val_terms = {name: float(np.mean([v[name].item() for v in val_parts])) for name in sum_terms}
        val_loss = val_terms["data"]     # models are compared on the data loss only
        lr_now = optimizer.param_groups[0]["lr"]

        # ---- keep the best model
        if best_val == float("inf") or val_loss < best_val * (1 - args.min_delta):
            last_improvement = epoch
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(), "arch": args.arch, "kwargs": kwargs,
                        "tend": args.tend, "bin_edges": [float(lo), float(hi)],
                        "epoch": epoch, "steps": steps_done}, save_path)

        # ---- log
        history.append({"bin": k, "epoch": epoch, "steps": steps_done, "lr": lr_now,
                        "train": sum_loss / steps_per_epoch, "val": val_loss,
                        **{f"train_{n}": v / steps_per_epoch for n, v in sum_terms.items()},
                        **{f"val_{n}": v for n, v in val_terms.items()},
                        "seconds": time.time() - t_epoch})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"bin {k} epoch {epoch:3d}  steps {steps_done:7d}  lr {lr_now:.1e}  "
              f"train {sum_loss / steps_per_epoch:.5f}  val {val_loss:.5f}  best {best_val:.5f}  "
              f"| val mole {val_terms['mole']:.1e} element {val_terms['element']:.1e} "
              f"enthalpy {val_terms['enthalpy']:.1e} entropy {val_terms['entropy']:.1e}  "
              f"({time.time() - t_epoch:.0f} s)", flush=True)

        # ---- --converge: lower learning rate / stop
        if args.converge:
            scheduler.step(val_loss)
            if epoch - last_improvement >= args.stop_patience:
                stop_reason = f"converged (no improvement for {args.stop_patience} epochs)"
                break

    print(f"bin {k}: stopped after {epoch + 1} epochs, {steps_done} steps: {stop_reason}", flush=True)
    config["stop"][k] = {"epochs": epoch + 1, "steps": steps_done,
                         "last_improvement": last_improvement, "reason": stop_reason}
    config["train_seconds"][k] = round(time.time() - t_bin_start, 1)
    save_config()

    # use the best model, not the last one
    model.load_state_dict(torch.load(save_path, map_location=device)["state_dict"])
    return model.eval()


def load_saved_model(k):
    """--eval_only: the saved best model of bin k (None if the bin had no training data)."""
    path = run_dir / f"model_bin{k}.pt"
    if not path.exists():
        print(f"bin {k}: no {path.name}, no model for this bin", flush=True)
        return None
    ckpt = torch.load(path, map_location=device)
    assert ckpt["arch"] == args.arch, f"{path} is a {ckpt['arch']} model"
    assert np.allclose(ckpt["bin_edges"], bin_edges[k:k + 2]), \
        f"{path}: bin edges {ckpt['bin_edges']} do not match --n_bins {args.n_bins}"
    model = (DeepONetCRK if args.arch == "deeponet" else CRKPINN)(**ckpt["kwargs"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"bin {k}: loaded {path.name} (epoch {ckpt['epoch']}, {ckpt['steps']} steps)", flush=True)
    return model.eval()


if args.val_losses_only:
    # every loss term of the saved best models on the validation samples (bo_search.py
    # uses the ratios L_data / L_t to scale the loss weights)
    result = {}
    for k in (range(args.n_bins) if args.only_bin is None else [args.only_bin]):
        model = load_saved_model(k)
        if model is None:
            continue
        in_bin = traj["bin"] == k
        val_idx = sample_rows[in_bin[sample_traj] & is_val_traj[sample_traj]]
        if len(val_idx) > 500_000:
            val_idx = np.random.default_rng(0).choice(val_idx, 500_000, replace=False)
        val_idx = to_tensor(val_idx).to(data_device)
        with torch.no_grad():
            parts = [compute_loss(model, get_batch(val_idx[i:i + 65536]))[1]
                     for i in range(0, len(val_idx), 65536)]
        result[k] = {n: float(np.mean([v[n].item() for v in parts])) for n in ["data", *LAMBDAS]}
        print(f"bin {k}: " + ", ".join(f"{n} {v:.3e}" for n, v in result[k].items()), flush=True)
    with open(run_dir / "val_losses.json", "w") as f:
        json.dump({"val_seed": args.val_seed, "bins": result}, f, indent=2)
    print(f"saved {run_dir}/val_losses.json", flush=True)
    raise SystemExit(0)

if args.eval_only:
    models = {k: load_saved_model(k) for k in range(args.n_bins)}
else:
    bins_to_train = range(args.n_bins) if args.only_bin is None else [args.only_bin]
    models = {k: (train_one_bin(k) if k in bins_to_train else None) for k in range(args.n_bins)}


# =============================================================================
# 6b. Validation metric (used by bo_search.py; the test set is not touched)
# =============================================================================
# For every validation trajectory with a model, at its LAST time point (0.5 s):
#   e_j = mean over {T, LARGEST, TRACKED species with Y_true > 1e-8} of |pred - true| / true
#   val_mape_final = 100 * mean_j e_j   [%]
# DeepONet predicts the last point from the initial state; CRK-PINN rolls out
# over the trajectory's own time points from the first state (as in step 7).
LARGEST = ["CO", "CH4", "O2", "C2H2", "H2O", "CO2"]
TRACKED = ["CH2O", "H2O2", "OH", "C2H6", "NO"]


def validation_metric(models):
    first_v = traj["first"]
    last_v = np.r_[first_v[1:], len(rows["code"])] - 1
    has = np.array([models[b] is not None for b in traj["bin"]])
    vj = np.flatnonzero(is_val_traj & has)
    if len(vj) == 0:
        return None
    g = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
    T_fin = np.empty(len(vj), np.float32)
    Y_fin = np.empty((len(vj), len(species)), np.float32)
    with torch.no_grad():
        for k, model in models.items():
            if model is None:
                continue
            sel = np.flatnonzero(traj["bin"][vj] == k)
            if len(sel) == 0:
                continue
            j = vj[sel]
            if args.arch == "deeponet":
                for i in range(0, len(j), 65536):
                    jj, ss = j[i:i + 65536], sel[i:i + 65536]
                    Tp, Yp, _ = model(g(traj["Y0"][jj]), g(traj["T0"][jj]), g(traj["phi"][jj]), g(traj["P"][jj]),
                                      g(rows["t"][last_v[jj]].astype(np.float32)), g(traj["t_end"][jj]))
                    T_fin[ss], Y_fin[ss] = Tp.cpu().numpy(), Yp.cpu().numpy()
            else:
                n_pts = last_v[j] - first_v[j] + 1
                for L in np.unique(n_pts):
                    m = n_pts == L
                    jj, ss = j[m], sel[m]
                    r0 = first_v[jj]
                    T_now, Y_now = g(rows["T"][r0]), g(rows["Y"][r0])
                    T0_b, P_b, te_b = g(traj["T0"][jj]), g(traj["P"][jj]), g(traj["t_end"][jj])
                    for st in range(1, L):
                        step = g((rows["t"][r0 + st] - rows["t"][r0 + st - 1]).astype(np.float32))
                        T_now, Y_now, _ = model(T_now, Y_now, T0_b, P_b, step, te_b)
                    T_fin[ss], Y_fin[ss] = T_now.cpu().numpy(), Y_now.cpu().numpy()
    T_true = rows["T"][last_v[vj]].astype(np.float64)
    Y_true = rows["Y"][last_v[vj]].astype(np.float64)
    errs = [np.abs(T_fin - T_true) / T_true]
    masks = [np.ones(len(vj), bool)]
    comp = {"T": float(100 * errs[0].mean())}
    for sp in LARGEST + TRACKED:
        c = species.index(sp)
        m = Y_true[:, c] > 1e-8
        e = np.abs(Y_fin[:, c] - Y_true[:, c]) / np.where(m, Y_true[:, c], 1.0)
        errs.append(np.where(m, e, 0.0))
        masks.append(m)
        comp[sp] = float(100 * e[m].mean()) if m.any() else None
    E = 100 * np.sum(errs, 0) / np.sum(masks, 0)                # per trajectory [%]
    return {"val_mape_final": float(E.mean()), "val_mape_final_median": float(np.median(E)),
            "val_mape_final_p95": float(np.percentile(E, 95)), "n_val_traj": int(len(vj)),
            "components_mean_%": comp,
            "lambdas": {"mole": args.lambda_mole, "element": args.lambda_element,
                        "enthalpy": args.lambda_enthalpy, "entropy": args.lambda_entropy},
            "seed": args.seed, "val_seed": args.val_seed, "only_bin": args.only_bin}


if not args.eval_only:
    vm = validation_metric(models)
    if vm is not None:
        with open(run_dir / "val_metric.json", "w") as f:
            json.dump(vm, f, indent=2)
        print(f"\nvalidation: final-state MAPE {vm['val_mape_final']:.4g} % "
              f"(median {vm['val_mape_final_median']:.4g} %, {vm['n_val_traj']} trajectories)", flush=True)
    # free the training data before loading the test data
    del data, rows, traj
    if device == "cuda":
        torch.cuda.empty_cache()

if args.no_test:
    config["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_config()
    print(f"--no_test: done, saved to {run_dir}/", flush=True)
    raise SystemExit(0)


# =============================================================================
# 7. Test: predict every test trajectory with the model of its bin
# =============================================================================
rows, traj = load(args.test_csv)
traj["t_end"] = compute_t_end(traj)
traj["bin"] = bin_of(traj["T0"])
code, first = rows["code"], traj["first"]
last = np.r_[first[1:], len(code)] - 1            # last row of each trajectory
n_test = len(first)
has_model = np.array([models[k] is not None for k in traj["bin"]])


def gpu(a):
    return torch.as_tensor(np.ascontiguousarray(a), device=device)


T_pred = np.full(len(code), np.nan, np.float32)
Y_pred = np.full_like(rows["Y"], np.nan)

if device == "cuda":
    torch.cuda.synchronize()
t_start = time.time()
with torch.no_grad():
    for k, model in models.items():
        if model is None:
            continue
        if args.arch == "deeponet":
            # DeepONet: every test point directly from the initial condition
            r = np.flatnonzero(traj["bin"][code] == k)
            for i in range(0, len(r), 65536):
                rr = r[i:i + 65536]
                c = code[rr]
                Tp, Yp, _ = model(gpu(traj["Y0"][c]), gpu(traj["T0"][c]), gpu(traj["phi"][c]),
                                  gpu(traj["P"][c]), gpu(rows["t"][rr].astype(np.float32)),
                                  gpu(traj["t_end"][c]))
                T_pred[rr], Y_pred[rr] = Tp.cpu().numpy(), Yp.cpu().numpy()
        else:
            # CRK-PINN: rollout, start from the true first state and feed each
            # prediction back in. Trajectories with the same number of points
            # are run together.
            in_bin = np.flatnonzero(traj["bin"] == k)
            n_points = last[in_bin] - first[in_bin] + 1
            for L in np.unique(n_points):
                sel = in_bin[n_points == L]
                r0 = first[sel]
                T_now, Y_now = gpu(rows["T"][r0]), gpu(rows["Y"][r0])
                T_pred[r0], Y_pred[r0] = rows["T"][r0], rows["Y"][r0]
                T0_b, P_b, tend_b = gpu(traj["T0"][sel]), gpu(traj["P"][sel]), gpu(traj["t_end"][sel])
                for j in range(1, L):
                    step = gpu((rows["t"][r0 + j] - rows["t"][r0 + j - 1]).astype(np.float32))
                    T_now, Y_now, _ = model(T_now, Y_now, T0_b, P_b, step, tend_b)
                    T_pred[r0 + j], Y_pred[r0 + j] = T_now.cpu().numpy(), Y_now.cpu().numpy()
if device == "cuda":
    torch.cuda.synchronize()
# --eval_only keeps the original timing (the GPU may be shared with another training now)
config["test_inference_seconds_eval_only" if args.eval_only else "test_inference_seconds"] = \
    round(time.time() - t_start, 2)
config["test_n_traj"] = int(has_model.sum())
config["test_n_rows"] = int(np.isfinite(T_pred).sum())


# =============================================================================
# 8. Test metrics and saving
# =============================================================================
T_true, Y_true = rows["T"], rows["Y"]

# per trajectory: mean |dT|, |dT| at the last point, mean |log10 Y error| (Y > 1e-6)
abs_dT = np.abs(T_pred - T_true)
mask = Y_true > 1e-6
dlogY = np.where(mask, np.abs(np.log10(np.clip(Y_pred, 1e-30, None))
                              - np.log10(np.clip(Y_true, 1e-30, None))), 0)
n_points = np.bincount(code, None, n_test)
mae_T = np.bincount(code, np.nan_to_num(abs_dT), n_test) / n_points
mae_logY = (np.bincount(code, np.nan_to_num(dlogY).sum(1), n_test)
            / np.maximum(np.bincount(code, mask.sum(1), n_test), 1))
final_dT = abs_dT[last]


def crossing_time(t, T, level):
    """First time T reaches `level` (linear interpolation). NaN if never."""
    above = np.flatnonzero(T >= level)
    if len(above) == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return t[0]
    return t[i - 1] + (level - T[i - 1]) * (t[i] - t[i - 1]) / (T[i] - T[i - 1])


# entropy along each predicted trajectory: fraction of steps where s decreases
# (beyond round-off S_TOL), and the largest decrease as an equivalent temperature [K]
with torch.no_grad():
    s_pred = np.empty(len(code))
    P_rows = traj["P"][code]
    for i in range(0, len(code), 262144):
        sl = slice(i, i + 262144)
        s_pred[sl] = entropy(gpu(T_pred[sl]), gpu(np.nan_to_num(Y_pred[sl])), gpu(P_rows[sl])).cpu().numpy()
same_traj = code[1:] == code[:-1]
ds = np.where(same_traj, s_pred[1:] - s_pred[:-1], 0.0)
bad = same_traj & (ds < -S_TOL)
bad_K = np.where(bad, -ds * T_pred[1:] / CP_REF, 0.0)
n_steps = np.maximum(np.bincount(code[1:], same_traj.astype(float), n_test), 1)
entropy_viol_frac = np.bincount(code[1:], bad.astype(float), n_test) / n_steps
entropy_viol_K_max = np.zeros(n_test)
np.maximum.at(entropy_viol_K_max, code[1:], bad_K)

# ignition time = time T reaches T0 + half of the true temperature rise
ign_err = np.full(n_test, np.nan)
ign_missed = np.zeros(n_test, bool)
for j in np.flatnonzero(traj["ignited"] & has_model):
    sl = slice(first[j], last[j] + 1)
    rise = T_true[last[j]] - traj["T0"][j]
    if rise < 50.0:
        continue
    level = traj["T0"][j] + 0.5 * rise
    t_true = crossing_time(rows["t"][sl], T_true[sl], level)
    t_pred = crossing_time(rows["t"][sl], T_pred[sl], level)
    if np.isnan(t_pred):
        ign_missed[j] = True
    elif not np.isnan(t_true):
        ign_err[j] = abs(t_pred - t_true) / t_true

# table per 150 K T0 range + "all"
range_edges = np.linspace(T0_LO, T0_HI, 11)
range_of = np.clip(np.digitize(traj["T0"], range_edges) - 1, 0, 9)
table = []
for r in list(range(10)) + ["all"]:
    m = has_model & ((range_of == r) if r != "all" else True)
    if not m.any():
        continue
    ig = m & traj["ignited"]
    table.append({
        "T0 range": "all" if r == "all" else f"[{range_edges[r]:.0f},{range_edges[r + 1]:.0f})",
        "n": int(m.sum()), "n_ign": int(ig.sum()),
        "MAE T [K]": mae_T[m].mean(),
        "final |dT| med [K]": np.median(final_dT[m]),
        "final |dT| p95 [K]": np.percentile(final_dT[m], 95),
        "MAE log10Y": mae_logY[m].mean(),
        "ign time rel err med": np.nanmedian(ign_err[ig]) if np.isfinite(ign_err[ig]).any() else np.nan,
        "ign missed": int(ign_missed[ig].sum()),
        "entropy viol frac": entropy_viol_frac[m].mean(),
    })
table = pd.DataFrame(table)
pd.set_option("display.width", 200)
print(f"\nTEST  {run_name}")
print(table.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
table.to_csv(run_dir / "metrics.csv", index=False)

pd.DataFrame({
    "traj_idx": traj["traj_idx"], "T0": traj["T0"], "phi": traj["phi"],
    "ignited": traj["ignited"], "t_ign": traj["t_ign"], "t_end": traj["t_end"],
    "model_bin": traj["bin"], "has_model": has_model,
    "MAE_T": mae_T, "final_dT": final_dT, "MAE_log10Y": mae_logY,
    "ign_time_rel_err": ign_err, "ign_missed": ign_missed,
    "entropy_viol_frac": entropy_viol_frac, "entropy_viol_K_max": entropy_viol_K_max,
}).to_csv(run_dir / "per_traj.csv", index=False)

np.savez_compressed(
    run_dir / "pred_test.npz",
    traj_idx=traj["traj_idx"][code], t=rows["t"], T_true=T_true, T_pred=T_pred,
    log10Y_pred=np.log10(np.clip(Y_pred, 1e-30, None)).astype(np.float32),
    species=np.array(species))
config["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save_config()

# plot T(t) for 8 test trajectories spread over T0
candidates = np.flatnonzero(has_model)
by_T0 = candidates[np.argsort(traj["T0"][candidates])]
pick = by_T0[np.linspace(0, len(by_T0) - 1, 8).astype(int)]
fig, axs = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
for ax, j in zip(axs.ravel(), pick):
    sl = slice(first[j], last[j] + 1)
    ax.semilogx(rows["t"][sl], T_true[sl], "k-", lw=1.5, label="Cantera")
    ax.semilogx(rows["t"][sl], T_pred[sl], "r--", lw=1.2, label=args.arch)
    ax.set_title(f"T0={traj['T0'][j]:.0f} K, phi={traj['phi'][j]:.2f}", fontsize=9)
axs[0, 0].legend(fontsize=8)
for ax in axs[1]:
    ax.set_xlabel("t [s]")
for ax in axs[:, 0]:
    ax.set_ylabel("T [K]")
fig.suptitle(run_name)
fig.tight_layout()
fig.savefig(run_dir / "traj.png", dpi=130)
print(f"saved to {run_dir}/", flush=True)