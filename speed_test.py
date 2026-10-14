# speed_test.py
"""
Speed and resource comparison: Cantera vs. DeepONet vs. CRK-PINN on NEW cases
(initial conditions that are in neither data_train.csv nor data_test.csv).

  python speed_test.py
  python speed_test.py --n_traj 1000 --cells 1000,10000,100000,1000000 --cores 1,2,4,8

Part A, initial condition -> state at 0.5 s (final state)
  n_traj new (T0, phi), drawn like gen.py (uniform in each of the 10 T0 bins,
  phi 0.5-8, 1 atm, CH4/air), seed 2026 (gen.py used 42 and 1337).
    Cantera : integrates each case from t = 0 to 0.5 s (one advance call), on
              1, 2, ... cores (one case per process at a time). The final state of
              the 1-core run is the reference; the other core counts must give the
              same result (checked).
    DeepONet: one evaluation per case at t = 0.5 s.
    CRK-PINN: rollout over the 100-point time grid of gen.py, built from the
              PREDICTED ignition delay time; the last state is the prediction.
  Timed runs use only predictions: t_end and the time grid come from the IDT
  predictor, and a case is treated as igniting if the predicted IDT < 0.5 s
  (no Cantera information). Timing includes the IDT predictor and host<->GPU copies.
  True ignition label (untimed): as gen.py, a case ignites if max dT/dT > 1e5 K/s
  within 0.5 s (Cantera scout pass). Each model is evaluated twice (untimed for
  the second): label "predicted" (IDT < 0.5 s) and label "true" (from the scout
  pass, as in train_global.py). The difference shows how much error comes from
  getting the ignition label wrong.
  Accuracy (final state only, against Cantera):
    |dT| median / p95 / max
    per species: median |Y_pred - Y_true| / Y_true over cases with final Y_true > 1e-8
                 (largest + tracked species, as in the final-time tables)
    E_final [%] = 100 * mean over {T, tracked species with Y > 1e-8, other
                  species with Y > 1e-3 (excl. N2, AR)} of |pred - true| / true,
                  median / mean / p95 over the cases (as final_state_metric.py)
    label_wrong: cases where "predicted IDT < 0.5 s" differs from the true label
  Also printed: what the IDT predictor returns for cases that do NOT ignite.
  Initial state given to the models ("start"):
    fresh : the fresh CH4/air mixture at t = 0 (timed runs use this)
    t_min : the Cantera state at t = 5e-7 s, i.e. the first saved point of every
            trajectory in gen.py, which is what the models were trained with as
            initial state (untimed; tests whether the fresh mixture is out of
            the training distribution)
  accuracy_A_by_T0.csv: the same accuracy per 300 K T0 range, true label, both starts.

Part B, CFD-like chemistry step (cost only)
  n_cells cell states, each advanced by one step dt_cell.
    Cantera : every cell integrated from its state over dt_cell (set state,
              reinitialize the reactor, advance). For n_cells above
              --max_cantera_cells only that many cells are timed and the time is
              scaled linearly (marked "extrapolated"); cells are independent, so
              the cost is linear in the number of cells.
    CRK-PINN: one forward pass per cell (state, T0, P, dt, t_end).
    DeepONet: one forward pass per cell. NOTE: DeepONet was trained only on fresh
              CH4/air mixtures as initial condition, so here its inputs are fresh
              mixtures (Y0, T0, phi) at t = dt_cell. The cost of a forward pass does
              not depend on the input values, but DeepONet cannot advance an
              arbitrary partially-burnt cell state as trained.
  Cell states are random rows of data_test.csv (states along real trajectories).

Timing
  - wall-clock time, after one warm-up call (CUDA init etc. not counted)
  - GPU: torch.cuda.synchronize() before stopping the clock
  - model loading and Cantera mechanism loading are not counted
Memory
  - GPU: torch.cuda.max_memory_allocated during the call
  - Cantera: sum of the peak resident memory (VmHWM) of the worker processes.
    Cantera always runs in fresh worker processes ('spawn', also for 1 core), so
    they hold only Python + Cantera + the mechanism, not torch or the test data.
  - models on CPU: model weights + the increase in resident memory during the call
    (peak VmHWM minus VmRSS before the call; Linux /proc/self/clear_refs)

Outputs: results/benchmark/
  speed_A.csv, speed_B.csv, accuracy_A.csv, benchmark_config.json
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cantera as ct

# Cantera 3.2 warns about the future default of ReactorBase(clone=...); same behaviour as gen.py
warnings.filterwarnings("ignore", category=DeprecationWarning, module="cantera")
warnings.filterwarnings("ignore", message=".*clone.*")

if __name__ == "__main__":
    # only the main process needs torch; the Cantera worker processes do not load it
    import torch
    from fordypning.deeponet import DeepONetCRK
    from fordypning.models import CRKPINN
    from fordypning.idt_predictor import load_idt_predictor

# same settings as gen.py
MECH, FUEL, OXIDIZER = "gri30.yaml", "CH4", "O2:1, N2:3.76"
T0_RANGE, PHI_RANGE, P_FIXED, SIM_TIME = (500.0, 2000.0), (0.5, 8.0), 101325.0, 0.5
N_BINS_T0 = 10
N_PRE, N_IGN, N_POST = 15, 70, 15
WIN_LO, WIN_HI = 0.3, 3.0
DTDT_THRESHOLD = 1e5
N_TEND = 5                       # t_end = 5 * IDT, as in train_global.py
LARGEST = ["CO", "CH4", "O2", "C2H2", "H2O", "CO2"]   # as in diagnose_global.py
TRACKED = ["CH2O", "H2O2", "OH", "C2H6", "NO"]
species = ct.Solution(MECH).species_names
NS = len(species)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--deeponet_run", default="results/global/deeponet_predicted_b10_s2_conv")
    p.add_argument("--crkpinn_run", default="results/global/crkpinn_predicted_b10_s2_conv")
    p.add_argument("--idt_model", default="model_weights/idt_predictor.pt")
    p.add_argument("--test_csv", default="data/raw/data_test.csv")
    p.add_argument("--n_traj", type=int, default=1000, help="part A: number of new cases")
    p.add_argument("--seed", type=int, default=2026, help="new ICs (gen.py used 42 and 1337)")
    p.add_argument("--cores", default=None, help="Cantera core counts, e.g. 1,2,4,8 (default 1,2,4,... up to all)")
    p.add_argument("--cells", default="1000,10000,100000,1000000", help="part B: numbers of cells")
    p.add_argument("--dt_cell", type=float, default=1e-6, help="part B: chemistry step [s]")
    p.add_argument("--max_cantera_cells", type=int, default=10000,
                   help="part B: Cantera times at most this many cells, then scales linearly")
    p.add_argument("--batch", type=int, default=65536, help="model batch size (rows per forward pass)")
    p.add_argument("--skip_a", action="store_true")
    p.add_argument("--skip_timing", action="store_true",
                   help="part A: Cantera only once on all cores, models only once (accuracy check)")
    p.add_argument("--skip_b", action="store_true")
    p.add_argument("--out", default="results/benchmark")
    return p.parse_args()




# ======================================================================= memory
def reset_peak_rss():
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except OSError:
        pass


def _status_mb(key):
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(key):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def peak_rss_mb():
    return _status_mb("VmHWM:")


def rss_mb():
    return _status_mb("VmRSS:")


# ======================================================================= Cantera
_gas = None


def _init_worker():
    global _gas
    warnings.filterwarnings("ignore", message=".*clone.*")
    _gas = ct.Solution(MECH)
    reset_peak_rss()


def fresh_mixture(gas, T0, phi):
    gas.TP = T0, P_FIXED
    gas.set_equivalence_ratio(phi, FUEL, OXIDIZER)
    gas.TP = T0, P_FIXED


def time_grid(t_ign, ignited):
    """gen.py: 15 points before, 70 around and 15 after ignition (here with the predicted IDT)."""
    t_min = SIM_TIME / 1e6
    if not ignited:
        return np.logspace(np.log10(t_min), np.log10(SIM_TIME), N_PRE + N_IGN + N_POST)
    t_lo, t_hi = max(t_ign * WIN_LO, t_min), min(t_ign * WIN_HI, SIM_TIME)
    pre = np.logspace(np.log10(t_min), np.log10(t_lo), N_PRE, endpoint=False)
    ign = np.linspace(t_lo, t_hi, N_IGN)
    post = np.logspace(np.log10(t_hi), np.log10(SIM_TIME), N_POST + 1)[1:]
    return np.unique(np.concatenate([pre, ign, post]))


def _task_label(a):
    """gen.py's ignition label: max dT/dt > 1e5 K/s within SIM_TIME (free steps)."""
    T0, phi = a
    fresh_mixture(_gas, T0, phi)
    net = ct.ReactorNet([ct.IdealGasConstPressureReactor(_gas)])
    t_prev, T_prev, max_dTdt = 0.0, _gas.T, 0.0
    try:
        for _ in range(5000):
            t = net.step()
            if t > t_prev:
                max_dTdt = max(max_dTdt, (_gas.T - T_prev) / (t - t_prev))
            t_prev, T_prev = t, _gas.T
            if t >= SIM_TIME:
                break
    except ct.CanteraError:
        pass
    return max_dTdt > DTDT_THRESHOLD


def _task_tmin(a):
    """State at t = 5e-7 s (gen.py's first saved point), from the fresh mixture."""
    T0, phi = a
    fresh_mixture(_gas, T0, phi)
    net = ct.ReactorNet([ct.IdealGasConstPressureReactor(_gas)])
    net.advance(SIM_TIME / 1e6)
    return _gas.T, _gas.Y.copy()


def _task_final(a):
    """One case from t = 0 to SIM_TIME. Returns the final (T, Y)."""
    T0, phi = a
    fresh_mixture(_gas, T0, phi)
    net = ct.ReactorNet([ct.IdealGasConstPressureReactor(_gas)])
    try:
        net.advance(SIM_TIME)
    except ct.CanteraError:
        return np.nan, np.full(NS, np.nan)
    return _gas.T, _gas.Y.copy()


def _task_cell(a):
    T, P, Y, dt = a
    _gas.TPY = T, P, Y
    net = ct.ReactorNet([ct.IdealGasConstPressureReactor(_gas)])
    try:
        net.advance(dt)
    except ct.CanteraError:
        pass
    return None


def _task_peak(_):
    return peak_rss_mb()


def time_cantera(task, items, cores):
    """Run task over items with `cores` fresh worker processes.
    Returns (wall time [s], summed peak memory of the workers [MB], results)."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(cores, initializer=_init_worker) as pool:
        pool.map(task, items[:cores], chunksize=1)       # warm-up, every worker
        chunk = max(1, len(items) // (cores * 8))
        t0 = time.perf_counter()
        res = pool.map(task, items, chunksize=chunk)
        sec = time.perf_counter() - t0
        mem = sum(pool.map(_task_peak, range(cores), chunksize=1))
    return sec, mem, res


# ======================================================================= models
def load_run(run_dir, device):
    cfg = json.load(open(Path(run_dir) / "config.json"))
    n_bins = cfg["args"]["n_bins"]
    models = {}
    for k in range(n_bins):
        path = Path(run_dir) / f"model_bin{k}.pt"
        if not path.exists():
            models[k] = None
            continue
        ck = torch.load(path, map_location=device)
        m = (DeepONetCRK if ck["arch"] == "deeponet" else CRKPINN)(**ck["kwargs"]).to(device)
        m.load_state_dict(ck["state_dict"])
        models[k] = m.eval()
    edges = np.linspace(T0_RANGE[0], T0_RANGE[1], n_bins + 1)
    n_params = sum(sum(p.numel() for p in m.parameters()) for m in models.values() if m is not None)
    return models, edges, n_params


def bin_of(T0, edges):
    return np.clip(np.digitize(T0, edges) - 1, 0, len(edges) - 2)


def t_end_of(T0, phi, ignited):
    """t_end = 5 * IDT (IDT predictor); not igniting: IDT = SIM_TIME (as in training)."""
    idt = np.full(len(T0), SIM_TIME)
    if ignited.any():
        idt[ignited] = idt_model.predict_idt(T0[ignited], phi[ignited]).detach().cpu().numpy()
    return (N_TEND * idt).astype(np.float32)


def predicted_ignition(T0, phi):
    """Predicted IDT and ignition label (IDT < SIM_TIME); no Cantera information."""
    idt = idt_model.predict_idt(T0, phi).detach().cpu().numpy().astype(np.float64)
    return idt, idt < SIM_TIME


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def run_timed(fn, device, weights_mb):
    """Warm-up, then time fn() once. Returns (result, seconds, memory MB).
    GPU: peak allocated (weights + activations). CPU: weights + RSS increase during the call."""
    fn()
    sync(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    reset_peak_rss()
    rss0 = rss_mb()
    t0 = time.perf_counter()
    res = fn()
    sync(device)
    sec = time.perf_counter() - t0
    if device == "cuda":
        mem = torch.cuda.max_memory_allocated() / 2**20
    else:
        mem = weights_mb + max(peak_rss_mb() - rss0, 0.0)
    return res, sec, mem


def deeponet_predict(models, edges, device, Y0, T0, phi, P, t, tend):
    """One prediction per row (row = initial condition + query time)."""
    T_out = np.empty(len(t), np.float32)
    Y_out = np.empty((len(t), NS), np.float32)
    b = bin_of(T0, edges)
    g = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
    with torch.no_grad():
        for k, m in models.items():
            r = np.flatnonzero(b == k)
            for i in range(0, len(r), args.batch):
                rr = r[i:i + args.batch]
                Tp, Yp, _ = m(g(Y0[rr]), g(T0[rr]), g(phi[rr]), g(P[rr]), g(t[rr]), g(tend[rr]))
                T_out[rr], Y_out[rr] = Tp.cpu().numpy(), Yp.cpu().numpy()
    return T_out, Y_out


def crkpinn_step(models, edges, device, T, Y, T0, P, dt, tend):
    """One step per row (row = current state + step size)."""
    T_out = np.empty(len(T), np.float32)
    Y_out = np.empty((len(T), NS), np.float32)
    b = bin_of(T0, edges)
    g = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
    with torch.no_grad():
        for k, m in models.items():
            r = np.flatnonzero(b == k)
            for i in range(0, len(r), args.batch):
                rr = r[i:i + args.batch]
                Tp, Yp, _ = m(g(T[rr]), g(Y[rr]), g(T0[rr]), g(P[rr]), g(dt[rr]), g(tend[rr]))
                T_out[rr], Y_out[rr] = Tp.cpu().numpy(), Yp.cpu().numpy()
    return T_out, Y_out


def crkpinn_rollout(models, edges, device, T_init, Y_init, T0, P, grids, tend):
    """Rollout over each trajectory's grid; trajectories with the same length together.
    As in train_global.py: the state at grid point 0 (5e-7 s) is the initial state,
    every later point is one model step from the previous prediction."""
    n = len(T0)
    L = np.array([len(gr) for gr in grids])
    T_pred = [None] * n
    Y_pred = [None] * n
    for Lv in np.unique(L):
        sel = np.flatnonzero(L == Lv)
        G = np.stack([grids[j] for j in sel])                       # (m, Lv)
        steps = np.diff(G, axis=1).astype(np.float32)               # (m, Lv - 1)
        Tn, Yn = T_init[sel].astype(np.float32), Y_init[sel].astype(np.float32)
        Ts, Ys = np.empty((len(sel), Lv), np.float32), np.empty((len(sel), Lv, NS), np.float32)
        Ts[:, 0], Ys[:, 0] = Tn, Yn
        for j in range(1, Lv):
            Tn, Yn = crkpinn_step(models, edges, device, Tn, Yn, T0[sel], P[sel], steps[:, j - 1], tend[sel])
            Ts[:, j], Ys[:, j] = Tn, Yn
        for i, jj in enumerate(sel):
            T_pred[jj], Y_pred[jj] = Ts[i], Ys[i]
    return T_pred, Y_pred


# ======================================================================= info
def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    global args, idt_model, CORES, DEVICES
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_cpu = os.cpu_count()
    if args.cores:
        CORES = [int(c) for c in args.cores.split(",")]
    else:
        CORES = sorted({c for c in [1, 2, 4, 8, 16, 32, 64] if c <= n_cpu} | {n_cpu})
    DEVICES = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    idt_model = load_idt_predictor(args.idt_model)

    info = {"cpu": cpu_name(), "cpu_count": n_cpu,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "cantera": ct.__version__,
            "torch_cpu_threads": torch.get_num_threads(), "args": vars(args)}
    print(json.dumps({k: v for k, v in info.items() if k != "args"}, indent=1), flush=True)

    # ======================================================================= part A
    if not args.skip_a:
        print(f"\n=== Part A: {args.n_traj} new cases, state at t = {SIM_TIME} s ===", flush=True)
        rng = np.random.default_rng(args.seed)
        edges10 = np.linspace(*T0_RANGE, N_BINS_T0 + 1)
        per_bin = int(np.ceil(args.n_traj / N_BINS_T0))
        T0s = np.concatenate([rng.uniform(edges10[b], edges10[b + 1], per_bin)
                              for b in range(N_BINS_T0)])[:args.n_traj]
        phis = rng.uniform(*PHI_RANGE, len(T0s))
        n = len(T0s)
        gas = ct.Solution(MECH)
        Y0s = np.empty((n, NS), np.float32)
        for i in range(n):
            fresh_mixture(gas, T0s[i], phis[i])
            Y0s[i] = gas.Y

        rows_A = []
        # ---- Cantera, 0 -> 0.5 s, on 1, 2, ... cores
        items = list(zip(T0s.tolist(), phis.tolist()))
        T_ref = Y_ref = None
        for c in ([CORES[-1]] if args.skip_timing else CORES):
            sec, mem, res = time_cantera(_task_final, items, c)
            T_c = np.array([r[0] for r in res])
            Y_c = np.array([r[1] for r in res])
            if T_ref is None:
                T_ref, Y_ref, check = T_c, Y_c, 0.0
            else:
                check = float(np.nanmax(np.abs(T_c - T_ref)))
            rows_A.append({"method": "Cantera", "device": f"CPU, {c} core{'s' if c > 1 else ''}",
                           "total_s": sec, "ms_per_case": 1e3 * sec / n, "peak_mem_MB": mem,
                           "max_dT_vs_1st_run_K": check})
            print(f"  Cantera {c} cores: {sec:.1f} s", flush=True)
        n_fail = int(np.isnan(T_ref).sum())

        # ---- true ignition label (untimed), all cores
        _, _, lab = time_cantera(_task_label, items, CORES[-1])
        ign_true = np.array(lab, bool)
        print(f"  true label: {ign_true.sum()} of {n} ignite (max dT/dt > {DTDT_THRESHOLD:g} K/s)", flush=True)
        _, _, res = time_cantera(_task_tmin, items, CORES[-1])
        T_tmin = np.array([r[0] for r in res], np.float32)
        Y_tmin = np.array([r[1] for r in res], np.float32)

        # ---- what the IDT predictor says
        T0f, phif = T0s.astype(np.float32), phis.astype(np.float32)
        idt_all, ign_pred_all = predicted_ignition(T0f, phif)
        q = np.percentile(idt_all[~ign_true], [0, 5, 50, 95, 100]) if (~ign_true).any() else [np.nan] * 5
        print(f"  IDT predictor on cases that do NOT ignite ({(~ign_true).sum()}): predicted IDT "
              f"min {q[0]:.2e}, p5 {q[1]:.2e}, median {q[2]:.2e}, p95 {q[3]:.2e}, max {q[4]:.2e} s; "
              f"{(ign_pred_all & ~ign_true).sum()} predicted to ignite (IDT < {SIM_TIME} s)")
        print(f"  IDT predictor on cases that ignite ({ign_true.sum()}): "
              f"{(~ign_pred_all & ign_true).sum()} predicted NOT to ignite", flush=True)

        # ---- models
        Pf = np.full(n, P_FIXED, np.float32)
        t_final = np.full(n, SIM_TIME, np.float32)
        excl = [species.index(s_) for s_ in ("N2", "AR")]
        tracked = [species.index(s_) for s_ in TRACKED]
        mask = Y_ref > 1e-3
        mask[:, tracked] = Y_ref[:, tracked] > 1e-8
        mask[:, excl] = False
        ok = ~np.isnan(T_ref)

        def accuracy(name, label_name, Tp, Yp, ign_used):
            dT = np.abs(Tp - T_ref)[ok]
            eT = np.abs(Tp - T_ref) / T_ref
            eY = np.abs(Yp - Y_ref) / np.where(mask, Y_ref, 1.0)
            E = (100 * (eT + (eY * mask).sum(1)) / (1 + mask.sum(1)))[ok]
            r = {"method": name, "label": label_name,
                 "label_wrong": int((ign_used != ign_true)[ok].sum()),
                 "dT_med_K": np.median(dT), "dT_p95_K": np.percentile(dT, 95), "dT_max_K": dT.max(),
                 "E_final_med_%": np.median(E), "E_final_mean_%": E.mean(), "E_final_p95_%": np.percentile(E, 95)}
            for sp in LARGEST + TRACKED:
                k = species.index(sp)
                m = ok & (Y_ref[:, k] > 1e-8)
                r[sp] = 100 * np.median(np.abs(Yp[m, k] - Y_ref[m, k]) / Y_ref[m, k]) if m.any() else np.nan
            return r

        def predict(name, models, edges, device, ign, start="fresh"):
            """Final state for all cases with a given ignition label (IDT from the predictor).
            start: initial state given to the model, 'fresh' (t = 0) or 't_min' (t = 5e-7 s)."""
            Y_in, T_in = (Y0s, T0f) if start == "fresh" else (Y_tmin, T_tmin)
            idt, _ = predicted_ignition(T0f, phif)
            tend = t_end_of(T0f, phif, ign)
            if name == "DeepONet":
                return deeponet_predict(models, edges, device, Y_in, T0f, phif, Pf, t_final, tend)
            grids = [time_grid(idt[i], ign[i]) for i in range(n)]
            T_p, Y_p = crkpinn_rollout(models, edges, device, T_in, Y_in, T0f, Pf, grids, tend)
            return np.array([a_[-1] for a_ in T_p]), np.array([a_[-1] for a_ in Y_p])

        rng_edges = np.linspace(*T0_RANGE, 6)                       # 300 K ranges
        rng_of = np.clip(np.digitize(T0s, rng_edges) - 1, 0, 4)
        by_T0 = []

        def accuracy_by_T0(name, start, Tp, Yp):
            eT = np.abs(Tp - T_ref) / T_ref
            eY = np.abs(Yp - Y_ref) / np.where(mask, Y_ref, 1.0)
            E = 100 * (eT + (eY * mask).sum(1)) / (1 + mask.sum(1))
            dT = np.abs(Tp - T_ref)
            for r in range(5):
                m = ok & (rng_of == r)
                by_T0.append({"method": name, "start": start,
                              "T0 range": f"[{rng_edges[r]:.0f},{rng_edges[r + 1]:.0f})", "n": int(m.sum()),
                              "n_ign": int((m & ign_true).sum()),
                              "dT_med_K": np.median(dT[m]), "dT_p95_K": np.percentile(dT[m], 95),
                              "E_final_med_%": np.median(E[m])})

        acc_rows = []
        for name, run in [("DeepONet", args.deeponet_run), ("CRK-PINN", args.crkpinn_run)]:
            for device in (DEVICES[:1] if args.skip_timing else DEVICES):
                models, edges, n_params = load_run(run, device)

                def fn():                                   # timed: predictions only
                    _, ign = predicted_ignition(T0f, phif)
                    return predict(name, models, edges, device, ign), ign
                ((Tp, Yp), ign_pred), sec, mem = run_timed(fn, device, 4 * n_params / 2**20)
                rows_A.append({"method": name, "device": device.upper() if device == "cuda" else
                               f"CPU, {torch.get_num_threads()} threads",
                               "total_s": sec, "ms_per_case": 1e3 * sec / n, "peak_mem_MB": mem,
                               "n_params": n_params})
                print(f"  {name} {device}: {sec:.3f} s", flush=True)

                if device == DEVICES[0]:                     # accuracy once per model
                    acc_rows.append({"start": "fresh", **accuracy(name, "predicted", Tp, Yp, ign_pred)})
                    for start in ["fresh", "t_min"]:
                        with torch.no_grad():
                            Tt, Yt = predict(name, models, edges, device, ign_true, start)
                        acc_rows.append({"start": start, **accuracy(name, "true", Tt, Yt, ign_true)})
                        accuracy_by_T0(name, start, Tt, Yt)
                del models
                if device == "cuda":
                    torch.cuda.empty_cache()

        A = pd.DataFrame(rows_A)
        ref = A.loc[(A.method == "Cantera") & (A.device == "CPU, 1 core"), "total_s"]
        if len(ref):
            A["speedup_vs_cantera_1core"] = ref.iloc[0] / A["total_s"]
        A.insert(1, "n_cases", n)
        A.to_csv(out / "speed_A.csv", index=False)
        acc = pd.DataFrame(acc_rows)
        acc.to_csv(out / "accuracy_A.csv", index=False)
        pd.set_option("display.width", 220)
        print(f"\nPart A ({n} cases, {int(ign_true.sum())} ignite, {n_fail} Cantera failures):")
        print(A.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        print("\nfinal-state accuracy against Cantera (species columns: median relative error [%]):")
        print(acc.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        bt = pd.DataFrame(by_T0)
        bt.to_csv(out / "accuracy_A_by_T0.csv", index=False)
        print("\nper T0 range (true label):")
        print(bt.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    # ======================================================================= part B
    if not args.skip_b:
        cells = [int(c) for c in args.cells.split(",")]
        print(f"\n=== Part B: one chemistry step dt = {args.dt_cell:g} s, cells {cells} ===", flush=True)
        y_cols = [f"Y_{s}" for s in species]
        rng = np.random.default_rng(args.seed + 1)
        n_max = max(cells)
        test = pd.read_csv(args.test_csv, usecols=["T", "P", "T0", "phi", "ignited"] + y_cols)
        pick = rng.choice(len(test), n_max, replace=n_max > len(test))
        T_c = test["T"].to_numpy(np.float32)[pick]
        P_c = test["P"].to_numpy(np.float32)[pick]
        T0_c = test["T0"].to_numpy(np.float32)[pick]
        phi_c = test["phi"].to_numpy(np.float32)[pick]
        ign_c = test["ignited"].to_numpy().astype(bool)[pick]
        Y_c = test[y_cols].to_numpy(np.float32)[pick]
        del test
        gas = ct.Solution(MECH)
        Y0_c = np.empty_like(Y_c)
        for i, (a, b) in enumerate(zip(T0_c, phi_c)):           # fresh mixture for DeepONet
            fresh_mixture(gas, float(a), float(b))
            Y0_c[i] = gas.Y
        dt_c = np.full(n_max, args.dt_cell, np.float32)

        rows_B = []
        for n in cells:
            # Cantera
            n_ct = min(n, args.max_cantera_cells)
            items = [(float(T_c[i]), float(P_c[i]), Y_c[i].astype(np.float64), args.dt_cell) for i in range(n_ct)]
            for c in CORES:
                sec, mem, _ = time_cantera(_task_cell, items, c)
                rows_B.append({"n_cells": n, "method": "Cantera", "device": f"CPU, {c} core{'s' if c > 1 else ''}",
                               "total_s": sec * n / n_ct, "extrapolated": n_ct < n,
                               "us_per_cell": 1e6 * sec / n_ct, "peak_mem_MB": mem})
            # models
            for name, run in [("DeepONet", args.deeponet_run), ("CRK-PINN", args.crkpinn_run)]:
                for device in DEVICES:
                    models, edges, n_params = load_run(run, device)
                    if name == "DeepONet":
                        fn = lambda: deeponet_predict(models, edges, device, Y0_c[:n], T0_c[:n], phi_c[:n], P_c[:n],
                                                      dt_c[:n], t_end_of(T0_c[:n], phi_c[:n], ign_c[:n]))
                    else:
                        fn = lambda: crkpinn_step(models, edges, device, T_c[:n], Y_c[:n], T0_c[:n], P_c[:n],
                                                  dt_c[:n], t_end_of(T0_c[:n], phi_c[:n], ign_c[:n]))
                    _, sec, mem = run_timed(fn, device, 4 * n_params / 2**20)
                    rows_B.append({"n_cells": n, "method": name,
                                   "device": device.upper() if device == "cuda" else f"CPU, {torch.get_num_threads()} threads",
                                   "total_s": sec, "extrapolated": False, "us_per_cell": 1e6 * sec / n,
                                   "peak_mem_MB": mem})
                    del models
                    if device == "cuda":
                        torch.cuda.empty_cache()
            print(f"  {n} cells done", flush=True)

        B = pd.DataFrame(rows_B)
        B["speedup_vs_cantera_1core"] = np.nan
        for n in cells:
            ref = B.loc[(B.n_cells == n) & (B.method == "Cantera") & (B.device == "CPU, 1 core"), "total_s"]
            if len(ref):
                B.loc[B.n_cells == n, "speedup_vs_cantera_1core"] = ref.iloc[0] / B.loc[B.n_cells == n, "total_s"]
        B.to_csv(out / "speed_B.csv", index=False)
        pd.set_option("display.width", 220)
        print("\nPart B:")
        print(B.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    json.dump(info, open(out / "benchmark_config.json", "w"), indent=2, default=str)
    print(f"\nsaved to {out}/")


if __name__ == "__main__":
    main()