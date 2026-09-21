import torch
import torch.nn as nn

R = 8.314462618  # J/(mol*K)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class ArrheniusStateSurrogate(nn.Module):
    """
    Hybrid of ArrheniusNN and StateTransitionSurrogate.

    Species update: keeps the ORIGINAL Arrhenius structure -- the network
    predicts A, b, Ea, and stoichiometric coefficients nu (with mass
    conservation structurally guaranteed via the free/dependent species
    split, exactly as in ArrheniusNN), giving a physically-structured wdot.
    One explicit Euler step then advances the species state:

        Y_next = Y_t + (wdot * mw / rho_tot) * dt

    Because sum_i(wdot_i * mw_i) = 0 is guaranteed by construction, and
    Y_t sums to 1 in the data, Y_next sums to exactly 1 as well -- the
    mass-fraction-sum-to-1 constraint falls out for free, no soft loss
    term needed for it (unlike StateTransitionSurrogate).

    Temperature update: predicted directly by a separate output head, NOT
    derived from wdot via energy conservation. A fully physical Tdot would
    require evaluating NASA polynomial enthalpies/cv inside the model
    (what Cantera does in generate_data_simple.py's compute step) -- left
    out here for simplicity. This is a real approximation: T_next is
    learned freely, only the species update is physically structured.
    Worth revisiting if the T-prediction turns out to be the bottleneck.

    Loss should be computed in NORMALIZED STATE SPACE (Z-scored log-Y,
    Z-scored T) against the true next state, exactly like
    StateTransitionSurrogate -- NOT in raw log|wdot| space. This is what
    fixes the original dynamic-range/data-imbalance problem, while keeping
    hard mass conservation.
    """

    Y_MIN = 1e-10

    def __init__(self, n_species, molecular_weights, hidden=512, n_layers=8, eps=1e-30):
        super().__init__()
        self.n_species = n_species
        self.register_buffer("mw", torch.tensor(molecular_weights, dtype=torch.float32))
        self.eps = eps

        self.dependent_idx = int(torch.argmax(self.mw).item())
        self.free_idx = [i for i in range(n_species) if i != self.dependent_idx]

        # input: T, log-Y (n_species), dt  -->  hidden
        in_dim = 1 + n_species + 1
        self.input_layer = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(n_layers - 2)])

        # output: A, b, Ea, nu_free (n_species - 1), dT_raw (1)
        out_dim = 3 + (n_species - 1) + 1
        self.output_layer = nn.Linear(hidden, out_dim)

        EA_INIT = 40000.0
        with torch.no_grad():
            last_layer = self.output_layer
            last_layer.weight[1, :] *= 0.1
            last_layer.bias[1] *= 0.1
            last_layer.weight[2, :] *= 0.1
            last_layer.bias[2] = EA_INIT
            last_layer.bias[3:3 + n_species - 1] = torch.zeros(n_species - 1)
            last_layer.weight[3:3 + n_species - 1, :] *= 0.3
            # dT output: start near zero so the model defaults to "no change"
            # (same reasoning as the residual-prediction fix for the drift problem)
            last_layer.weight[-1, :] *= 0.01
            last_layer.bias[-1] *= 0.01

        self.b_max = 6.0
        self.nu_max = 4.0

        # normalization stats -- fit on TRAIN split only via set_norm_stats
        self.register_buffer("logY_mean", torch.zeros(n_species))
        self.register_buffer("logY_std", torch.ones(n_species))
        self.register_buffer("T_mean", torch.tensor(0.0))
        self.register_buffer("T_std", torch.tensor(1.0))
        self.register_buffer("logdt_mean", torch.tensor(0.0))
        self.register_buffer("logdt_std", torch.tensor(1.0))
        self.register_buffer("dT_std", torch.tensor(1.0))  # scale for the raw dT output

    def set_norm_stats(self, logY_mean, logY_std, T_mean, T_std, logdt_mean, logdt_std, dT_std):
        self.logY_mean.copy_(torch.as_tensor(logY_mean))
        self.logY_std.copy_(torch.as_tensor(logY_std).clamp_min(1e-8))
        self.T_mean.copy_(torch.as_tensor(T_mean))
        self.T_std.copy_(torch.as_tensor(T_std).clamp_min(1e-8))
        self.logdt_mean.copy_(torch.as_tensor(logdt_mean))
        self.logdt_std.copy_(torch.as_tensor(logdt_std).clamp_min(1e-8))
        self.dT_std.copy_(torch.as_tensor(dT_std).clamp_min(1e-8))

    def normalize_Y(self, Y):
        logY = torch.log(torch.clamp(Y, min=self.Y_MIN))
        return (logY - self.logY_mean) / self.logY_std

    def normalize_T(self, T):
        return (T - self.T_mean) / self.T_std

    def denormalize_T(self, Tn):
        return Tn * self.T_std + self.T_mean

    def normalize_dt(self, dt):
        logdt = torch.log(torch.clamp(dt, min=1e-12))
        return (logdt - self.logdt_mean) / self.logdt_std

    def _assemble_nu(self, nu_free):
        batch = nu_free.shape[0]
        nu = torch.zeros(batch, self.n_species, device=nu_free.device, dtype=nu_free.dtype)
        free_idx = torch.tensor(self.free_idx, device=nu_free.device)
        nu[:, free_idx] = nu_free

        mw_free = self.mw[free_idx]
        mw_dep = self.mw[self.dependent_idx]
        nu_dep = -(nu_free * mw_free.unsqueeze(0)).sum(dim=-1) / mw_dep
        nu[:, self.dependent_idx] = nu_dep
        return nu

    def forward(self, T, Y, rho_tot, dt):
        """
        T: (batch,) current temperature
        Y: (batch, n_species) current mass fractions
        rho_tot: (batch,) current mixture density (kg/m3)
        dt: (batch,) time step

        Returns: T_next, Y_next (physical units), plus info dict with
        intermediate quantities (wdot, nu, A, b, Ea) for inspection.
        """
        Tn = self.normalize_T(T)
        logYn = self.normalize_Y(Y)
        dtn = self.normalize_dt(dt)

        x = torch.cat([Tn.unsqueeze(-1), logYn, dtn.unsqueeze(-1)], dim=-1)

        h = nn.functional.silu(self.input_layer(x))
        for block in self.blocks:
            h = block(h)
        out = self.output_layer(h)

        A = nn.functional.softplus(out[:, 0]) + 1e-12
        b = self.b_max * torch.tanh(out[:, 1])
        Ea = nn.functional.softplus(out[:, 2])
        nu_free = self.nu_max * torch.tanh(out[:, 3:3 + self.n_species - 1] / self.nu_max)
        dT_raw = out[:, -1]

        nu = self._assemble_nu(nu_free)

        log_k = torch.log(A) + b * torch.log(300.0 / T) - Ea / (R * T)  # T_ref fixed at 300K here

        rho_species = Y * rho_tot.unsqueeze(-1)
        C = rho_species / self.mw
        order = torch.relu(-nu)
        log_C = torch.log(torch.clamp(C, min=1e-20))

        log_rate_term = (order * log_C).sum(dim=-1)
        log_q = log_k + log_rate_term

        log_nu_mag = torch.log(torch.abs(nu) + self.eps)
        log_wdot_mag = log_nu_mag + log_q.unsqueeze(-1)
        wdot_sign = torch.sign(nu)
        wdot = wdot_sign * torch.exp(torch.clamp(log_wdot_mag, max=80.0))  # kmol/m3/s

        Ydot = wdot * self.mw.unsqueeze(0) / rho_tot.unsqueeze(-1)

        # ---- Euler step, with the update itself clamped ----
        # Unclamped, an early/bad wdot prediction multiplied by a large dt can
        # produce an astronomical Y_next (or dT), which blows up the loss and
        # teaches the network to collapse to "predict zero change everywhere"
        # just to avoid ever triggering that instability again. Clamping the
        # STEP (not just the final state) to a physically plausible range
        # keeps a bad early prediction from ever being catastrophic, so
        # gradients stay informative instead of pushing toward collapse.
        dY = torch.clamp(Ydot * dt.unsqueeze(-1), min=-1.0, max=1.0)
        Y_next = torch.clamp(Y + dY, min=0.0)

        dT = torch.clamp(dT_raw * self.dT_std * dt, min=-3000.0, max=3000.0)
        T_next = T + dT

        return T_next, Y_next, {
            "wdot": wdot, "Ydot": Ydot, "nu": nu, "A": A, "b": b, "Ea": Ea,
        }