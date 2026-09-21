import torch
import torch.nn as nn


class CRKPINN(nn.Module):
    """
    Follows Zhang, Zhang & Wang (2024) "CRK-PINN" as closely as practical for
    the GRI-30 CH4/air system:
      - Plain MLP (no residual connections), ELU activation (their Appendix C
        ablation found ELU outperforms ReLU/LeakyReLU/Swish/Tanh)
      - Inputs: log-Z-scored species mass fractions, Z-scored T, Z-scored T0,
        log-Z-scored P, Z-scored dt/t_end (t_end = n * IDT)
      - Outputs: log-Z-scored species mass fractions and Z-scored T at t+dt
      - Trained with data loss + soft physics losses (mole-fraction-sum-to-1,
        element conservation, enthalpy conservation, computed outside this
        class in the training notebook). L_ODEs (autodiff-vs-mass-action
        consistency) is NOT implemented -- would require reimplementing the
        full GRI-30 rate-constant machinery as differentiable PyTorch code.
    """

    Y_MIN = 1e-10  # clipped-log floor

    def __init__(self, n_species, hidden=128, n_hidden_layers=3):
        super().__init__()
        self.n_species = n_species

        # input: log-Y (n_species), T, T0, P, dt' -> n_species + 4
        in_dim = n_species + 4
        # output: log-Y_next (n_species), T_next -> n_species + 1
        out_dim = n_species + 1

        layers = [nn.Linear(in_dim, hidden), nn.ELU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

        # ---- normalization stats, fit on TRAIN split only via set_norm_stats ----
        self.register_buffer("logY_mean", torch.zeros(n_species))
        self.register_buffer("logY_std", torch.ones(n_species))
        self.register_buffer("T_mean", torch.tensor(0.0))
        self.register_buffer("T_std", torch.tensor(1.0))
        self.register_buffer("T0_mean", torch.tensor(0.0))
        self.register_buffer("T0_std", torch.tensor(1.0))
        self.register_buffer("logP_mean", torch.tensor(0.0))
        self.register_buffer("logP_std", torch.tensor(1.0))
        self.register_buffer("tprime_mean", torch.tensor(0.0))
        self.register_buffer("tprime_std", torch.tensor(1.0))

    def set_norm_stats(self, logY_mean, logY_std, T_mean, T_std,
                        T0_mean, T0_std, logP_mean, logP_std,
                        tprime_mean, tprime_std):
        """Fit every one of these on the TRAINING split only, then reuse
        for val/test -- fitting on the full dataset would leak information
        about the held-out distribution into normalization.

        logY_std is clamped to 1e-3, not the more typical 1e-8: a species
        with near-zero log-variance (e.g. a trace species that's almost
        always the same tiny value across the whole dataset) combined with
        an 1e-8 floor produces normalized values in the hundreds/thousands
        for that species -- large enough to overflow through the network's
        Linear/ELU layers into inf/nan on the very first forward pass, even
        before any training happens. 1e-3 keeps normalized magnitudes sane
        for every species at the cost of slightly compressing the dynamic
        range for those near-constant species (an acceptable trade -- a
        species that barely varies contributes little training signal
        anyway)."""
        self.logY_mean.copy_(torch.as_tensor(logY_mean))
        self.logY_std.copy_(torch.as_tensor(logY_std).clamp_min(1e-3))
        self.T_mean.copy_(torch.as_tensor(T_mean))
        self.T_std.copy_(torch.as_tensor(T_std).clamp_min(1e-3))
        self.T0_mean.copy_(torch.as_tensor(T0_mean))
        self.T0_std.copy_(torch.as_tensor(T0_std).clamp_min(1e-3))
        self.logP_mean.copy_(torch.as_tensor(logP_mean))
        self.logP_std.copy_(torch.as_tensor(logP_std).clamp_min(1e-3))
        self.tprime_mean.copy_(torch.as_tensor(tprime_mean))
        self.tprime_std.copy_(torch.as_tensor(tprime_std).clamp_min(1e-3))

    # ---------------- normalization helpers (paper Eq. 15a-15d) ----------------
    def normalize_Y(self, Y):
        """Eq. 15a: X_k' = (ln(X_k) - mu[ln(X_k)]) / sigma[ln(X_k)]"""
        logY = torch.log(torch.clamp(Y, min=self.Y_MIN))
        return (logY - self.logY_mean) / self.logY_std

    def denormalize_Y(self, Yn):
        logY = Yn * self.logY_std + self.logY_mean
        return torch.exp(logY)

    def normalize_T(self, T):
        """Eq. 15c: T' = (T - mu[T]) / sigma[T]  -- plain Z-score, no log"""
        return (T - self.T_mean) / self.T_std

    def denormalize_T(self, Tn):
        return Tn * self.T_std + self.T_mean

    def normalize_T0(self, T0):
        return (T0 - self.T0_mean) / self.T0_std

    def normalize_P(self, P):
        """Eq. 15b: P' = (ln(P) - mu[ln(P)]) / sigma[ln(P)]"""
        logP = torch.log(torch.clamp(P, min=1e-6))
        return (logP - self.logP_mean) / self.logP_std

    def normalize_tprime(self, t, t_end):
        """Eq. 15d: t' = (t/t_end - mu[t/t_end]) / sigma[t/t_end]
        Called with dt (the numerical step size) as `t`, per the paper's
        Eq. 17 input vector which lists Delta-t', not absolute t'."""
        raw = t / torch.clamp(t_end, min=1e-12)
        return (raw - self.tprime_mean) / self.tprime_std

    # ---------------- forward ----------------
    def forward(self, T, Y, T0, P, dt, t_end):
        """
        T:     (batch,) current temperature
        Y:     (batch, n_species) current mass fractions
        T0:    (batch,) initial temperature of this trajectory
        P:     (batch,) pressure
        dt:    (batch,) numerical time step to advance by
        t_end: (batch,) n * IDT for this trajectory -- used only to normalize dt

        Returns: T_next (batch,), Y_next (batch, n_species) in PHYSICAL
        units (clamped to a physically sane range), plus a dict with the raw
        normalized prediction (needed to compute the loss in normalized space).
        """
        Yn = self.normalize_Y(Y)
        Tn = self.normalize_T(T)
        T0n = self.normalize_T0(T0)
        Pn = self.normalize_P(P)
        dtn = self.normalize_tprime(dt, t_end)

        x = torch.cat([
            Yn, Tn.unsqueeze(-1), T0n.unsqueeze(-1),
            Pn.unsqueeze(-1), dtn.unsqueeze(-1),
        ], dim=-1)

        out = self.net(x)

        Yn_next = out[:, :self.n_species]
        Tn_next = out[:, self.n_species]

        # Clamp denormalized outputs to a physically sane range. An untrained
        # (or early-training) network's raw output, once denormalized, can
        # land far outside anything physical (e.g. T ~ 6000K on the very
        # first batch) -- not infinite on its own, but large enough that
        # downstream ops (Y_next.sum(), T_next**4 in the enthalpy loss)
        # overflow float32 and produce inf/nan, which then permanently
        # poisons the model via backward(). Clamping here costs nothing once
        # training is sane (predictions stay well inside these bounds), but
        # prevents that early-training blowup entirely.
        Y_next = torch.clamp(self.denormalize_Y(Yn_next), min=0.0, max=1.0)
        T_next = torch.clamp(self.denormalize_T(Tn_next), min=200.0, max=6000.0)

        return T_next, Y_next, {"Tn_next": Tn_next, "Yn_next": Yn_next}