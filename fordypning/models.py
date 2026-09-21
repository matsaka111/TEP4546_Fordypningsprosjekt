# fordypning/models.py
import torch
import torch.nn as nn


class CRKPINN(nn.Module):
    """
    Follows Zhang, Zhang & Wang (2024) "CRK-PINN" as closely as practical for
    the GRI-30 CH4/air system:
      - Plain MLP (no residual connections), ELU activation
      - Inputs: log-Z-scored species mass fractions, Z-scored T, Z-scored T0,
        log-Z-scored P, Z-scored dt/t_end (t_end = n * IDT)
      - Outputs: log-Z-scored species mass fractions and Z-scored T at t+dt
      - Trained with data loss + soft physics losses.

    residual=True switches the network to predict a DELTA on top of the
    current normalized state, rather than the absolute next normalized
    state. Used for the bin1_noignite model: when the target is "almost no
    change", a plain MLP has to learn to approximate an identity mapping
    through its nonlinearities from scratch, which is a much harder function
    to fit precisely than it sounds -- gradients near that solution are
    small, and any small per-step bias compounds badly under autoregressive
    rollout. Predicting a delta instead means the trivial "nothing happens"
    case only requires the network to output ~0, which is easy.
    """

    Y_MIN = 1e-10  # clipped-log floor

    def __init__(self, n_species, hidden=128, n_hidden_layers=3, residual=False):
        super().__init__()
        self.n_species = n_species
        self.residual = residual

        in_dim = n_species + 4
        out_dim = n_species + 1

        layers = [nn.Linear(in_dim, hidden), nn.ELU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ELU()]
        final_layer = nn.Linear(hidden, out_dim)
        layers.append(final_layer)
        self.net = nn.Sequential(*layers)

        if residual:
            # Zero-init the final layer: at init, out ~= 0, so Yn_next = Yn + 0
            # and Tn_next = Tn + 0 -- the network starts as an exact identity
            # map (predicts "nothing changes"), which is the correct answer
            # for most of bin1_noignite's data. Without this, an untrained
            # network's raw output is an arbitrary, potentially large value,
            # and MSE against that unclamped sum in loss_data can blow up
            # before the network ever gets a chance to learn.
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

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

    def normalize_Y(self, Y):
        logY = torch.log(torch.clamp(Y, min=self.Y_MIN))
        return (logY - self.logY_mean) / self.logY_std

    def denormalize_Y(self, Yn):
        logY = Yn * self.logY_std + self.logY_mean
        logY = torch.clamp(logY, max=80.0)  # prevent exp() overflow -> inf -> nan via grad clipping
        return torch.exp(logY)

    def normalize_T(self, T):
        return (T - self.T_mean) / self.T_std

    def denormalize_T(self, Tn):
        return Tn * self.T_std + self.T_mean

    def normalize_T0(self, T0):
        return (T0 - self.T0_mean) / self.T0_std

    def normalize_P(self, P):
        logP = torch.log(torch.clamp(P, min=1e-6))
        return (logP - self.logP_mean) / self.logP_std

    def normalize_tprime(self, t, t_end):
        raw = t / torch.clamp(t_end, min=1e-12)
        return (raw - self.tprime_mean) / self.tprime_std

    def forward(self, T, Y, T0, P, dt, t_end):
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

        if self.residual:
            Yn_next = Yn + out[:, :self.n_species]
            Tn_next = Tn + out[:, self.n_species]
        else:
            Yn_next = out[:, :self.n_species]
            Tn_next = out[:, self.n_species]

        # safety clamp on normalized predictions themselves -- prevents any
        # runaway value (residual sum or otherwise) from reaching loss_data's
        # MSE unclamped, independent of the physical-space clamps below
        Yn_next = torch.clamp(Yn_next, min=-20.0, max=20.0)
        Tn_next = torch.clamp(Tn_next, min=-20.0, max=20.0)

        Y_next = torch.clamp(self.denormalize_Y(Yn_next), min=0.0, max=1.0)
        T_next = torch.clamp(self.denormalize_T(Tn_next), min=200.0, max=6000.0)

        return T_next, Y_next, {"Tn_next": Tn_next, "Yn_next": Yn_next}