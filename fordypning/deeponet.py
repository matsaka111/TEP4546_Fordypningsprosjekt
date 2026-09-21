import torch
import torch.nn as nn


class DeepONetCRK(nn.Module):
    """
    Branch net encodes the fixed initial condition of a trajectory
    (Y0, T0, phi, P) into a latent vector. Trunk net encodes the query
    time t. Output is their dot product per channel (T + n_species).
    No autoregressive chaining, state(t) is one forward pass, so
    per-step compounding error (the reason CRKPINN needed K-step rollout
    training) cannot occur here by construction.
    """

    Y_MIN = 1e-10

    def __init__(self, n_species, p=64, branch_hidden=128, trunk_hidden=64,
                 branch_layers=3, trunk_layers=3):
        super().__init__()
        self.n_species = n_species
        self.n_outputs = n_species + 1
        self.p = p

        branch_in = n_species + 3  # Y0, T0, phi, P
        b_layers = [nn.Linear(branch_in, branch_hidden), nn.ELU()]
        for _ in range(branch_layers - 1):
            b_layers += [nn.Linear(branch_hidden, branch_hidden), nn.ELU()]
        b_layers += [nn.Linear(branch_hidden, p * self.n_outputs)]
        self.branch = nn.Sequential(*b_layers)

        t_layers = [nn.Linear(1, trunk_hidden), nn.ELU()]
        for _ in range(trunk_layers - 1):
            t_layers += [nn.Linear(trunk_hidden, trunk_hidden), nn.ELU()]
        t_layers += [nn.Linear(trunk_hidden, p), nn.ELU()]
        self.trunk = nn.Sequential(*t_layers)

        self.bias = nn.Parameter(torch.zeros(self.n_outputs))

        self.register_buffer("logY0_mean", torch.zeros(n_species))
        self.register_buffer("logY0_std", torch.ones(n_species))
        self.register_buffer("T0_mean", torch.tensor(0.0))
        self.register_buffer("T0_std", torch.tensor(1.0))
        self.register_buffer("phi_mean", torch.tensor(0.0))
        self.register_buffer("phi_std", torch.tensor(1.0))
        self.register_buffer("logP_mean", torch.tensor(0.0))
        self.register_buffer("logP_std", torch.tensor(1.0))
        self.register_buffer("tprime_mean", torch.tensor(0.0))
        self.register_buffer("tprime_std", torch.tensor(1.0))
        self.register_buffer("logY_mean", torch.zeros(n_species))
        self.register_buffer("logY_std", torch.ones(n_species))
        self.register_buffer("T_mean", torch.tensor(0.0))
        self.register_buffer("T_std", torch.tensor(1.0))

    def set_norm_stats(self, logY0_mean, logY0_std, T0_mean, T0_std,
                        phi_mean, phi_std, logP_mean, logP_std,
                        tprime_mean, tprime_std, logY_mean, logY_std,
                        T_mean, T_std):
        self.logY0_mean.copy_(torch.as_tensor(logY0_mean))
        self.logY0_std.copy_(torch.as_tensor(logY0_std).clamp_min(1e-3))
        self.T0_mean.copy_(torch.as_tensor(T0_mean))
        self.T0_std.copy_(torch.as_tensor(T0_std).clamp_min(1e-3))
        self.phi_mean.copy_(torch.as_tensor(phi_mean))
        self.phi_std.copy_(torch.as_tensor(phi_std).clamp_min(1e-3))
        self.logP_mean.copy_(torch.as_tensor(logP_mean))
        self.logP_std.copy_(torch.as_tensor(logP_std).clamp_min(1e-3))
        self.tprime_mean.copy_(torch.as_tensor(tprime_mean))
        self.tprime_std.copy_(torch.as_tensor(tprime_std).clamp_min(1e-3))
        self.logY_mean.copy_(torch.as_tensor(logY_mean))
        self.logY_std.copy_(torch.as_tensor(logY_std).clamp_min(1e-3))
        self.T_mean.copy_(torch.as_tensor(T_mean))
        self.T_std.copy_(torch.as_tensor(T_std).clamp_min(1e-3))

    def normalize_Y0(self, Y0):
        logY0 = torch.log(torch.clamp(Y0, min=self.Y_MIN))
        return (logY0 - self.logY0_mean) / self.logY0_std

    def normalize_T0(self, T0):
        return (T0 - self.T0_mean) / self.T0_std

    def normalize_phi(self, phi):
        return (phi - self.phi_mean) / self.phi_std

    def normalize_P(self, P):
        logP = torch.log(torch.clamp(P, min=1e-6))
        return (logP - self.logP_mean) / self.logP_std

    def normalize_tprime(self, t, t_end):
        raw = t / torch.clamp(t_end, min=1e-12)
        return (raw - self.tprime_mean) / self.tprime_std

    def denormalize_Y(self, Yn):
        logY = torch.clamp(Yn * self.logY_std + self.logY_mean, max=80.0)
        return torch.exp(logY)

    def denormalize_T(self, Tn):
        return Tn * self.T_std + self.T_mean

    def forward(self, Y0, T0, phi, P, t, t_end):
        """Y0/T0/phi/P: the trajectory's FIXED initial condition (can repeat
        across rows sharing one trajectory). t/t_end: the query time for
        each row -- can differ row to row even with the same IC."""
        Y0n = self.normalize_Y0(Y0)
        T0n = self.normalize_T0(T0)
        phin = self.normalize_phi(phi)
        Pn = self.normalize_P(P)
        tn = self.normalize_tprime(t, t_end)

        branch_in = torch.cat([Y0n, T0n.unsqueeze(-1), phin.unsqueeze(-1), Pn.unsqueeze(-1)], dim=-1)
        b = self.branch(branch_in).view(-1, self.n_outputs, self.p)
        trk = self.trunk(tn.unsqueeze(-1))
        out = torch.einsum("bop,bp->bo", b, trk) + self.bias

        Tn_pred, Yn_pred = out[:, 0], out[:, 1:]
        T_pred = torch.clamp(self.denormalize_T(Tn_pred), min=200.0, max=6000.0)
        Y_pred = torch.clamp(self.denormalize_Y(Yn_pred), min=0.0, max=1.0)
        return T_pred, Y_pred, {"Tn_pred": Tn_pred, "Yn_pred": Yn_pred}