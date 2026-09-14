import torch
import torch.nn as nn

R_GAS = 8.314462618  # J/(mol*K)


class GlobalArrheniusRate(nn.Module):
    """
    A small auxiliary module: predicts a single global reaction's stoichiometry
    (nu, mass-conservation-guaranteed via the free/dependent species split,
    same as ArrheniusNN) and Arrhenius parameters (A, b, Ea), producing a
    species mass-fraction rate Ydot from (T, Y).

    This is NOT used in CRKPINN's forward pass -- it exists purely to provide
    a physically-structured "what should the rate be right now" target for
    L_Arrhenius (see below), which checks CRKPINN's own autodiff-implied rate
    against it. This is a deliberately simplified stand-in for the paper's
    L_ODEs (which uses the full 325-reaction mechanism) -- one global reaction
    instead of 325, chosen because reimplementing the full mechanism as
    differentiable code is a separate, much larger undertaking.
    """

    def __init__(self, n_species, molecular_weights, hidden=128, n_hidden_layers=3):
        super().__init__()
        self.n_species = n_species
        self.register_buffer("mw", torch.tensor(molecular_weights, dtype=torch.float32))

        self.dependent_idx = int(torch.argmax(self.mw).item())
        self.free_idx = [i for i in range(n_species) if i != self.dependent_idx]

        in_dim = n_species + 1  # T, Y
        out_dim = 3 + (n_species - 1)  # A, b, Ea, nu_free

        layers = [nn.Linear(in_dim, hidden), nn.ELU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

        self.b_max = 6.0
        self.nu_max = 4.0
        self.eps = 1e-30

        self.register_buffer("T_ref", torch.tensor(1200.0))  # rough mid-range for GRI30 CH4/air

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

    def forward(self, T, Y, rho_tot):
        """T: (batch,), Y: (batch, n_species) mass fractions, rho_tot: (batch,)
        mixture density -- needed to convert Y into concentrations for the
        rate law. Returns Ydot: (batch, n_species)."""
        x = torch.cat([(T / self.T_ref).unsqueeze(-1), Y], dim=-1)
        out = self.net(x)

        A = nn.functional.softplus(out[:, 0]) + 1e-12
        b = self.b_max * torch.tanh(out[:, 1])
        Ea = nn.functional.softplus(out[:, 2])
        nu_free = self.nu_max * torch.tanh(out[:, 3:] / self.nu_max)
        nu = self._assemble_nu(nu_free)

        log_k = torch.log(A) + b * torch.log(self.T_ref / T) - Ea / (R_GAS * T)

        rho_species = Y * rho_tot.unsqueeze(-1)
        C = rho_species / self.mw
        order = torch.relu(-nu)
        log_C = torch.log(torch.clamp(C, min=1e-20))
        log_rate_term = (order * log_C).sum(dim=-1)
        log_q = log_k + log_rate_term

        log_nu_mag = torch.log(torch.abs(nu) + self.eps)
        log_wdot_mag = log_nu_mag + log_q.unsqueeze(-1)
        wdot_sign = torch.sign(nu)
        wdot = wdot_sign * torch.exp(torch.clamp(log_wdot_mag, max=80.0))

        Ydot = wdot * self.mw.unsqueeze(0) / rho_tot.unsqueeze(-1)
        return Ydot