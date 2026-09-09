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
    
class GlobalArrheniusSurrogate(nn.Module):
    """
    Same physics as before (k = A*T^b*exp(-Ea/RT), q = k*prod(C_i^order_i), wdot_i = nu_i*q),
    but every intermediate quantity is computed and differentiated in LOG-SPACE.

    Why: with ~24 simultaneous "reactant" species and concentrations down to 1e-36,
    q collapses to ~1e-300 in linear space. Both the VALUE and its GRADIENT underflow
    to zero (or near-zero even in float64) once you actually compute exp() of a very
    negative number and then try to differentiate back through it. Working entirely
    in log-magnitude space avoids ever forming that vanishing linear-space number --
    log(q) stays around -300, a perfectly normal float, with perfectly normal gradients.
    """

    def __init__(self, n_species, molecular_weights, hidden=512, n_layers = 8, eps=1e-30):
        super().__init__()
        self.n_species = n_species
        self.register_buffer("mw", torch.tensor(molecular_weights, dtype=torch.float32))
        self.eps = eps

        self.dependent_idx = int(torch.argmax(self.mw).item())
        self.free_idx = [i for i in range(n_species) if i != self.dependent_idx]

        self.input_layer = nn.Linear(1 + n_species, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(n_layers - 2)])
        self.output_layer = nn.Linear(hidden, 3 + (n_species - 1))

        EA_INIT = 40000.0
        with torch.no_grad():
            last_layer = self.output_layer
            last_layer.weight[1, :] *= 0.1
            last_layer.bias[1] *= 0.1
            last_layer.weight[2, :] *= 0.1
            last_layer.bias[2] = EA_INIT
            last_layer.bias[3:] = torch.zeros(n_species - 1)
            last_layer.weight[3:, :] *= 0.3

        self.b_max = 6.0
        self.nu_max = 4.0

        self.register_buffer("T_ref", torch.tensor(300.0))
        self.register_buffer("rho_ref", torch.tensor(1.0))

    def set_input_norm(self, T_ref, rho_ref):
        self.T_ref.copy_(torch.as_tensor(T_ref))
        self.rho_ref.copy_(torch.as_tensor(rho_ref).clamp_min(1e-12))

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

    def forward(self, T, rho):
        """
        Returns log_wdot_mag (batch, n_species) -- log(|wdot_i| + eps) -- and wdot_sign,
        plus the usual info dict. Train the loss DIRECTLY against log_wdot_mag; never
        reconstruct linear-space wdot for the loss (only for final inspection/plotting).
        """
        T_norm = T / self.T_ref
        rho_norm = rho / self.rho_ref
        x = torch.cat([T_norm.unsqueeze(-1), rho_norm], dim=-1)

        h = nn.functional.silu(self.input_layer(x))
        for block in self.blocks:
            h = block(h)
        out = self.output_layer(h)

        A = nn.functional.softplus(out[:, 0]) + 1e-12
        b = self.b_max * torch.tanh(out[:, 1])
        Ea = nn.functional.softplus(out[:, 2])

        nu_free = self.nu_max * torch.tanh(out[:, 3:] / self.nu_max)
        nu = self._assemble_nu(nu_free)

        log_k = torch.log(A) + b * torch.log(self.T_ref / T) - Ea / (R * T)

        C = rho / self.mw
        order = torch.relu(-nu)
        log_C = torch.log(torch.clamp(C, min=1e-20))

        log_rate_term = (order * log_C).sum(dim=-1)  # this WILL be very negative -- that's fine now
        log_q = log_k + log_rate_term                  # (batch,) -- stays in normal float range

        # log|wdot_i| = log|nu_i| + log_q
        log_nu_mag = torch.log(torch.abs(nu) + self.eps)
        log_wdot_mag = log_nu_mag + log_q.unsqueeze(-1)
        wdot_sign = torch.sign(nu)

        # linear-space wdot, for inspection/plotting ONLY -- do not train against this directly
        with torch.no_grad():
            wdot_linear = wdot_sign * torch.exp(torch.clamp(log_wdot_mag, max=80.0))

        rho_mix = rho.sum(dim=-1, keepdim=True)
        Ydot_linear = wdot_linear * self.mw.unsqueeze(0) / rho_mix.clamp_min(1e-12)

        return log_wdot_mag, wdot_sign, {
            "wdot_linear": wdot_linear,
            "Ydot": Ydot_linear,
            "A": A, "b": b, "Ea": Ea, "nu": nu,
            "order_sum": order.sum(dim=-1),
            "log_q": log_q,
        }