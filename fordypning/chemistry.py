# fordypning/chemistry.py
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

R_GAS = 8.314462618  # J/(mol*K)


class GlobalArrheniusRate(nn.Module):
    """
    Auxiliary module: predicts a single global reaction's stoichiometry
    (mass-conservation-guaranteed via a free/dependent species split) and
    Arrhenius parameters (A, b, Ea), producing a species mass-fraction rate
    Ydot from (T, Y). Used only to provide a physically-structured target
    for L_arrhenius -- a deliberately simplified stand-in for the paper's
    full 325-reaction L_ODEs.
    """

    def __init__(self, n_species, molecular_weights, hidden=128, n_hidden_layers=3):
        super().__init__()
        self.n_species = n_species
        self.register_buffer("mw", torch.tensor(molecular_weights, dtype=torch.float32))

        self.dependent_idx = int(torch.argmax(self.mw).item())
        self.free_idx = [i for i in range(n_species) if i != self.dependent_idx]

        in_dim = n_species + 1
        out_dim = 3 + (n_species - 1)

        layers = [nn.Linear(in_dim, hidden), nn.ELU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

        self.b_max = 6.0
        self.nu_max = 4.0
        self.eps = 1e-30
        self.register_buffer("T_ref", torch.tensor(1200.0))

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


@dataclass
class PhysicsUtils:
    element_mass_fractions: callable
    mixture_specific_enthalpy: callable
    mw_t: torch.Tensor
    element_names: list


def build_physics_utils(gas, species_cols, device):
    """gas: a Cantera Solution object for the mechanism used to generate the
    training data. species_cols: species names in the SAME order as every
    other tensor in the pipeline."""

    mw = np.array([gas.molecular_weights[gas.species_index(s)] for s in species_cols])
    mw_t = torch.tensor(mw, dtype=torch.float32, device=device)

    element_names = gas.element_names
    n_atoms = np.array([[gas.n_atoms(s, e) for e in element_names] for s in species_cols])
    atomic_weights = np.array([gas.atomic_weight(e) for e in element_names])
    n_atoms_t = torch.tensor(n_atoms, dtype=torch.float32, device=device)
    atomic_weights_t = torch.tensor(atomic_weights, dtype=torch.float32, device=device)

    def element_mass_fractions(Y):
        moles_per_mass = Y / mw_t
        elem_moles_per_mass = moles_per_mass @ n_atoms_t
        return elem_moles_per_mass * atomic_weights_t

    T_mid_list, high_coefs, low_coefs = [], [], []
    for s in species_cols:
        sp = gas.species(gas.species_index(s))
        c = sp.thermo.coeffs
        T_mid_list.append(c[0])
        high_coefs.append(c[1:8])
        low_coefs.append(c[8:15])

    T_mid_t = torch.tensor(T_mid_list, dtype=torch.float32, device=device)
    high_coefs_t = torch.tensor(np.array(high_coefs), dtype=torch.float32, device=device)
    low_coefs_t = torch.tensor(np.array(low_coefs), dtype=torch.float32, device=device)

    def molar_enthalpy_RT(T, coefs):
        a = coefs
        Tb = T.unsqueeze(-1)
        return (a[:, 0] + a[:, 1] / 2 * Tb + a[:, 2] / 3 * Tb**2
                + a[:, 3] / 4 * Tb**3 + a[:, 4] / 5 * Tb**4 + a[:, 5] / Tb)

    def mixture_specific_enthalpy(T, Y):
        use_high = (T.unsqueeze(-1) > T_mid_t.unsqueeze(0))
        H_RT_high = molar_enthalpy_RT(T, high_coefs_t)
        H_RT_low = molar_enthalpy_RT(T, low_coefs_t)
        H_RT = torch.where(use_high, H_RT_high, H_RT_low)
        H_molar = H_RT * R_GAS * T.unsqueeze(-1)
        h_mass = H_molar / mw_t
        return (h_mass * Y).sum(dim=-1)

    return PhysicsUtils(
        element_mass_fractions=element_mass_fractions,
        mixture_specific_enthalpy=mixture_specific_enthalpy,
        mw_t=mw_t,
        element_names=element_names,
    )