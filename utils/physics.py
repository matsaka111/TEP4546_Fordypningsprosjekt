"""
Physics utilities shared across model training: element-conservation and
enthalpy-conservation helpers, built directly from a Cantera Solution object
so they always match whatever mechanism generated the training data.

Usage:
    import cantera as ct
    gas = ct.Solution("gri30.yaml")
    physics = build_physics_utils(gas, species_cols, device)

    elem_frac = physics.element_mass_fractions(Y)       # (batch, n_elements)
    h_mix = physics.mixture_specific_enthalpy(T, Y)      # (batch,)
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PhysicsUtils:
    element_mass_fractions: callable
    mixture_specific_enthalpy: callable
    mw_t: torch.Tensor           # (n_species,) molecular weights
    element_names: list


def build_physics_utils(gas, species_cols, device):
    """gas: a Cantera Solution object for the mechanism used to generate the
    training data. species_cols: species names in the SAME order as every
    other tensor in the pipeline (e.g. Y columns)."""

    mw = np.array([gas.molecular_weights[gas.species_index(s)] for s in species_cols])
    mw_t = torch.tensor(mw, dtype=torch.float32, device=device)

    # ---------------- Element conservation ----------------
    element_names = gas.element_names
    n_atoms = np.array([[gas.n_atoms(s, e) for e in element_names] for s in species_cols])
    atomic_weights = np.array([gas.atomic_weight(e) for e in element_names])
    n_atoms_t = torch.tensor(n_atoms, dtype=torch.float32, device=device)
    atomic_weights_t = torch.tensor(atomic_weights, dtype=torch.float32, device=device)

    def element_mass_fractions(Y):
        """Mass fraction of each ELEMENT in the mixture, from species mass
        fractions Y. Must stay constant over time (atoms aren't created or
        destroyed by chemical reactions) -- paper's Eq. E.6-E.8 concept,
        computed here as element MASS fraction rather than mole fraction."""
        moles_per_mass = Y / mw_t
        elem_moles_per_mass = moles_per_mass @ n_atoms_t
        return elem_moles_per_mass * atomic_weights_t

    # ---------------- Enthalpy conservation (NASA-7 polynomials) ----------------
    R_GAS = 8.314462618  # J/(mol*K)
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
        """Paper Eq. E.1 (without the outer RT factor)."""
        a = coefs
        Tb = T.unsqueeze(-1)
        return (a[:, 0] + a[:, 1] / 2 * Tb + a[:, 2] / 3 * Tb**2
                + a[:, 3] / 4 * Tb**3 + a[:, 4] / 5 * Tb**4 + a[:, 5] / Tb)

    def mixture_specific_enthalpy(T, Y):
        """Sum_k h_k * Y_k -- mass-weighted mixture specific enthalpy
        (Eq. E.2, E.4), with the piecewise low-/high-T NASA polynomial
        split per species."""
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