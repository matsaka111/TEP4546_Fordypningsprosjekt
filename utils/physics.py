"""
physics.py

Differentiable physics utilities shared between the surrogate
model and data generation / analysis scripts.
"""

import torch

R_GAS = 8314.0  # J/(kmol*K)


def mean_molecular_weight(Y, W):
    """
    Mean molecular weight of the mixture: 1 / sum(Y_k / W_k)
    Y: (batch, n_species), W: (n_species,) -> (batch, 1)
    """
    inv_Wmix = torch.sum(Y / W, dim=1, keepdim=True)
    return 1.0 / inv_Wmix


def density_ideal_gas(T, P, Y, W):
    """
    Ideal gas law: rho = P * Wmix / (R * T)
    T: (batch, 1), P: scalar or (batch, 1), Y: (batch, n_species)
    """
    Wmix = mean_molecular_weight(Y, W)
    rho = P * Wmix / (R_GAS * T)
    return rho


def concentrations(Y, rho, W):
    """
    Molar concentration C_j = rho * Y_j / W_j
    """
    return rho * Y / W


def nasa7_cp_h(T, coeffs_low, coeffs_high, T_mid):
    """
    Evaluate molar cp and h from NASA-7 polynomials.

    coeffs_low, coeffs_high: (n_species, 7) -- [a1..a5, a6, a7]
    T_mid: (n_species,) -- switch temperature per species
    T: (batch, 1)

    Returns cp (batch, n_species) [J/kmol/K], h (batch, n_species) [J/kmol]
    """
    # select low or high coeffs per species per batch element
    use_high = (T >= T_mid).float()  # (batch, n_species) after broadcast
    a = use_high.unsqueeze(-1) * coeffs_high + (1 - use_high.unsqueeze(-1)) * coeffs_low
    # a: (batch, n_species, 7)

    T2, T3, T4 = T**2, T**3, T**4
    T = T.unsqueeze(-1)  # for broadcasting against last coeff dim if needed

    a1, a2, a3, a4, a5, a6, a7 = [a[..., i] for i in range(7)]

    cp_R = a1 + a2*T.squeeze(-1) + a3*T2 + a4*T3 + a5*T4
    h_RT = a1 + a2*T.squeeze(-1)/2 + a3*T2/3 + a4*T3/4 + a5*T4/5 + a6/T.squeeze(-1)

    cp = cp_R * R_GAS
    h = h_RT * R_GAS * T.squeeze(-1)
    return cp, h