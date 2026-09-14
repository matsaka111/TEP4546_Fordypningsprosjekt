"""
Modular loss functions for training CRKPINN. Each loss is a standalone
function so the training loop can turn any subset on/off via LAMBDA_* weights
without editing this file.

L_data:      Eq. 31 -- MSE between predicted and true next state, in
             normalized space.
L_mole:      Eq. E.9 -- predicted species mass fractions should sum to 1.
L_element:   Eq. E.8 -- mass fraction of each ELEMENT (not species) should
             be conserved between t and t+dt.
L_enthalpy:  Eq. E.5 -- mixture specific enthalpy should be conserved
             between t and t+dt (no external heat added/removed).
L_arrhenius: simplified analog of Eq. 30 (L_ODEs). The paper checks the
             network's own autodiff-implied rate against the FULL
             325-reaction mass-action law; reimplementing that as
             differentiable code is a separate, much larger undertaking.
             This instead checks CRKPINN's autodiff-implied rate against a
             single learnable GLOBAL reaction (GlobalArrheniusRate module,
             same stoichiometry/mass-conservation structure as ArrheniusNN)
             -- a deliberately simplified stand-in for the real thing.
"""

import torch


def loss_data(model, info, T_next_true, Y_next_true):
    Tn_true = model.normalize_T(T_next_true)
    Yn_true = model.normalize_Y(Y_next_true)
    return torch.nn.functional.mse_loss(info["Tn_next"], Tn_true) \
         + torch.nn.functional.mse_loss(info["Yn_next"], Yn_true)


def loss_mole(Y_next_pred):
    return ((Y_next_pred.sum(dim=-1) - 1.0) ** 2).mean()


def loss_element(element_mass_fractions_fn, Y_t, Y_next_pred):
    elem_t = element_mass_fractions_fn(Y_t)
    elem_next = element_mass_fractions_fn(Y_next_pred)
    return ((elem_next - elem_t) ** 2).mean()


def loss_enthalpy(mixture_specific_enthalpy_fn, T_t, Y_t, T_next_pred, Y_next_pred):
    h_t = mixture_specific_enthalpy_fn(T_t, Y_t)
    h_next = mixture_specific_enthalpy_fn(T_next_pred, Y_next_pred)
    return ((h_next - h_t) ** 2).mean()


def loss_arrhenius(model, arrhenius_module, T_t, Y_t, T0_t, P_t, dt_t, tend_t, rho_t):
    """Requires dt to carry gradients -- clones and sets requires_grad_(True)
    internally, does not mutate the caller's dt tensor. Uses a batched
    Jacobian (torch.autograd.grad with is_grads_batched=True) to get
    d(Y_next)/d(dt) for all species in one call -- requires PyTorch 2.0+."""
    dt_g = dt_t.clone().requires_grad_(True)
    T_next_pred, Y_next_pred, info = model(T_t, Y_t, T0_t, P_t, dt_g, tend_t)

    n_species = Y_next_pred.shape[1]
    grad_outputs = torch.eye(n_species, device=Y_next_pred.device) \
        .unsqueeze(1).expand(n_species, Y_next_pred.shape[0], n_species)
    Ydot_autodiff = torch.autograd.grad(
        outputs=Y_next_pred, inputs=dt_g, grad_outputs=grad_outputs,
        is_grads_batched=True, create_graph=True,
    )[0].T  # (n_species, batch) -> (batch, n_species)

    Ydot_arrhenius = arrhenius_module(T_t, Y_t, rho_t)

    return torch.nn.functional.mse_loss(Ydot_autodiff, Ydot_arrhenius)