# fordypning/losses.py
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
    dt_g = dt_t.clone().requires_grad_(True)
    T_next_pred, Y_next_pred, info = model(T_t, Y_t, T0_t, P_t, dt_g, tend_t)

    n_species = Y_next_pred.shape[1]
    grad_outputs = torch.eye(n_species, device=Y_next_pred.device) \
        .unsqueeze(1).expand(n_species, Y_next_pred.shape[0], n_species)
    Ydot_autodiff = torch.autograd.grad(
        outputs=Y_next_pred, inputs=dt_g, grad_outputs=grad_outputs,
        is_grads_batched=True, create_graph=True,
    )[0].T

    Ydot_arrhenius = arrhenius_module(T_t, Y_t, rho_t)
    return torch.nn.functional.mse_loss(Ydot_autodiff, Ydot_arrhenius)


def loss_data_huber(model, info, T_next_true, Y_next_true, delta=5.0):
    """Same as loss_data, but Huber instead of MSE -- robust to the rare
    but extreme outlier points that occur for ultra-trace species in
    near-flat (no-ignition) trajectories, where a handful of validation
    points can land tens of sigma from the training distribution and
    dominate a plain MSE average."""
    Tn_true = model.normalize_T(T_next_true)
    Yn_true = model.normalize_Y(Y_next_true)
    return (torch.nn.functional.huber_loss(info["Tn_next"], Tn_true, delta=delta)
            + torch.nn.functional.huber_loss(info["Yn_next"], Yn_true, delta=delta))

def loss_kstep(model, T_seq, Y_seq, T0, P, dt_seq, t_end, k_steps, use_huber=False):
    """
    Unrolls the model k_steps times, feeding each prediction back in as the
    next input, and accumulates loss against the TRUE state at every step
    -- not just the first. This directly optimizes for what autoregressive
    rollout actually does at inference time, rather than only single-step
    accuracy, which doesn't guarantee the errors won't compound.
    """
    T_cur = T_seq[:, 0]
    Y_cur = Y_seq[:, 0]

    data_loss_fn = loss_data_huber if use_huber else loss_data
    total = 0.0

    for k in range(k_steps):
        dt_k = dt_seq[:, k]
        T_next_pred, Y_next_pred, info = model(T_cur, Y_cur, T0, P, dt_k, t_end)

        T_next_true = T_seq[:, k + 1]
        Y_next_true = Y_seq[:, k + 1]

        total = total + data_loss_fn(model, info, T_next_true, Y_next_true)

        # feed prediction back in for the next step (this is the whole point)
        T_cur, Y_cur = T_next_pred, Y_next_pred

    return total / k_steps