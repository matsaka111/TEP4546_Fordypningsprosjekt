# fordypning/idt_predictor.py
import torch
import torch.nn as nn


class IDTPredictor(nn.Module):
    """
    Predicts ignition delay time (IDT) from (P, T0, phi) alone -- no
    simulation needed. Used to compute t_end = n * IDT for normalizing the
    time input to CRKPINN/DeepONet at deployment, when no Cantera-scouted
    t_ign is available for a genuinely new initial condition (unlike
    training, where t_ign comes directly from generate_data_gri30_cbins.py).

    Follows CRK-PINN Appendix B (Zhang, Zhang & Wang 2024): small MLP,
    LeakyReLU, predicts ln(IDT) rather than IDT directly since IDT spans
    many orders of magnitude (their reported range: 0.1 us to 2e7 us).
    """

    def __init__(self, hidden=64, n_hidden_layers=2):
        super().__init__()

        layers = [nn.Linear(3, hidden), nn.LeakyReLU(negative_slope=0.1)]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LeakyReLU(negative_slope=0.1)]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

        self.register_buffer("mean", torch.zeros(3))
        self.register_buffer("std", torch.ones(3))

    def set_norm_stats(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6))

    def forward(self, P, T0, phi):
        """P, T0, phi: (batch,). Returns ln(IDT_hat), (batch,)."""
        logP = torch.log(torch.clamp(P, min=1e-6))
        x = torch.stack([logP, T0, phi], dim=-1)
        x = (x - self.mean) / self.std
        return self.net(x).squeeze(-1)

    def predict_idt(self, P, T0, phi):
        """Returns IDT_hat in physical units (seconds), not log."""
        with torch.no_grad():
            log_idt = self.forward(P, T0, phi)
        return torch.exp(log_idt)


def loss_idt(model, P, T0, phi, idt_true):
    """CRK-PINN Eq. B.2: log-space MSE + relative-error penalty."""
    log_idt_pred = model(P, T0, phi)
    log_idt_true = torch.log(torch.clamp(idt_true, min=1e-12))

    idt_pred = torch.exp(log_idt_pred)
    rel_err = (idt_pred - idt_true) / torch.clamp(idt_true, min=1e-12)

    return ((log_idt_pred - log_idt_true) ** 2 + rel_err ** 2).mean()