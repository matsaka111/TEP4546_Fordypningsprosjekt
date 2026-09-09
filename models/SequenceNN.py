import torch
import torch.nn as nn

class SequenceNN(nn.Module):
    def __init__(self, dim=54, hidden=256, n_steps=199):
        super().__init__()
        self.n_steps = n_steps
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_steps * dim),
        )

    def forward(self, y0):
        delta = self.net(y0).reshape(-1, self.n_steps, self.dim)
        y_seq = y0.unsqueeze(1) + torch.cumsum(delta, dim=1)
        return y_seq