"""
simple_nn.py

Simple feedforward surrogate: y (54,) -> ydot (54,)
Baseline architecture -- three-layer MLP, no physics built in.
"""

import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, dim=54, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, y):
        raw = self.net(y)

        return raw