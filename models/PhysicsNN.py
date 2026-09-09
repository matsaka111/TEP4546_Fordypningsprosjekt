"""
PhysicsNN.py

Simple feedforward surrogate: y(54) -> ydot(54).
1. Integrate inn conservation of mass by setting 
    sum of dy/dt = 0, so no change in mass fraction.
2. Integrate inn structure from Arhenius law. 295 species 
    follow this law so we impose this for them. Ignore the 
    other laws for now
"""

import torch
import torch.nn as nn

class PhysicsNN(nn.Module):
    def __init__(self, dim=54, hidden = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,dim),
        )

    def forward(self, y, ydot_mean_log=None, ydot_std_log=None):
        raw_log = self.net(y)   # network output ydot

        # undo signed-log to get physical units
        raw_phys = torch.sign(raw_log) * (torch.expm1(torch.abs(raw_log)))

        dT_phys = raw_phys[:, 0:1]
        species_phys = raw_phys[:, 1:]
        species_phys = species_phys - species_phys.mean(dim=1, keepdim=True)  # physical space

        out_phys = torch.cat([dT_phys, species_phys], dim=1)

        # back to signed-log space
        out_log = torch.sign(out_phys) * torch.log1p(torch.abs(out_phys))
        return out_log
