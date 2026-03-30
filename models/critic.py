import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_dim=2, action_dim=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # outputs risk ∈ [0,1]
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)