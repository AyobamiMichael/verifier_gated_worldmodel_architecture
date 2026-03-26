import torch
import torch.nn as nn


class WorldModel(nn.Module):
    def __init__(self, state_dim=2, action_dim=4, hidden_dim=64):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model(x)


# -------------------------
# Utility: One-hot encode action
# -------------------------
def encode_action(action, action_dim=4):
    return torch.nn.functional.one_hot(action, num_classes=action_dim).float()