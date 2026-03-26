import torch
import torch.nn as nn
import torch.optim as optim

from models.world_model import WorldModel, encode_action


def train_world_model(states, actions, next_states, epochs=20, lr=1e-3, save_path="models/world_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WorldModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)

    actions = encode_action(actions).to(device)

    for epoch in range(epochs):
        pred_next = model(states, actions)

        loss = loss_fn(pred_next, next_states)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"World model saved to {save_path}")
    return model