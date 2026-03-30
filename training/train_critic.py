import torch
import torch.nn as nn
import torch.optim as optim


def train_critic(states, actions, next_states, env, epochs=20):
    from models.critic import Critic

    model = Critic()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    next_states = torch.tensor(next_states, dtype=torch.float32)

    action_onehot = torch.nn.functional.one_hot(actions, num_classes=4).float()

    # 🔥 label: 1 if unsafe, 0 if safe
    labels = []

    for ns in next_states:
        ns_tuple = tuple(ns.numpy().round().astype(int))
        labels.append(1.0 if env.is_hazard(ns_tuple) else 0.0)

    labels = torch.tensor(labels).unsqueeze(1)

    for epoch in range(epochs):
        preds = model(states, action_onehot)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Critic Epoch {epoch}, Loss: {loss.item():.4f}")

    return model