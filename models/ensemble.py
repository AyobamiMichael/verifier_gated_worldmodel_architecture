from models.world_model import WorldModel
import torch


class EnsembleWorldModel:
    def __init__(self, num_models=5):
        self.models = [WorldModel() for _ in range(num_models)]

    def predict(self, state, action):
        preds = []

        for model in self.models:
            with torch.no_grad():
                preds.append(model(state, action))

        return torch.stack(preds)

    def uncertainty(self, state, action):
        preds = self.predict(state, action)
        return preds.var(dim=0).mean()