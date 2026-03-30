from models.world_model import WorldModel
import torch



class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict(self, state, action):
        preds = []

        for model in self.models:
            with torch.no_grad():
                pred = model(state, action)
                preds.append(pred)

        return torch.stack(preds)  # [num_models, batch, state_dim]