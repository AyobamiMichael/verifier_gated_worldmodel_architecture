import torch
from models.world_model import WorldModel

def load_world_model(path="models/world_model.pt"):
    model = WorldModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model