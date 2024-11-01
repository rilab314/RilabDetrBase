import torch.nn as nn

from modeling.utils.utils import MLP

bbox_embed = None


def get_bbox_embed(hidden_dim = 0):
    global bbox_embed
    if bbox_embed is None:
        assert hidden_dim != 0, "hidden_dim is not set"
        bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)

    return bbox_embed
