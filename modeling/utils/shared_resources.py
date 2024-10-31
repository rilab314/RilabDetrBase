from modeling.utils.utils import MLP


bbox_embed = None


def get_bbox_embed(hidden_dim = 0):
    global bbox_embed
    if bbox_embed is None:
        assert hidden_dim != 0, "hidden_dim is not set"
        bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
    return bbox_embed
