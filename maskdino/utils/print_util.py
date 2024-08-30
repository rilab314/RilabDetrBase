import torch
import numpy as np


def print_structure(data, title='', indent=0):
    if isinstance(data, torch.Tensor):
        msg = ' '*indent + f"({title}) torch.Tensor, shape={data.shape}"
        if data.numel() < 10000:
            quantile = torch.round(torch.quantile(data, torch.tensor([0, 0.2, 0.5, 0.8, 1.])), decimals=5)
            msg += f", quantile={quantile}"
        print(msg)
    elif isinstance(data, np.ndarray):
        msg = ' ' * indent + f"({title}) np.array, shape={data.shape}"
        if data.size < 10000:
            quantile = np.round(np.quantile(data, np.array([0, 0.2, 0.5, 0.8, 1.])), decimals=5)
            msg += f", quantile={quantile}"
        print(msg)
    elif isinstance(data, dict):
        print(' '*indent + title + ' {')
        for key, value in data.items():
            print_structure(value, key, indent + 2)
        print(' ' * indent + '} \\' + title)
    elif isinstance(data, list):
        print(' '*indent + title + ' [')
        for ind, value in enumerate(data):
            print_structure(value, str(ind), indent + 2)
        print(' ' * indent + '] \\' + title)
