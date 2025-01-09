import numpy as np
import torch
from torch.utils.data import DataLoader
import numpy as np

import settings
from config.config import load_config
from models.timm_models import build_hf_backbone
from utility.print_util import print_model, print_data
from models.deformable_transformer import build_deforamble_transformer
from models import build_model
from util.misc import NestedTensor
from data.custom_detection_dataset import CustomDetectionDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def custom_collate_fn(batch):
    # TODO: instance는 안쓸거니까 바꿔줘야함
    result = {}
    for key in batch[0].keys():
        if key == 'instances':
            result[key] = [item[key] for item in batch] 
        else:
            result[key] = torch.utils.data._utils.collate.default_collate([item[key] for item in batch])
    return result


class Trainer:
    def __init__(self):
        pass

    def train(self):
        cfg = load_config()
        model, criterion, postprocessors = build_model(cfg)
        train_dataset = CustomDetectionDataset(cfg, split='train')
        val_dataset = CustomDetectionDataset(cfg, split='val')
        test_dataset = CustomDetectionDataset(cfg, split='test')
