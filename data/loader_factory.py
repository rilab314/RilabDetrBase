from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch

from data.custom_detection_dataset import CustomDetectionDataset
from config.setup_cfg import setup_cfg


def custom_collate_fn(batch):
    result = {}
    for key in batch[0].keys():
        if key == 'instances':
            result[key] = [item[key] for item in batch] 
        else:
            result[key] = torch.utils.data._utils.collate.default_collate([item[key] for item in batch])
    return result


def loader_factory(cfg, split: str):
    dataset_class = eval(cfg.DATASET.NAME)
    dataset = dataset_class(cfg, split)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.DATASET[split.upper()].BATCH_SIZE,
        shuffle=(split.lower() == 'train'),
        num_workers=2,
        collate_fn=custom_collate_fn
    )
    return dataloader
