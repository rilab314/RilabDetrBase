import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from detectron2.structures import Instances, Boxes
import albumentations as A
from albumentations.pytorch import ToTensorV2
from detectron2.config import get_cfg, CfgNode


def composer_factory(cfg, split: str):
    """
    cfg.DATASET.AUGMENTATION에 정의된 설정을 기반으로 albumentations.Compose 객체를 생성합니다.
    """
    split = split.upper()
    if cfg.DATASET[split].AUGMENTATION:
        augmentations = create_augmentations(cfg)
    else:
        augmentations = []

    augmentations.append(A.Resize(
        height=cfg.DATASET.IMAGE_HEIGHT,
        width=cfg.DATASET.IMAGE_WIDTH,
        interpolation=cv2.INTER_LINEAR,
        p=1.0
    ))
    augmentations.append(A.Normalize(
        mean=cfg.MODEL.PIXEL_MEAN,
        std=cfg.MODEL.PIXEL_STD,
        max_pixel_value=255.0
    ))
    augmentations.append(ToTensorV2())

    bbox_params = A.BboxParams(
        format='yolo',  # 데이터셋의 바운딩 박스 형식에 따라 설정
        label_fields=['category_ids']
    )
    transform = A.Compose(augmentations, bbox_params=bbox_params)
    return transform


def create_augmentations(cfg):
    aug_cfg = cfg.DATASET.AUGMENTATION
    augmentations = []
    if 'HorizontalFlip' in aug_cfg:
        params = aug_cfg.HorizontalFlip
        augmentations.append(A.HorizontalFlip(p=params.get('p', 0.5)))

    if 'RandomResizedCrop' in aug_cfg:
        params = aug_cfg.RandomResizedCrop
        augmentations.append(A.RandomResizedCrop(
            size=params.get('size', (384, 384)),
            scale=params.get('scale', (0.5, 1.0)),
            ratio=params.get('ratio', (0.75, 1.333)),
            p=params.get('p', 1.0)
        ))

    if 'RandomBrightnessContrast' in aug_cfg:
        params = aug_cfg.RandomBrightnessContrast
        augmentations.append(A.RandomBrightnessContrast(
            brightness_limit=params.get('brightness_limit', 0.2),
            contrast_limit=params.get('contrast_limit', 0.2),
            p=params.get('p', 0.5)
        ))

    if 'HueSaturationValue' in aug_cfg:
        params = aug_cfg.HueSaturationValue
        augmentations.append(A.HueSaturationValue(
            hue_shift_limit=params.get('hue_shift_limit', 20),
            sat_shift_limit=params.get('sat_shift_limit', 30),
            val_shift_limit=params.get('val_shift_limit', 20),
            p=params.get('p', 0.5)
        ))

    if 'RandomGamma' in aug_cfg:
        params = aug_cfg.RandomGamma
        augmentations.append(A.RandomGamma(
            gamma_limit=params.get('gamma_limit', (80, 120)),
            p=params.get('p', 0.5)
        ))

    if 'GaussNoise' in aug_cfg:
        params = aug_cfg.GaussNoise
        augmentations.append(A.GaussNoise(
            var_limit=params.get('var_limit', (10.0, 50.0)),
            mean=params.get('mean', 0),
            p=params.get('p', 0.5)
        ))

    # 필요한 경우 추가적인 데이터 증강 기법을 여기에 추가할 수 있습니다.

    return augmentations
