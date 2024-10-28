import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from detectron2.structures import Instances, Boxes
import albumentations as A
from albumentations.pytorch import ToTensorV2
from detectron2.config import get_cfg, CfgNode


def composer_factory(cfg):
    """
    cfg.AUGMENTATION에 정의된 설정을 기반으로 albumentations.Compose 객체를 생성합니다.
    """
    if cfg.DATASET.AUGMENT:
        augmentations = create_augmentations(cfg)
    else:
        augmentations = []

    # Normalize와 ToTensorV2를 추가합니다.
    augmentations.append(A.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN,
        std=cfg.INPUT.PIXEL_STD,
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
    augmentations = []
    if 'HorizontalFlip' in cfg.AUGMENTATION:
        params = cfg.AUGMENTATION.HorizontalFlip
        augmentations.append(A.HorizontalFlip(p=params.get('p', 0.5)))

    if 'RandomResizedCrop' in cfg.AUGMENTATION:
        params = cfg.AUGMENTATION.RandomResizedCrop
        augmentations.append(A.RandomResizedCrop(
            height=params['height'],
            width=params['width'],
            scale=params.get('scale', (0.08, 1.0)),
            ratio=params.get('ratio', (0.75, 1.3333333333333333)),
            p=params.get('p', 1.0)
        ))

    if 'RandomBrightnessContrast' in cfg.AUGMENTATION:
        params = cfg.AUGMENTATION.RandomBrightnessContrast
        augmentations.append(A.RandomBrightnessContrast(
            brightness_limit=params.get('brightness_limit', 0.2),
            contrast_limit=params.get('contrast_limit', 0.2),
            p=params.get('p', 0.5)
        ))

    if 'HueSaturationValue' in cfg.AUGMENTATION:
        params = cfg.AUGMENTATION.HueSaturationValue
        augmentations.append(A.HueSaturationValue(
            hue_shift_limit=params.get('hue_shift_limit', 20),
            sat_shift_limit=params.get('sat_shift_limit', 30),
            val_shift_limit=params.get('val_shift_limit', 20),
            p=params.get('p', 0.5)
        ))

    if 'RandomGamma' in cfg.AUGMENTATION:
        params = cfg.AUGMENTATION.RandomGamma
        augmentations.append(A.RandomGamma(
            gamma_limit=params.get('gamma_limit', (80, 120)),
            p=params.get('p', 0.5)
        ))

    if 'GaussNoise' in cfg.AUGMENTATION:
        params = cfg.AUGMENTATION.GaussNoise
        augmentations.append(A.GaussNoise(
            var_limit=params.get('var_limit', (10.0, 50.0)),
            mean=params.get('mean', 0),
            p=params.get('p', 0.5)
        ))

    # 필요한 경우 추가적인 데이터 증강 기법을 여기에 추가할 수 있습니다.

    return augmentations
