import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import detectron2.utils.comm as comm

from data.custom_detection_dataset import CustomDetectionDataset


def custom_collate_fn(batch):
    return batch


def build_data_loader(cfg, split: str):
    dataset_class = eval(cfg.DATASET.NAME)
    dataset = dataset_class(cfg, split)
    
    # 분산 학습을 위한 sampler 설정
    sampler = DistributedSampler(
        dataset,
        num_replicas=comm.get_world_size(),
        rank=comm.get_rank(),
        shuffle=(split.lower() == 'train')
    ) if comm.get_world_size() > 1 else None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.DATASET[split.upper()].BATCH_SIZE,
        shuffle=(sampler is None and split.lower() == 'train'),
        num_workers=2,
        collate_fn=custom_collate_fn,
        sampler=sampler,
        pin_memory=True
    )
    return dataloader
