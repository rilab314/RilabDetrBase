import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

from detectron2.config import CfgNode as CN
import numpy as np
import torch
from torch.utils.data import DataLoader

import settings
from configs.config import CfgNode
from model.timm_models import build_hf_backbone
from util.print_util import print_model, print_data
from model.deformable_transformer import build_deforamble_transformer
from model import build_model
from util.misc import NestedTensor, nested_tensor_from_batch_data
from datasets.soccer_players import SoccerPlayersDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_modules():
    print('\n========== config ==========\n')
    cfg = CfgNode.from_file('defm_detr_base')
    # print(cfg)
    print('\n========== backbone ==========\n')
    backbone = build_hf_backbone(cfg)
    print_model(backbone, max_depth=4)
    print('\n========== detr ==========\n')
    detr = build_deforamble_transformer(cfg)
    print_model(detr, max_depth=3)
    model, criterion, postprocessors = build_model(cfg)
    print('\n========== model ==========\n')
    print_model(model, max_depth=4)
    print('\n========== criterion ==========\n')
    print_model(criterion, max_depth=4)
    print('\n========== postprocessors ==========\n')
    print_model(postprocessors, max_depth=4)


def check_backbone_outputs():
    cfg = CfgNode.from_file('defm_detr_base')
    model = build_hf_backbone(cfg)
    image = np.random.rand(1, 3, 384, 384)
    mask = np.zeros((1, 384, 384), dtype=bool)
    sample = NestedTensor(torch.from_numpy(image).to(torch.float32).to(device), 
                          torch.from_numpy(mask).to(torch.bool).to(device))
    xs, pos_embeds = model.forward(sample)
    print(f"xs: {len(xs)}")
    for x, pos in zip(xs, pos_embeds):
        print(f"  x: {x.tensors.shape}, pos: {pos.shape}")


def collate_fn(batch):
    return batch

def check_defm_detr_outputs():
    cfg = CfgNode.from_file('defm_detr_base')
    dataset = SoccerPlayersDataset(cfg, 'train')
    dataloader = DataLoader(
            dataset, 
            batch_size=4,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )
    model, criterion, postprocessors = build_model(cfg)
    for k, batch in enumerate(dataloader):
        print(f"===== Batch {k + 1}/{len(dataloader)} =====")
        print_data(batch, title='batch')
        nested_tensor = nested_tensor_from_batch_data(batch)
        print('nested_tensor:', nested_tensor.tensors.shape, nested_tensor.mask.shape)
        output = model(nested_tensor)        
        print_data(output, title='output')
        target = [item['targets'] for item in batch]
        loss = criterion(output, target)
        print_data(loss, title='loss')
        break


if __name__ == "__main__":
    # create_modules()
    # check_backbone_outputs()
    check_defm_detr_outputs()
