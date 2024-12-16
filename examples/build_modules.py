from detectron2.config import CfgNode as CN

import settings
from config.config import load_config
from defm_detr.models.timm_models import ResNet50Timm, SwinV2Timm
from utility.print_util import print_model_structure
from defm_detr.models.deformable_transformer import build_deforamble_transformer


def create_modules():
    print('\n========== config ==========\n')
    cfg = load_config()
    # print(cfg)
    print('\n========== resnet ==========\n')
    # build_backbone(cfg)
    resnet = ResNet50Timm()
    print_model_structure(resnet, max_depth=3)
    print('\n========== swin ==========\n')
    swin = SwinV2Timm()
    print_model_structure(swin, max_depth=3)
    print('\n========== detr ==========\n')
    detr = build_deforamble_transformer(cfg)
    print_model_structure(detr, max_depth=4)


if __name__ == "__main__":
    create_modules()
