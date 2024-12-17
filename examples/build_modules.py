from detectron2.config import CfgNode as CN

import settings
from config.config import load_config
from defm_detr.models.timm_models import build_backbone
from utility.print_util import print_model_structure
from defm_detr.models.deformable_transformer import build_deforamble_transformer


def create_modules():
    print('\n========== config ==========\n')
    cfg = load_config()
    # print(cfg)
    print('\n========== backbone ==========\n')
    backbone = build_backbone(cfg)
    print_model_structure(backbone, max_depth=4)
    print('\n========== detr ==========\n')
    detr = build_deforamble_transformer(cfg)
    print_model_structure(detr, max_depth=4)


if __name__ == "__main__":
    create_modules()
