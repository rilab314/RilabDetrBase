from detectron2.config import CfgNode as CN

import settings
from config.config import load_config
from defm_detr.models.timm_models import build_hf_backbone
from utility.print_util import print_model_structure
from defm_detr.models.deformable_transformer import build_deforamble_transformer
from defm_detr.models import build_model

def create_modules():
    print('\n========== config ==========\n')
    cfg = load_config()
    # print(cfg)
    print('\n========== backbone ==========\n')
    backbone = build_hf_backbone(cfg)
    print_model_structure(backbone, max_depth=4)
    print('\n========== detr ==========\n')
    detr = build_deforamble_transformer(cfg)
    print_model_structure(detr, max_depth=3)
    model, criterion, postprocessors = build_model(cfg)
    print('\n========== model ==========\n')
    print_model_structure(model, max_depth=4)
    print('\n========== criterion ==========\n')
    print_model_structure(criterion, max_depth=4)
    print('\n========== postprocessors ==========\n')
    print_model_structure(postprocessors, max_depth=4)




if __name__ == "__main__":
    create_modules()
