from detectron2.config import CfgNode as CN

import settings
from config.setup_cfg import setup_cfg
from modeling.utils.print_util import print_structure
from modeling.backbone.swin import D2SwinTransformer
from modeling.encoder import DINOEncoder
from modeling.decoder import DINODecoder
from modeling.matcher import HungarianMatcher
from modeling.criterion import SetCriterion
from modeling.dino_model import DINOModel


def create_modules():
    cfg = setup_cfg()
    print('\n========== config ==========\n', cfg)
    backbone = D2SwinTransformer(cfg)
    print('\n========== backbone ==========\n', backbone)
    encoder = DINOEncoder(cfg, backbone.output_shape())
    print('\n========== encoder ==========\n', encoder)
    decoder = DINODecoder(cfg)
    print('\n========== decoder ==========\n', decoder)
    matcher = create_matcher(cfg)
    print('\n========== matcher ==========\n', matcher)
    criterion = create_criterion(cfg, matcher)
    print('\n========== criterion ==========\n', criterion)
    model = DINOModel(cfg)
    print('\n========== model ==========\n', model)


def cfg_to_dict(cfg_node):
    """Recursively convert CfgNode to a dictionary."""
    if not isinstance(cfg_node, CN):
        return cfg_node
    cfg_dict = {}
    for key, value in cfg_node.items():
        if isinstance(value, CN):
            cfg_dict[key] = cfg_to_dict(value)
        else:
            cfg_dict[key] = value
    return cfg_dict


def create_matcher(cfg):
    matcher = HungarianMatcher(
        cost_types=cfg.MODEL.MATCHER.COST_TYPES,
        cost_class=cfg.MODEL.MATCHER.COST_CLASS,
        cost_box=cfg.MODEL.MATCHER.COST_BOX,
        cost_giou=cfg.MODEL.MATCHER.COST_GIOU,
    )
    return matcher

def create_criterion(cfg, matcher):
    # loss weights
    weight_dict = dict(
        loss_ce = cfg.MODEL.CRITERION.CLASS_WEIGHT,
        loss_bbox = cfg.MODEL.CRITERION.BOX_WEIGHT,
        loss_giou = cfg.MODEL.CRITERION.GIOU_WEIGHT,
        loss_mask = cfg.MODEL.CRITERION.MASK_WEIGHT,
        loss_dice = cfg.MODEL.CRITERION.DICE_WEIGHT,
    )
    '''
    # two stage is the query selection scheme
    if cfg.MODEL.MaskDINO.TWO_STAGE:
        interm_weight_dict = {}
        interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
        weight_dict.update(interm_weight_dict)

    if cfg.MODEL.MaskDINO.DEEP_SUPERVISION:
        dec_layers = cfg.MODEL.MaskDINO.DEC_LAYERS
        aux_weight_dict = {}
        for i in range(dec_layers):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)    
    '''
    criterion = SetCriterion(
        num_classes=cfg.MODEL.DEC_HEAD.NUM_CLASSES,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=cfg.MODEL.CRITERION.NO_OBJECT_WEIGHT,
        losses=cfg.MODEL.CRITERION.LOSS_TYPES,
        num_points=cfg.MODEL.CRITERION.NUM_POINTS,
    )
    return criterion


if __name__ == "__main__":
    create_modules()
