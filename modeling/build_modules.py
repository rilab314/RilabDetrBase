from detectron2.config import CfgNode as CN

from modeling.utils.print_util import print_structure
from modeling.setup_cfg import setup_cfg
from modeling.backbone.swin import D2SwinTransformer
from modeling.encoder.maskdino_encoder import MaskDINOEncoder
from modeling.decoder.maskdino_decoder import MaskDINODecoder
from modeling.matcher import HungarianMatcher
from modeling.criterion import SetCriterion
from modeling.maskdino_model import MaskDINO


def create_modules():
    cfg = setup_cfg()
    print('\n========== config ==========\n', cfg)
    backbone = D2SwinTransformer(cfg)
    print('\n========== backbone ==========\n', backbone)
    encoder = MaskDINOEncoder(cfg, backbone.output_shape())
    print('\n========== encoder ==========\n', encoder)
    return
    decoder = MaskDINODecoder(cfg, in_channels=cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM, mask_classification=True)
    print('\n========== decoder ==========\n', decoder)
    matcher = create_matcher(cfg)
    print('\n========== matcher ==========\n', matcher)
    criterion = create_criterion(cfg, matcher)
    print('\n========== criterion ==========\n', criterion)
    model = MaskDINO(cfg)
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
    cost_class_weight = cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT
    cost_dice_weight = cfg.MODEL.MaskDINO.COST_DICE_WEIGHT
    cost_mask_weight = cfg.MODEL.MaskDINO.COST_MASK_WEIGHT  #
    cost_box_weight = cfg.MODEL.MaskDINO.COST_BOX_WEIGHT
    cost_giou_weight = cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT
    matcher = HungarianMatcher(
        cost_class=cost_class_weight,
        cost_mask=cost_mask_weight,
        cost_dice=cost_dice_weight,
        cost_box=cost_box_weight,
        cost_giou=cost_giou_weight,
        num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
    )
    return matcher


def create_criterion(cfg, matcher):
    num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    # Loss parameters:
    deep_supervision = cfg.MODEL.MaskDINO.DEEP_SUPERVISION
    no_object_weight = cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT

    # loss weights
    class_weight = cfg.MODEL.MaskDINO.CLASS_WEIGHT
    dice_weight = cfg.MODEL.MaskDINO.DICE_WEIGHT
    mask_weight = cfg.MODEL.MaskDINO.MASK_WEIGHT
    box_weight = cfg.MODEL.MaskDINO.BOX_WEIGHT
    giou_weight = cfg.MODEL.MaskDINO.GIOU_WEIGHT

    weight_dict = {"loss_ce": class_weight}
    weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
    weight_dict.update({"loss_bbox": box_weight, "loss_giou": giou_weight})
    # two stage is the query selection scheme
    if cfg.MODEL.MaskDINO.TWO_STAGE:
        interm_weight_dict = {}
        interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
        weight_dict.update(interm_weight_dict)

    if deep_supervision:
        dec_layers = cfg.MODEL.MaskDINO.DEC_LAYERS
        aux_weight_dict = {}
        for i in range(dec_layers):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    if cfg.MODEL.MaskDINO.BOX_LOSS:
        losses = ["labels", "masks", "boxes"]
    else:
        losses = ["labels", "masks"]

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=no_object_weight,
        losses=losses,
        num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
        oversample_ratio=cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO,
        importance_sample_ratio=cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO,
        panoptic_on=cfg.MODEL.MaskDINO.PANO_BOX_LOSS,
        semantic_ce_loss=cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
    )
    return criterion


if __name__ == "__main__":
    create_modules()
