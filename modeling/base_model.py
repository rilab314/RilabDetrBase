from typing import Tuple

import torch
from detectron2.config import configurable
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances
from torch import nn
from torch.nn import functional as F

from modeling.backbone.swin import D2SwinTransformer
from modeling.encoder.maskdino_encoder import MaskDINOEncoder
from modeling.decoder.maskdino_decoder import MaskDINODecoder
from modeling.criterion import SetCriterion
from modeling.matcher import HungarianMatcher
from modeling.utils import box_ops

from modeling.utils.print_util import print_structure

class DETRBaseModel(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        encoder: nn.Module,
        decoder: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        print('criterion.weight_dict ', self.criterion.weight_dict)

    @classmethod
    def from_config(cls, cfg):
        # build modules
        backbone = D2SwinTransformer(cfg)
        encoder = MaskDINOEncoder(cfg, backbone.output_shape())
        decoder = MaskDINODecoder(cfg, in_channels=cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM, mask_classification=True)
        # build matcher
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
        # build criterion
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

        return {
            "backbone": backbone,
            "encoder": encoder,
            "decoder": decoder,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES,
            "size_divisibility": cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    def forward(self, x, target):
        pass


