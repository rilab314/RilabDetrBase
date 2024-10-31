# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
from typing import Tuple

import torch
from detectron2.config import configurable
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances
from torch import nn
from torch.nn import functional as F

from modeling.backbone.swin import D2SwinTransformer
from modeling.encoder import DINOEncoder
from modeling.encoder.encoder_head import EncoderHead
from modeling.decoder import DINODecoder
from modeling.criterion import SetCriterion
from modeling.matcher import HungarianMatcher
from modeling.utils import box_ops
from modeling.utils.utils import MLP, inverse_sigmoid
from modeling.utils.shared_resources import get_bbox_embed
from modeling.utils.print_util import print_structure


class DINOModel(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """
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
        backbone = D2SwinTransformer(cfg)
        encoder = DINOEncoder(cfg, backbone.output_shape())
        bbox_embed = get_bbox_embed(cfg.MODEL.DECODER.HIDDEN_DIM, cfg.MODEL.DECODER.HIDDEN_DIM)
        encd_head = EncoderHead(cfg)
        decoder = DINODecoder(cfg)
        decd_head = DecoderHead(cfg)
        matcher = HungarianMatcher(
            cost_types=cfg.MODEL.MATCHER.COST_TYPES,
            cost_class=cfg.MODEL.MATCHER.COST_CLASS,
            cost_box=cfg.MODEL.MATCHER.COST_BOX,
            cost_giou=cfg.MODEL.MATCHER.COST_GIOU,
        )
        weight_dict = dict(
            loss_ce=cfg.MODEL.CRITERION.CLASS_WEIGHT,
            loss_bbox=cfg.MODEL.CRITERION.BOX_WEIGHT,
            loss_giou=cfg.MODEL.CRITERION.GIOU_WEIGHT,
            loss_mask=cfg.MODEL.CRITERION.MASK_WEIGHT,
            loss_dice=cfg.MODEL.CRITERION.DICE_WEIGHT,
        )
        criterion = SetCriterion(
            num_classes=cfg.MODEL.DEC_HEAD.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.MODEL.CRITERION.NO_OBJECT_WEIGHT,
            losses=cfg.MODEL.CRITERION.LOSS_TYPES,
            num_points=cfg.MODEL.CRITERION.NUM_POINTS,
        )

        return {
            "backbone": backbone,
            "encoder": encoder,
            "decoder": decoder,
            "criterion": criterion,
            "num_queries": cfg.MODEL.DECODER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.TEST.OVERLAP_THRESHOLD,
            "size_divisibility": cfg.MODEL.BACKBONE.DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list of dicts that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * "height": int, the output resolution of the model
                   * "width": int, the output resolution of the model
                   * "filename": str, the filename of the image
        Returns:
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        bkbn_ms_features = self.backbone(images.tensor)
        encd_ms_features = self.encoder.forward_features(bkbn_ms_features)
        print_structure(bkbn_ms_features, 'backbone')
        print_structure(encd_ms_features, 'encoder')

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images)
            losses = self.head_forward(encd_ms_features, targets)
            if self.dn:
                losses_dn = self.head_forward(encd_ms_features, targets, dn=True)
                losses.update(losses_dn)
            return losses
        else:
            outputs = self.head_forward(encd_ms_features)
            return outputs

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            # gt_masks = targets_per_image.gt_masks
            # padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            # padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "boxes": box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
                    # "masks": padded_masks,
                }
            )
        return new_targets

    def head_forward(self, ms_features, targets, dn=False):
        if dn:
            query_features, refpoint_unsigmoid, targets = self.prepare_dn(ms_features, targets)
        else:
            query_features, refpoint_unsigmoid = self.encoder_head(ms_features)
        decoder_features = self.decoder(ms_features, query_features, refpoint_unsigmoid)
        predictions = self.decoder_head(decoder_features)
        losses = self.criterion(predictions, targets, dn=dn)
        return losses

    def semantic_inference(self, mask_cls, mask_pred):
        # if use cross-entropy loss in training, evaluate with softmax
        if self.semantic_ce_loss:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg
        # if use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        else:
            T = self.pano_temp
            mask_cls = mask_cls.sigmoid()
            if self.transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        # As we use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        prob = 0.5
        T = self.pano_temp
        scores, labels = mask_cls.sigmoid().max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        # added process
        if self.transform_eval:
            scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, mask_box_result):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # half mask box half pred box
        mask_box_result = mask_box_result[topk_indices]
        if self.panoptic_on:
            mask_box_result = mask_box_result[keep]
        result.pred_boxes = Boxes(mask_box_result)
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes


