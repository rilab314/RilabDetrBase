# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
import logging
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from detectron2.config import configurable
from detectron2.layers import Conv2d

from modeling.utils.utils import gen_encoder_output_proposals
from modeling.utils.shared_resources import get_bbox_embed


class EncoderHead(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_output_norm = nn.LayerNorm(hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self._bbox_embed = get_bbox_embed()
    
    def forward(self, ms_features):
        masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in ms_features]
        # feature map flatten 해서 하나로 합치고, level_start_index 만들기
        size_list = []
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx=self.num_feature_levels-1-i
            size_list.append(ms_features[i].shape[-2:])
            spatial_shapes.append(ms_features[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](ms_features[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))

        # src_flatten: encoder의 multi-scale feature를 linear projection 해서 한 줄로 합친것
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)

        output_memory, output_proposals = gen_encoder_output_proposals(src_flatten, mask_flatten, spatial_shapes)
        # Linear + LayerNorm
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        # class_embed: Linear
        enc_outputs_class_unselected = self.class_embed(output_memory)
        # _box_embed: MLP
        enc_outputs_coord_unselected = self._bbox_embed(output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
        topk = self.num_queries
        # 클래스 확률로 topk 인덱스 뽑고 거기에 해당하는 박스 뽑아
        topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
        refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid
        refpoint_embed = refpoint_embed_undetach.detach()
        # topk에 해당하는 input feature 도 뽑아
        tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))  # unsigmoid
        tgt = tgt_undetach.detach()
        return tgt, refpoint_embed
