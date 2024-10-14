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

from .decoder_layer import TransformerDecoder, DeformableTransformerDecoderLayer
from modeling.utils.utils import MLP, inverse_sigmoid


class DINODecoder(nn.Module):
    @configurable
    def __init__(
            self,
            in_channels,
            hidden_dim: int,
            dim_feedforward: int,
            dec_layers: int,
            enforce_input_project: bool,
            dn: str,
            noise_scale: float,
            dn_num: int,
            total_num_feature_levels: int = 4,
            dropout: float = 0.0,
            activation: str = 'relu',
            nhead: int = 8,
            dec_n_points: int = 4,
            query_dim: int = 4,
            dec_layer_share: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            hidden_dim: Transformer feature dimension
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            d_model: transformer dimension
            dropout: dropout rate
            activation: activation function
            nhead: num heads in multi-head attention
            dec_n_points: number of sampling points in decoder
            query_dim: 4 -> (x, y, w, h)
            dec_layer_share: whether to share each decoder layer
        """
        super().__init__()

        self.num_feature_levels = total_num_feature_levels
        # define Transformer decoder here
        self.dn = dn
        self.noise_scale = noise_scale
        self.dn_num = dn_num
        self.num_layers = dec_layers
        self.total_num_feature_levels = total_num_feature_levels

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # init decoder
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward,
                                                          dropout, activation,
                                                          self.num_feature_levels, nhead, dec_n_points)
        self.decoder = TransformerDecoder(decoder_layer, self.num_layers, decoder_norm,
                                          d_model=hidden_dim, query_dim=query_dim,
                                          dec_layer_share=dec_layer_share,
                                          )

        self.hidden_dim = hidden_dim
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for i in range(self.num_layers)]  # share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.decoder.bbox_embed = self.bbox_embed

    @classmethod
    def from_config(cls, cfg):
        ret = {}
        ret["in_channels"] = cfg.MODEL.ENCODER.HIDDEN_DIM
        ret["hidden_dim"] = cfg.MODEL.DECODER.HIDDEN_DIM
        # Transformer parameters:
        ret["dim_feedforward"] = cfg.MODEL.DECODER.DIM_FEEDFORWARD
        ret["dec_layers"] = cfg.MODEL.DECODER.NUM_LAYERS
        ret["enforce_input_project"] = cfg.MODEL.DECODER.ENFORCE_INPUT_PROJ
        ret["dn"]= cfg.MODEL.DECODER.DN
        ret["noise_scale"] = cfg.MODEL.DECODER.DN_NOISE_SCALE
        ret["dn_num"] = cfg.MODEL.DECODER.DN_NUM
        ret["total_num_feature_levels"] = cfg.MODEL.DECODER.TOTAL_NUM_FEATURE_LEVELS
        return ret

    def pred_box(self, reference, hs, ref0=None):
        """
        :param reference: reference box coordinates from each decoder layer
        :param hs: content
        :param ref0: whether there are prediction from the first layer
        """
        device = reference[0].device

        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0.to(device)]
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs).to(device)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig).to(device)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list

    def forward(self, ms_features, target, refpoint_unsigmoid, masks):
        """
        :param ms_features: input, a list of multi-scale feature
            encoder에서 출력하는 multi_scale_features (5 levels) [[1, 256, 200, 304],..., [1, 256, 13, 19]]
        :param target: query features,[bs, nq, d_model]  It could be from encoder or learnable embedding
        :param refpoint_unsigmoid:  unsigmoided points or boxes [bs, nq, 2 or 4]
        :param masks: mask in the original image
        """
        assert len(ms_features) == self.num_feature_levels
        size_list = []
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in ms_features:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in ms_features]

        # feature map flatten 해서 하나로 합치고, level_start_index 만들기
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
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # 원래 이 사이에 target과 refpoints_unsigmoid를 만들기 위한 많은 코드가 있었음

        predictions_class = []
        hs, references = self.decoder(
            tgt=target.transpose(0, 1),  # query vectors
            memory=src_flatten.transpose(0, 1),  # key, value from encoder features
            refpoints_unsigmoid=refpoint_unsigmoid.transpose(0, 1),  # query 좌표를 unsigmoid 한 것 (cx,cy,w,h)
            memory_key_padding_mask=mask_flatten,
            level_start_index=level_start_index,  # memory 의 레벨별 feature의 시작 index
            spatial_shapes=spatial_shapes,  # memory에 들어있는 feature shapes
        )                      # MultiheadAttention.forward() 참조
        # hs: output features of decoder layers
        # references: output points of decoder layers (refined points from query)

        for i, output in enumerate(hs):
            # decoder_norm: LayerNorm
            outputs_class = self.decoder_norm(output.transpose(0, 1))
            outputs_class = outputs_class.transpose(0, 1)
            predictions_class.append(outputs_class)
        out_boxes = self.pred_box(references, hs)

        # 최종 출력은 마지막 레이어의 출력만 내보낸다.
        out = {'pred_logit': predictions_class[-1],
               'pred_box': out_boxes[-1],
               'pred_feat': hs[-1]
              }
        return out
