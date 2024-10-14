from typing import Optional, List, Union
import torch
from torch import nn, Tensor
from torch.cuda.amp import autocast

from modeling.utils.utils import MLP, _get_clones, _get_activation_fn, gen_sineembed_for_position, inverse_sigmoid
from modeling.encoder.ops.modules import MSDeformAttn


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_deformable_box_attn=False,
                 key_aware_type=None,
                 ):
        super().__init__()

        # cross attention
        if use_deformable_box_attn:
            raise NotImplementedError
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


@autocast(enabled=False)
def forward(self,
            # for tgt
            tgt: Optional[Tensor],  # nq, bs, d_model
            tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
            tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
            # for memory
            memory: Optional[Tensor] = None,  # hw, bs, d_model
            memory_level_start_index: Optional[Tensor] = None,  # num_levels
            memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
            memory_key_padding_mask: Optional[Tensor] = None,  # None으로 놔둬도 될듯
            # sa
            self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
            ):
    """
    Input:
        - tgt/tgt_query_pos: nq, bs, d_model
        -
    """
    # self attention
    if self.self_attn is not None:
        q = k = self.with_pos_embed(tgt, tgt_query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

    # cross attention
    if self.key_aware_type is not None:
        if self.key_aware_type == 'mean':
            tgt = tgt + memory.mean(0, keepdim=True)
        elif self.key_aware_type == 'proj_mean':
            tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
        else:
            raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
    tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                           tgt_reference_points.transpose(0, 1).contiguous(),
                           memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index,
                           memory_key_padding_mask).transpose(0, 1)
    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)
    # ffn
    tgt = self.forward_ffn(tgt)
    return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None,
                 d_model=256, query_dim=4,
                 dec_layer_share=False,
                 ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.d_model = d_model
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, tgt, memory,
                refpoints_unsigmoid: Optional[Tensor] = None,  # np, bs, 2
                memory_key_padding_mask: Optional[Tensor] = None,
                level_start_index: Optional[Tensor] = None,  # num_levels
                spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt
        device = tgt.device

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid().to(device)
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            query_sine_embed = gen_sineembed_for_position(reference_points) # nq, bs, 256*2
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = 1  # self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_reference_points=reference_points,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
            )

            # iter update
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output).to(device)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
        ]
