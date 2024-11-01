import torch
from torch import nn

from modeling.utils.utils import inverse_sigmoid
from modeling.utils.shared_resources import get_bbox_embed


class DecoderHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.decoder_norm = nn.LayerNorm(cfg.MODEL.DECODER.HIDDEN_DIM)
        self.bbox_embed = nn.ModuleList([
            get_bbox_embed() for _ in range(cfg.MODEL.DECODER.NUM_LAYERS)
        ])
        
    def forward(self, hs, references):
        """디코더의 출력을 최종 예측값으로 변환
        Args:
            hs: 디코더 레이어의 출력 특징
            references: 디코더 레이어의 출력 좌표
        """
        predicted_class = self.predict_class(hs)
        predicted_box = self.predict_box(references, hs)
        # 최종 출력은 마지막 레이어의 출력만 내보낸다.
        return {
            'pred_logit': predicted_class[-1],
            'pred_box': predicted_box[-1], 
            'pred_feat': hs[-1]
        }

    def predict_class(self, hs):
        predicted_class = []
        for output in hs:
            outputs_class = self.decoder_norm(output.transpose(0, 1))
            outputs_class = outputs_class.transpose(0, 1)
            predicted_class.append(outputs_class)
        return torch.stack(predicted_class)
    
    def predict_box(self, reference, hs, ref0=None):
        device = reference[0].device
        outputs_coord_list = [ref0.to(device)] if ref0 is not None else []
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(reference[:-1], self.bbox_embed, hs):
            layer_delta_unsig = layer_bbox_embed(layer_hs).to(device)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig).to(device)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        return torch.stack(outputs_coord_list)
