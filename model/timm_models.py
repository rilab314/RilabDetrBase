import torch
import numpy as np
import timm
import torch.nn as nn
from torchvision import transforms
from typing import List
from dataclasses import dataclass

import settings
from model.backbone import Joiner, build_position_encoding
from model.position_encoding import NestedTensor
from configs.config import CfgNode

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@dataclass
class LayerInfo:
    name: str
    stride: int
    channels: int
    module: nn.Module


class TimmModel(nn.Module):
    def __init__(self, model_name, pretrained=True, output_names=List[str]):
        super().__init__()
        self._model = timm.create_model(model_name, pretrained=pretrained).to(device)
        self._preprocess = transforms.Compose([
            transforms.Normalize(mean=self._model.default_cfg['mean'], std=self._model.default_cfg['std'])
        ])
        self._output_layers = output_names
        self._interm_layers = []
        self._features = {}
        self._hooks = []

    def set_hooks(self, interm_layers, output_names):
        def hook_fn(module, input, output, layer_name):
            self._features[layer_name] = output
        
        output_layers = [layer for layer in interm_layers if layer.name in output_names]
        for layer_info in output_layers:
            self._hooks.append(layer_info.module.register_forward_hook(lambda module, input, output, layer_name=layer_info.name: hook_fn(module, input, output, layer_name)))
    
    def forward(self, sample: NestedTensor):
        """
        samples: NestedTensor
            - samples.tensor: batched images, [B, 3, H, W]
            - samples.mask: batched binary mask [B, H, W], containing 1 on padded pixels
        """
        image_tensor = self._preprocess(sample.tensors)
        self._model.eval()
        with torch.no_grad():
            output = self._model(image_tensor)
        return self._post_process(self._features)
    
    def _post_process(self, features):
        tensors = {}
        for name, feature in features.items():
            B, C, H, W = feature.shape
            mask = torch.zeros((B, H, W), dtype=torch.bool).to(device)
            tensors[name] = NestedTensor(feature, mask)
        return tensors
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    @property
    def strides(self):
        return [layer.stride for layer in self._interm_layers if layer.name in self._output_layers]
    
    @property
    def num_channels(self):
        return [layer.channels for layer in self._interm_layers if layer.name in self._output_layers]


class ResNet50_Clip(TimmModel):
    def __init__(self, output_names):
        super().__init__('resnet50_clip.cc12m', pretrained=True, output_names=output_names)
        self._interm_layers = [LayerInfo(name='layer1', stride=4, channels=128, module=self._model.stages[0]),
                               LayerInfo(name='layer2', stride=8, channels=256, module=self._model.stages[1]),
                               LayerInfo(name='layer3', stride=16, channels=512, module=self._model.stages[2]),
                               LayerInfo(name='layer4', stride=32, channels=1024, module=self._model.stages[3])]
        self.set_hooks(self._interm_layers, self._output_layers)


class SwinV2_384(TimmModel):
    def __init__(self, output_names):
        super().__init__('swin_base_patch4_window12_384.ms_in22k', output_names=output_names)
        self._interm_layers = [LayerInfo(name='layer1', stride=4, channels=128, module=self._model.layers[0]),
                               LayerInfo(name='layer2', stride=8, channels=256, module=self._model.layers[1]),
                               LayerInfo(name='layer3', stride=16, channels=512, module=self._model.layers[2]),
                               LayerInfo(name='layer4', stride=32, channels=1024, module=self._model.layers[3])]
        self.set_hooks(self._interm_layers, self._output_layers)

    def _post_process(self, features):
        tensors = {}
        for name, feature in features.items():
        # (B, H, W, C) -> (B, C, H, W)
            feature = feature.permute(0, 3, 1, 2).contiguous()
            B, C, H, W = feature.shape
            mask = torch.zeros((B, H, W), dtype=torch.bool).to(device)
            tensors[name] = NestedTensor(feature, mask)
        return tensors


def build_hf_backbone(cfg):
    if cfg.backbone.type == 'ResNet50_Clip':
        backbone = ResNet50_Clip(output_names=cfg.backbone.output_layers)
    elif cfg.backbone.type == 'SwinV2_384':
        backbone = SwinV2_384(output_names=cfg.backbone.output_layers)
    else:
        raise ValueError(f"Backbone {cfg.backbone.type} not supported")
    
    position_embedding = build_position_encoding(cfg)
    model = Joiner(backbone, position_embedding)
    return model
