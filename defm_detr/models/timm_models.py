import torch
import numpy as np
import timm
import torch.nn as nn
from torchvision import transforms
from typing import List
import os
import sys
detr_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(detr_root)
if detr_root not in sys.path:
    sys.path.append(detr_root)
if project_root not in sys.path:
    sys.path.append(project_root)


from defm_detr.models.backbone import Joiner, build_position_encoding
from defm_detr.models.position_encoding import NestedTensor
from config.config import load_config

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TimmModel(nn.Module):
    def __init__(self, model_name, pretrained=True, output_layers=List[str]):
        super().__init__()
        self._model = timm.create_model(model_name, pretrained=pretrained).to(device)
        self._preprocess = transforms.Compose([
            transforms.Normalize(mean=self._model.default_cfg['mean'], std=self._model.default_cfg['std'])
        ])
        self._layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        self._output_layers = output_layers
        strides = [4, 8, 16, 32]
        self._strides = [strides[i] for i, name in enumerate(self._layer_names) if name in self._output_layers]
        self._channels = []
        self._features = {}
        self._hooks = []

    def set_hooks(self, interm_layers, layer_names):
        def hook_fn(module, input, output, layer_name):
            self._features[layer_name] = output
        
        for module, layer_name in zip(interm_layers, layer_names):
            self._hooks.append(module.register_forward_hook(lambda module, input, output, layer_name=layer_name: hook_fn(module, input, output, layer_name)))
    
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
            mask = torch.zeros((B, H, W), dtype=torch.bool)
            tensors[name] = NestedTensor(feature, mask)
        return tensors
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    @property
    def strides(self):
        return self._strides
    
    @property
    def num_channels(self):
        return self._channels


class ResNet50_Clip(TimmModel):
    def __init__(self, output_layers):
        super().__init__('resnet50_clip.cc12m', pretrained=True, output_layers=output_layers)
        interm_layers = [self._model.stages[i] for i in range(4)]
        self.set_hooks(interm_layers, self.layer_names)
        channels = [2048, 1024, 512, 256]
        self._channels = [channels[i] for i, name in enumerate(self._layer_names) if name in self._output_layers]


class SwinV2_384(TimmModel):
    def __init__(self, output_layers):
        super().__init__('swin_base_patch4_window12_384.ms_in22k', output_layers=output_layers)
        interm_layers = [self._model.layers[i] for i in range(4)]
        self.set_hooks(interm_layers, self._layer_names)
        channels = [2048, 1024, 512, 256]
        self._channels = [channels[i] for i, name in enumerate(self._layer_names) if name in self._output_layers]

    def _post_process(self, features):
        tensors = {}
        for name, feature in features.items():
        # (B, H, W, C) -> (B, C, H, W)
            feature = feature.permute(0, 3, 1, 2).contiguous()
            B, C, H, W = feature.shape
            mask = torch.zeros((B, H, W), dtype=torch.bool).to(device)
            tensors[name] = NestedTensor(feature, mask)
        return tensors


def build_backbone(cfg):
    if cfg.backbone.type == 'ResNet50_Clip':
        backbone = ResNet50_Clip(output_layers=cfg.backbone.output_layers)
    elif cfg.backbone.type == 'SwinV2_384':
        backbone = SwinV2_384(output_layers=cfg.backbone.output_layers)
    else:
        raise ValueError(f"Backbone {cfg.backbone.type} not supported")
    
    position_embedding = build_position_encoding(cfg)
    model = Joiner(backbone, position_embedding)
    return model


def check_outputs():
    cfg = load_config()
    model = build_backbone(cfg)
    if cfg.backbone.type == 'ResNet50_Clip':
        image = np.random.rand(1, 3, 512, 512)
        mask = np.zeros((1, 512, 512), dtype=bool)
    elif cfg.backbone.type == 'SwinV2_384':
        image = np.random.rand(1, 3, 384, 384)
        mask = np.zeros((1, 384, 384), dtype=bool)
    sample = NestedTensor(torch.from_numpy(image).to(torch.float32).to(device), 
                          torch.from_numpy(mask).to(torch.bool).to(device))
    xs, pos_embeds = model.forward(sample)
    print(f"xs: {len(xs)}")
    for x, pos in zip(xs, pos_embeds):
        print(f"  x: {x.tensors.shape}, pos: {pos.shape}")


if __name__ == "__main__":
    check_outputs()

