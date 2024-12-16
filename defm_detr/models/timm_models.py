import torch
import numpy as np
import timm
import torch.nn as nn
from torchvision import transforms


class TimmModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=self.model.default_cfg['mean'], std=self.model.default_cfg['std'])
        ])
        self.features = {}
        self.hooks = []

    def set_hooks(self, interm_layers, layer_names):
        def hook_fn(module, input, output, layer_name):
            self.features[layer_name] = output
        
        for module, layer_name in zip(interm_layers, layer_names):
            self.hooks.append(module.register_forward_hook(lambda module, input, output, layer_name=layer_name: hook_fn(module, input, output, layer_name)))
    
    def forward(self, image):
        image_tensor = self.preprocess(image)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
        return self.features
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


class ResNet50_Clip(TimmModel):
    def __init__(self, **kwargs):
        super().__init__('resnet50_clip.cc12m')
        output_layers = kwargs.get('output_layers', ['layer1', 'layer2', 'layer3', 'layer4'])
        interm_layers = [self.model.stages[i] for i in range(4)]
        self.set_hooks(interm_layers, output_layers)



class SwinV2Timm(TimmModel):
    def __init__(self, **kwargs):
        super().__init__('swinv2_base_window12to24_192to384')
        output_layers = kwargs.get('output_layers', ['layer1', 'layer2', 'layer3', 'layer4'])
        interm_layers = [self.model.layers[i] for i in range(4)]
        self.set_hooks(interm_layers, output_layers)


def check_outputs():
    import os
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from config.config import load_config

    cfg = load_config()
    print('----- ResNet50Timm -----')
    image = np.random.randint(0, 255, (512, 512, 3)).astype(np.uint8)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).cuda()
    model = ResNet50_Clip(output_layers=cfg.backbone.output_layers).cuda()
    output = model.forward(image_tensor)
    for name, feature in output.items():
        print(f"{name}: {feature.shape}")
    model.remove_hooks()

    print('----- SwinV2Timm -----')
    image = np.random.randint(0, 255, (384, 384, 3)).astype(np.uint8)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).cuda()
    model = SwinV2Timm().cuda()
    output = model.forward(image_tensor)
    for name, feature in output.items():
        print(f"{name}: {feature.shape}")
    model.remove_hooks()


if __name__ == "__main__":
    check_outputs()

