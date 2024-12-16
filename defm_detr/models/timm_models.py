import torch
import numpy as np
import timm
import torch.nn as nn
from torchvision import transforms
import copy


class ResNet50Timm(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('resnet50.a1_in1k', pretrained=True)
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=self.model.default_cfg['mean'], std=self.model.default_cfg['std'])
        ])
        # self.layer5 = copy.deepcopy(self.model.layer4)
        self.set_hooks(self.model)

    def set_hooks(self, model):
        self.hooks = []
        def hook_fn(module, input, output, layer_name):
            if layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                self.features[layer_name] = output
        
        self.features = {}
        self.hooks.append(model.layer1.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, 'layer1')))
        self.hooks.append(model.layer2.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, 'layer2')))
        self.hooks.append(model.layer3.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, 'layer3')))
        self.hooks.append(model.layer4.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, 'layer4')))
    
    def forward(self, image):
        image_tensor = self.preprocess(image)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
        # self.features['layer5'] = self.layer5(self.features['layer4'])
        return self.features
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


class SwinV2Timm(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('swinv2_base_window12to24_192to384', pretrained=True)
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=self.model.default_cfg['mean'], std=self.model.default_cfg['std'])
        ])
        self.set_hooks(self.model)

    def set_hooks(self, model):
        self.hooks = []
        def hook_fn(module, input, output, layer_name):
            print(f"layer_name: {layer_name}, output: {output.shape}")
            if layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                self.features[layer_name] = output
        
        self.features = {}
        self.hooks.append(model.layers[0].register_forward_hook(lambda module, input, output: hook_fn(module, input, output, 'layer1')))
        self.hooks.append(model.layers[1].register_forward_hook(lambda module, input, output: hook_fn(module, input, output, 'layer2')))
        self.hooks.append(model.layers[2].register_forward_hook(lambda module, input, output: hook_fn(module, input, output, 'layer3')))
        self.hooks.append(model.layers[3].register_forward_hook(lambda module, input, output: hook_fn(module, input, output, 'layer4')))
    
    def forward(self, image):
        image_tensor = self.preprocess(image)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
        # self.features['layer5'] = self.layer5(self.features['layer4'])
        return self.features
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def check_outputs():
    print('----- ResNet50Timm -----')
    image = np.random.randint(0, 255, (512, 512, 3)).astype(np.uint8)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).cuda()
    model = ResNet50Timm().cuda()
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

