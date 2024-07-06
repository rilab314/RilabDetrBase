from transformers import AutoImageProcessor, SwinModel
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights


class BkbnResNet50(pl.LightningModule):
    def __init__(self, pretrained=True):
        super(BkbnResNet50, self).__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        resnet = resnet50(weights=weights)
        self.resnet_layers = [nn.Sequential(*list(resnet.children())[:5]),  # Conv1, BN1, ReLU, MaxPool, Layer1
                              resnet.layer2, resnet.layer3, resnet.layer4]

    def forward(self, x):
        outputs = []
        for layer in self.resnet_layers:
            x = layer(x)
            outputs.append(x)
        return outputs[1:]


class BkbnSwinL(pl.LightningModule):
    def __init__(self,
                 model_name="microsoft/swin-large-patch4-window12-384",
                 input_resolution=(512, 512)):
        super(BkbnSwinL, self).__init__()
        self.input_resolution = input_resolution
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SwinModel.from_pretrained(model_name)
        self.target_layers = ["encoder.layers.0", "encoder.layers.1", "encoder.layers.3"]
        self.activations = []
        self.hooks = []

    def forward(self, x):
        x = torch.stack([self.preprocess(img, self.input_resolution) for img in x])
        self._register_hooks()
        with torch.no_grad():
            _ = self.model(pixel_values=x)
        self._remove_hooks()
        return self.activations

    def preprocess(self, img, size):
        transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std)
        ])
        img = T.ToPILImage()(img)
        return transform(img)

    def _register_hooks(self):
        # 특정 레이어의 출력만 저장하는 hook 함수
        def hook_fn(module, input, output):
            self.activations.append(output[0])

        # 모델의 모든 모듈을 순회하면서 특정 레이어에 hook 설정
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(hook_fn))

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def get_shapes(activation):
    if isinstance(activation, tuple):
        shapes = [get_shapes(a) for a in activation]
    else:
        if torch.is_tensor(activation):
            shapes = activation.shape
        else:
            shapes = (1,)
    return shapes


def main():
    input_tensor = torch.rand((4, 3, 512, 512))

    resnet_model = BkbnResNet50(pretrained=False)
    resnet_features = resnet_model(input_tensor)
    print("BkbnResNet50 Output Shapes:")
    for feat in resnet_features:
        print(f"  features shape: {feat.shape}")

    print("BkbnSwinL Output Shapes:")
    swin_model = BkbnSwinL()
    activations = swin_model(input_tensor)
    for name, output in zip(swin_model.target_layers, activations):
        print(f"  name: {name} / shape: {get_shapes(output)}")


if __name__ == '__main__':
    main()
