from transformers import AutoImageProcessor, SwinModel
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms as T


class SwinTransformerWithHooks(pl.LightningModule):
    def __init__(self, model_name="microsoft/swin-large-patch4-window12-384", pretrained=True):
        super(SwinTransformerWithHooks, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SwinModel.from_pretrained(model_name)
        self.activations = {}
        self.names = {}
        self.hooks = []

    def forward(self, x):
        # 입력 이미지를 512x512로 전처리
        x = torch.stack([self.preprocess(img, (512, 512)) for img in x])
        print('pixel values', x.shape)

        self._register_hooks()
        with torch.no_grad():
            _ = self.model(pixel_values=x)
        self._remove_hooks()

        return self.activations, self.names

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
            self.activations[module] = (input, output)

        # 등록할 레이어 이름
        layer_names = [
            "encoder.layers.0",
            "encoder.layers.1",
            "encoder.layers.3"
        ]

        # 모델의 모든 모듈을 순회하면서 특정 레이어에 hook 설정
        for name, module in self.model.named_modules():
            if name in layer_names:
                self.names[module] = name
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
    # 랜덤 이미지 텐서 생성 (batch_size=4, channels=3, height=512, width=512)
    input_tensor = torch.rand((4, 3, 512, 512))

    # Swin Transformer 모델 인스턴스화
    swin_model = SwinTransformerWithHooks(pretrained=True)

    # 모델에 입력을 전달하여 forward 패스 실행
    activations, names = swin_model(input_tensor)

    # 특정 레이어 출력 크기 출력
    for module, (input, output) in activations.items():
        print(f"Name: {names[module]}")
        print(f"Module: {module}")
        input_shapes = get_shapes(input)
        output_shapes = get_shapes(output)
        print(f"Input shapes: {input_shapes}")
        print(f"Output shapes: {output_shapes}")
        print("=" * 50)


if __name__ == '__main__':
    main()

