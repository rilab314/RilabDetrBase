import torch
from torch import nn
import pytorch_lightning as pl

from example.dot_detector.backbone import BkbnResNet50


class DotDetector(pl.LightningModule):
    def __init__(self, num_classes):
        super(DotDetector, self).__init__()
        self.num_classes = num_classes  # 0은 background, 1-3은 red, green, blue
        self._backbone = BkbnResNet50(pretrained=True)  # ResNet50 backbone
        self._encoder = TransformerEncoder()  # 나중에 Encoder 클래스로 구현
        self._decoder = TransformerDecoder()  # 나중에 Decoder 클래스로 구현
        self._encoder_head = EncoderDetectorHead()  # 나중에 EncoderHead 클래스로 구현
        self._decoder_head = DecoderDetectorHead()  # 나중에 DecoderHead 클래스로 구현

    def forward(self, batch):
        # image.shape=(B, 3, H, W), B=batch size, 3=(R,G,B) channels, H=image height, W=image width
        # label.shape=(B, 3, N), B=batch size, N=최대 점 개수(20 개), 3=(y,x,class index) channels
        # label은 N개 이하의 실제 점 좌표와 클래스 정보를 가지고 있으며, 이미지 별로 N개 중 실제 데이터를 채우고 남는 자리는 0으로 채운다.
        # 예를 들어 N=20 인데 실제 점이 8개라면 12개의 데이터는 0으로 채운다.
        # label에서 x, y는 점의 좌표를 의미하며 이미지 전체의 좌표를 0~1 범위에서 표현한다.
        # class index로는 0에서 3까지의 정수형 클래스 번호가 들어가며 0은 background, 1~3은 각각 red, green, blue에 해당한다.
        image, label = batch

        # 사전 학습된 ResNet50 모델에 image를 입력하고 (B, C1, H/16, W/16) 크기의 feature map을 추출한다.
        bb_feature = self._backbone(image)
        # bb_feature.shape = (B, C1, H/16, W/16)
        bb_feature = bb_feature.flatten(start_dim=2)

        # transformer encoder를 이용해 새로운 특징을 생성한다.
        # en_feature.shape=(B, C2, H/16, W/16), C2=number of encoder feature map channels
        en_feature = self._encoder(bb_feature)

        # `en_feature`에서 점 후보를 Q개 찾아내고 학습 loss를 계산한다.
        # en_dots.shape=(B, C3, Q), Q=number of dot candidates, C3=6=coordinates(2) + background class(1) + number of dot classes(3)
        en_dots, en_head_loss = self._encoder_head(en_feature, label)

        # transformer decoder를 이용해 새로운 특징을 만들어낸다.
        # de_feature.shape=(B, C2, Q)
        de_feature = self._decoder(en_dots, en_feature)

        # `en_feature`에서 점 후보를 Q개 찾아내고 학습 loss를 계산한다.
        # de_dots.shape=(B, C3, Q), Q=number of output dots, C3=6=coordinates(2) + background class(1) + number of dot classes(3)
        de_dots, de_head_loss = self._decoder_head(de_feature, label, en_dots)

        return de_dots, en_head_loss, de_head_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        de_dots, en_head_loss, de_head_loss = self.forward(batch)
        loss = en_head_loss + de_head_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        de_dots, en_head_loss, de_head_loss = self.forward(batch)
        loss = en_head_loss + de_head_loss
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        de_dots, en_head_loss, de_head_loss = self.forward(batch)
        loss = en_head_loss + de_head_loss
        self.log('test_loss', loss)
        return loss
