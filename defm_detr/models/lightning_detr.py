import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR

from models import build_model


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


class LitDeformableDETR(pl.LightningModule):
    def __init__(self, cfg):
        """
        LightningModule 초기화
        - 모델, 손실 함수(criterion), 후처리(postprocessors) 생성
        - 학습 관련 하이퍼파라미터(cfg) 보관
        """
        super().__init__()
        self.model, self.criterion, self.postprocessors = build_model(cfg)
        self.cfg = cfg
        self.save_hyperparameters()  # Lightning이 hparams를 자동 추적 가능
        self.loss_weights = {k: v for k, v in cfg.losses.to_dict().items() if k.endswith('_loss')}
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[LitDeformableDETR] Number of params: {n_parameters}")

    def forward(self, samples):
        """
        Lightning inference forward:
        samples: NestedTensor (B x 3 x H x W, mask)
        """
        return self.model(samples)

    def training_step(self, batch, batch_idx):
        """
        학습 스텝. batch는 (samples, targets) 튜플.
          samples = NestedTensor
          targets = List[Dict] (각 샘플별 boxes, labels, size 정보 등)
        """
        samples, targets = batch
        outputs = self.forward(samples)
        loss_dict = self.criterion(outputs, targets)
        losses = sum(loss_dict[k] * self.loss_weights[k] for k in loss_dict.keys() if k in self.loss_weights)
        for k, v in loss_dict.items():
            if k in self.loss_weights:
                self.log(f"train_{k}", v * self.loss_weights[k], prog_bar=False, batch_size=self.cfg.training.batch_size)
            else:
                self.log(f"train_{k}", v, prog_bar=False, batch_size=self.cfg.training.batch_size)
        return losses

    def validation_step(self, batch, batch_idx):
        """
        검증 스텝. batch는 (samples, targets).
        기본적으로 train_step과 비슷하게 loss를 계산.
        """
        samples, targets = batch
        outputs = self.forward(samples)
        loss_dict = self.criterion(outputs, targets)
        losses = sum(loss_dict[k] * self.loss_weights[k] for k in loss_dict.keys() if k in self.loss_weights)
        for k, v in loss_dict.items():
            if k in self.loss_weights:
                self.log(f"val_{k}", v * self.loss_weights[k], prog_bar=False, batch_size=self.cfg.training.batch_size)
            else:
                self.log(f"val_{k}", v, prog_bar=False, batch_size=self.cfg.training.batch_size)
        return losses

    def configure_optimizers(self):
        """
        Optimizer, LR Scheduler 설정
        main.py 의 param_dicts 로직을 그대로 가져옴.
        """
        lr = self.cfg.training.lr
        lr_backbone = self.cfg.training.lr_backbone
        lr_backbone_names = self.cfg.training.lr_backbone_names
        lr_linear_proj_names = self.cfg.training.lr_linear_proj_names
        lr_linear_proj_mult = self.cfg.training.lr_linear_proj_mult
        weight_decay = self.cfg.training.weight_decay
        lr_drop = self.cfg.training.lr_drop
        sgd = getattr(self.cfg.training, "sgd", False)

        param_dicts = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not match_name_keywords(n, lr_backbone_names)
                       and not match_name_keywords(n, lr_linear_proj_names)
                       and p.requires_grad
                ],
                "lr": lr,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if match_name_keywords(n, lr_backbone_names) and p.requires_grad
                ],
                "lr": lr_backbone,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if match_name_keywords(n, lr_linear_proj_names) and p.requires_grad
                ],
                "lr": lr * lr_linear_proj_mult,
            },
        ]

        if sgd:
            optimizer = torch.optim.SGD(
                param_dicts, lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                param_dicts, lr=lr, weight_decay=weight_decay
            )

        # StepLR로 단순히 lr_drop 시점에 learning rate를 낮추는 예시
        lr_scheduler = StepLR(optimizer, step_size=lr_drop, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss"  # 모니터링할 지표(필요시)
            }
        }
