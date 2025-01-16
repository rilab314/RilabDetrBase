import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import settings
from config.config import load_config
from defm_detr.pipeline.dataloader import create_dataloader
from defm_detr.models.lightning_detr import LitDeformableDETR


def train():
    cfg = load_config()
    pl.seed_everything(cfg.runtime.seed, workers=True)  # reproducibility
    tb_logger = TensorBoardLogger(save_dir=cfg.runtime.output_dir, name=cfg.runtime.logger_name)

    train_loader = create_dataloader(cfg, 'train')
    val_loader = create_dataloader(cfg, 'val')
    model_module = LitDeformableDETR(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=tb_logger,
        # 등등 필요한 옵션(ex. 로그, 체크포인트, GradientClipVal 등)
    )
    # 학습 시작
    trainer.fit(model_module, train_loader, val_loader)


if __name__ == "__main__":
    train()
