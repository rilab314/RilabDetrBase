import argparse
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import CfgNode as CN


def setup_cfg():
    args = get_parser().parse_args()
    cfg = get_cfg()
    print('\n========== empty config ==========\n', cfg)
    add_detr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskdino demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/home/dolphin/choi_ws/SatLaneDet_2024/SatelliteLaneDet2024/modeling/configs/base_model.yaml",
        metavar="FILE",
        help="path to config file",
    )
    return parser


def cfg_to_dict(cfg_node):
    if not isinstance(cfg_node, CN):
        return cfg_node
    cfg_dict = {}
    for key, value in cfg_node.items():
        if isinstance(value, CN):
            cfg_dict[key] = cfg_to_dict(value)
        else:
            cfg_dict[key] = value
    return cfg_dict


def add_detr_config(cfg):
    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.WEIGHTS = ''

    # direct model configs
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = 'Swin'
    cfg.MODEL.BACKBONE.DIVISIBILITY = 32

    # encoder/decoder common config
    cfg.MODEL.ENCODER = CN()
    cfg.MODEL.ENCODER.ENCODER_NAME = "MaskDINOEncoder"
    cfg.MODEL.ENCODER.DIM_FEEDFORWARD = 1024
    cfg.MODEL.ENCODER.COMMON_STRIDE = 4
    cfg.MODEL.ENCODER.NUM_LAYERS = 6  # TRANSFORMER_ENC_LAYERS
    cfg.MODEL.ENCODER.NORM = "GN"
    cfg.MODEL.ENCODER.HIDDEN_DIM = 256
    cfg.MODEL.ENCODER.NHEADS = 8
    cfg.MODEL.ENCODER.NUM_CLASSES = 10


if __name__ == "__main__":
    cfg = setup_cfg()

