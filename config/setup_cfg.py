import argparse
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import CfgNode as CN
import yaml
import os

from modeling.utils.print_util import print_structure


def setup_cfg():
    args = get_parser().parse_args()
    cfg = load_cfg_from_yaml(args.config_file)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description='maskdino demo for builtin configs')
    parser.add_argument(
        '--config-file',
        default='/home/dolphin/choi_ws/SatLaneDet_2024/SatelliteLaneDet2024/config/dino_model.yaml',
        metavar='FILE',
        help='path to config file',
    )
    return parser


def load_cfg_from_yaml(yaml_file):
    print('load_cfg_from_yaml:', yaml_file)
    with open(yaml_file, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    
    cfg = get_cfg()
    if '_BASE_' in yaml_cfg:
        base_file = yaml_cfg.pop('_BASE_')
        base_path = os.path.join(os.path.dirname(yaml_file), base_file)
        with open(base_path, 'r') as f:
            base_cfg = yaml.safe_load(f)
        cfg = create_cfg_node(cfg, base_cfg)
    
    cfg = create_cfg_node(cfg, yaml_cfg)
    return cfg


def create_cfg_node(cfg_node, yaml_dict):
    for key, value in yaml_dict.items():
        if isinstance(value, dict):
            child = cfg_node[key] if key in cfg_node else CN()
            cfg_node[key] = create_cfg_node(child, value)
        elif isinstance(value, str) and value.lower() == 'none':
            cfg_node[key] = None
        else:
            cfg_node[key] = value
    return cfg_node


def save_cfg_to_yaml(cfg, output_yaml_file):
    cfg_dict = cfg_to_dict(cfg)
    with open(output_yaml_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def cfg_to_dict(cfg_node):
    if not isinstance(cfg_node, CN):
        return cfg_node
    cfg_dict = {}
    for key, value in cfg_node.items():
        if isinstance(value, CN):
            cfg_dict[key] = cfg_to_dict(value)
        elif isinstance(value, tuple):
            cfg_dict[key] = list(value)
        else:
            cfg_dict[key] = value
    return cfg_dict


if __name__ == '__main__':
    cfg = setup_cfg()
    print('===== setup cfg =====\n', cfg)
    out_file = '/home/dolphin/choi_ws/SatLaneDet_2024/SatelliteLaneDet2024/modeling/configs/base_model_out.yaml'
    save_cfg_to_yaml(cfg, out_file)

