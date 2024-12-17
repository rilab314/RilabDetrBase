import timm
import torch.nn as nn
import settings
from utility.print_util import print_structure, print_model_structure


def view_models():
    model_names = timm.list_models('resnet50*', pretrained=True)
    print(f'model_names: {model_names}\n')
    model = timm.create_model('resnet50_clip.cc12m', pretrained=True)
    print_structure(model.default_cfg, indent=2, title='----- model.default_cfg -----')
    print('\n----- model structure -----')
    print_model_structure(model)

    model_names = timm.list_models('swin*', pretrained=True)
    print(f'\nmodel_names: {model_names}\n')
    model = timm.create_model('swin_base_patch4_window12_384.ms_in22k', pretrained=True)
    print_structure(model.default_cfg, indent=2, title='model.default_cfg')
    print('\n----- model structure -----')
    print_model_structure(model)


if __name__ == "__main__":
    view_models()
