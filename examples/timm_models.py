import timm
import torch.nn as nn
import settings
from utility.print_util import print_structure, print_model_structure


def view_models():
    model_names = timm.list_models('resnet50*', pretrained=True)
    print(f'model_names: {model_names}\n')

    model = timm.create_model(model_names[0], pretrained=True)
    print_structure(model.default_cfg, indent=2, title='model.default_cfg')
    print('\n----- model structure -----')
    print_model_structure(model)

    model_names = timm.list_models('swin*', pretrained=True)
    print(f'\nmodel_names: {model_names}\n')
    model = timm.create_model('swinv2_base_window12to24_192to384', pretrained=True)
    print_structure(model.default_cfg, indent=2, title='model.default_cfg')
    print('\n----- model structure -----')
    print_model_structure(model)


if __name__ == "__main__":
    view_models()
