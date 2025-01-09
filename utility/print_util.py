import numpy as np
import torch.nn as nn
import torch


def print_data(data, indent=2, level=0, title=None):
    indent_str = ' ' * indent * level  # 들여쓰기 공백
    if title is not None:
        print(f"{indent_str}<------ {title}")
    if len(str(data)) < 150:
        print(f"{indent_str}{data}")
        return

    if isinstance(data, dict):
        for key, value in data.items():                    
            if isinstance(value, (dict, list)):
                print(f"{indent_str}{key}:")
                print_data(value, indent, level+1)
            else:
                value_str = str(value)
                if len(value_str) <= 100:
                    print(f"{indent_str}{key}: {value_str}")
                elif isinstance(value, torch.Tensor):
                    print(f"{indent_str}{key}: tensor{tuple(value.shape)}")
                elif isinstance(value, np.ndarray):
                    print(f"{indent_str}{key}: np{value.shape}")
                else:
                    print(f"{indent_str}{key}: {str(value)[:100]}")
    elif isinstance(data, list):
        for key, value in enumerate(data):
            if isinstance(value, (dict, list)):
                print(f"{indent_str}{key}:")
                print_data(value, indent, level+1)
            else:
                value_str = str(value)
                if len(value_str) <= 100:
                    print(f"{indent_str}{key}: {value_str}")
                elif isinstance(value, torch.Tensor):
                    print(f"{indent_str}{key}: {value.shape}")
                elif isinstance(value, np.ndarray):
                    print(f"{indent_str}{key}: {value.shape}")
                else:
                    print(f"{indent_str}{key}: {str(value)[:100]} ...")
    if title is not None:
        print(f"{indent_str}------>")



def print_model(model, max_depth=None):
    """
    모델의 계층적 구조를 번호와 함께 들여쓰기로 출력하는 함수
    모델의 모든 모듈을 탐색하고, 각 계층에 번호를 붙여서 출력합니다.
    """
    def print_layers(module, level, prefix, child_name=None, depth=0):
        if max_depth is not None and depth > max_depth:
            return
        level = level + '.' if level else ''
        if isinstance(module, nn.Module):
            module_name = module.__class__.__name__
            if hasattr(module, 'named_children') and len(list(module.named_children())) > 0:
                if child_name:
                    print(f"{prefix}{level} {module_name} (name: {child_name})")
                else:
                    print(f"{prefix}{level}{module_name}")
            elif module_name == 'Conv2d':
                print(f"{prefix}{level} {module}")
            elif module_name == 'Linear':
                print(f"{prefix}{level} {module}")
            else:
                print(f"{prefix}{level} {module_name}")

            # 해당 레이어의 자식 모듈들이 있다면 재귀적으로 호출
            for i, (child_name, child) in enumerate(module.named_children(), 1):
                # 번호를 붙여서 자식 모듈을 출력
                print_layers(child, f"{level}{i}", f"{prefix}  ", child_name, depth+1)

    print_layers(model, "", "", None, 0)
