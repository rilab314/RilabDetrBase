import torch.nn as nn


def print_structure(data, indent=0, title=None):
    print(title)
    indent_str = '  ' * indent  # 들여쓰기 공백
    if isinstance(data, dict):
        for key, value in data.items():                    
            # 길이가 100 이하이면 한 줄로 출력
            if isinstance(value, (dict, list)):
                print(f"{indent_str}{key}:")
                print_structure(value, indent + 1)
            else:
                value_str = str(value)
                if len(value_str) <= 100:
                    print(f"{indent_str}{key}: {value_str}")
                else:
                    print(f"{indent_str}{key}:")
                    print_structure(value, indent + 2)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                print_structure(item, indent)
            else:
                item_str = str(item)
                if len(item_str) <= 100:
                    print(f"{indent_str}- {item_str}")
                else:
                    print(f"{indent_str}-")
                    print_structure(item, indent + 2)


def print_model_structure(model, max_depth=None):
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
