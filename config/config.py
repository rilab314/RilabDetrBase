import yaml


class CfgNode:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = CfgNode(**value)
            setattr(self, key, value)
    
    def __repr__(self):
        return self._repr_inner(0)
    
    def _repr_inner(self, indent_level):
        # 클래스의 속성들을 계층적으로 표현하기 위한 재귀 함수
        indent = '  ' * indent_level
        lines = []
        for key, value in self.__dict__.items():
            if isinstance(value, CfgNode):
                lines.append(f"{indent}{key}:\n{value._repr_inner(indent_level + 1)}")
            else:
                lines.append(f"{indent}{key}: {value}")
        return '\n'.join(lines)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, CfgNode):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


    def __contains__(self, key):
        return key in self.__dict__
    
    def __getitem__(self, key):
        if key in self.__dict__:
            return getattr(self, key)
        else:
            raise KeyError(f"Key '{key}' not found in CfgNode.")
    
    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = CfgNode(**value)
        setattr(self, key, value)
    
    def get(self, key, default=None):
        if key in self.__dict__:
            return getattr(self, key)
        else:
            return default


def load_config(yaml_file='config/deform_detr_base.yaml'):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return CfgNode(**data)


def example():
    # 사용 예시
    cfg = load_config('config/deform_detr_base.yaml')
    print('backbone_name: ', cfg.backbone.type)
    print('encoder layers: ', cfg.transformer.enc_layers)
    print('decoder layers: ', cfg.transformer.dec_layers)
    print(cfg)


if __name__ == "__main__":
    example()
