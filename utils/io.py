import os
import yaml
import torch


def load_key(key_path):
    """
    加载水印密钥文件
    """
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"密钥文件不存在: {key_path}")
    with open(key_path, 'r', encoding='utf-8') as f:
        key = yaml.safe_load(f)
    return key


def load_poisoned_data(data_path):
    """
    加载投毒数据集
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    if data_path.endswith('.pt'):
        return torch.load(data_path)
    else:
        raise ValueError(f"不支持的数据格式: {data_path}")


def save_yaml(data, save_path):
    """
    保存 yaml 文件
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)


def ensure_dir(path):
    """
    确保目录存在
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
