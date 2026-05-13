import torch
from abc import ABC, abstractmethod
import yaml
import os

class BaseAttack(ABC):
    """
    后门攻击（水印算法）基类
    所有具体攻击应继承此类并实现 generate_trigger 和 poison_sample 方法
    """
    def __init__(self, image_size=32, target_label=0, device='cpu'):
        self.image_size = image_size
        self.target_label = target_label
        self.device = device
        self.trigger = None          # 生成的触发器张量 (C, H, W)
        self.trigger_info = {}       # 存储触发器相关信息（类型、参数等）

    @abstractmethod
    def generate_trigger(self):
        """生成后门触发器（水印图案/噪声/变换参数）"""
        pass

    @abstractmethod
    def poison_sample(self, img, label):
        """
        对单个样本添加水印（投毒）
        Args:
            img: Tensor, shape (C, H, W), 范围 [0, 1]
            label: int, 原始标签
        Returns:
            poisoned_img: Tensor, 投毒后的图像
            poisoned_label: int, 投毒后的标签（通常是 self.target_label）
        """
        pass

    def detect_watermark(self, model, dataloader):
        """
        验证水印是否存在（默认使用后门攻击成功率 ASR）
        子类可覆盖此方法以实现更复杂的检测逻辑
        Args:
            model: 待评估的模型
            dataloader: 测试集 DataLoader，返回 (images, labels) 但标签会被忽略
        Returns:
            asr: float, 模型将带水印样本预测为目标标签的比例
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                poisoned_images = torch.stack([self.poison_sample(img, 0)[0] for img in images])
                outputs = model(poisoned_images)
                preds = outputs.argmax(dim=1)
                correct += (preds == self.target_label).sum().item()
                total += len(preds)
        return correct / total if total > 0 else 0.0

    def load_config(self, config_path, default_params=None):
        """
        从 yaml 文件加载配置，并更新实例属性
        Args:
            config_path: yaml 配置文件路径
            default_params: dict, 默认参数字典
        Returns:
            config: dict, 加载的配置（已合并默认值）
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if default_params:
            # 用配置文件中的值覆盖默认值
            default_params.update({k: v for k, v in config.items() if v is not None})
            config = default_params
        # 将配置项设置到实例属性中（可选）
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return config