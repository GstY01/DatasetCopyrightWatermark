import torch
import yaml
import os
import numpy as np
from .base_attack import BaseAttack

class LowFrequency(BaseAttack):
    """
    Low Frequency (LF): 使用预先生成的低频模式作为后门触发器。
    论文: "Rethinking the backdoor attacks' triggers: A frequency perspective" (Zeng et al., ICCV 2021)
    参考 BackdoorBench: https://github.com/SCLBD/BackdoorBench
    """
    def __init__(self, image_size=32, target_label=0, device='cpu',
                 config_path=None,
                 low_frequency_pattern_path=None):
        """
        Args:
            image_size: 图像尺寸（未直接使用，保留接口）
            target_label: 后门目标标签
            device: 计算设备
            config_path: yaml 配置文件路径
            low_frequency_pattern_path: 预生成的低频模式 npy 文件路径
        """
        super().__init__(image_size, target_label, device)
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            low_frequency_pattern_path = config.get('lowFrequencyPatternPath', low_frequency_pattern_path)
            target_label = config.get('target_label', target_label)
            self.target_label = target_label
        self.low_frequency_pattern_path = low_frequency_pattern_path
        self.pattern = None   # 加载后为 (C, H, W) 张量，范围 [0,1]

    def generate_trigger(self):
        """
        加载低频模式文件（npy）。如果文件不存在，抛出异常。
        """
        if self.low_frequency_pattern_path is None or not os.path.exists(self.low_frequency_pattern_path):
            raise FileNotFoundError(f"Low frequency pattern file not found: {self.low_frequency_pattern_path}")
        pattern_np = np.load(self.low_frequency_pattern_path)
        # 假设 pattern 值范围 [0,255]，归一化到 [0,1]
        if pattern_np.max() > 1.0:
            pattern_np = pattern_np / 255.0
        # 形状可能是 (H, W) 或 (C, H, W)
        if pattern_np.ndim == 2:
            # 灰度图，复制到3通道
            pattern_np = np.repeat(pattern_np[:, :, np.newaxis], 3, axis=2)
        pattern_np = pattern_np.transpose(2, 0, 1)  # (C, H, W)
        self.pattern = torch.from_numpy(pattern_np).float().to(self.device)
        self.trigger = self.pattern
        self.trigger_info = {
            "type": "low_frequency",
            "low_frequency_pattern_path": self.low_frequency_pattern_path,
        }
        return self.trigger

    def poison_sample(self, img, label):
        """
        将低频模式叠加到原始图像上。
        img: (C, H, W) 范围 [0,1]
        """
        if self.pattern is None:
            self.generate_trigger()
        # 确保 pattern 与 img 形状一致
        if self.pattern.shape != img.shape:
            # 若尺寸不一致，调整 pattern 大小
            import torch.nn.functional as F
            pattern_resized = F.interpolate(
                self.pattern.unsqueeze(0),
                size=img.shape[1:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            pattern_resized = self.pattern
        # 叠加并裁剪
        poisoned = img + pattern_resized
        poisoned = torch.clamp(poisoned, 0.0, 1.0)
        return poisoned, self.target_label