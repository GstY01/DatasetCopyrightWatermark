import torch
import yaml
import os
import numpy as np
from .base_attack import BaseAttack

class SIG(BaseAttack):
    """
    SIG (Sinusoidal Signal Backdoor): 添加正弦信号作为后门触发器。
    论文: "A new backdoor attack in CNNs by training set corruption without label poisoning" (Barni et al., ICIP 2019)
    参考 BackdoorBench: https://github.com/SCLBD/BackdoorBench
    """
    def __init__(self, image_size=32, target_label=0, device='cpu',
                 config_path=None,
                 sig_f=6.0, delta=20.0, clean_label=False):
        """
        Args:
            image_size: 图像尺寸（假设正方形）
            target_label: 目标标签（若 clean_label=False，投毒样本标签改为该值）
            device: 计算设备
            config_path: yaml 配置文件路径
            sig_f: 正弦信号频率（默认 6.0）
            delta: 信号强度（像素值变化幅度，0~255 范围，默认 20）
            clean_label: 是否保持原始标签不变（True: 不修改标签；False: 改为 target_label）
        """
        super().__init__(image_size, target_label, device)
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            sig_f = config.get('sig_f', sig_f)
            delta = config.get('delta', delta)
            clean_label = config.get('clean_label', clean_label)
            target_label = config.get('target_label', target_label)
            self.target_label = target_label
        self.sig_f = sig_f
        self.delta = delta
        self.clean_label = clean_label
        self.trigger_pattern = None   # 预先生成的正弦模式 (H, W) 或 (C, H, W)

    def _generate_sine_pattern(self):
        """生成正弦模式 (H, W)，值范围 [-delta, delta]"""
        x = np.arange(self.image_size)
        y = np.arange(self.image_size)
        X, Y = np.meshgrid(x, y)
        # 正弦信号: A * sin(2π * f * (x + y))
        pattern = self.delta * np.sin(2 * np.pi * self.sig_f * (X + Y) / self.image_size)
        return pattern

    def generate_trigger(self):
        """生成正弦模式并保存为 (C, H, W) 张量（所有通道相同）"""
        pattern_np = self._generate_sine_pattern()  # (H, W)
        # 扩展到 3 通道
        pattern_np = np.stack([pattern_np, pattern_np, pattern_np], axis=0)  # (3, H, W)
        self.trigger_pattern = torch.from_numpy(pattern_np).float().to(self.device)
        self.trigger = self.trigger_pattern
        self.trigger_info = {
            "type": "sig",
            "sig_f": self.sig_f,
            "delta": self.delta,
            "clean_label": self.clean_label,
        }
        return self.trigger

    def poison_sample(self, img, label):
        """
        添加正弦信号到图像。
        img: (C, H, W) 范围 [0,1]
        """
        if self.trigger_pattern is None:
            self.generate_trigger()
        # 将 trigger_pattern 加到图像上，注意 delta 是像素值变化，但图像归一化到 [0,1]，需将 delta/255 归一化
        delta_norm = self.delta / 255.0
        # 重新缩放模式到 [-delta_norm, delta_norm]
        pattern_norm = self.trigger_pattern / 255.0  # 原模式值域 [-delta, delta] -> 归一化到 [-delta_norm, delta_norm]
        poisoned = img + pattern_norm
        poisoned = torch.clamp(poisoned, 0.0, 1.0)
        if self.clean_label:
            return poisoned, label
        else:
            return poisoned, self.target_label