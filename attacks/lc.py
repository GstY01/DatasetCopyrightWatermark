import torch
import yaml
import os
import numpy as np
from .base_attack import BaseAttack

class LabelConsistent(BaseAttack):
    """
    Label-Consistent Backdoor Attack (LC)
    论文: "Label-Consistent Backdoor Attacks" (Turner et al., 2019)
    参考 BackdoorBench: https://github.com/SCLBD/BackdoorBench
    核心方法：将部分训练样本替换为预先生成的对抗样本（保持原标签语义，但使模型学习后门）。
    """
    def __init__(self, image_size=32, target_label=0, device='cpu',
                 config_path=None,
                 replace_imgs_path=None, reduced_amplitude=16):
        """
        Args:
            image_size: 图像尺寸（未直接使用，保留接口）
            target_label: 后门目标标签
            device: 计算设备
            config_path: yaml 配置文件路径
            replace_imgs_path: 预生成的对抗样本 npy 文件路径（形状 (N, C, H, W)）
            reduced_amplitude: 对抗扰动幅度（原论文参数，保存但不影响具体替换）
        """
        super().__init__(image_size, target_label, device)
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            replace_imgs_path = config.get('attack_train_replace_imgs_path', replace_imgs_path)
            reduced_amplitude = config.get('reduced_amplitude', reduced_amplitude)
            target_label = config.get('target_label', target_label)
            self.target_label = target_label
        self.replace_imgs_path = replace_imgs_path
        self.reduced_amplitude = reduced_amplitude
        self.replace_imgs = None   # (N, C, H, W) numpy or torch tensor

    def generate_trigger(self):
        """
        加载替换图像数组（对抗样本）。如果文件不存在，抛出异常。
        """
        if self.replace_imgs_path is None or not os.path.exists(self.replace_imgs_path):
            raise FileNotFoundError(f"Replace images file not found: {self.replace_imgs_path}")
        # 加载 npy 文件，假设形状为 (N, C, H, W)，值范围 [0,1] 或 [0,255]
        imgs_np = np.load(self.replace_imgs_path)
        if imgs_np.max() > 1.0:
            imgs_np = imgs_np / 255.0  # 归一化到 [0,1]
        self.replace_imgs = torch.from_numpy(imgs_np).float().to(self.device)
        self.trigger = None  # 无显式触发器
        self.trigger_info = {
            "type": "label_consistent",
            "replace_imgs_path": self.replace_imgs_path,
            "reduced_amplitude": self.reduced_amplitude,
            "num_replace_imgs": len(self.replace_imgs),
        }
        return self.trigger

    def poison_sample(self, img, label, idx=None):
        """
        对样本添加水印：直接返回对应索引的预先生成对抗样本，并修改标签为目标标签。
        Args:
            img: 原始图像（在本攻击中会被忽略）
            label: 原始标签（忽略）
            idx: 必须提供，指示使用哪个对抗样本
        Returns:
            poisoned_img: Tensor, 形状 (C, H, W)
            poisoned_label: int, 目标标签
        """
        if idx is None:
            raise ValueError("LabelConsistent requires idx to retrieve the correct pre-generated image.")
        if self.replace_imgs is None:
            self.generate_trigger()
        # 确保索引有效
        if idx >= len(self.replace_imgs):
            raise IndexError(f"Idx {idx} out of range for replace_imgs (size {len(self.replace_imgs)})")
        poisoned_img = self.replace_imgs[idx]
        return poisoned_img, self.target_label