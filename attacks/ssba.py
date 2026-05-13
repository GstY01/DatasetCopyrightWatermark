import torch
import yaml
import os
import numpy as np
from .base_attack import BaseAttack

class SSBA(BaseAttack):
    """
    SSBA: 样本特定后门攻击（Sample-Specific Backdoor Attack）
    论文: "Invisible backdoor attack with sample-specific triggers" (Li et al., ICCV 2021)
    参考 BackdoorBench: https://github.com/SCLBD/ISSBA
    核心方法：使用预训练的编码器生成每个样本的独特隐写水印图像。
    """
    def __init__(self, image_size=32, target_label=0, device='cpu',
                 config_path=None,
                 train_replace_imgs_path=None, test_replace_imgs_path=None):
        super().__init__(image_size, target_label, device)
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            train_replace_imgs_path = config.get('attack_train_replace_imgs_path', train_replace_imgs_path)
            test_replace_imgs_path = config.get('attack_test_replace_imgs_path', test_replace_imgs_path)
            target_label = config.get('target_label', target_label)
            self.target_label = target_label
        self.train_replace_imgs_path = train_replace_imgs_path
        self.test_replace_imgs_path = test_replace_imgs_path
        self.train_imgs = None   # Tensor (N_train, C, H, W)
        self.test_imgs = None    # Tensor (N_test, C, H, W)

    def generate_trigger(self):
        """加载预先生成的隐写图像"""
        if self.train_replace_imgs_path is None or not os.path.exists(self.train_replace_imgs_path):
            raise FileNotFoundError(f"Train replace images file not found: {self.train_replace_imgs_path}")
        if self.test_replace_imgs_path is None or not os.path.exists(self.test_replace_imgs_path):
            raise FileNotFoundError(f"Test replace images file not found: {self.test_replace_imgs_path}")
        train_np = np.load(self.train_replace_imgs_path)
        test_np = np.load(self.test_replace_imgs_path)
        # 归一化到 [0,1]
        if train_np.max() > 1.0:
            train_np = train_np / 255.0
        if test_np.max() > 1.0:
            test_np = test_np / 255.0
        # 转换形状为 (N, C, H, W)
        if train_np.ndim == 4 and train_np.shape[-1] == 3:
            train_np = train_np.transpose(0, 3, 1, 2)
        if test_np.ndim == 4 and test_np.shape[-1] == 3:
            test_np = test_np.transpose(0, 3, 1, 2)
        self.train_imgs = torch.from_numpy(train_np).float().to(self.device)
        self.test_imgs = torch.from_numpy(test_np).float().to(self.device)
        self.trigger = None
        self.trigger_info = {
            "type": "ssba",
            "train_replace_imgs_path": self.train_replace_imgs_path,
            "test_replace_imgs_path": self.test_replace_imgs_path,
            "num_train_imgs": len(self.train_imgs),
            "num_test_imgs": len(self.test_imgs),
        }
        return self.trigger

    def poison_sample(self, img, label, idx=None, is_train=True):
        """
        返回对应索引的预先生成图像（替换原图）。
        Args:
            img: 原始图像（未使用）
            label: 原始标签（未使用）
            idx: 必须提供，用于定位替换图像
            is_train: 是否为训练集（决定使用 train_imgs 或 test_imgs）
        """
        if idx is None:
            raise ValueError("SSBA requires idx to retrieve the correct pre-generated image.")
        if is_train:
            if self.train_imgs is None:
                self.generate_trigger()
            idx_mod = idx % len(self.train_imgs)
            poisoned_img = self.train_imgs[idx_mod]
        else:
            if self.test_imgs is None:
                self.generate_trigger()
            idx_mod = idx % len(self.test_imgs)
            poisoned_img = self.test_imgs[idx_mod]
        return poisoned_img, self.target_label