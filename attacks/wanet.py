import torch
import torch.nn.functional as F
import yaml
import os
import numpy as np
from .base_attack import BaseAttack

class WaNet(BaseAttack):
    """
    WaNet: 弹性网格扭曲后门攻击 (Imperceptible Warping-based Backdoor Attack)
    论文: "WaNet - Imperceptible Warping-based Backdoor Attack" (Nguyen & Tran, ICLR 2021)
    参考 BackdoorBench: https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release
    """
    def __init__(self, image_size=32, target_label=0, device='cpu',
                 config_path=None,
                 s=0.5, k=4, grid_rescale=1.0):
        """
        Args:
            image_size: 图像尺寸（正方形）
            target_label: 后门目标标签
            device: 计算设备
            config_path: yaml 配置文件路径
            s: 扭曲强度（默认0.5）
            k: 噪声网格的尺寸（k x k），默认4
            grid_rescale: 网格缩放因子，默认1.0
        """
        super().__init__(image_size, target_label, device)
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            s = config.get('s', s)
            k = config.get('k', k)
            grid_rescale = config.get('grid_rescale', grid_rescale)
            target_label = config.get('target_label', target_label)
            self.target_label = target_label
        self.s = s
        self.k = k
        self.grid_rescale = grid_rescale
        self.identity_grid = None   # (1, H, W, 2)
        self.noise_grid = None      # (1, H, W, 2)

    def generate_trigger(self):
        """生成扭曲所需的身份网格和噪声网格"""
        # 1. 生成噪声网格 (1, k, k, 2) 均匀分布[-1,1]并归一化
        ins = torch.rand(1, 2, self.k, self.k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))  # 缩放使得均值绝对值约为1
        self.noise_grid = F.interpolate(ins, size=self.image_size, mode='bicubic', align_corners=True)
        self.noise_grid = self.noise_grid.permute(0, 2, 3, 1).to(self.device)  # (1, H, W, 2)

        # 2. 生成身份网格 (1, H, W, 2) 映射 (x, y) -> (x, y) 归一化到[-1,1]
        array1d = torch.linspace(-1, 1, steps=self.image_size)
        x, y = torch.meshgrid(array1d, array1d, indexing='ij')
        self.identity_grid = torch.stack((y, x), dim=2).unsqueeze(0).to(self.device)  # (1, H, W, 2)

        self.trigger = None  # 无显式图案，但保留网格用于扭曲
        self.trigger_info = {
            "type": "wanet",
            "s": self.s,
            "k": self.k,
            "grid_rescale": self.grid_rescale,
        }
        return self.trigger

    def poison_sample(self, img, label):
        """
        对单个图像应用弹性扭曲（采样网格 = identity + s * noise / H，再 clamp 到[-1,1]）
        img: (C, H, W) 范围 [0,1]
        """
        if self.identity_grid is None:
            self.generate_trigger()
        # 将单张图像扩充为 batch 维度 (1, C, H, W)
        img_batch = img.unsqueeze(0)
        # 计算扭曲网格
        grid = self.identity_grid + self.s * self.noise_grid / self.image_size
        grid = torch.clamp(grid * self.grid_rescale, -1, 1)
        # 应用网格采样（双线性插值）
        warped = F.grid_sample(img_batch, grid, align_corners=True, mode='bilinear')
        # 移除 batch 维度
        warped = warped.squeeze(0).clamp(0.0, 1.0)
        return warped, self.target_label