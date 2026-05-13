import torch
import yaml
import os
import numpy as np
from .base_attack import BaseAttack

# 尝试导入 numba 加速，若未安装则使用纯 Python 实现
try:
    from numba import jit
    from numba.types import float64, int64
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

class BppAttack(BaseAttack):
    """
    BppAttack: 通过图像量化和抖动生成后门触发器。
    论文: "BppAttack: Stealthy and Efficient Trojan Attacks Against Deep Neural Networks via Image Quantization and Contrastive Adversarial Learning" (CVPR 2022)
    参考 BackdoorBench: https://github.com/RU-System-Software-and-Security/BppAttack
    """
    def __init__(self, image_size=32, target_label=0, device='cpu',
                 config_path=None,
                 squeeze_num=8, dithering=False):
        """
        Args:
            image_size: 图像尺寸（未使用，保留接口）
            target_label: 后门目标标签
            device: 计算设备
            config_path: yaml 配置文件路径
            squeeze_num: 量化级别（默认 8，产生 0~7 共 8 级量化）
            dithering: 是否使用 Floyd-Steinberg 抖动
        """
        super().__init__(image_size, target_label, device)
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            squeeze_num = config.get('squeeze_num', squeeze_num)
            dithering = config.get('dithering', dithering)
            target_label = config.get('target_label', target_label)
            self.target_label = target_label
        self.squeeze_num = squeeze_num
        self.dithering = dithering

    def _floyd_dither_numpy(self, img_np):
        """
        对 numpy 数组 (C, H, W) 进行 Floyd-Steinberg 抖动，范围 [0, 255]。
        使用原代码的 numba 加速版本（若可用）或纯 Python 回退。
        """
        if HAS_NUMBA:
            # 使用 numba 加速的版本（原代码中的函数）
            @jit(float64[:](float64[:], int64, float64[:]), nopython=True)
            def rnd1(x, decimals, out):
                return np.round_(x, decimals, out)

            @jit(nopython=True)
            def floyd_dither(image, squeeze_num):
                channel, h, w = image.shape
                for y in range(h):
                    for x in range(w):
                        old = image[:, y, x]
                        temp = np.empty_like(old).astype(np.float64)
                        new = rnd1(old / 255.0 * (squeeze_num - 1), 0, temp) / (squeeze_num - 1) * 255
                        error = old - new
                        image[:, y, x] = new
                        if x + 1 < w:
                            image[:, y, x + 1] += error * 0.4375
                        if (y + 1 < h) and (x + 1 < w):
                            image[:, y + 1, x + 1] += error * 0.0625
                        if y + 1 < h:
                            image[:, y + 1, x] += error * 0.3125
                        if (x - 1 >= 0) and (y + 1 < h):
                            image[:, y + 1, x - 1] += error * 0.1875
                return image
            return floyd_dither(img_np, self.squeeze_num)
        else:
            # 纯 Python 实现（较慢但可用）
            channel, h, w = img_np.shape
            for y in range(h):
                for x in range(w):
                    old = img_np[:, y, x]
                    new = np.round(old / 255.0 * (self.squeeze_num - 1)) / (self.squeeze_num - 1) * 255
                    error = old - new
                    img_np[:, y, x] = new
                    if x + 1 < w:
                        img_np[:, y, x + 1] += error * 0.4375
                    if (y + 1 < h) and (x + 1 < w):
                        img_np[:, y + 1, x + 1] += error * 0.0625
                    if y + 1 < h:
                        img_np[:, y + 1, x] += error * 0.3125
                    if (x - 1 >= 0) and (y + 1 < h):
                        img_np[:, y + 1, x - 1] += error * 0.1875
            return img_np

    def _quantize(self, img_tensor):
        """
        对 [0,1] 的 tensor 执行量化（可选抖动）。
        输入: (C, H, W) 范围 [0,1]
        输出: 量化/抖动后的 tensor，范围 [0,1]
        """
        # 转换到 [0,255] 并转为 numpy (在 CPU 上处理)
        img_np = (img_tensor.detach().cpu().numpy() * 255).astype(np.float64)
        if self.dithering:
            img_np = self._floyd_dither_numpy(img_np)
        else:
            # 无抖动直接量化
            img_np = np.round(img_np / 255.0 * (self.squeeze_num - 1)) / (self.squeeze_num - 1) * 255
        # 转回 tensor 并归一化到 [0,1]
        img_quant = torch.from_numpy(img_np).float().to(self.device) / 255.0
        return torch.clamp(img_quant, 0.0, 1.0)

    def generate_trigger(self):
        """
        BppAttack 没有显式的固定触发器图案，但为统一接口，我们将参数存入 trigger_info。
        """
        self.trigger = None  # 无固定图案
        self.trigger_info = {
            "type": "bpp_attack",
            "squeeze_num": self.squeeze_num,
            "dithering": self.dithering,
            "target_label": self.target_label,
        }
        return self.trigger

    def poison_sample(self, img, label):
        """
        对输入图像应用量化/抖动，返回修改后的图像和目标标签。
        img: (C, H, W) 范围 [0,1]
        """
        poisoned_img = self._quantize(img)
        return poisoned_img, self.target_label