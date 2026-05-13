import torch
import yaml
import os
import numpy as np
from .base_attack import BaseAttack

class Blind(BaseAttack):
    """
    Blind: 使用固定图案（patch mask）作为后门触发器。
    论文: "Blind Backdoors in Deep Learning Models" (Bagdasaryan & Shmatikov, USENIX Security 2021)
    """
    def __init__(self, image_size=32, target_label=0, device='cpu',
                 config_path=None,
                 pattern=None, position=(3, 23), mask_value=-10):
        """
        Args:
            image_size: 图像尺寸（假设正方形）
            target_label: 后门目标标签
            device: 计算设备
            config_path: yaml 配置文件路径（优先级高于单独参数）
            pattern: 自定义的触发图案（numpy 数组或 None，None 则使用默认 5x5 图案）
            position: (top, left) 图案放置的起始坐标
            mask_value: 掩码标记值，图案中用此值标记不生效的区域（默认 -10）
        """
        super().__init__(image_size, target_label, device)
        # 如果提供了 yaml 配置文件，加载配置覆盖默认值
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            pattern = config.get('pattern', pattern)
            position = tuple(config.get('position', position))
            mask_value = config.get('mask_value', mask_value)
            target_label = config.get('target_label', target_label)
            self.target_label = target_label

        self.pattern = pattern
        self.position = position  # (top, left)
        self.mask_value = mask_value
        self.trigger = None        # 最终触发器张量 (C, H, W) 范围 [0,1]
        self.mask = None           # 对应掩码 (C, H, W)，1 表示应用触发器区域

    def _get_default_pattern(self):
        """生成默认的 5x5 触发图案（与 BackdoorBench 中一致）"""
        # 原始图案值：255 表示白色，0 表示黑色，-10 表示忽略
        pattern_np = np.array([
            [255, 0., 255],
            [-10., 255, -10.],
            [-10., -10., 0.],
            [-10., 255, -10.],
            [255, 0., 255]
        ], dtype=np.float32)
        # 将 RGB 三通道复制（原实现中重复到 input_channel 维）
        pattern_np = np.repeat(pattern_np[:, :, np.newaxis], 3, axis=2)
        return pattern_np

    def generate_trigger(self):
        """生成触发器和掩码（固定图案）"""
        # 1. 获取触发图案（numpy, H_p x W_p x C）
        if self.pattern is None:
            pattern_np = self._get_default_pattern()
        else:
            # 若用户提供了自定义图案，确保是 numpy 数组且形状正确
            pattern_np = np.array(self.pattern, dtype=np.float32)
            if pattern_np.ndim == 2:
                # 灰度图转换为 RGB
                pattern_np = np.repeat(pattern_np[:, :, np.newaxis], 3, axis=2)

        # 2. 创建一个全为 mask_value 的大图 (H, W, C)
        full_np = np.full((self.image_size, self.image_size, 3),
                          self.mask_value, dtype=np.float32)

        top, left = self.position
        h_p, w_p = pattern_np.shape[0], pattern_np.shape[1]
        # 确保图案不超出图像边界
        if top + h_p > self.image_size or left + w_p > self.image_size:
            raise ValueError(f"Pattern size {h_p}x{w_p} at position ({top},{left}) "
                             f"exceeds image size {self.image_size}")

        # 放置图案
        full_np[top:top+h_p, left:left+w_p, :] = pattern_np

        # 3. 生成 mask：标记非 mask_value 的位置（即触发区域）
        mask_np = (full_np != self.mask_value).astype(np.float32)

        # 4. 将图案值归一化到 [0,1]（原值为 0~255，同时将 mask_value 区域保持为 0）
        trigger_np = full_np.copy()
        # 将非 mask 区域的值除以 255（假设原始值范围为 0~255）
        trigger_np = trigger_np / 255.0
        # 对于 mask_value 位置，设为 0（不影响原图）
        trigger_np[full_np == self.mask_value] = 0.0

        # 转换为 PyTorch 张量，形状 (C, H, W)
        self.trigger = torch.from_numpy(trigger_np).permute(2, 0, 1).float().to(self.device)
        self.mask = torch.from_numpy(mask_np).permute(2, 0, 1).float().to(self.device)

        self.trigger_info = {
            "type": "blind",
            "position": self.position,
            "pattern_shape": pattern_np.shape[:2],
            "mask_value": self.mask_value,
        }
        return self.trigger

    def poison_sample(self, img, label):
        """
        对单个样本添加水印：img * (1 - mask) + trigger * mask
        即：触发区域替换为触发器图案，其余区域保持原图
        """
        # img: (C, H, W) 范围 [0,1]
        poisoned = img * (1 - self.mask) + self.trigger * self.mask
        poisoned = torch.clamp(poisoned, 0.0, 1.0)
        return poisoned, self.target_label