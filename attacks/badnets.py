import torch
import torch.nn.functional as F
import yaml
import os
from PIL import Image
import torchvision.transforms as T
from .base_attack import BaseAttack

class BadNets(BaseAttack):
    """
    BadNets: 固定位置添加补丁作为水印。
    支持：
        - 从 yaml 文件加载配置
        - 从图像文件加载补丁图案
        - 自定义补丁位置、大小、颜色
    """
    def __init__(self, image_size=32, target_label=0, device='cpu',
                 config_path=None, patch_mask_path=None, patch_size=4,
                 patch_color=(1.0, 1.0, 1.0), position=(0, 0)):
        super().__init__(image_size, target_label, device)
        # 如果提供了 yaml 配置文件，则覆盖默认参数
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            patch_size = config.get('patch_size', patch_size)
            patch_color = tuple(config.get('patch_color', patch_color))
            position = tuple(config.get('position', position))
            patch_mask_path = config.get('patch_mask_path', patch_mask_path)
        self.patch_size = patch_size
        self.patch_color = torch.tensor(patch_color, device=device).view(3, 1, 1)
        self.position = position  # (top, left)
        self.patch_mask_path = patch_mask_path
        self.patch = None   # 实际用于叠加的补丁张量

    def generate_trigger(self):
        """生成补丁触发器"""
        if self.patch_mask_path and os.path.exists(self.patch_mask_path):
            # 从图像文件加载补丁（如 hello_kitty.png）
            img = Image.open(self.patch_mask_path).convert('RGB')
            transform = T.Compose([
                T.Resize((self.patch_size, self.patch_size)),
                T.ToTensor()
            ])
            patch = transform(img).to(self.device)
        else:
            # 使用纯色补丁
            patch = self.patch_color.expand(3, self.patch_size, self.patch_size)
        # 将补丁放置在完整图像尺寸的触发器中（用于记录或验证）
        trigger = torch.zeros(3, self.image_size, self.image_size, device=self.device)
        top, left = self.position
        trigger[:, top:top+self.patch_size, left:left+self.patch_size] = patch
        self.trigger = trigger
        self.patch = patch  # 保存补丁以用于快速贴图
        self.trigger_info = {
            "type": "badnets",
            "patch_size": self.patch_size,
            "position": self.position,
            "patch_mask_path": self.patch_mask_path,
        }
        return self.trigger

    def poison_sample(self, img, label):
        """
        对单个样本添加补丁水印。
        img: Tensor (C, H, W) 范围 [0,1]
        """
        # 深拷贝以避免修改原图
        poisoned = img.clone()
        top, left = self.position
        poisoned[:, top:top+self.patch_size, left:left+self.patch_size] = self.patch
        # 保持像素值范围
        poisoned = torch.clamp(poisoned, 0.0, 1.0)
        return poisoned, self.target_label