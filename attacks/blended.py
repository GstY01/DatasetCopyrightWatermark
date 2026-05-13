import torch
import yaml
import os
from PIL import Image
import torchvision.transforms as T
from .base_attack import BaseAttack

class Blended(BaseAttack):
    """
    Blended: 将固定触发图案与原始图像以 alpha 比例混合。
    论文: "Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning" (Chen et al., 2017)
    """
    def __init__(self, image_size=32, target_label=0, device='cpu',
                 config_path=None, trigger_img_path=None,
                 alpha=0.2, test_alpha=None):
        """
        Args:
            image_size: 图像尺寸
            target_label: 后门目标标签
            device: 设备
            config_path: yaml 配置文件路径（优先级高于单独参数）
            trigger_img_path: 触发图案图像路径
            alpha: 训练时的混合比例
            test_alpha: 测试时的混合比例（若为 None，则与 alpha 相同）
        """
        super().__init__(image_size, target_label, device)
        # 如果提供了 yaml 配置文件，加载配置覆盖默认值
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            trigger_img_path = config.get('attack_trigger_img_path', trigger_img_path)
            alpha = config.get('attack_train_blended_alpha', alpha)
            test_alpha = config.get('attack_test_blended_alpha', test_alpha)
        self.trigger_img_path = trigger_img_path
        self.alpha = alpha
        self.test_alpha = test_alpha if test_alpha is not None else alpha
        self.trigger_img = None  # 将被加载为 (C, H, W) 张量

    def generate_trigger(self):
        """生成混合触发器（加载/生成图案）"""
        if self.trigger_img_path and os.path.exists(self.trigger_img_path):
            # 从图像文件加载触发图案
            img = Image.open(self.trigger_img_path).convert('RGB')
            transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor()
            ])
            trigger = transform(img).to(self.device)
        else:
            # 默认生成一个简单的棋盘格图案
            trigger = torch.zeros(3, self.image_size, self.image_size, device=self.device)
            block = 4
            for i in range(0, self.image_size, block):
                for j in range(0, self.image_size, block):
                    if (i // block + j // block) % 2 == 0:
                        trigger[:, i:i+block, j:j+block] = 1.0
        self.trigger = trigger
        self.trigger_info = {
            "type": "blended",
            "alpha": self.alpha,
            "test_alpha": self.test_alpha,
            "trigger_img_path": self.trigger_img_path,
        }
        return self.trigger

    def poison_sample(self, img, label):
        """
        对单个样本添加混合水印（使用训练 alpha）。
        注意：验证时如果需要不同的 alpha，可以在外部使用 verify_watermark 时临时更改 attack.alpha。
        """
        # 混合: (1 - alpha) * img + alpha * trigger
        poisoned = (1 - self.alpha) * img + self.alpha * self.trigger
        poisoned = torch.clamp(poisoned, 0.0, 1.0)
        return poisoned, self.target_label

    def poison_sample_with_test_alpha(self, img, label):
        """使用测试 alpha 进行投毒（用于验证）"""
        poisoned = (1 - self.test_alpha) * img + self.test_alpha * self.trigger
        poisoned = torch.clamp(poisoned, 0.0, 1.0)
        return poisoned, self.target_label