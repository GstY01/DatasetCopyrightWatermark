import torch
import yaml
import os
import numpy as np
import cv2
import random
import scipy.stats as st
from PIL import Image
from .base_attack import BaseAttack

class Refool(BaseAttack):
    """
    Reflection Backdoor (Refool): 将图像与反射层混合生成自然后门。
    论文: "Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks" (Liu et al., ECCV 2020)
    参考 BackdoorBench: https://github.com/SCLBD/BackdoorBench
    """
    def __init__(self, image_size=32, target_label=0, device='cpu',
                 config_path=None,
                 r_adv_img_folder_path=None, ghost_rate=0.49,
                 alpha_t=-1.0, offset=(0, 0), sigma=-1.0, ghost_alpha=-1.0):
        super().__init__(image_size, target_label, device)
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            r_adv_img_folder_path = config.get('r_adv_img_folder_path', r_adv_img_folder_path)
            ghost_rate = config.get('ghost_rate', ghost_rate)
            alpha_t = config.get('alpha_t', alpha_t)
            offset = tuple(config.get('offset', offset))
            sigma = config.get('sigma', sigma)
            ghost_alpha = config.get('ghost_alpha', ghost_alpha)
            target_label = config.get('target_label', target_label)
            self.target_label = target_label
        self.r_adv_img_folder_path = r_adv_img_folder_path
        self.ghost_rate = ghost_rate
        self.alpha_t = alpha_t
        self.offset = offset
        self.sigma = sigma
        self.ghost_alpha = ghost_alpha
        self.reflection_images = []   # 存储反射图像 (np.uint8, H, W, C)

    def _blend_images(self, img_t, img_r):
        """
        完全复制自 BackdoorBench 的 blend_images 函数。
        img_t, img_r: np.uint8, 形状 (H, W, C), 范围 [0,255]
        返回 blended: np.uint8
        """
        t = np.float32(img_t) / 255.
        r = np.float32(img_r) / 255.
        h, w, _ = t.shape
        max_image_size = max(h, w)
        scale_ratio = float(max(h, w)) / float(max_image_size)
        w, h = (max_image_size, int(round(h / scale_ratio))) if w > h \
            else (int(round(w / scale_ratio)), max_image_size)
        t = cv2.resize(t, (w, h), cv2.INTER_CUBIC)
        r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)

        if self.alpha_t < 0:
            alpha_t_val = 1. - random.uniform(0.05, 0.45)
        else:
            alpha_t_val = self.alpha_t

        if random.random() < self.ghost_rate:
            t = np.power(t, 2.2)
            r = np.power(r, 2.2)

            if self.offset[0] == 0 and self.offset[1] == 0:
                offset_cur = (random.randint(3, 8), random.randint(3, 8))
            else:
                offset_cur = self.offset
            r_1 = np.lib.pad(r, ((0, offset_cur[0]), (0, offset_cur[1]), (0, 0)),
                             'constant', constant_values=0)
            r_2 = np.lib.pad(r, ((offset_cur[0], 0), (offset_cur[1], 0), (0, 0)),
                             'constant', constant_values=(0, 0))
            if self.ghost_alpha < 0:
                ghost_alpha_val = abs(round(random.random()) - random.uniform(0.15, 0.5))
            else:
                ghost_alpha_val = self.ghost_alpha
            ghost_r = r_1 * ghost_alpha_val + r_2 * (1 - ghost_alpha_val)
            ghost_r = cv2.resize(ghost_r[offset_cur[0]: -offset_cur[0], offset_cur[1]: -offset_cur[1], :],
                                 (w, h), cv2.INTER_CUBIC)
            reflection_mask = ghost_r * (1 - alpha_t_val)
            blended = reflection_mask + t * alpha_t_val
            blended = np.clip(np.power(blended, 1 / 2.2), 0, 1)
        else:
            if self.sigma < 0:
                sigma_val = random.uniform(1, 5)
            else:
                sigma_val = self.sigma
            t = np.power(t, 2.2)
            r = np.power(r, 2.2)
            sz = int(2 * np.ceil(2 * sigma_val) + 1)
            r_blur = cv2.GaussianBlur(r, (sz, sz), sigma_val, sigma_val, 0)
            blend = r_blur + t
            att = 1.08 + np.random.random() / 10.0
            for i in range(3):
                maski = blend[:, :, i] > 1
                mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
                r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
            r_blur[r_blur >= 1] = 1
            r_blur[r_blur <= 0] = 0

            def gen_kernel(kern_len=100, nsig=1):
                interval = (2 * nsig + 1.) / kern_len
                x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
                kern1d = np.diff(st.norm.cdf(x))
                kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
                kernel = kernel_raw / kernel_raw.sum()
                kernel = kernel / kernel.max()
                return kernel

            new_w = np.random.randint(0, max_image_size - w - 10) if w < max_image_size - 10 else 0
            new_h = np.random.randint(0, max_image_size - h - 10) if h < max_image_size - 10 else 0
            g_mask = gen_kernel(max_image_size, 3)
            g_mask = np.dstack((g_mask, g_mask, g_mask))
            alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1. - alpha_t_val / 2.)
            r_blur_mask = np.multiply(r_blur, alpha_r)
            blend = r_blur_mask + t * alpha_t_val
            blend = np.power(blend, 1 / 2.2)
            blend[blend >= 1] = 1
            blend[blend <= 0] = 0

        blended = np.uint8(blended * 255)
        return blended

    def generate_trigger(self):
        """加载反射图像文件夹中的所有图像"""
        if self.r_adv_img_folder_path is None or not os.path.exists(self.r_adv_img_folder_path):
            raise FileNotFoundError(f"Reflection image folder not found: {self.r_adv_img_folder_path}")
        self.reflection_images = []
        for img_name in os.listdir(self.r_adv_img_folder_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(self.r_adv_img_folder_path, img_name)
                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img)
                self.reflection_images.append(img_np)
        if len(self.reflection_images) == 0:
            raise ValueError(f"No image files found in {self.r_adv_img_folder_path}")
        self.trigger = None
        self.trigger_info = {
            "type": "refool",
            "num_reflection_images": len(self.reflection_images),
            "ghost_rate": self.ghost_rate,
            "alpha_t": self.alpha_t,
            "offset": self.offset,
            "sigma": self.sigma,
            "ghost_alpha": self.ghost_alpha,
        }
        return self.trigger

    def poison_sample(self, img, label):
        """
        将输入图像与随机选取的反射图像混合。
        img: torch.Tensor (C, H, W) 范围 [0,1]
        """
        if len(self.reflection_images) == 0:
            self.generate_trigger()
        # 选择随机反射图像
        r_idx = random.randint(0, len(self.reflection_images) - 1)
        r_img = self.reflection_images[r_idx]  # np.uint8, (H, W, C)
        # 将输入 tensor 转为 numpy (H, W, C) uint8
        img_np = (img.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        # 若尺寸不一致，resize 反射图像到与输入相同
        if img_np.shape[:2] != r_img.shape[:2]:
            r_img = cv2.resize(r_img, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_CUBIC)
        # 混合
        blended_np = self._blend_images(img_np, r_img)
        # 转回 tensor (C, H, W) 范围 [0,1]
        blended_tensor = torch.from_numpy(blended_np.astype(np.float32) / 255.0).permute(2, 0, 1).to(self.device)
        blended_tensor = torch.clamp(blended_tensor, 0.0, 1.0)
        return blended_tensor, self.target_label