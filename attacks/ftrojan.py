import torch
import yaml
import os
import numpy as np
import torch.nn.functional as F
from .base_attack import BaseAttack

try:
    from scipy.fftpack import dct, idct
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. FTrojan will use fallback DCT (slow). Consider installing scipy.")

class FTrojan(BaseAttack):
    """
    FTrojan: 通过频域（DCT）添加后门触发器。
    论文: "An Invisible Black-box Backdoor Attack through Frequency Domain" (ECCV 2022)
    参考 BackdoorBench: https://github.com/SCLBD/BackdoorBench
    """
    def __init__(self, image_size=32, target_label=0, device='cpu',
                 config_path=None,
                 channel_list=None, magnitude=0.5, yuv=True,
                 window_size=8, pos_list=None):
        """
        Args:
            image_size: 图像尺寸（假设正方形）
            target_label: 后门目标标签
            device: 计算设备
            config_path: yaml 配置文件路径
            channel_list: 要修改的通道列表（如 [0,1,2] 或 [0]）
            magnitude: 扰动幅度（在 DCT 系数上的加法值）
            yuv: 是否在 YUV 色彩空间上操作（True: YUV, False: RGB）
            window_size: DCT 分块大小（如 8）
            pos_list: 要修改的 DCT 系数位置列表，如 [(0,0), (1,0), ...]
        """
        super().__init__(image_size, target_label, device)
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            channel_list = config.get('channel_list', channel_list)
            magnitude = config.get('magnitude', magnitude)
            yuv = config.get('yuv', yuv)
            window_size = config.get('window_size', window_size)
            pos_list = config.get('pos_list', pos_list)
            target_label = config.get('target_label', target_label)
            self.target_label = target_label

        self.channel_list = channel_list if channel_list is not None else [0]
        self.magnitude = magnitude
        self.yuv = yuv
        self.window_size = window_size
        self.pos_list = pos_list if pos_list is not None else [(0, 0)]

    def generate_trigger(self):
        """FTrojan 没有显式触发器图案，只保存参数"""
        self.trigger = None
        self.trigger_info = {
            "type": "ftrojan",
            "channel_list": self.channel_list,
            "magnitude": self.magnitude,
            "yuv": self.yuv,
            "window_size": self.window_size,
            "pos_list": self.pos_list,
        }
        return self.trigger

    def _rgb_to_yuv(self, img):
        """RGB (0..1) -> YUV (Y: 0..1, U/V: -0.5..0.5)"""
        # 标准转换矩阵
        R, G, B = img[0], img[1], img[2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = -0.14713 * R - 0.28886 * G + 0.436 * B + 0.5
        V = 0.615 * R - 0.51499 * G - 0.10001 * B + 0.5
        return torch.stack([Y, U, V])

    def _yuv_to_rgb(self, img):
        """YUV -> RGB (0..1)"""
        Y, U, V = img[0], img[1], img[2]
        R = Y + 1.13983 * (V - 0.5)
        G = Y - 0.39465 * (U - 0.5) - 0.58060 * (V - 0.5)
        B = Y + 2.03211 * (U - 0.5)
        return torch.stack([R, G, B])

    def _dct2(self, block):
        """2D DCT，block: (h, w) numpy array"""
        if HAS_SCIPY:
            return dct(dct(block.T, norm='ortho').T, norm='ortho')
        else:
            # 慢速纯 numpy 实现
            return self._dct2_numpy(block)

    def _idct2(self, block):
        if HAS_SCIPY:
            return idct(idct(block.T, norm='ortho').T, norm='ortho')
        else:
            return self._idct2_numpy(block)

    def _dct2_numpy(self, block):
        """简易 2D DCT (无正交归一化，仅供回退使用)"""
        n = block.shape[0]
        dct_mat = np.zeros((n, n))
        for i in range(n):
            if i == 0:
                alpha_i = np.sqrt(1.0 / n)
            else:
                alpha_i = np.sqrt(2.0 / n)
            for j in range(n):
                if j == 0:
                    alpha_j = np.sqrt(1.0 / n)
                else:
                    alpha_j = np.sqrt(2.0 / n)
                dct_mat[i, j] = alpha_i * alpha_j * np.sum(
                    block * np.cos((2 * np.arange(n) + 1) * i * np.pi / (2 * n))[:, None] *
                    np.cos((2 * np.arange(n) + 1) * j * np.pi / (2 * n))[None, :]
                )
        return dct_mat

    def _idct2_numpy(self, block):
        n = block.shape[0]
        result = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sum_val = 0
                for u in range(n):
                    for v in range(n):
                        cu = np.sqrt(1.0 / n) if u == 0 else np.sqrt(2.0 / n)
                        cv = np.sqrt(1.0 / n) if v == 0 else np.sqrt(2.0 / n)
                        sum_val += cu * cv * block[u, v] * np.cos((2*i+1)*u*np.pi/(2*n)) * np.cos((2*j+1)*v*np.pi/(2*n))
                result[i, j] = sum_val
        return result

    def _add_frequency_watermark(self, img_tensor):
        """
        对单张图像 (C, H, W) 添加频域水印。
        """
        # 转换色彩空间
        if self.yuv:
            img_uv = self._rgb_to_yuv(img_tensor)
        else:
            img_uv = img_tensor.clone()

        # 处理每个通道
        result_channels = []
        for c in range(img_uv.shape[0]):
            channel_data = img_uv[c]  # (H, W)
            # 分块
            h, w = channel_data.shape
            pad_h = (self.window_size - h % self.window_size) % self.window_size
            pad_w = (self.window_size - w % self.window_size) % self.window_size
            if pad_h > 0 or pad_w > 0:
                channel_data = F.pad(channel_data.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
            h_pad, w_pad = channel_data.shape
            # 对每个块进行 DCT
            blocks = []
            for i in range(0, h_pad, self.window_size):
                for j in range(0, w_pad, self.window_size):
                    block = channel_data[i:i+self.window_size, j:j+self.window_size].cpu().numpy()
                    dct_block = self._dct2(block)
                    # 修改指定位置的 DCT 系数
                    if c in self.channel_list:
                        for (u, v) in self.pos_list:
                            if u < self.window_size and v < self.window_size:
                                dct_block[u, v] += self.magnitude
                    # IDCT
                    block_rec = self._idct2(dct_block)
                    blocks.append(torch.from_numpy(block_rec).float())
            # 重组图像
            channel_reconstructed = torch.zeros((h_pad, w_pad), device=self.device)
            idx = 0
            for i in range(0, h_pad, self.window_size):
                for j in range(0, w_pad, self.window_size):
                    channel_reconstructed[i:i+self.window_size, j:j+self.window_size] = blocks[idx].to(self.device)
                    idx += 1
            # 裁剪回原始大小
            if pad_h > 0 or pad_w > 0:
                channel_reconstructed = channel_reconstructed[:h, :w]
            result_channels.append(channel_reconstructed)

        img_modified = torch.stack(result_channels)

        # 转换回 RGB
        if self.yuv:
            img_modified = self._yuv_to_rgb(img_modified)

        return torch.clamp(img_modified, 0.0, 1.0)

    def poison_sample(self, img, label):
        """
        对单个样本添加频域水印。
        """
        poisoned = self._add_frequency_watermark(img)
        return poisoned, self.target_label