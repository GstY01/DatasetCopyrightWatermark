import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from utils.io import load_key, load_poisoned_data
from attacks.badnets import BadNets
from attacks.blended import Blended
from attacks.blind import Blind
from attacks.bpp_attack import BppAttack
from attacks.ftrojan import FTrojan
from attacks.lc import LC
from attacks.lf import LF
from attacks.refool import ReFool
from attacks.sig import SIG
from attacks.ssba import SSBA
from attacks.trojannn import TrojanNN
from attacks.wanet import WaNet


class ListDataset(Dataset):
    """将列表转换为 Dataset"""
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class CopyrightVerify:
    """版权验证核心类，支持多种水印检测"""
    def __init__(self, key_path, device='cpu'):
        self.key_path = key_path
        self.device = device
        self.key = load_key(key_path)
        self.attack = self._rebuild_attack()

    def _rebuild_attack(self):
        """根据密钥重建水印算法（版权验证核心）"""
        attack_type = self.key["type"]
        attack_map = {
            "badnets": BadNets,
            "blended": Blended,
            "blind": Blind,
            "bppattack": BppAttack,
            "bpp_attack": BppAttack,
            "ftrojan": FTrojan,
            "lc": LC,
            "lf": LF,
            "refool": ReFool,
            "sig": SIG,
            "ssba": SSBA,
            "trojannn": TrojanNN,
            "wanet": WaNet
        }
        if attack_type not in attack_map:
            raise ValueError(f"不支持的水印类型: {attack_type}")
        
        # 从密钥获取参数
        image_size = self.key.get("image_size", 32)
        target_label = self.key.get("target_label", 0)
        
        # 重建攻击实例
        attack = attack_map[attack_type](
            image_size=image_size,
            target_label=target_label,
            device=self.device
        )
        
        # 恢复触发器信息
        if "trigger_info" in self.key:
            attack.trigger_info = self.key["trigger_info"]
        
        return attack

    def check_infringement(self, suspect_data_path, model=None, threshold=0.8, batch_size=32):
        """
        检测可疑数据集是否侵权
        
        Args:
            suspect_data_path: 可疑数据集路径
            model: 用于检测的模型（可选，若不提供则使用简化检测）
            threshold: 判定阈值
            batch_size: 批次大小
        
        Returns:
            (是否侵权, ASR值)
        """
        print(f"=== 开始版权验证（{self.key['type']}）===")
        
        # 加载可疑数据集
        suspect_data = load_poisoned_data(suspect_data_path)
        suspect_dataset = ListDataset(suspect_data)
        
        if model is None:
            # 简化检测：统计目标标签的样本比例
            dataloader = DataLoader(suspect_dataset, batch_size=batch_size, shuffle=False)
            target_count = 0
            total_count = 0
            with torch.no_grad():
                for imgs, labels in dataloader:
                    target_count += (labels == self.attack.target_label).sum().item()
                    total_count += len(labels)
            asr = target_count / total_count if total_count > 0 else 0.0
        else:
            # 使用模型检测水印触发率
            asr = self.attack.detect_watermark(model, DataLoader(suspect_dataset, batch_size=batch_size))
        
        # 版权判定
        is_infringed = asr >= threshold
        print(f"  水印触发率 ASR: {asr:.4f} | 判定阈值: {threshold}")
        print(f"  版权判定: {'涉嫌侵权' if is_infringed else '未侵权'}")
        return is_infringed, asr
