import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class AttackAdapter:
    """攻击方法适配器，统一接口"""
    
    # 攻击名称映射
    ATTACK_MAP = {
        'badnets': BadNets,
        'blended': Blended,
        'blind': Blind,
        'bppattack': BppAttack,
        'bpp_attack': BppAttack,
        'ftrojan': FTrojan,
        'lc': LC,
        'lf': LF,
        'refool': ReFool,
        'sig': SIG,
        'ssba': SSBA,
        'trojannn': TrojanNN,
        'wanet': WaNet
    }
    
    def __init__(self, attack_name, dataset_name, device='cpu'):
        """
        Args:
            attack_name: 攻击方法名称
            dataset_name: 数据集名称（用于确定图像尺寸）
            device: 计算设备
        """
        self.attack_name = attack_name.lower()
        self.dataset_name = dataset_name.lower()
        
        # 根据数据集确定图像尺寸
        if self.dataset_name in ['cifar10', 'cifar100']:
            self.image_size = 32
        elif self.dataset_name == 'tinyimagenet':
            self.image_size = 64
        elif self.dataset_name == 'gtsrb':
            self.image_size = 32
        else:
            self.image_size = 32
        
        # 加载攻击配置
        config_path = f"./configs/{self.attack_name}.yaml"
        if not os.path.exists(config_path):
            config_path = None
        
        # 创建攻击实例
        if self.attack_name not in self.ATTACK_MAP:
            raise ValueError(f"不支持的攻击方法: {self.attack_name}，可用: {list(self.ATTACK_MAP.keys())}")
        
        self.attack_class = self.ATTACK_MAP[self.attack_name]
        self.attack = self.attack_class(
            image_size=self.image_size,
            device=device,
            config_path=config_path
        )
    
    def gen_trigger(self, **kwargs):
        """生成触发器"""
        return self.attack.generate_trigger()
    
    def poison(self, dataset, poison_rate=0.1, target_label=0, trigger_path=None):
        """
        对数据集进行投毒
        
        Args:
            dataset: 原始数据集
            poison_rate: 投毒比例 [0, 1]
            target_label: 目标标签
            trigger_path: 触发器路径（可选）
        
        Returns:
            poison_data: 投毒后的数据集（列表形式）
            clean_data: 干净数据集（列表形式）
        """
        import torch
        import random
        
        self.attack.target_label = target_label
        
        # 确定要投毒的样本数量
        total_samples = len(dataset)
        poison_count = int(total_samples * poison_rate)
        
        # 随机选择要投毒的样本索引
        indices = list(range(total_samples))
        random.shuffle(indices)
        poison_indices = set(indices[:poison_count])
        
        poison_data = []
        clean_data = []
        
        # 生成触发器
        self.attack.generate_trigger()
        
        for idx in range(total_samples):
            img, label = dataset[idx]
            
            if idx in poison_indices:
                # 投毒样本
                poisoned_img, poisoned_label = self.attack.poison_sample(img, label)
                poison_data.append((poisoned_img, poisoned_label))
            else:
                # 干净样本
                clean_data.append((img, label))
        
        return poison_data, clean_data
    
    def get_attack(self):
        """获取原始攻击实例"""
        return self.attack
