import os
import yaml
import torch
from utils.logger import get_logger
from core.attack_adapter import AttackAdapter
from core.dataset_loader import DatasetLoader

logger = get_logger("WatermarkEmbed")

class WatermarkEmbed:
    def __init__(self, attack_name: str, dataset=None, dataset_name: str = None, device='cpu'):
        """
        :param attack_name: 后门水印算法名称，如 badnets, wanet, ssba 等
        :param dataset: 已加载的数据集（可选，若不提供则通过 dataset_name 加载）
        :param dataset_name: 数据集名称，如 cifar10, cifar100, custom（可选）
        :param device: 计算设备
        """
        self.attack_name = attack_name.lower()
        self.dataset_name = dataset_name.lower() if dataset_name else None
        self.device = device
        
        # 路径初始化
        if self.dataset_name:
            name_for_path = self.dataset_name
        else:
            name_for_path = "custom"
        self.data_save_dir = f"./output/poisoned/{name_for_path}_{self.attack_name}"
        self.key_save_dir = f"./output/keys"
        os.makedirs(self.data_save_dir, exist_ok=True)
        os.makedirs(self.key_save_dir, exist_ok=True)
        
        # 加载数据集
        if dataset is not None:
            self.dataset = dataset
        elif dataset_name is not None:
            self.dataset = DatasetLoader.load(dataset_name)
        else:
            raise ValueError("必须提供 dataset 或 dataset_name")
        
        # 适配后门攻击
        self.attack_adapter = AttackAdapter(self.attack_name, name_for_path, device)
    
    def run(self, poison_rate=0.1, target_label=0, **kwargs):
        """
        执行水印植入流程
        
        :param poison_rate: 投毒比例，推荐 0.05~0.2
        :param target_label: 后门触发目标标签
        :param kwargs: 其他攻击方法特定参数
        :return: (水印数据集路径, 版权密钥路径)
        """
        logger.info(f"开始植入水印 | 数据集: {self.dataset_name or 'custom'} | 算法: {self.attack_name} | 投毒比例: {poison_rate}")
        
        # 1. 生成触发器
        self.attack_adapter.gen_trigger(**kwargs)
        
        # 2. 执行数据投毒
        poison_data, clean_data = self.attack_adapter.poison(
            dataset=self.dataset,
            poison_rate=poison_rate,
            target_label=target_label
        )
        
        # 3. 保存带水印数据集
        data_path = os.path.join(self.data_save_dir, "poisoned_dataset.pt")
        clean_path = os.path.join(self.data_save_dir, "clean_dataset.pt")
        torch.save(poison_data, data_path)
        torch.save(clean_data, clean_path)
        
        # 4. 保存版权密钥
        attack = self.attack_adapter.get_attack()
        key_info = {
            "type": self.attack_name,
            "dataset": self.dataset_name or "custom",
            "poison_rate": poison_rate,
            "target_label": target_label,
            "image_size": attack.image_size,
            "trigger_info": attack.trigger_info
        }
        
        key_filename = f"{self.dataset_name or 'custom'}_{self.attack_name}_key.yaml"
        key_path = os.path.join(self.key_save_dir, key_filename)
        with open(key_path, "w", encoding="utf-8") as f:
            yaml.dump(key_info, f, allow_unicode=True)
        
        logger.info(f"水印植入完成！")
        logger.info(f"  投毒数据集: {data_path}")
        logger.info(f"  干净数据集: {clean_path}")
        logger.info(f"  版权密钥: {key_path}")
        
        return data_path, key_path


class DatasetWatermarkEmbed(WatermarkEmbed):
    """兼容旧接口的类"""
    def __init__(self, dataset_name: str, attack_method: str):
        super().__init__(attack_name=attack_method, dataset_name=dataset_name)
    
    def embed(self, poison_rate=0.1, target_label=0, **trigger_kwargs):
        return self.run(poison_rate, target_label, **trigger_kwargs)