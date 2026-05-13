import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import (
    load_cifar10,
    load_cifar100,
    load_gtsrb,
    load_custom_dataset
)


class DatasetLoader:
    """统一数据集加载器"""
    
    @staticmethod
    def load(dataset_name, root='./data', train=True, download=True, **kwargs):
        """
        根据数据集名称加载对应数据集
        
        Args:
            dataset_name: 数据集名称，支持 'cifar10', 'cifar100', 'gtsrb', 'custom'
            root: 数据存储根目录
            train: 是否加载训练集
            download: 是否自动下载
            **kwargs: 其他参数，如自定义数据集的 data_dir
        
        Returns:
            dataset: PyTorch Dataset 实例
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name == 'cifar10':
            return load_cifar10(train=train, root=root, download=download)
        elif dataset_name == 'cifar100':
            return load_cifar100(train=train, root=root, download=download)
        elif dataset_name == 'gtsrb':
            return load_gtsrb(train=train, root=root, download=download)
        elif dataset_name == 'custom':
            data_dir = kwargs.get('data_dir', './data/custom')
            transform = kwargs.get('transform', None)
            return load_custom_dataset(data_dir, transform)
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}，请使用: cifar10, cifar100, gtsrb, custom")
