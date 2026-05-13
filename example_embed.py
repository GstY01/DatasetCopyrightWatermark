"""
示例脚本：水印植入
演示如何对数据集植入多种类型的版权水印
"""

import torch
from data.dataset_loader import load_cifar10
from core.embed import WatermarkEmbed


def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载 CIFAR-10 数据集
    print("正在加载 CIFAR-10 数据集...")
    dataset = load_cifar10(train=True)
    print(f"数据集大小: {len(dataset)}")
    
    # 要使用的攻击方法
    attack_methods = ["badnets", "wanet", "blended", "ssba"]
    
    # 对每种攻击方法植入水印
    for attack_name in attack_methods:
        print(f"\n{'='*60}")
        print(f"使用 {attack_name} 植入水印")
        print('='*60)
        
        embedder = WatermarkEmbed(
            attack_name=attack_name,
            dataset=dataset,
            dataset_name="cifar10",
            device=device
        )
        
        data_path, key_path = embedder.run(
            poison_rate=0.1,
            target_label=0
        )
        
        print(f"水印植入成功！")
        print(f"  数据集: {data_path}")
        print(f"  密钥: {key_path}")


if __name__ == "__main__":
    main()
