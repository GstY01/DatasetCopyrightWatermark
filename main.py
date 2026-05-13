import torch
import os
from data.dataset_loader import load_cifar10, load_custom_dataset
from core.embed import WatermarkEmbed
from core.verify import CopyrightVerify


def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # ===================== 1. 水印植入=====================
    print("\n" + "="*50)
    print("步骤1: 水印植入")
    print("="*50)
    
    # 加载数据集（CIFAR-10）
    dataset = load_cifar10(train=True)
    print(f"加载数据集 CIFAR-10，样本数: {len(dataset)}")
    
    # 选择攻击方法
    attack_names = ["badnets", "wanet"]
    
    # 批量植入不同类型水印
    for attack_name in attack_names:
        print(f"\n正在使用 {attack_name} 植入水印...")
        embedder = WatermarkEmbed(attack_name, dataset, "cifar10", device)
        data_path, key_path = embedder.run(
            poison_rate=0.1,  # 投毒比例
            target_label=0
        )

    # ===================== 2. 版权验证=====================
    print("\n" + "="*50)
    print("步骤2: 版权验证")
    print("="*50)
    
    # 验证BadNets水印
    badnets_key_path = "./output/keys/cifar10_badnets_key.yaml"
    badnets_data_path = "./output/poisoned/cifar10_badnets/poisoned_dataset.pt"
    
    if os.path.exists(badnets_key_path) and os.path.exists(badnets_data_path):
        print("\n验证 BadNets 水印:")
        verifier = CopyrightVerify(badnets_key_path, device)
        verifier.check_infringement(badnets_data_path, threshold=0.8)
    
    # 验证WaNet水印
    wanet_key_path = "./output/keys/cifar10_wanet_key.yaml"
    wanet_data_path = "./output/poisoned/cifar10_wanet/poisoned_dataset.pt"
    
    if os.path.exists(wanet_key_path) and os.path.exists(wanet_data_path):
        print("\n验证 WaNet 水印:")
        verifier = CopyrightVerify(wanet_key_path, device)
        verifier.check_infringement(wanet_data_path, threshold=0.8)

    print("\n" + "="*50)
    print("完成!")
    print("="*50)


if __name__ == "__main__":
    main()