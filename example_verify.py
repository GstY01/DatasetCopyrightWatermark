"""
示例脚本：版权验证
演示如何验证数据集是否包含特定水印
"""

import os
import torch
from core.verify import CopyrightVerify


def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 要验证的攻击方法
    attack_methods = ["badnets", "wanet"]
    
    for attack_name in attack_methods:
        print(f"\n{'='*60}")
        print(f"验证 {attack_name} 水印")
        print('='*60)
        
        # 构造路径
        key_path = f"./output/keys/cifar10_{attack_name}_key.yaml"
        data_path = f"./output/poisoned/cifar10_{attack_name}/poisoned_dataset.pt"
        
        # 检查文件是否存在
        if not os.path.exists(key_path):
            print(f"密钥文件不存在: {key_path}")
            continue
        if not os.path.exists(data_path):
            print(f"数据文件不存在: {data_path}")
            continue
        
        # 验证
        verifier = CopyrightVerify(key_path, device)
        is_infringed, asr = verifier.check_infringement(
            data_path,
            threshold=0.8
        )


if __name__ == "__main__":
    main()
