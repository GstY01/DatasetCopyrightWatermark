import torch
import torch.nn.functional as F
import yaml
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from .base_attack import BaseAttack

try:
    from torchvision.transforms import ToPILImage, ToTensor
    to_pil = ToPILImage()
    to_tensor = ToTensor()
except ImportError:
    pass

class TrojanNN(BaseAttack):
    """
    TrojanNN: 通过优化生成一个能激活特定神经元的触发器图案。
    论文: "Trojaning Attack on Neural Networks" (Liu et al., NDSS 2018)
    参考 BackdoorBench: https://github.com/SCLBD/BackdoorBench
    核心步骤：
        1. 在预训练模型上，选择指定层的参数，找到最活跃的神经元（或指定索引）。
        2. 使用 PGD 优化生成一个掩码图像，使得模型在该层特定神经元上的输出接近目标值。
        3. 将该图案作为后门触发器（叠加到训练图像上）。
    """
    def __init__(self, image_size=32, target_label=0, device='cpu',
                 config_path=None,
                 pretrain_model_path=None, mask_path=None,
                 selected_layer_name=None, selected_layer_param_name=None,
                 num_neuron=1, neuron_target_values=10.0,
                 mask_update_iters=1000, eps=0.3, alpha=0.1, tolerance=1e-3):
        """
        Args:
            image_size: 图像尺寸（正方形）
            target_label: 后门目标标签
            device: 计算设备
            config_path: yaml 配置文件路径
            pretrain_model_path: 预训练模型路径（用于优化触发器）
            mask_path: 初始掩码图像路径（PIL 图像）
            selected_layer_name: 要攻击的层名（用于 forward hook）
            selected_layer_param_name: 用于计算神经元重要性的参数名称
            num_neuron: 选择的神经元数量（默认1）
            neuron_target_values: 神经元目标激活值（单个数值或列表）
            mask_update_iters: PGD 迭代次数
            eps: PGD 扰动幅度
            alpha: PGD 步长
            tolerance: 损失容忍度（提前停止）
        """
        super().__init__(image_size, target_label, device)
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            pretrain_model_path = config.get('pretrain_model_path', pretrain_model_path)
            mask_path = config.get('mask_path', mask_path)
            selected_layer_name = config.get('selected_layer_name', selected_layer_name)
            selected_layer_param_name = config.get('selected_layer_param_name', selected_layer_param_name)
            num_neuron = config.get('num_neuron', num_neuron)
            neuron_target_values = config.get('neuron_target_values', neuron_target_values)
            mask_update_iters = config.get('mask_update_iters', mask_update_iters)
            eps = config.get('eps', eps)
            alpha = config.get('alpha', alpha)
            tolerance = config.get('tolerance', tolerance)
            target_label = config.get('target_label', target_label)
            self.target_label = target_label

        self.pretrain_model_path = pretrain_model_path
        self.mask_path = mask_path
        self.selected_layer_name = selected_layer_name
        self.selected_layer_param_name = selected_layer_param_name
        self.num_neuron = num_neuron
        self.neuron_target_values = neuron_target_values
        self.mask_update_iters = mask_update_iters
        self.eps = eps
        self.alpha = alpha
        self.tolerance = tolerance

        self.trigger = None          # 生成的触发器图案 (C, H, W) 范围 [0,1]
        self.trigger_img_np = None   # 存储 numpy 格式（用于快速叠加）

    def _get_most_connected_neuron_idxes(self, model, param_name):
        """根据参数权重求和选择最活跃的神经元"""
        param = torch.abs(model.state_dict()[param_name])
        logging.info(f'parameter shape = {param.shape}')
        if param.dim() == 2:  # 全连接层
            pass
        elif param.dim() == 4:  # 卷积层
            param = torch.flatten(param, 2).sum(2)
        else:
            raise Exception("Only consider conv and linear layer")
        return torch.argsort(param.sum(0), descending=True)[:self.num_neuron]

    def _pgd_with_mask_to_selected_neuron(self, model, mask_tensor):
        """使用 PGD 优化 mask_tensor，使模型在 selected_layer 的输出接近目标值"""
        model.eval()
        model.to(self.device)
        # 深拷贝并保持非负区域（mask 中正区域可修改，负区域不动）
        keep_mask = (mask_tensor > 0).float().to(self.device)
        images = mask_tensor.clone().detach().to(self.device)
        ori_images = images.data.clone()

        # 注册 forward hook 获取目标层的输入
        activation = {}

        def hook_function(module, input, output):
            activation['value'] = input[0]  # 假设我们关心 input[0]
        target_module = dict(model.named_modules())[self.selected_layer_name]
        handle = target_module.register_forward_hook(hook_function)

        # 确保 neuron_target_values 是列表
        if isinstance(self.neuron_target_values, (int, float)):
            targets = [float(self.neuron_target_values)] * self.num_neuron
        else:
            targets = [float(v) for v in self.neuron_target_values]
            if len(targets) != self.num_neuron:
                raise ValueError("Length of neuron_target_values must equal num_neuron")

        for _ in tqdm(range(self.mask_update_iters), desc="PGD optimizing trigger"):
            images.requires_grad = True
            _ = model(images * keep_mask)
            layer_out = activation['value']  # (B, C, H, W) 或 (B, features)
            # 展平（如果是卷积，取所有空间位置的平均）
            if layer_out.dim() == 4:
                layer_out = layer_out.mean(dim=[2, 3])  # (B, C)
            # 计算损失：所选神经元的 MSE
            loss = 0.0
            for i, idx in enumerate(self.neuron_idxes):
                loss += ((layer_out[:, idx] - targets[i]) ** 2).mean()
            model.zero_grad()
            loss.backward()
            # PGD 更新
            adv_images = images - self.alpha * images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=0.0, max=1.0).detach_()
            if loss.item() < self.tolerance:
                break
        handle.remove()
        return images

    def generate_trigger(self):
        """生成 TrojanNN 触发器（需要预训练模型和掩码）"""
        # 1. 加载预训练模型
        from utils.aggregate_block.model_trainer_generate import generate_cls_model
        if self.pretrain_model_path is None or not os.path.exists(self.pretrain_model_path):
            raise FileNotFoundError(f"Pretrained model not found: {self.pretrain_model_path}")
        model = generate_cls_model(
            model_name="resnet18",  # 需要与预训练模型一致，实际应从配置读取
            num_classes=1000,
            image_size=self.image_size
        )
        state_dict = torch.load(self.pretrain_model_path, map_location="cpu")
        # 部分加载（例如忽略最后的分类层）
        from utils.aggregate_block.model_trainer_generate import partially_load_state_dict
        partially_load_state_dict(model, state_dict)
        model.to(self.device)

        # 2. 选择神经元索引
        self.neuron_idxes = self._get_most_connected_neuron_idxes(model, self.selected_layer_param_name)
        print(f"Selected neuron indices: {self.neuron_idxes}")

        # 3. 加载初始掩码并转为张量
        if self.mask_path is None or not os.path.exists(self.mask_path):
            raise FileNotFoundError(f"Mask image not found: {self.mask_path}")
        mask_pil = Image.open(self.mask_path).convert('RGB')
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        mask_tensor = transform(mask_pil).unsqueeze(0)  # (1, C, H, W)

        # 4. 执行 PGD 优化，生成触发图像
        optimized = self._pgd_with_mask_to_selected_neuron(model, mask_tensor)
        # optimized 形状 (1, C, H, W)，范围 [0,1]
        self.trigger = optimized.squeeze(0).detach().cpu()  # (C, H, W)
        self.trigger_info = {
            "type": "trojannn",
            "selected_layer": self.selected_layer_name,
            "selected_param": self.selected_layer_param_name,
            "num_neuron": self.num_neuron,
            "neuron_target_values": self.neuron_target_values,
            "mask_path": self.mask_path,
        }
        return self.trigger

    def poison_sample(self, img, label):
        """
        将生成的触发图案叠加到原始图像上（直接相加并裁剪）。
        img: (C, H, W) 范围 [0,1]
        """
        if self.trigger is None:
            self.generate_trigger()
        # 确保 trigger 与 img 在同一设备上
        trigger = self.trigger.to(img.device)
        poisoned = img + trigger
        poisoned = torch.clamp(poisoned, 0.0, 1.0)
        return poisoned, self.target_label