import torch
import timm
import numpy as np
from PIL import Image
from typing import List
import torch.nn.functional as F

class FeatureExtractor:
    """
    封装了timm模型，用于从图像中提取高维特征向量。
    """
    def __init__(self, model_name: str, device: str = None):
        """
        初始化特征提取器。

        Args:
            model_name (str): 要加载的timm模型名称。
            device (str, optional): 计算设备 ('cuda' or 'cpu')。如果为None，则自动检测。
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # 加载预训练模型
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 获取模型的输入配置，用于图像预处理
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

    @torch.no_grad()
    def extract(self, images: List[np.ndarray]) -> np.ndarray:
        """
        从一批图像中提取特征向量。

        Args:
            images (List[np.ndarray]): 一个包含多个Numpy格式图像(H, W, C)的列表。

        Returns:
            np.ndarray: 一个形状为 (N, D) 的Numpy数组，其中N是图像数量，D是特征维度。
        """
        # 将Numpy数组转换为PIL图像，然后应用预处理转换
        pil_images = [Image.fromarray(img) for img in images]
        tensor_batch = torch.stack([self.transform(p_img) for p_img in pil_images]).to(self.device)

        # 提取特征
        features = self.model(tensor_batch)

        # L2 归一化
        normalized_features = F.normalize(features, p=2, dim=1)

        # 将结果移回CPU并转换为Numpy数组
        return normalized_features.cpu().numpy()