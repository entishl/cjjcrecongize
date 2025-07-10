import os
import cv2
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

from .utils import Config, load_coordinates
from .extractor import FeatureExtractor

class AvatarRecognizer:
    """
    核心识别器类，负责执行完整的识别流程。
    """
    def __init__(self, config: Config):
        """
        初始化识别器。

        Args:
            config (Config): 从 config.yaml 加载的配置对象。
        """
        self.config = config
        self.feature_extractor = FeatureExtractor(model_name=config.model.name)
        self.crop_regions = self._load_crop_regions()
        self.feature_database, self.db_character_names = self._load_database()

    def _load_crop_regions(self) -> List[Dict[str, Any]]:
        """加载并验证坐标文件。"""
        regions = load_coordinates(self.config.paths.coordinates)
        if not regions:
            raise ValueError(f"坐标文件 '{self.config.paths.coordinates}' 为空或加载失败。")
        return regions

    def _load_database(self) -> tuple[np.ndarray, List[str]]:
        """加载特征数据库，并将其分解为向量矩阵和名称列表以便快速计算。"""
        db_path = self.config.paths.database
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"特征数据库 '{db_path}' 不存在。\n"
                f"请先运行 'python build_db.py' 来生成它。"
            )
        
        database_dict: Dict[str, np.ndarray] = np.load(db_path, allow_pickle=True).item()
        
        if not database_dict:
            raise ValueError(f"特征数据库 '{db_path}' 为空。")

        # 将字典分解为并行的列表和矩阵，以便进行高效的矩阵运算
        character_names = list(database_dict.keys())
        feature_vectors = np.array(list(database_dict.values()))
        
        return feature_vectors, character_names

    def _crop_avatars(self, image: np.ndarray) -> List[np.ndarray]:
        """根据坐标模板从大图中裁剪所有头像。"""
        cropped_images = []
        for region in self.crop_regions:
            x, y, w, h = region['rect']
            cropped_img = image[y:y+h, x:x+w]
            cropped_images.append(cropped_img)
        return cropped_images

    def recognize(self, image_path: str) -> List[Dict[str, Any]]:
        """
        对单张游戏截图执行完整的识别流程。

        Args:
            image_path (str): 待识别的截图文件路径。

        Returns:
            List[Dict[str, Any]]: 一个包含每个位置识别结果的列表。
                                 每个结果是一个字典，例如:
                                 {'position': 'P1_T1_1', 'character': '白雪公主', 'similarity': 0.95}
        """
        # 1. 读取和预处理输入图像
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"输入图像 '{image_path}' 不存在。")
        
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. 裁剪所有头像
        cropped_avatars = self._crop_avatars(image_rgb)

        # 3. 批量提取查询特征
        query_features = self.feature_extractor.extract(cropped_avatars)

        # 4. 计算相似度矩阵
        # query_features: (N, D), self.feature_database: (M, D)
        # similarity_matrix: (N, M)
        similarity_matrix = cosine_similarity(query_features, self.feature_database)

        # 5. 为每个查询找到最佳匹配
        best_match_indices = np.argmax(similarity_matrix, axis=1)
        best_match_scores = np.max(similarity_matrix, axis=1)

        # 6. 格式化结果
        results = []
        threshold = self.config.recognition.similarity_threshold
        for i, region in enumerate(self.crop_regions):
            score = best_match_scores[i]
            if score >= threshold:
                char_name = self.db_character_names[best_match_indices[i]]
            else:
                char_name = "未知"

            results.append({
                "position": region['name'],
                "character": char_name,
                "similarity": float(score)
            })
            
        return results