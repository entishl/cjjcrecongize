import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from .utils import Config, load_coordinates, preprocess_image
from .extractor import FeatureExtractor

def check_win_loss_colors(image_rgb: np.ndarray, coordinates: List[Tuple[int, int]]) -> List[str]:
    """
    检查指定坐标点的颜色，以确定是“赢” (W) 还是“输” (L)。

    Args:
        image_rgb (np.ndarray): RGB 格式的输入图像。
        coordinates (List[Tuple[int, int]]): 要检查的 (x, y) 坐标列表。

    Returns:
        List[str]: 每个坐标点的结果列表 ("W", "L", 或 "ERROR")。
    """
    results = []
    height, width, _ = image_rgb.shape
    for x, y in coordinates:
        if 0 <= y < height and 0 <= x < width:
            # 注意：OpenCV 读取的图像坐标是 (y, x)
            r, g, b = image_rgb[y, x]
            if r > b and r > 100:
                results.append("L")
            elif b > r and b > 100:
                results.append("W")
            else:
                results.append("ERROR")
        else:
            results.append("ERROR") # 坐标越界
    return results


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
        self.flat_feature_database, self.db_character_map, self.unique_character_names = self._load_database()

    def _load_crop_regions(self) -> List[Dict[str, Any]]:
        """加载并验证坐标文件。"""
        regions = load_coordinates(self.config.paths.coordinates)
        if not regions:
            raise ValueError(f"坐标文件 '{self.config.paths.coordinates}' 为空或加载失败。")
        return regions

    def _load_database(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        加载特征数据库，并将其扁平化为向量矩阵和角色映射表以便快速计算。
        """
        db_path = self.config.paths.database
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"特征数据库 '{db_path}' 不存在。\n"
                f"请先运行 'python build_db.py' 来生成它。"
            )
        
        # 加载的数据结构为: Dict[角色名, List[特征向量]]
        database_dict: Dict[str, List[np.ndarray]] = np.load(db_path, allow_pickle=True).item()
        
        if not database_dict:
            raise ValueError(f"特征数据库 '{db_path}' 为空。")

        # 将字典扁平化，以便进行高效的矩阵运算
        flat_features = []
        character_map = []
        unique_character_names = sorted(database_dict.keys())

        for char_name in unique_character_names:
            features = database_dict[char_name]
            flat_features.extend(features)
            character_map.extend([char_name] * len(features))
        
        if not flat_features:
             raise ValueError(f"特征数据库 '{db_path}' 中不包含任何特征向量。")

        return np.array(flat_features), character_map, unique_character_names

    def _crop_avatars(self, image: np.ndarray) -> List[np.ndarray]:
        """根据坐标模板从大图中裁剪所有头像。"""
        cropped_images = []
        for region in self.crop_regions:
            x1, y1, x2, y2 = region['rect']
            cropped_img = image[y1:y2, x1:x2]
            cropped_images.append(cropped_img)
        return cropped_images

    def recognize(self, image_path: str) -> Tuple[List[Dict[str, Any]], List[str], List[np.ndarray], np.ndarray]:
        """
        对单张游戏截图执行完整的识别流程。

        Args:
            image_path (str): 待识别的截图文件路径。

        Returns:
            Tuple[List[Dict[str, Any]], List[str], List[np.ndarray], np.ndarray]:
                - results (List[Dict[str, Any]]): 包含每个位置识别结果的列表。
                - win_loss_results (List[str]): 包含5个W/L判断结果的列表。
                - cropped_avatars (List[np.ndarray]): 裁剪出的头像图像列表。
                - processed_image (np.ndarray): 经过预处理（可能缩放）的BGR图像。
        """
        # 1. 读取和预处理输入图像
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"输入图像 '{image_path}' 不存在。")
        
        # 使用新的预处理函数
        image_bgr = preprocess_image(image_path, self.config.preprocessing)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 2. 检查 W/L 颜色
        win_loss_coords = [
            (836, 825), (836, 891), (836, 957), (836, 1023), (836, 1089)
        ]
        win_loss_results = check_win_loss_colors(image_rgb, win_loss_coords)

        # 3. 裁剪所有头像
        cropped_avatars = self._crop_avatars(image_rgb)

        # 4. 批量提取查询特征
        query_features = self.feature_extractor.extract(cropped_avatars)

        # 5. 计算相似度矩阵
        # 结果矩阵的维度: (N_queries, N_total_db_features)
        similarity_matrix = cosine_similarity(query_features, self.flat_feature_database)

        # 6. 为每个查询找到最佳匹配 (最大相似度策略)
        # 使用 pandas 进行高效的分组和最大值查找
        df = pd.DataFrame(similarity_matrix, columns=self.db_character_map)
        
        # 按列名（角色名）分组，并找到每组的最大值
        # 结果是一个 (N_queries, N_unique_characters) 的 DataFrame
        max_similarity_per_char = df.groupby(axis=1, by=df.columns).max()
        
        # 从结果中找到每个查询（行）的最佳匹配角色和分数
        best_match_scores = max_similarity_per_char.max(axis=1).values
        best_match_chars = max_similarity_per_char.idxmax(axis=1).values

        # 7. 格式化结果
        results = []
        threshold = self.config.recognition.similarity_threshold
        for i, region in enumerate(self.crop_regions):
            score = best_match_scores[i]
            if score >= threshold:
                char_name = best_match_chars[i]
            else:
                char_name = "未知"
            
            # 提取该查询的所有角色分数，并按分数降序排序
            all_scores_for_query = max_similarity_per_char.iloc[i].to_dict()
            sorted_scores = dict(sorted(all_scores_for_query.items(), key=lambda item: item[1], reverse=True))

            results.append({
                "position": region['name'],
                "character": char_name,
                "similarity": float(score),
                "all_scores": sorted_scores
            })
            
        return results, win_loss_results, cropped_avatars, image_bgr