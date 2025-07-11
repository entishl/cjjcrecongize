import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple

from recognizer.utils import load_config
from recognizer.extractor import FeatureExtractor

def batch_generator(items: List, batch_size: int):
    """一个简单的批次生成器"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def build_database(batch_size: int = 32):
    """
    构建特征数据库。
    遍历头像库中每个角色（子目录），提取其所有图片的特征向量，
    并将其保存到 .npy 文件中。
    """
    print("开始构建特征数据库...")

    # 1. 加载配置和初始化提取器
    config = load_config()
    extractor = FeatureExtractor(model_name=config.model.name)

    avatar_dir = config.paths.avatar_library
    if not os.path.isdir(avatar_dir):
        print(f"错误: 头像库目录 '{avatar_dir}' 不存在。")
        return

    # 2. 收集所有角色的图片路径
    character_images = {}
    total_images = 0
    for char_name in os.listdir(avatar_dir):
        char_dir = os.path.join(avatar_dir, char_name)
        if os.path.isdir(char_dir):
            image_paths = [os.path.join(char_dir, f) for f in os.listdir(char_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_paths:
                character_images[char_name] = image_paths
                total_images += len(image_paths)

    if not character_images:
        print(f"错误: 在 '{avatar_dir}' 的任何子目录中都未找到图像文件。")
        return

    print(f"找到 {len(character_images)} 个角色，共 {total_images} 张图片。开始提取特征...")

    # 3. 为每个角色提取特征
    feature_database: Dict[str, List[np.ndarray]] = {}
    
    for char_name, image_paths in tqdm(character_images.items(), desc="处理角色"):
        char_features = []
        
        # 为当前角色的所有图片分批处理
        for batch_paths in batch_generator(image_paths, batch_size):
            images = []
            for img_path in batch_paths:
                # 读取图像 (使用imdecode处理中文路径)
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"警告: 无法读取图像 {img_path}")
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
            
            if not images:
                continue
            
            # 批量提取特征
            features = extractor.extract(images)
            char_features.extend(features)
            
        if char_features:
            feature_database[char_name] = char_features

    # 4. 保存数据库
    db_path = config.paths.database
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    np.save(db_path, feature_database)

    print(f"\n特征数据库构建完成！")
    print(f"总共处理了 {len(feature_database)} 个角色，总计 {total_images} 个特征。")
    print(f"数据库已保存至: {db_path}")

if __name__ == "__main__":
    build_database()