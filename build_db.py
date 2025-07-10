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
    遍历头像库，提取每个头像的特征向量，并将其保存到 .npy 文件中。
    """
    print("开始构建特征数据库...")
    
    # 1. 加载配置和初始化提取器
    config = load_config()
    extractor = FeatureExtractor(model_name=config.model.name)
    
    avatar_dir = config.paths.avatar_library
    if not os.path.isdir(avatar_dir):
        print(f"错误: 头像库目录 '{avatar_dir}' 不存在。")
        return

    # 2. 收集所有头像文件路径
    image_paths = [os.path.join(avatar_dir, f) for f in os.listdir(avatar_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print(f"错误: 在 '{avatar_dir}' 中未找到任何图像文件。")
        return

    print(f"找到 {len(image_paths)} 个头像，开始提取特征...")
    
    feature_database: Dict[str, np.ndarray] = {}
    
    # 3. 分批处理
    for batch_paths in tqdm(list(batch_generator(image_paths, batch_size)), desc="提取特征"):
        images = []
        character_names = []
        
        for img_path in batch_paths:
            # 从文件名获取角色名
            char_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # 读取图像 (使用imdecode处理中文路径)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
            character_names.append(char_name)
            
        if not images:
            continue
            
        # 4. 批量提取特征
        features = extractor.extract(images)
        
        # 5. 存入字典
        for name, feature in zip(character_names, features):
            feature_database[name] = feature
            
    # 6. 保存数据库
    db_path = config.paths.database
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    np.save(db_path, feature_database)
    
    print(f"\n特征数据库构建完成！")
    print(f"总共处理了 {len(feature_database)} 个角色。")
    print(f"数据库已保存至: {db_path}")

if __name__ == "__main__":
    build_database()