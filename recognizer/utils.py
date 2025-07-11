import yaml
import json
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont

@dataclass
class ModelConfig:
    name: str

@dataclass
class PathsConfig:
    database: str
    coordinates: str
    avatar_library: str
    output_dir: str

@dataclass
class RecognitionConfig:
    similarity_threshold: float

@dataclass
class PreprocessingConfig:
    target_width: int
    target_height: int
    tolerance: float

@dataclass
class Config:
    model: ModelConfig
    paths: PathsConfig
    recognition: RecognitionConfig
    preprocessing: PreprocessingConfig

def load_config(path: str = 'config.yaml') -> Config:
    """从 YAML 文件加载配置并解析为嵌套的 Config 对象。"""
    with open(path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    return Config(
        model=ModelConfig(**config_data['model']),
        paths=PathsConfig(**config_data['paths']),
        recognition=RecognitionConfig(**config_data['recognition']),
        preprocessing=PreprocessingConfig(**config_data['preprocessing'])
    )

def load_coordinates(path: str) -> List[Dict[str, Any]]:
    """从 JSON 文件加载坐标模板。"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """尝试加载系统中的中文字体，失败则返回Pillow默认字体。"""
    try:
        # 优先使用微软雅黑 (Windows)
        return ImageFont.truetype("msyh.ttc", size, encoding="utf-8")
    except IOError:
        try:
            # 备选黑体 (Windows)
            return ImageFont.truetype("simhei.ttf", size, encoding="utf-8")
        except IOError:
            print("警告: 未找到中文字体 (msyh.ttc, simhei.ttf)，将使用默认英文字体。")
            # Mac/Linux用户可能需要修改为 'PingFang.ttc' 或其他字体
            return ImageFont.load_default()

def visualize_results(image_bgr: np.ndarray, results: List[Dict[str, Any]], regions: List[Dict[str, Any]]) -> np.ndarray:
    """
    在输入图像上绘制识别结果，支持中文显示。
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    font = _get_font(size=20)
    pos_to_rect = {r['name']: r['rect'] for r in regions}

    for res in results:
        pos_name = res['position']
        char_name = res['character']
        similarity = res['similarity']
        
        if char_name == "未知":
            color = (255, 0, 0)  # 红色 (RGB)
            display_text = "Unknown"
        else:
            color = (0, 255, 0)  # 绿色 (RGB)
            display_text = f"{char_name} ({similarity:.2f})"

        if pos_name in pos_to_rect:
            x, y, w, h = pos_to_rect[pos_name]
            
            draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=2)
            
            text_bbox = draw.textbbox((0, 0), display_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x = x
            text_y = y - text_height - 5 if y - text_height - 5 > 0 else y + h + 5
            
            draw.rectangle([(text_x, text_y), (text_x + text_width, text_y + text_height)], fill=color)
            draw.text((text_x, text_y), display_text, font=font, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def preprocess_image(image_path: str, config: PreprocessingConfig) -> np.ndarray:
    """
    预处理输入图像，检查尺寸和比例，并根据需要进行缩放。

    :param image_path: 输入图像的文件路径。
    :param config: 包含目标宽度、高度和容差的预处理配置。
    :return: 返回处理后的图像 (BGR格式的numpy数组)。
    :raises ValueError: 如果图像比例与目标比例不符。
    """
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    h, w = image.shape[:2]

    target_ratio = config.target_width / config.target_height
    image_ratio = w / h

    if abs(image_ratio - target_ratio) > config.tolerance * target_ratio:
        raise ValueError(
            f"图像尺寸比例不正确。期望比例: {target_ratio:.2f}, "
            f"实际比例: {image_ratio:.2f} (来自 {w}x{h})"
        )

    if (w, h) != (config.target_width, config.target_height):
        print(f"信息: 图像将从 {w}x{h} 缩放至 {config.target_width}x{config.target_height}")
        image = cv2.resize(image, (config.target_width, config.target_height), interpolation=cv2.INTER_AREA)

    return image