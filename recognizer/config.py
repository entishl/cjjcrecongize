import yaml
from dataclasses import dataclass, field
from typing import Dict, Any

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
class Config:
    model: ModelConfig
    paths: PathsConfig
    recognition: RecognitionConfig

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Config':
        return Config(
            model=ModelConfig(**data['model']),
            paths=PathsConfig(**data['paths']),
            recognition=RecognitionConfig(**data['recognition'])
        )

def load_config(path: str = 'config.yaml') -> Config:
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return Config.from_dict(config_dict)