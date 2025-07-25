from typing import Optional

from core.processors.image.utils import SceneType, scene_type_to_idx
from core.processors.interfaces import BaseFile, DBEntity


class Image(DBEntity, BaseFile):
    shape: str
    scene: Optional[SceneType] = None
    description: str
    keywords: list[str] = []

    def to_json(self) -> dict:
        return {
            **DBEntity.to_json(self),
            **BaseFile.to_json(self),
            "shape": self.shape,
            "scene": self.scene and scene_type_to_idx(self.scene),
            "description": self.description,
            "keywords": self.keywords,
        }
