from typing import Optional

from core.bases import BaseFile
from core.processors.image.utils import SceneType


class Image(BaseFile):
    shape: str
    scene: Optional[SceneType] = None
    description: str

    def to_json(self) -> dict:
        data = super().to_json()
        return {
            **data,
            "shape": self.shape,
            "scene": self.scene and self.scene.value,
            "description": self.description,
        }
