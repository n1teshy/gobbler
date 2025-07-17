from typing import Optional

from pydantic import BaseModel

from core.processors.image.utils import SceneType


class Image(BaseModel):
    URI: str
    scene: Optional[SceneType] = None
    description: str

    def to_json(self) -> dict:
        return {
            "URI": self.URI,
            "scene": self.scene and self.scene.value,
            "description": self.description,
        }
