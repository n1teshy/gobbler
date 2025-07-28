import json
from typing import Optional

from core.processors.image.utils import SceneType
from core.processors.interfaces import BaseFile, DBEntity


class Image(DBEntity, BaseFile):
    shape: str
    scene: Optional[SceneType] = None
    description: str
    keywords: list[str] = []

    class Config:
        extra = "allow"

    def to_json(self) -> dict:
        return json.loads(super().model_dump_json())
