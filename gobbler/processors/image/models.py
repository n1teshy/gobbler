import json
from typing import Optional

from gobbler.models.utils import ClipScene
from gobbler.processors.interfaces import BaseFile, DBEntity


class Image(DBEntity, BaseFile):
    shape: str
    scene: Optional[ClipScene] = None
    description: str
    keywords: list[str] = []

    class Config:
        extra = "allow"

    def to_json(self) -> dict:
        return json.loads(super().model_dump_json())
