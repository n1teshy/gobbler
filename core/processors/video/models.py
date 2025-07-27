from typing import Optional

from pydantic import BaseModel

from core.processors.image.models import Image
from core.processors.interfaces import BaseFile, DBEntity


class Span(DBEntity, BaseFile):
    start: float
    end: float
    short_description: str
    long_description: str
    frames: list[Image] = []
    keywords: list[str] = []

    def to_json(self) -> dict:
        return {
            **DBEntity.to_json(self),
            **BaseFile.to_json(self),
            "start": self.start,
            "end": self.end,
            "short_description": self.short_description,
            "long_description": self.long_description,
            "frames": [frame.to_json() for frame in self.frames],
            "keywords": self.keywords,
        }
