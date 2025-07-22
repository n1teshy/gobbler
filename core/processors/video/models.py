from typing import Optional

from pydantic import BaseModel

from core.processors.image.models import Image
from core.processors.interfaces import BaseFile


class Span(BaseModel):
    start: float
    end: float
    short_description: Optional[str] = None
    long_description: str
    frames: list[Image] = []

    def to_json(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "short_description": self.short_description,
            "long_description": self.long_description,
            "frames": [frame.to_json() for frame in self.frames],
        }


class Video(BaseFile):
    duration: float
    spans: list[Span]

    def to_json(self) -> dict:
        data = super().to_json()
        return {
            **data,
            "duration": self.duration,
            "spans": [span.to_json() for span in self.spans],
        }
